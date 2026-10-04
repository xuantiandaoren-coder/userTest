"""扫描版 PDF 的 OCR 兜底：逐页渲染成图片，交给视觉 OCR 模型识别。

对外只暴露 ``ocr_pdf``，内部三步（均为本文件私有实现）：

1. ``pypdfium2`` 按 ``OCR_RENDER_DPI`` 把每页渲染成 PNG 位图
2. PNG 转 base64 data URI，和提示词一起上送视觉 OCR 模型（OpenAI 兼容协议，
   默认是 MaaS 上的 ``vanchin/deepseek-ocr``）
3. 按页序、按行把识别结果拼接成全文

配置统一走 ``app.core.config.settings``：
- ``dashscope_api_key``：OCR 模型 Key（与文本向量化共用同一个 DashScope Key）
- ``ocr_base_url`` / ``ocr_model`` / ``ocr_timeout``：OCR 服务地址、模型名、单次请求超时
- ``ocr_render_dpi`` / ``ocr_max_pages``：渲染清晰度与单文档识别页数上限

返回形态兼容两种：网关直接给纯文本就按纯文本用；给 JSON（形如
``{"data": {"line_texts": [...]}}``）则取 ``line_texts`` 按行拼接。
"""

from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium
from openai import OpenAI

from app.core.config import settings
from app.core.exceptions import SystemError

logger = logging.getLogger("app.rag.ocr")

__all__ = ["ocr_pdf"]

# 识别提示词：只要求读出图里的文字，避免模型补解释性文字污染正文
OCR_PROMPT = "Read all the text in the image."

# PDF 用户空间单位是 1/72 英寸，DPI 需要换算成渲染缩放倍数
POINTS_PER_INCH = 72


class OcrNotConfiguredError(SystemError):
    """系统异常：OCR 未配置（缺 Key）。与向量化共用 Key，语义等同 RAG 未配置。"""

    code = "RAG_NOT_CONFIGURED"
    message = "向量化服务未配置，请联系管理员"


class OcrError(SystemError):
    """系统异常：PDF 渲染或 OCR 调用失败。"""

    code = "OCR_FAILED"
    message = "扫描件识别失败，请稍后重试"


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------
def ocr_pdf(source: str | Path | bytes) -> str:
    """把扫描版 PDF 识别成全文文本。

    入参 ``source`` 支持文件路径与 PDF 字节内容（上传链路手上只有 bytes）。
    返回按页序、按行拼接的全文；一页都没识别出内容时返回空串。

    用法::

        ocr_pdf("/data/scanned.pdf")
        ocr_pdf(pdf_bytes)
    """
    images = _render_pages(_read_bytes(source))
    if not images:
        return ""

    client = _build_client()
    lines: list[str] = []
    for page_number, png in enumerate(images, start=1):
        page_text = _recognize(client, png)
        if not page_text:
            logger.warning("OCR 第 %s 页未识别出文本", page_number)
            continue
        lines.extend(_split_lines(page_text))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 渲染：pypdfium2 逐页 -> PNG
# ---------------------------------------------------------------------------
def _read_bytes(source: str | Path | bytes) -> bytes:
    """路径或字节内容统一成 bytes（pypdfium2 只接受 bytes，bytearray 会报错）。"""
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    return Path(source).read_bytes()


def _render_pages(data: bytes) -> list[bytes]:
    """PDF 每页 -> PNG 字节；超过 ``ocr_max_pages`` 时只渲染前 N 页。"""
    scale = max(settings.ocr_render_dpi, 1) / POINTS_PER_INCH
    limit = max(settings.ocr_max_pages, 1)

    try:
        document = pdfium.PdfDocument(data)
    except Exception as exc:  # noqa: BLE001 - PDF 损坏 / 加密统一归为识别失败
        raise OcrError(detail=f"PDF 打开失败：{exc}") from exc

    images: list[bytes] = []
    with document:
        page_count = len(document)
        if page_count > limit:
            logger.warning(
                "PDF 共 %s 页，超过 OCR 单文档上限 %s，只识别前 %s 页",
                page_count,
                limit,
                limit,
            )
        for index in range(min(page_count, limit)):
            page = document[index]
            try:
                bitmap = page.render(scale=scale)
                try:
                    images.append(_to_png(bitmap))
                finally:
                    bitmap.close()  # 单页位图按 150DPI 也有几 MB，逐页释放
            finally:
                page.close()
    return images


def _to_png(bitmap: pdfium.PdfBitmap) -> bytes:
    """位图 -> PNG 字节（PNG 无损，扫描件文字边缘不会二次劣化）。"""
    image = bitmap.to_pil()
    buffer = io.BytesIO()
    try:
        image.save(buffer, format="PNG")
    finally:
        image.close()
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# 识别：单页 PNG -> 文本
# ---------------------------------------------------------------------------
def _build_client() -> OpenAI:
    """构造 OCR 客户端（未发请求，纯本地行为）。"""
    api_key = settings.dashscope_api_key.get_secret_value().strip()
    if not api_key:
        raise OcrNotConfiguredError(
            detail="缺少 dashscope_api_key，OCR 无法调用；请在 .env / 环境变量中配置 DASHSCOPE_API_KEY",
        )
    return OpenAI(
        api_key=api_key,
        base_url=settings.ocr_base_url,
        timeout=settings.ocr_timeout,
        max_retries=2,
    )


def _recognize(client: OpenAI, png: bytes) -> str:
    """单页图片 -> 识别文本（一页一次请求，便于按页定位失败）。"""
    content = [
        {"type": "image_url", "image_url": {"url": _to_data_uri(png), "detail": "high"}},
        {"type": "text", "text": OCR_PROMPT},
    ]
    try:
        completion = client.chat.completions.create(
            model=settings.ocr_model,
            messages=[{"role": "user", "content": content}],
        )
    except Exception as exc:  # noqa: BLE001 - 网络 / 鉴权 / 配额统一归为识别失败
        raise OcrError(detail=f"OCR 调用失败：{exc}") from exc

    raw = (completion.choices[0].message.content or "").strip()
    return _extract_text(raw)


def _to_data_uri(png: bytes) -> str:
    """PNG -> ``data:image/png;base64,...``，兼容模式按 data URI 收图。"""
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


# ---------------------------------------------------------------------------
# 结果整形
# ---------------------------------------------------------------------------
def _extract_text(raw: str) -> str:
    """统一成纯文本：网关给纯文本就原样用，给 JSON 就取 ``line_texts``。"""
    if not raw.startswith("{"):
        return raw
    try:
        payload: Any = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if not isinstance(payload, dict):
        return raw

    line_texts = _line_texts(payload)
    return "\n".join(line_texts) if line_texts else raw


def _line_texts(payload: dict[str, Any]) -> list[str]:
    """取 ``{"line_texts": [...]}`` 或 ``{"data": {"line_texts": [...]}}`` 里的行文本。"""
    for candidate in (payload.get("data"), payload):
        if isinstance(candidate, dict):
            lines = candidate.get("line_texts")
            if isinstance(lines, list):
                return [str(line).strip() for line in lines if str(line).strip()]
    return []


def _split_lines(text: str) -> list[str]:
    """按行拆开并丢掉空行，保证不同页 / 不同行之间不粘在一起。"""
    return [line.strip() for line in text.splitlines() if line.strip()]
