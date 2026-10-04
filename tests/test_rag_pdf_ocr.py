"""PDF 解析链路：pypdfium2 抽文本层、扫描件判定、OCR 兜底（app/rag/ocr.py）。

用例不连真实 OCR 服务：渲染 / 识别 / 客户端分别用假实现替换，只验证判定口径、
图片送法（base64 data URI）与按页按行拼接；真实调用由手工联调覆盖。
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw, ImageFont
from pydantic import SecretStr

from app.core.config import settings
from app.rag import core as rag_core
from app.rag import ocr as rag_ocr

# 手写的最小文本 PDF：一页，带真实文本层（"Hello pypdfium2 text layer 12345"）
TEXT_PDF_BYTES = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj
4 0 obj << /Length 90 >> stream
BT /F1 18 Tf 20 150 Td (Hello pypdfium2 text layer 12345) Tj ET
endstream
endobj
5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
trailer << /Root 1 0 R >>
%%EOF
"""


def _page_image(text: str) -> Image.Image:
    image = Image.new("RGB", (1240, 300), "white")
    ImageDraw.Draw(image).text(
        (40, 80), text, fill="black", font=ImageFont.load_default(size=48)
    )
    return image


def _scanned_pdf_bytes(*pages: str) -> bytes:
    """造"扫描件"：每页只有图片、没有文本层。不传页内容时造一页。"""
    images = [_page_image(text) for text in (pages or ("Scanned Resume Sample",))]
    buffer = BytesIO()
    images[0].save(
        buffer,
        format="PDF",
        resolution=150,
        save_all=True,
        append_images=images[1:],
    )
    for image in images:
        image.close()
    return buffer.getvalue()


class _FakeCompletions:
    """假 completions：记录调用参数，返回预设的 message.content。"""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeClient:
    """假 OpenAI 客户端（只用到 chat.completions.create）。"""

    def __init__(self, reply: str) -> None:
        self.completions = _FakeCompletions(reply)
        self.chat = SimpleNamespace(completions=self.completions)


@pytest.fixture()
def fake_ocr(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """替换 core 里的 OCR 入口，记录传进去的 PDF 字节。"""
    calls: list[bytes] = []

    def _fake_ocr_pdf(source: str | Path | bytes) -> str:
        calls.append(source if isinstance(source, bytes) else source.encode())
        return "OCR 识别结果"

    monkeypatch.setattr(rag_core, "ocr_pdf", _fake_ocr_pdf)
    return SimpleNamespace(calls=calls)


# ---------------------------------------------------------------------------
# 判定：电子版走文本层，扫描件走 OCR 兜底
# ---------------------------------------------------------------------------
def test_extract_pdf_text_returns_text_and_page_count() -> None:
    """文本层抽取返回（全文, 页数），页数用来算平均每页字符数。"""
    text, page_count = rag_core._extract_pdf_text(TEXT_PDF_BYTES)

    assert page_count == 1
    assert text.strip() == "Hello pypdfium2 text layer 12345"


def test_parse_pdf_uses_text_layer_for_digital_pdf(fake_ocr: SimpleNamespace) -> None:
    """有文本层的电子版 PDF 直接返回文本，不触发 OCR（省时省钱）。"""
    text = rag_core._parse_pdf(TEXT_PDF_BYTES)

    assert "Hello pypdfium2 text layer 12345" in text
    assert fake_ocr.calls == []


def test_parse_pdf_falls_back_to_ocr_for_scanned_pdf(fake_ocr: SimpleNamespace) -> None:
    """文本层为空（扫描件）时转 OCR，并把原始字节交给 OCR 层。"""
    scanned = _scanned_pdf_bytes()

    assert rag_core._parse_pdf(scanned) == "OCR 识别结果"
    assert fake_ocr.calls == [scanned]


def test_parse_pdf_threshold_is_average_chars_per_page(
    monkeypatch: pytest.MonkeyPatch, fake_ocr: SimpleNamespace
) -> None:
    """判定看"平均每页字符数"：两页共 30 字符（平均 15 < 20）算扫描件。"""
    monkeypatch.setattr(rag_core, "_extract_pdf_text", lambda data: ("x" * 30, 2))
    scanned = _scanned_pdf_bytes()

    assert rag_core._parse_pdf(scanned) == "OCR 识别结果"
    assert fake_ocr.calls == [scanned]


def test_parse_pdf_keeps_text_layer_when_ocr_disabled(
    monkeypatch: pytest.MonkeyPatch, fake_ocr: SimpleNamespace
) -> None:
    """OCR 关闭后扫描件不再兜底，只返回文本层内容（这里是空串）。"""
    monkeypatch.setattr(settings, "ocr_enabled", False)

    assert rag_core._parse_pdf(_scanned_pdf_bytes()) == ""
    assert fake_ocr.calls == []


def test_parse_pdf_keeps_sparse_text_layer_when_ocr_disabled(
    monkeypatch: pytest.MonkeyPatch, fake_ocr: SimpleNamespace
) -> None:
    """已判定为扫描件但 OCR 关闭时，仍把抽到的零散文本层还回去，不要丢内容。"""
    monkeypatch.setattr(settings, "ocr_enabled", False)
    monkeypatch.setattr(rag_core, "_extract_pdf_text", lambda data: ("watermark", 3))

    assert rag_core._parse_pdf(b"whatever") == "watermark"
    assert fake_ocr.calls == []


# ---------------------------------------------------------------------------
# OCR：渲染 -> 识别 -> 拼接
# ---------------------------------------------------------------------------
def test_ocr_pdf_joins_pages_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """多页按页序拼接，页内按行拼接并丢掉空行。"""
    monkeypatch.setattr(rag_ocr, "_render_pages", lambda data: [b"page-1", b"page-2"])
    monkeypatch.setattr(rag_ocr, "_build_client", lambda: object())
    replies = iter(["第一页\n\nline a", "第二页\n\nline b"])
    monkeypatch.setattr(rag_ocr, "_recognize", lambda client, png: next(replies))

    assert rag_ocr.ocr_pdf(b"fake-pdf") == "第一页\nline a\n第二页\nline b"


def test_ocr_pdf_sends_base64_data_uri_with_configured_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """送图用 base64 data URI，模型名与提示词走配置。"""
    monkeypatch.setattr(rag_ocr, "_render_pages", lambda data: [b"\x89PNG-fake"])
    client = _FakeClient("识别文本")
    monkeypatch.setattr(rag_ocr, "_build_client", lambda: client)

    assert rag_ocr.ocr_pdf(b"fake-pdf") == "识别文本"

    call = client.completions.calls[0]
    assert call["model"] == settings.ocr_model
    content = call["messages"][0]["content"]  # type: ignore[index]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1] == {"type": "text", "text": rag_ocr.OCR_PROMPT}


def test_ocr_pdf_accepts_line_texts_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """网关返回 JSON（data.line_texts）时取行文本，空白行丢掉。"""
    monkeypatch.setattr(rag_ocr, "_render_pages", lambda data: [b"page-1"])
    payload = '{"data": {"line_texts": ["第一行", "  ", "第二行"]}, "code": "0"}'
    monkeypatch.setattr(rag_ocr, "_build_client", lambda: _FakeClient(payload))

    assert rag_ocr.ocr_pdf(b"fake-pdf") == "第一行\n第二行"


def test_ocr_pdf_accepts_flat_line_texts_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """``line_texts`` 在顶层（没有 data 包裹）时同样识别。"""
    monkeypatch.setattr(rag_ocr, "_render_pages", lambda data: [b"page-1"])
    monkeypatch.setattr(
        rag_ocr, "_build_client", lambda: _FakeClient('{"line_texts": ["只有一行"]}')
    )

    assert rag_ocr.ocr_pdf(b"fake-pdf") == "只有一行"


def test_ocr_pdf_keeps_plain_text_as_is(monkeypatch: pytest.MonkeyPatch) -> None:
    """以花括号开头但不是合法 JSON 时，按纯文本返回而不是丢内容。"""
    monkeypatch.setattr(rag_ocr, "_render_pages", lambda data: [b"page-1"])
    monkeypatch.setattr(rag_ocr, "_build_client", lambda: _FakeClient("{not json"))

    assert rag_ocr.ocr_pdf(b"fake-pdf") == "{not json"


def test_ocr_pdf_returns_empty_without_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    """空 PDF（0 页）直接返回空串，不构造客户端、不发请求。"""
    monkeypatch.setattr(rag_ocr, "_render_pages", lambda data: [])
    monkeypatch.setattr(rag_ocr, "_build_client", lambda: pytest.fail("空 PDF 不该建客户端"))

    assert rag_ocr.ocr_pdf(b"fake-pdf") == ""


def test_ocr_pdf_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """没配 Key 时按 RAG 未配置处理（OCR 与向量化共用同一个 Key）。"""
    monkeypatch.setattr(settings, "dashscope_api_key", SecretStr(""))

    with pytest.raises(rag_ocr.OcrNotConfiguredError) as excinfo:
        rag_ocr.ocr_pdf(_scanned_pdf_bytes())

    assert excinfo.value.code == "RAG_NOT_CONFIGURED"


def test_ocr_pdf_wraps_broken_pdf_as_ocr_error() -> None:
    """PDF 损坏时抛 OCR_FAILED，不把底层 pdfium 异常直接冒到上层。"""
    with pytest.raises(rag_ocr.OcrError) as excinfo:
        rag_ocr.ocr_pdf(b"not a pdf at all")

    assert excinfo.value.code == "OCR_FAILED"


def test_ocr_pdf_wraps_model_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """模型调用失败（网络 / 鉴权 / 配额）归为 OCR_FAILED。"""
    monkeypatch.setattr(rag_ocr, "_render_pages", lambda data: [b"page-1"])
    client = _FakeClient("")

    def _boom(**kwargs: object) -> object:
        raise RuntimeError("rate limited")

    monkeypatch.setattr(client.completions, "create", _boom)
    monkeypatch.setattr(rag_ocr, "_build_client", lambda: client)

    with pytest.raises(rag_ocr.OcrError) as excinfo:
        rag_ocr.ocr_pdf(b"fake-pdf")

    assert excinfo.value.code == "OCR_FAILED"
    assert "rate limited" in (excinfo.value.detail or "")


def test_ocr_pdf_reads_from_path_and_bytes(tmp_path: Path) -> None:
    """支持路径与字节两种入参（上传链路只有 bytes，联调常用路径）。"""
    path = tmp_path / "scanned.pdf"
    path.write_bytes(_scanned_pdf_bytes())

    assert rag_ocr._read_bytes(path) == path.read_bytes()
    assert rag_ocr._read_bytes(str(path)) == path.read_bytes()
    assert rag_ocr._read_bytes(b"abc") == b"abc"


def test_ocr_max_pages_caps_rendering(monkeypatch: pytest.MonkeyPatch) -> None:
    """超过页数上限时只渲染前 N 页，避免大文档把上传请求拖死。"""
    monkeypatch.setattr(settings, "ocr_max_pages", 2)

    pages = rag_ocr._render_pages(_scanned_pdf_bytes("Page one", "Page two", "Page three"))

    assert len(pages) == 2
    assert all(png.startswith(b"\x89PNG") for png in pages)
