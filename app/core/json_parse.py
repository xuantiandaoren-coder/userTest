"""模型返回文本的 JSON 解析工具（统一封装，避免各处重复写正则）。

大模型经常把 JSON 包在 Markdown 代码块里（```json ... ```），或在前后带解释文字，
甚至返回被截断 / 多个 JSON 拼接的内容。这里统一做「代码块优先 -> 整段文本 ->
第一个 {..}/[..]」的容错提取，并提供严格解析入口。

用法：

    data = extract_json_object(raw)          # 宽松：拿不到返回 None
    quiz = loads_json(raw, expect=dict)      # 严格：失败抛 JsonParseError
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any

__all__ = [
    "JsonParseError",
    "extract_json",
    "extract_json_object",
    "extract_json_array",
    "loads_json",
]

# Markdown 代码块：```json ... ``` 或 ``` ... ```
_CODE_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class JsonParseError(ValueError):
    """模型输出无法解析为合法 JSON（或类型不符合预期）。"""


def extract_json(raw: str | None) -> Any | None:
    """从模型输出里提取 JSON（dict / list / 标量）；找不到合法 JSON 返回 None。"""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    for candidate in _candidates(text):
        parsed = _try_parse(candidate)
        if parsed is not None:
            return parsed
    return None


def extract_json_object(raw: str | None) -> dict[str, Any] | None:
    """提取 JSON 对象；结果是数组 / 标量则返回 None。"""
    value = extract_json(raw)
    return value if isinstance(value, dict) else None


def extract_json_array(raw: str | None) -> list[Any] | None:
    """提取 JSON 数组；结果是对象 / 标量则返回 None。"""
    value = extract_json(raw)
    return value if isinstance(value, list) else None


def loads_json(raw: str | None, *, expect: type | tuple[type, ...] | None = None) -> Any:
    """严格解析：找不到 JSON 或类型不符时抛 ``JsonParseError``。"""
    value = extract_json(raw)
    if value is None:
        raise JsonParseError("模型输出中未找到合法 JSON")
    if expect is not None and not isinstance(value, expect):
        raise JsonParseError(f"JSON 类型不符：期望 {expect}，实际 {type(value).__name__}")
    return value


# ---------------------------------------------------------------------------
# 内部实现
# ---------------------------------------------------------------------------
def _candidates(text: str) -> Iterator[str]:
    """候选文本：代码块 -> 整段 -> 第一个 {..} / [..] 片段。"""
    for block in _CODE_FENCE_PATTERN.findall(text):
        block = block.strip()
        if block:
            yield block
    yield text
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        end = text.rfind(closer)
        if end > start:
            yield text[start : end + 1]


def _try_parse(text: str) -> Any | None:
    """单段候选文本解析：先整体 loads，再用 raw_decode 吃最长的合法 JSON。"""
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    # 从第一个 { 或 [ 开始，raw_decode 只吃合法 JSON，自动忽略尾随解释文字
    starts = [index for index in (text.find("{"), text.find("[")) if index != -1]
    if not starts:
        return None
    try:
        value, _ = decoder.raw_decode(text[min(starts) :])
    except json.JSONDecodeError:
        return None
    return value
