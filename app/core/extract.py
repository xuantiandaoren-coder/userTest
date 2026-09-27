"""文件内容提取：storage_scene=2 时只把文件内容读出来，不保存原文件。

优先纯文本类格式（txt / md / csv / json / 代码等）按 UTF-8 解码；
其余格式做一次容错解码，拿到可打印文本即为内容，不做格式级解析。
"""

from __future__ import annotations

from pathlib import Path

# 纯文本类扩展名：直接按文本解码
_TEXT_EXTENSIONS = frozenset(
    {
        "txt", "md", "markdown", "csv", "json", "xml", "html", "htm", "yml", "yaml",
        "log", "ini", "conf", "sql", "py", "js", "ts", "java", "go", "rs", "c", "cpp", "h",
    }
)

MAX_EXTRACTED_CHARS = 1_000_000  # 防止超大文本撑爆上下文与响应体


def extract_text(file_name: str, data: bytes) -> str:
    """提取文本内容；拿不到可读文本时返回空串（不抛错，调用方按“无内容”处理）。"""
    if Path(file_name).suffix.lstrip(".").lower() in _TEXT_EXTENSIONS:
        return _truncate(data.decode("utf-8", errors="replace"))

    # 二进制格式：丢掉控制字符后容错解码，能读出多少算多少
    printable = bytes(byte for byte in data if byte in (9, 10, 13) or 32 <= byte < 127 or byte >= 128)
    return _truncate(printable.decode("utf-8", errors="ignore"))


def _truncate(text: str) -> str:
    """按字符数截断，避免超长内容撑大响应与数据库。"""
    return text[:MAX_EXTRACTED_CHARS]
