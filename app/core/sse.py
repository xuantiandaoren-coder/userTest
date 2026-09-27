"""SSE（Server-Sent Events）协议工具：把事件序列化成 text/event-stream 帧。

帧格式（每条事件以空行结束）：

    event: delta
    data: {"content": "你"}

约定：`data` 始终是一行 JSON（`ensure_ascii=False`，中文不转义），
前端用 `JSON.parse(e.data)` 即可，无需再拆包。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

# 反向代理下必须关掉缓冲，否则流会被攒成一坨再下发
SSE_HEADERS: dict[str, str] = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

SSE_MEDIA_TYPE = "text/event-stream"

# 事件类型：meta（本次请求的元信息）-> delta（增量）-> done / error
EVENT_META = "meta"
EVENT_DELTA = "delta"
EVENT_DONE = "done"
EVENT_ERROR = "error"


def sse_event(event: str, data: Mapping[str, Any] | None = None) -> str:
    """构造一个 SSE 帧；data 为空时也要写 `data: {}`，否则部分前端库不触发事件。"""
    payload = json.dumps(dict(data or {}), ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def sse_comment(text: str) -> str:
    """SSE 注释帧（心跳用），前端收到后直接忽略。"""
    return f": {text}\n\n"
