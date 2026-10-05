"""记忆层：短期记忆读取（InstantMemory）+ 搜索增强，统一入口 MemoryService。

上层只依赖 ``MemoryService.load()``；工作记忆 / 长期记忆后续在 service 里扩展。
"""

from app.memory.instant import InstantMemory, InstantMemoryContext, build_chat_history_from_rows
from app.memory.memory import MemoryConfig, MemoryContext, SearchHit
from app.memory.service import MemoryService

__all__ = [
    "MemoryService",
    "MemoryConfig",
    "MemoryContext",
    "SearchHit",
    "InstantMemory",
    "InstantMemoryContext",
    "build_chat_history_from_rows",
]
