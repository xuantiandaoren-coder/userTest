"""记忆层（第三层）公共构件：配置、数据结构与搜索增强后端。

真正干活的入口拆成两个模块，本文件只放两边共用的东西：

- ``InstantMemory``（app/memory/instant.py）：短期记忆，读本会话最近 N 轮 -> LangChain 消息
- ``MemoryService``（app/memory/service.py）：统一入口，短期记忆 + 搜索增强组装成 ``MemoryContext``

本文件提供：

1. ``MemoryConfig``：记忆层参数（轮数 / 字符预算 / 检索条数）
2. ``MemoryContext``：记忆层产出结构（历史消息 + 检索增强文本），提示词层只认它
3. 搜索增强：检索出的相关片段 -> 文本块（拼进 system，给模型当参考资料）

检索后端是可替换的：``SearchBackend`` 协议 + 默认的 ``DatabaseSearchBackend``
（MySQL/SQLite 关键词召回）。要换成向量库 / ES / 外部搜索时，
只要实现同样的 ``search()`` 并在 ``MemoryService`` 里注入，记忆层其余逻辑不用改。

不做：不选模型（模型层）、不拼最终提示词（提示词层）、不算版本（模板管理）。
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from langchain_core.messages import BaseMessage

from app.db.chat_message_repository import ChatMessageRepository
from app.db.models import ChatMessage

logger = logging.getLogger("app.memory")

# 关键词抽取：英文/数字标识符 + 连续中文片段
WORD_PATTERN = re.compile(r"[A-Za-z0-9_+#.]{2,}")
CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,}")
WHITESPACE_PATTERN = re.compile(r"\s+")

MAX_KEYWORDS = 12          # 单次检索最多用几个关键词（防止 SQL 条件爆炸）
MIN_KEYWORD_LEN = 2
CJK_GRAM_LEN = 2           # 中文无空格，用 2-gram 做粗粒度召回


@dataclass(frozen=True)
class MemoryConfig:
    """记忆层参数（默认值取自配置，可按需覆盖）。"""

    max_turns: int = 10            # 注入的历史轮数上限
    max_chars: int = 6000          # 历史文本总长度上限，超出丢弃最旧的轮次
    search_enabled: bool = True    # 是否启用搜索增强
    search_top_k: int = 3          # 最终注入的检索片段条数
    search_candidate_limit: int = 60  # 召回候选上限
    snippet_chars: int = 240       # 单条片段长度


@dataclass(frozen=True)
class SearchHit:
    """一条检索命中：来自哪个会话 / 哪条消息，命中了什么。"""

    message_id: int
    session_id: int
    snippet: str
    score: int
    source: str = "history"  # history=问答正文，file=文件提取文本

    def as_dict(self) -> dict[str, object]:
        """转成可 JSON 序列化的字典（SSE 元事件 / 调试用）。"""
        return {
            "message_id": self.message_id,
            "session_id": self.session_id,
            "score": self.score,
            "source": self.source,
            "snippet": self.snippet,
        }


@dataclass
class MemoryContext:
    """记忆层产出：历史消息 + 检索增强文本。"""

    messages: list[BaseMessage] = field(default_factory=list)     # 历史（已截断）
    turns: list[tuple[str, str]] = field(default_factory=list)     # 原始 (提问, 回答)
    search_context: str = ""                                       # 检索增强文本块
    search_hits: list[SearchHit] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def memory_text(self) -> str:
        """需要拼进 system 的部分（历史走 messages，不进 system）。"""
        return self.search_context

    @property
    def summary(self) -> str:
        """一行摘要，用于日志与 SSE 元事件。"""
        return f"history_turns={len(self.turns)} search_hits={len(self.search_hits)}"


class SearchBackend(Protocol):
    """检索后端协议：换向量库 / ES / 外部搜索时实现这一个方法即可。"""

    def search(
        self,
        *,
        query: str,
        user_id: int,
        session_id: int,
        limit: int,
    ) -> list[SearchHit]: ...


class DatabaseSearchBackend:
    """默认检索后端：在用户自己的历史消息里做关键词召回 + 打分。

    只用 SQL LIKE，不引入向量库，MySQL / SQLite 行为一致，测试无需额外依赖。
    打分规则：命中关键词按长度加权累加（长词更具体，权重更高），同分按消息 id 倒序（新的优先）。
    """

    def __init__(self, messages: ChatMessageRepository) -> None:
        self.messages = messages

    def search(
        self,
        *,
        query: str,
        user_id: int,
        session_id: int,
        limit: int,
    ) -> list[SearchHit]:
        keywords = extract_keywords(query)
        if not keywords:
            return []
        candidates = self.messages.search_text(
            user_id,
            keywords,
            exclude_session_id=session_id,
            limit=limit,
        )
        return rank_hits(candidates, keywords)


def rank_hits(candidates: Sequence[ChatMessage], keywords: Sequence[str]) -> list[SearchHit]:
    """对候选消息打分并生成片段（纯函数，便于单测）。"""
    hits: list[SearchHit] = []
    for message in candidates:
        text = build_searchable_text(message)
        if not text:
            continue
        matched = [keyword for keyword in keywords if keyword.lower() in text.lower()]
        if not matched:
            continue
        score = sum(len(keyword) for keyword in matched)
        hits.append(
            SearchHit(
                message_id=message.id,
                session_id=message.session_id,
                snippet=make_snippet(text, matched[0]),
                score=score,
                source="file" if message.file_extracted_text and matched[0].lower() in message.file_extracted_text.lower() else "history",
            )
        )
    hits.sort(key=lambda hit: (-hit.score, -hit.message_id))
    return hits


def build_searchable_text(message: ChatMessage) -> str:
    """把一条消息揉成可检索文本（提问 + 回答 + 文件提取内容）。"""
    parts = [message.request_text or "", message.response_text or "", message.file_extracted_text or ""]
    return WHITESPACE_PATTERN.sub(" ", " ".join(part for part in parts if part)).strip()


def make_snippet(text: str, keyword: str, *, width: int | None = None) -> str:
    """截取命中关键词附近的片段（带省略号），避免把整条长消息塞进提示词。"""
    width = width or MemoryConfig.snippet_chars
    if len(text) <= width:
        return text
    index = text.lower().find(keyword.lower())
    start = max(0, index - width // 3) if index >= 0 else 0
    snippet = text[start : start + width].strip()
    return f"{'…' if start > 0 else ''}{snippet}{'…' if start + width < len(text) else ''}"


def extract_keywords(query: str, *, max_keywords: int = MAX_KEYWORDS) -> list[str]:
    """从提问里抽出检索关键词。

    英文按词；中文没有空格，取连续中文片段本身 + 2-gram，兼顾「算法」这类短词与长片段。
    """
    text = (query or "").strip()
    if not text:
        return []

    keywords: list[str] = []
    for match in WORD_PATTERN.finditer(text):
        _push(keywords, match.group(0))
    for match in CJK_PATTERN.finditer(text):
        chunk = match.group(0)
        _push(keywords, chunk)
        for index in range(len(chunk) - CJK_GRAM_LEN + 1):
            _push(keywords, chunk[index : index + CJK_GRAM_LEN])
    return keywords[:max_keywords]


def _push(bucket: list[str], keyword: str) -> None:
    """按长度降序偏好保留（长词更具体），去重且不超过上限。"""
    keyword = keyword.strip()
    if len(keyword) < MIN_KEYWORD_LEN or keyword in bucket:
        return
    if len(bucket) < MAX_KEYWORDS:
        bucket.append(keyword)


def render_search_context(hits: Sequence[SearchHit]) -> str:
    """把检索命中排成参考资料文本（编号 + 片段）。"""
    if not hits:
        return ""
    lines = ["以下是用户历史对话中检索到的相关资料，供你参考（可能不完整，不要编造）："]
    for index, hit in enumerate(hits, start=1):
        lines.append(f"[{index}] (会话 {hit.session_id}) {hit.snippet}")
    return "\n".join(lines)
