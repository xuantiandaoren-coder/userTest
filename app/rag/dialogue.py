"""RAG 对话链路：意图判断 -> 检索 -> 上下文注入 -> 来源（sources）组装与回显。

一次对话里这份模块负责四件事：

1. **意图判断**：``should_retrieve`` 决定这条消息要不要检索知识库——空输入、
   极短输入、确认语（好的 / 嗯）、礼貌语（你好 / 谢谢）、推进语（继续 / 下一题）
   都不值得为它花一次向量化 + 一次向量检索
2. **检索**：``retrieve_knowledge`` 按 user_id 隔离调 ``search_similar_chunks``
3. **注入**：``format_retrieved_chunks`` 把命中片段排成参考资料文本，
   ``compose_question`` 把它拼到本轮问题前；``build_rag_system_prompt``
   只在原 system 提示词后**补充** RAG 回答规则，不重写模板正文
4. **来源**：``build_sources`` 给出完整来源（chunk_id / resource_id / chunk_index /
   file_name / score / text），``build_references`` 给出落库用的精简引用
   （只有 chunk_id + score，**不在消息表里重复保存知识块全文**）；
   ``resolve_reference_sources`` 供历史消息接口按引用回查原文重新拼 sources

配置统一走 ``app.core.config.settings``，本模块不直接读环境变量。
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.rag.chunk_store import fetch_chunks_with_resources
from app.rag.retriever import ChunkHit, search_similar_chunks

logger = logging.getLogger("app.rag.dialogue")

__all__ = [
    "should_retrieve",
    "format_retrieved_chunks",
    "build_rag_system_prompt",
    "retrieve_knowledge",
    "compose_question",
    "build_sources",
    "build_references",
    "resolve_reference_sources",
    "RagContext",
]

# 一次对话召回几条知识片段
RAG_TOP_K = 3

# 短于该长度（去掉空白与标点后）的消息不检索：多是"嗯""好"这类没有信息量的输入
MIN_QUERY_CHARS = 4

# 只有短消息才做"纯确认语 / 礼貌语 / 推进语"判定，避免长句里恰好含"好的"就被跳过
MAX_SKIP_SCAN_CHARS = 10

# 参考资料块标题：拼在本轮提问前，与提示词层的【检索到的相关资料】区分开
RAG_CONTEXT_HEADING = "【参考资料】"
QUESTION_HEADING = "【用户问题】"

# 引用片段已被删除时的占位文本（历史回显用）
DELETED_CHUNK_TEXT = "该参考片段已经删除"

# 只补充回答规则，不改写模板正文
RAG_RULES = """【RAG 回答规则】
1. 优先依据【参考资料】回答，引用时用编号标注来源（如 [1]）；
2. 参考资料里没有的内容不要编造，明说"资料中没有相关信息"，需要补充通用知识时要标明；
3. 资料片段可能被截断，只按给出的内容作答，不要臆测上下文。"""

# 确认语：用户在回应上一轮，不是新的问题
CONFIRM_PHRASES = (
    "好的", "好", "嗯", "嗯嗯", "对", "对的", "是的", "是", "没错", "可以", "行", "行的",
    "收到", "明白", "明白了", "懂了", "知道了", "了解", "没问题", "ok", "okay", "yes", "sure",
)

# 礼貌语：打招呼 / 致谢，问答本身没有内容
POLITE_PHRASES = (
    "你好", "您好", "hi", "hello", "在吗", "在么", "谢谢", "谢谢你", "多谢", "感谢", "辛苦了",
    "麻烦你了", "麻烦了", "拜托", "thanks", "thankyou", "请", "早上好", "晚上好",
)

# 推进语：让助手继续，本身不携带新的检索意图
CONTINUE_PHRASES = (
    "继续", "接着", "然后呢", "然后", "接下来", "下一步", "还有吗", "还有呢", "再来", "再来一个",
    "下一题", "下一个", "开始吧", "开始", "goon", "next", "说下去", "讲下去",
)

SKIP_PHRASES = tuple(sorted({*CONFIRM_PHRASES, *POLITE_PHRASES, *CONTINUE_PHRASES}, key=len, reverse=True))

# 归一化：去掉空白与标点（保留中英文与数字），小写化后再比对
_NOISE_PATTERN = re.compile(r"[\s\W_]+", re.UNICODE)


@dataclass(frozen=True)
class RagContext:
    """一次检索的产物：命中片段 + 注入用文本 + 展示用来源 + 落库用引用。"""

    retrieved: bool = False                    # 是否真的发起了检索（意图判断没过就是 False）
    chunks: list[ChunkHit] = field(default_factory=list)
    retrieved_text: str = ""                   # format_retrieved_chunks 的结果，拼到问题上
    sources: list[dict[str, object]] = field(default_factory=list)     # SSE done / 前端展示
    references: list[dict[str, object]] = field(default_factory=list)  # 落库：chunk_id + score

    @property
    def has_context(self) -> bool:
        """是否有资料可以注入（决定要不要补 RAG 回答规则）。"""
        return bool(self.retrieved_text)


# ---------------------------------------------------------------------------
# 1. 意图判断
# ---------------------------------------------------------------------------
def should_retrieve(request_text: str) -> bool:
    """判断这条用户消息要不要检索知识库。

    跳过：空输入、极短输入（去掉空白标点后短于 ``MIN_QUERY_CHARS``）、
    确认语 / 礼貌语 / 推进语（含它们的短组合，如"好的，继续"）。
    """
    text = normalize_query(request_text)
    if not text or len(text) < MIN_QUERY_CHARS:
        return False
    if text in SKIP_PHRASES:
        return False
    if len(text) <= MAX_SKIP_SCAN_CHARS and _consumed_by_skip_phrases(text):
        return False
    return True


def normalize_query(text: str | None) -> str:
    """归一化：去掉空白、标点、表情并转小写（"好的，继续！" -> "好的继续"）。"""
    return _NOISE_PATTERN.sub("", (text or "").lower())


def _consumed_by_skip_phrases(text: str) -> bool:
    """短句是否完全由确认语 / 礼貌语 / 推进语拼成（长词优先，避免"你的"被"你"吃掉）。

    每剥掉一段就**重新从头扫**：否则"好的继续"里先剥掉"好的"之后，
    已经扫过的"继续"不会再被匹配到，结果会随短语表的迭代顺序变化（不稳定）。
    """
    remaining = text
    stripped = True
    while remaining and stripped:
        stripped = False
        for phrase in SKIP_PHRASES:
            if phrase and remaining.startswith(phrase):
                remaining = remaining[len(phrase) :]
                stripped = True
                break
    return not remaining


# ---------------------------------------------------------------------------
# 2. 检索
# ---------------------------------------------------------------------------
def retrieve_knowledge(
    query: str,
    *,
    user_id: int,
    db: Session,
    top_k: int = RAG_TOP_K,
) -> RagContext:
    """按需检索知识库：意图判断不过就不检索（``retrieved=False``）。

    检索失败不让对话挂掉：记日志 + 返回空上下文，本轮按无资料回答。
    """
    if not should_retrieve(query):
        logger.debug("意图判断跳过检索 user_id=%s query_len=%s", user_id, len(query or ""))
        return RagContext()

    try:
        chunks = search_similar_chunks(query.strip(), user_id=user_id, db=db, top_k=top_k)
    except Exception as exc:  # noqa: BLE001 - 检索是增强项，失败不能让聊天整体不可用
        logger.warning("知识库检索失败，本轮不注入资料 user_id=%s：%r", user_id, exc)
        return RagContext()

    return RagContext(
        retrieved=True,
        chunks=chunks,
        retrieved_text=format_retrieved_chunks(chunks),
        sources=build_sources(chunks),
        references=build_references(chunks),
    )


# ---------------------------------------------------------------------------
# 3. 注入
# ---------------------------------------------------------------------------
def format_retrieved_chunks(chunks: Sequence[ChunkHit]) -> str:
    """把命中片段排成参考资料文本（编号 + 出处 + 原文），拼到本轮问题前。"""
    if not chunks:
        return ""

    lines = [RAG_CONTEXT_HEADING]
    for index, chunk in enumerate(chunks, start=1):
        origin = chunk.file_name or "未命名文件"
        position = f"第 {chunk.chunk_index + 1} 块" if isinstance(chunk.chunk_index, int) else "分块序号未知"
        lines.append(f"[{index}] 来源：{origin}（{position}，相似度 {chunk.score:.2f}）")
        lines.append(chunk.text)
    return "\n".join(lines).strip()


def compose_question(retrieved_text: str, question: str) -> str:
    """把参考资料拼到本轮问题**前面**（历史与落库仍用原始问题）。"""
    text = (retrieved_text or "").strip()
    if not text:
        return question
    return f"{text}\n\n{QUESTION_HEADING}\n{question}"


def build_rag_system_prompt(system_prompt: str) -> str:
    """在已有 system 提示词后**只补充** RAG 回答规则（不重写模板正文）。"""
    base = (system_prompt or "").rstrip()
    return f"{base}\n\n{RAG_RULES}" if base else RAG_RULES


# ---------------------------------------------------------------------------
# 4. 来源（sources）与落库引用（references）
# ---------------------------------------------------------------------------
def build_sources(chunks: Sequence[ChunkHit]) -> list[dict[str, object]]:
    """完整来源：SSE done / 前端展示用（含原文，方便直接渲染引用卡片）。"""
    return [
        {
            "chunk_id": chunk.vector_id,
            "resource_id": chunk.resource_id,
            "chunk_index": chunk.chunk_index,
            "file_name": chunk.file_name,
            "score": round(chunk.score, 4),
            "text": chunk.text,
        }
        for chunk in chunks
    ]


def build_references(chunks: Sequence[ChunkHit]) -> list[dict[str, object]]:
    """落库引用：只存 chunk_id + score，正文靠回查，避免在消息表里重复保存知识块全文。"""
    return [{"chunk_id": chunk.vector_id, "score": round(chunk.score, 4)} for chunk in chunks]


def load_source_index(
    db: Session,
    chunk_ids: Sequence[str],
    *,
    user_id: int | None = None,
) -> dict[str, dict[str, object]]:
    """``chunk_id`` -> 出处信息（resource_id / chunk_index / file_name / text）。

    查不到的 chunk_id 不会出现在结果里（调用方按"已删除"处理）。
    """
    rows = fetch_chunks_with_resources(db, list(chunk_ids), user_id=user_id)
    return {
        chunk_id: {
            "resource_id": chunk.resource_id,
            "chunk_index": chunk.chunk_index,
            "file_name": resource.file_name,
            "text": chunk.text,
        }
        for chunk_id, (chunk, resource) in rows.items()
    }


def resolve_reference_sources(
    db: Session,
    references: Sequence[Mapping[str, object]] | None,
    *,
    user_id: int | None = None,
    index: Mapping[str, Mapping[str, object]] | None = None,
) -> list[dict[str, object]]:
    """历史回显：``reference_sources``（chunk_id + score）-> 完整 sources。

    原文按 chunk_id 回查 ``knowledge_chunks`` + ``resources``；片段已被删除
    （资源过期清理 / 重新上传覆盖）时正文用 ``DELETED_CHUNK_TEXT`` 占位、
    出处字段留空，前端据此提示"该参考片段已经删除"。

    ``index`` 可以传入 ``load_source_index`` 的结果，列表接口按页复用同一份查询结果。
    """
    cleaned = [reference for reference in (references or []) if _is_valid_reference(reference)]
    if not cleaned:
        return []

    source_index = index if index is not None else load_source_index(
        db, [str(reference["chunk_id"]) for reference in cleaned], user_id=user_id
    )

    sources: list[dict[str, object]] = []
    for reference in cleaned:
        chunk_id = str(reference["chunk_id"])
        found = source_index.get(chunk_id)
        sources.append(
            {
                "chunk_id": chunk_id,
                "resource_id": found.get("resource_id") if found else None,
                "chunk_index": found.get("chunk_index") if found else None,
                "file_name": found.get("file_name") if found else None,
                "score": reference.get("score"),
                "text": found.get("text") if found else DELETED_CHUNK_TEXT,
            }
        )
    return sources


def _is_valid_reference(reference: object) -> bool:
    """落库引用至少要有一个非空 chunk_id，脏数据直接跳过。"""
    return isinstance(reference, Mapping) and bool(str(reference.get("chunk_id") or "").strip())
