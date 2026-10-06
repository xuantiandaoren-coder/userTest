"""提示词层（第二层）：**只**负责 Prompt 与链路组装。

这一层把三样东西拼成最终可执行链路：

    模板（公共 + 私有，来自提示词层）
      + 变量注入结果（验证 / 转换 / 填充三层处理）
      + 记忆（历史对话 + 搜索增强，来自记忆层）
      + 模型实例（来自模型层）
        └── ChatPromptTemplate | ChatOpenAI  →  Runnable（可 invoke / astream）

不做：不管模板存哪、怎么回滚（prompt_template_manager），不管历史怎么取（memory），
不管 provider 怎么选（llm）。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Mapping
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable

from app.memory.memory import MemoryContext
from app.prompts.injector import InjectionResult, render
from app.prompts.prompt_template_manager import ComposedPrompt

# 兜底行为约束：模板没写要求时也能保证输出形态稳定（模板作者可在私有模板里覆盖）
DEFAULT_RULES = """【回答要求】
1. 用中文回答，结构清晰，必要时用小标题或列表；
2. 结合用户的目标岗位与薄弱点，先给结论再给依据；
3. 不确定的内容要明说，不要编造事实或引用不存在的资料。"""

MEMORY_HEADING = "【长期记忆】"
SEARCH_HEADING = "【检索到的相关资料】"


@dataclass
class PromptContext:
    """一次调用的完整入参（全部是纯数据，可跨线程 / 跨会话传递）。"""

    system: str
    question: str
    history: list[BaseMessage] = field(default_factory=list)


def build_system_prompt(
    composed: ComposedPrompt,
    variables: Mapping[str, object] | None = None,
    *,
    memory: MemoryContext | None = None,
    rules: str | None = None,
) -> InjectionResult:
    """渲染模板并追加记忆 / 行为约束，返回注入结果（含实际注入的变量与告警）。

    模板正文由「提示词变量注入」三层处理渲染；记忆块由记忆层拼好（这里只负责排版）。
    """
    result = render(composed.content, variables)
    blocks = [result.text]
    if memory is not None and memory.memory_text:
        blocks.append(memory.memory_text)
    blocks.append(rules or DEFAULT_RULES)
    text = "\n\n".join(block.strip() for block in blocks if block and block.strip())
    return InjectionResult(
        text=text,
        variables=result.variables,
        unfilled=result.unfilled,
        warnings=result.warnings,
    )


def build_chat_prompt() -> ChatPromptTemplate:
    """系统提示 + 历史消息 + 本轮提问 的标准对话模板。

    注意：模板变量在渲染后才注入 system 文本，因此提示词里出现的 `{...}`
    （比如 JSON 示例）不会被再次解析。
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", "{system}"),
            MessagesPlaceholder("history", optional=True),
            ("human", "{question}"),
        ]
    )


def merge_system_messages(
    system: str,
    history: list[BaseMessage],
) -> tuple[str, list[BaseMessage]]:
    """把历史消息里的 SystemMessage 合并进最终 system 提示词。

    记忆层的长期画像以 ``SystemMessage`` 形式插在历史消息最前面；但 LangChain 的
    对话模板只保留一条 system（模板占位那条），历史里的 system 若原样透传会被
    下游模型当作普通消息、甚至被部分 provider 丢弃。这里在构建最终 system_prompt
    之前把它们提取出来、从 ``history`` 中移除，再按出现顺序追加到 ``system`` 末尾，
    用两个换行分隔。

    返回 ``(合并后的 system, 已移除 SystemMessage 的历史)``；调用方必须使用返回的历史。
    """
    blocks = [system.strip()] if system and system.strip() else []
    kept: list[BaseMessage] = []
    for message in history:
        if isinstance(message, SystemMessage):
            content = _message_content(message)
            if content:
                blocks.append(content)
            continue
        kept.append(message)
    return "\n\n".join(blocks), kept


def _message_content(message: BaseMessage) -> str:
    """SystemMessage 内容拍平成文本（兼容内容块列表形式）。"""
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, Iterable):
        pieces: list[str] = []
        for item in content:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                pieces.append(item["text"])
        return "".join(pieces).strip()
    return str(content).strip()


def build_chain(llm: BaseChatModel, prompt: ChatPromptTemplate | None = None) -> Runnable:
    """链路组装：ChatPromptTemplate | ChatOpenAI（LCEL，天然支持 astream）。"""
    return (prompt or build_chat_prompt()) | llm


def format_memory_block(memory: MemoryContext) -> str:
    """把记忆层结果排版成可直接拼进 system 的文本（调试 / 预览接口用）。"""
    return f"{SEARCH_HEADING}\n{memory.search_context}" if memory.search_context else ""


def chunk_to_text(chunk: object) -> str:
    """把模型返回的 chunk 转成纯文本。

    多数 provider 的 `content` 是字符串；部分多模态 provider 会返回内容块列表，
    这里统一拍平成文本，下游（SSE / 落库）只处理字符串。
    """
    content = getattr(chunk, "content", chunk)
    if isinstance(content, str):
        return content
    if isinstance(content, Iterable):
        pieces: list[str] = []
        for item in content:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                pieces.append(item["text"])
        return "".join(pieces)
    return str(content)


async def stream_tokens(chain: Runnable, context: PromptContext) -> AsyncIterator[str]:
    """流式产出文本增量（上游 SSE 逐条下发）。"""
    payload = {"system": context.system, "history": context.history, "question": context.question}
    async for chunk in chain.astream(payload):
        text = chunk_to_text(chunk)
        if text:
            yield text


async def collect_answer(chain: Runnable, context: PromptContext) -> str:
    """一次性拿到完整回答（非流式场景，如离线评测 / 摘要）。"""
    message = await chain.ainvoke(
        {"system": context.system, "history": context.history, "question": context.question}
    )
    return chunk_to_text(message)
