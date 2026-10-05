"""业务层：流式聊天（`POST /sessions/{session_id}/stream-chat`，SSE）。

把三层组装起来跑一次对话：

    prepare()  鉴权 + 归属校验 + 模板拼接 + 变量注入 + 记忆构建  ——有 HTTP 错误语义，先跑
    stream()   SSE：meta -> delta* -> done/error，结束后落库        ——响应头已发出，错误只能走 error 事件

为什么分成两步：SSE 一旦开始下发就不能再改 HTTP 状态码，
所以「会话不存在（404）」「智能体不存在（422）」「没配模板（404）」这类校验
必须在返回 StreamingResponse 之前完成，才能给前端标准的 4xx JSON。

落库为什么要另开会话：请求级会话（get_db）在响应结束时才提交，
而流式响应的生成器可能在依赖清理之后才结束，因此最后一步落库用独立的会话工厂，
与 HTTP 依赖生命周期解耦。
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.core.config import (
    DEFAULT_SCENE,
    AgentSetting,
    agent_for_session_model,
    get_agent,
    resolve_agent_name,
)
from app.core.exceptions import BusinessError
from app.core.sse import EVENT_DELTA, EVENT_DONE, EVENT_ERROR, EVENT_META, sse_event
from app.db.chat_message_repository import ChatMessageRepository
from app.db.models import User
from app.db.prompt_template_repository import PromptTemplateRepository
from app.db.session_repository import ChatSessionRepository
from app.db.user_profile_repository import UserProfileRepository
from app.memory.memory import MemoryConfig, MemoryContext
from app.memory.service import MemoryService
from app.prompts.injector import profile_to_variables
from app.prompts.prompt_layer import PromptContext, build_chain, build_system_prompt, stream_tokens
from app.prompts.prompt_template_manager import PromptTemplateManager
from app.schemas.chat import StreamChatRequest
from app.services.chat_service import SessionNotFoundError

logger = logging.getLogger("app.stream_chat")


class AgentNotFoundError(BusinessError):
    """业务异常：请求里指定的智能体不在 AGENT_CONFIG 内。"""

    code = "AGENT_NOT_FOUND"
    http_status = 422
    message = "智能体不存在"


@dataclass
class StreamPlan:
    """prepare() 的产物：全是纯数据，脱离数据库会话也能安全使用。"""

    request_id: str
    user_id: int
    session_id: int
    agent_name: str
    scene: str
    system_prompt: str
    question: str                                   # 原始提问：落库 / 历史 / 前端展示
    prompt_question: str = ""                       # 注入参考资料后的问题：只进本轮提示词
    history: list[BaseMessage] = field(default_factory=list)
    model: BaseChatModel | None = None
    select_model: int = 0
    prompt_versions: str = ""
    search_hits: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)       # 知识库来源（SSE done 下发）
    references: list[dict[str, Any]] = field(default_factory=list)    # 落库引用：只有 chunk_id + score
    warnings: list[str] = field(default_factory=list)


class StreamChatService:
    """流式聊天业务逻辑：模板 -> 变量 -> 记忆 -> 模型链路。"""

    def __init__(
        self,
        *,
        sessions: ChatSessionRepository,
        messages: ChatMessageRepository,
        profiles: UserProfileRepository,
        prompts: PromptTemplateManager,
        model: BaseChatModel,
        persist_factory: Callable[[], Session] | None = None,
        model_factory: Callable[[str | None], BaseChatModel] | None = None,
        memory_config: MemoryConfig | None = None,
        chain: Runnable | None = None,
    ) -> None:
        self.sessions = sessions
        self.messages = messages
        self.profiles = profiles
        self.prompts = prompts
        self.model = model
        self.persist_factory = persist_factory
        self.model_factory = model_factory
        self.chain = chain
        self.memory = MemoryService(messages, config=memory_config)

    # ------------------------------------------------------------------
    # 第一步：准备（可以有 4xx）
    # ------------------------------------------------------------------
    async def prepare(self, user: User, session_id: int, payload: StreamChatRequest) -> StreamPlan:
        """鉴权 / 归属校验 + 模板拼接 + 变量注入 + 记忆构建（阻塞部分丢线程池）。"""
        return await run_in_threadpool(self._prepare_sync, user, session_id, payload)

    def _prepare_sync(self, user: User, session_id: int, payload: StreamChatRequest) -> StreamPlan:
        chat_session = self.sessions.get(session_id, user.id)
        if chat_session is None:
            raise SessionNotFoundError(detail=f"session_id={session_id} user_id={user.id}")

        agent_name, agent = _resolve_agent(payload.agent_name, chat_session.session_model, payload.scene)
        composed = self.prompts.compose(agent_name, agent.scene)  # 404：该组没有生效模板

        profile = self.profiles.get_by_user(user.id)
        variables = profile_to_variables(
            profile,
            user_name=user.user_name,
            agent_name=agent_name,
            agent_label=agent.label,
            scene=agent.scene,
        )

        memory = self.memory.load(
            user_id=user.id,
            session_id=session_id,
            query=payload.search_query or payload.message,
            use_search=payload.use_search,
        )
        rendered = build_system_prompt(composed, variables, memory=memory)

        # RAG：按需检索自己的知识库 -> 参考资料拼到本轮问题前 -> system 只补 RAG 回答规则
        system_prompt, prompt_question, sources, references = self._apply_rag(
            user_id=user.id,
            system_prompt=rendered.text,
            question=payload.message,
        )

        return StreamPlan(
            request_id=uuid.uuid4().hex,
            user_id=user.id,
            session_id=session_id,
            agent_name=agent_name,
            scene=agent.scene,
            system_prompt=system_prompt,
            question=payload.message,     # 原始提问落库 / 进历史，注入的参考资料不落库
            prompt_question=prompt_question,
            history=memory.messages,
            model=self._resolve_model(agent),
            select_model=agent.select_model if agent.select_model is not None else 0,
            prompt_versions=composed.version_label,
            search_hits=[hit.as_dict() for hit in memory.search_hits],
            sources=sources,
            references=references,
            warnings=[*rendered.warnings, *memory.warnings],
        )

    def _apply_rag(
        self,
        *,
        user_id: int,
        system_prompt: str,
        question: str,
    ) -> tuple[str, str, list[dict[str, Any]], list[dict[str, Any]]]:
        """按需检索知识库并注入本轮，返回 (system_prompt, question, sources, references)。

        意图判断不过（空输入 / 极短 / 确认语 / 礼貌语 / 推进语）就不检索；
        检索属于增强能力，依赖缺失或检索失败都只记日志，本轮按无资料回答。
        注意落库与历史用的仍是**原始问题**，注入的参考资料只进本轮提示词。
        """
        try:
            # 延迟导入：向量库 / 向量化依赖较重，缺失时只影响检索，不影响聊天
            from app.rag.dialogue import (
                RAG_TOP_K,
                build_rag_system_prompt,
                compose_question,
                retrieve_knowledge,
            )
        except ImportError as exc:
            logger.warning("RAG 依赖不可用，跳过知识库检索：%r", exc)
            return system_prompt, question, [], []

        try:
            rag = retrieve_knowledge(question, user_id=user_id, db=self.messages.session, top_k=RAG_TOP_K)
        except Exception as exc:  # noqa: BLE001 - 检索是增强项，失败不能让聊天整体不可用
            logger.warning("知识库检索异常，本轮不注入资料：%r", exc)
            return system_prompt, question, [], []

        if not rag.has_context:
            return system_prompt, question, rag.sources, rag.references
        logger.info("知识库注入 user_id=%s chunks=%s", user_id, len(rag.chunks))
        return (
            build_rag_system_prompt(system_prompt),
            compose_question(rag.retrieved_text, question),
            rag.sources,
            rag.references,
        )

    def _resolve_model(self, agent: AgentSetting) -> BaseChatModel:
        """智能体指定了 provider 就单独建一个（有缓存），否则复用注入的默认模型。"""
        if agent.provider and self.model_factory is not None:
            return self.model_factory(agent.provider)
        return self.model

    # ------------------------------------------------------------------
    # 第二步：SSE 流式输出
    # ------------------------------------------------------------------
    async def stream(self, plan: StreamPlan) -> AsyncIterator[str]:
        """产出 SSE 帧：meta -> delta* -> done / error。"""
        started = time.monotonic()
        yield sse_event(
            EVENT_META,
            {
                "request_id": plan.request_id,
                "session_id": plan.session_id,
                "agent_name": plan.agent_name,
                "scene": plan.scene,
                "prompt_versions": plan.prompt_versions,
                "history_turns": len(plan.history),
                "search_hits": plan.search_hits,
                "warnings": plan.warnings,
            },
        )

        # 进模型的是注入了参考资料的问题；plan.question 保持原始提问，落库与历史都用它
        context = PromptContext(
            system=plan.system_prompt,
            question=plan.prompt_question or plan.question,
            history=plan.history,
        )
        chain = self.chain or build_chain(plan.model or self.model)
        buffer: list[str] = []
        try:
            async for token in stream_tokens(chain, context):
                buffer.append(token)
                yield sse_event(EVENT_DELTA, {"content": token})
        except asyncio.CancelledError:
            # 客户端断开：把已生成的部分落库（best-effort），再让取消继续向上传播
            await self._persist_quietly(plan, "".join(buffer), reason="cancelled")
            raise
        except Exception as exc:
            logger.error("流式聊天失败 request_id=%s agent=%s：%r", plan.request_id, plan.agent_name, exc, exc_info=exc)
            yield sse_event(EVENT_ERROR, {"code": "LLM_ERROR", "message": "模型调用失败，请稍后重试"})
            return

        answer = "".join(buffer)
        message_id = await self._persist_quietly(plan, answer, reason="done")
        yield sse_event(
            EVENT_DONE,
            {
                "request_id": plan.request_id,
                "message_id": message_id,
                "session_id": plan.session_id,
                "answer_length": len(answer),
                "prompt_versions": plan.prompt_versions,
                "search_hits": len(plan.search_hits),
                "sources": plan.sources,   # 本轮回答引用的知识片段（含原文，供前端展示来源）
                "elapsed_ms": int((time.monotonic() - started) * 1000),
            },
        )

    # ------------------------------------------------------------------
    # 落库
    # ------------------------------------------------------------------
    async def _persist_quietly(self, plan: StreamPlan, answer: str, *, reason: str) -> int | None:
        """落库一轮问答；失败只记日志（回答已经发给用户了，不能因为落库失败报错）。"""
        if not answer:
            logger.info("空回答不落库 request_id=%s reason=%s", plan.request_id, reason)
            return None
        try:
            return await run_in_threadpool(self._persist_sync, plan, answer)
        except Exception as exc:
            logger.error("聊天记录落库失败 request_id=%s reason=%s：%r", plan.request_id, reason, exc, exc_info=exc)
            return None

    def _persist_sync(self, plan: StreamPlan, answer: str) -> int:
        """用独立会话写入一轮问答（不依赖请求级会话的生命周期）。"""
        if self.persist_factory is None:
            raise RuntimeError("未配置 persist_factory，无法落库")
        session = self.persist_factory()
        try:
            message = ChatMessageRepository(session).create(
                user_id=plan.user_id,
                session_id=plan.session_id,
                select_model=plan.select_model,
                request_id=plan.request_id,
                request_text=plan.question,
                response_text=answer,
                # 只存引用（chunk_id + score），知识块全文仍只在 knowledge_chunks 里存一份
                reference_sources=plan.references or None,
            )
            session.commit()
            return message.id
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


def _resolve_agent(
    agent_name: str | None,
    session_model: int,
    scene: str | None,
) -> tuple[str, AgentSetting]:
    """确定本次使用哪个智能体 / 场景：显式指定 > 按会话类型推导 > 默认智能体。"""
    name = (agent_name or "").strip() or agent_for_session_model(session_model)
    agent = get_agent(name)
    if agent is None:
        raise AgentNotFoundError(detail=f"agent_name={agent_name!r} 不在 AGENT_CONFIG 内")
    name = resolve_agent_name(name) or name  # 历史别名折叠成现行智能体名
    resolved_scene = (scene or "").strip() or agent.scene or DEFAULT_SCENE
    if agent.scene == resolved_scene:
        return name, agent
    # 允许调用方临时换场景（同一个智能体可以有多套场景模板）
    return name, AgentSetting(
        label=agent.label,
        scene=resolved_scene,
        description=agent.description,
        select_model=agent.select_model,
        provider=agent.provider,
    )
