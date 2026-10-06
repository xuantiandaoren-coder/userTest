"""业务层：学习测评工作流（跨两次 HTTP 请求的流程编排与状态持久化）。

一次测评拆成两段图、两次请求：

1. ``start``    ：prepare_learning_context -> generate_quiz -> END，
                  产出题目并落库（status=quiz_ready），返回 run_id + quiz；
2. ``submit``   ：evaluate_and_review -> END，
                  按 run_id 从 state_json 恢复题目，评分后落库（status=evaluated）。

持久化策略（对应 workflow_runs 表）：

- 每次进入图之前先把 run 置 running 并写入当前 state（调用方可读）；
- 图执行成功后写回完整 state + 目标状态（quiz_ready / evaluated）；
- 图执行失败：``mark_failed`` 直接提交（失败信息不能随异常回滚），再抛出系统异常。

图本身不碰数据库（见 app/workflows/learning_assessment.py），本层负责“读状态 -> 跑图 -> 存状态”。
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel

from app.core.exceptions import BusinessError
from app.db.models import User, WorkflowRun
from app.db.session_repository import ChatSessionRepository
from app.db.workflow_run_repository import WorkflowRunRepository
from app.schemas.learning_workflow import LearningWorkflowStartRequest, LearningWorkflowSubmitRequest
from app.services.chat_service import SessionNotFoundError
from app.workflows.learning_assessment import (
    STATUS_EVALUATED,
    STATUS_QUIZ_READY,
    STATUS_RUNNING,
    WORKFLOW_TYPE_LEARNING,
    LearningWorkflowState,
    WorkflowGenerationError,
    build_start_graph,
    build_submit_graph,
)

logger = logging.getLogger("app.learning_workflow")

__all__ = ["LearningWorkflowService", "WorkflowRunNotFoundError", "WorkflowStateError"]


class WorkflowRunNotFoundError(BusinessError):
    """业务异常：run_id 不存在或不属于当前用户。"""

    code = "WORKFLOW_RUN_NOT_FOUND"
    http_status = 404
    message = "测评任务不存在"


class WorkflowStateError(BusinessError):
    """业务异常：当前工作流状态不允许该操作（如重复提交 / 状态未就绪）。"""

    code = "WORKFLOW_STATE_INVALID"
    http_status = 409
    message = "当前测评状态不允许该操作"


class LearningWorkflowService:
    """学习测评业务逻辑：两段 LangGraph 流程 + workflow_runs 状态持久化。"""

    def __init__(
        self,
        *,
        runs: WorkflowRunRepository,
        sessions: ChatSessionRepository,
        model_factory: Callable[[str | None], BaseChatModel],
    ) -> None:
        self.runs = runs
        self.sessions = sessions
        self.model_factory = model_factory
        # 图只编译一次；provider 存在 state 里，节点执行时经 resolver 取模型实例
        self._start_graph = build_start_graph(self._resolve_model)
        self._submit_graph = build_submit_graph(self._resolve_model)

    def _resolve_model(self, state: LearningWorkflowState) -> BaseChatModel:
        """按 state.provider 解析模型（None -> 默认 provider）。"""
        return self.model_factory(state.get("provider"))

    # ------------------------------------------------------------------
    # 第一次请求：生成题目
    # ------------------------------------------------------------------
    def start(self, user: User, payload: LearningWorkflowStartRequest) -> WorkflowRun:
        """校验会话归属 -> 建 run -> 跑启动图 -> 落库并返回（含 quiz 的 run）。"""
        chat_session = self.sessions.get(payload.session_id, user.id)
        if chat_session is None:
            raise SessionNotFoundError(detail=f"session_id={payload.session_id} user_id={user.id}")

        run_id = uuid.uuid4().hex
        state: LearningWorkflowState = {
            "run_id": run_id,
            "user_id": user.id,
            "session_id": payload.session_id,
            "request_text": payload.request_text,
            "provider": payload.provider,
            "question_count": payload.question_count,
            "quiz": [],
            "answers": [],
            "evaluation": [],
            "score": None,
            "weaknesses": [],
            "review": "",
        }
        run = self.runs.create(
            run_id=run_id,
            user_id=user.id,
            session_id=payload.session_id,
            workflow_type=WORKFLOW_TYPE_LEARNING,
            status=STATUS_RUNNING,
            state=dict(state),
        )
        logger.info("学习测评开始 run_id=%s user_id=%s 题量=%s", run_id, user.id, payload.question_count)
        return self._run_graph(run, state, self._start_graph, action="生成题目")

    # ------------------------------------------------------------------
    # 第二次请求：提交答案并评分
    # ------------------------------------------------------------------
    def submit(self, user: User, run_id: str, payload: LearningWorkflowSubmitRequest) -> WorkflowRun:
        """按 run_id 恢复题目 -> 合并答案 -> 跑提交图 -> 落库并返回评分结果。"""
        run = self.runs.get(run_id, user.id)
        if run is None:
            raise WorkflowRunNotFoundError(detail=f"run_id={run_id} user_id={user.id}")
        if run.status == STATUS_EVALUATED:
            raise WorkflowStateError(message="本次测评已完成评分", detail=f"run_id={run_id} status={run.status}")
        if run.status != STATUS_QUIZ_READY:
            raise WorkflowStateError(
                message="当前测评状态不允许提交答案",
                detail=f"run_id={run_id} status={run.status}",
            )

        state: LearningWorkflowState = dict(run.state_json or {})
        quiz = state.get("quiz") or []
        answers = [answer.model_dump() for answer in payload.answers]
        known_ids = {str(question.get("question_id")) for question in quiz if isinstance(question, dict)}
        unknown = [answer["question_id"] for answer in answers if answer["question_id"] not in known_ids]
        if unknown:
            raise WorkflowStateError(message="答案包含未知题目", detail=f"unknown_question_ids={unknown}")

        state["answers"] = answers
        state["status"] = STATUS_RUNNING
        self.runs.update(run, status=STATUS_RUNNING, state=state)
        logger.info("学习测评提交 run_id=%s 作答数=%s", run_id, len(answers))
        return self._run_graph(run, state, self._submit_graph, action="评分")

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def get(self, user: User, run_id: str) -> WorkflowRun:
        """按 run_id 查当前用户的工作流运行（用于轮询 / 恢复页面）。"""
        run = self.runs.get(run_id, user.id)
        if run is None:
            raise WorkflowRunNotFoundError(detail=f"run_id={run_id} user_id={user.id}")
        return run

    # ------------------------------------------------------------------
    # 内部：跑图 + 持久化
    # ------------------------------------------------------------------
    def _run_graph(
        self,
        run: WorkflowRun,
        state: LearningWorkflowState,
        graph: Any,
        *,
        action: str,
    ) -> WorkflowRun:
        """执行一段图并写回结果；失败标记 run failed 后抛系统异常。"""
        try:
            result: dict[str, Any] = graph.invoke(state)
        except Exception as exc:  # noqa: BLE001 - 统一把图异常转成可持久化的失败
            self.runs.mark_failed(run, f"{action}失败：{exc}")
            logger.error("学习测评%s失败 run_id=%s：%r", action, run.run_id, exc, exc_info=exc)
            if isinstance(exc, WorkflowGenerationError):
                raise
            raise WorkflowGenerationError(detail=f"{action}失败：{exc}") from exc

        status = str(result.get("status") or run.status)
        completed_at = int(time.time()) if status == STATUS_EVALUATED else None
        return self.runs.update(run, status=status, state=result, completed_at=completed_at)
