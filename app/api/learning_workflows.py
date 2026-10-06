"""路由层：学习测评工作流接口。

- POST /learning-workflows/start              发起测评，返回 run_id + 标准题目
- POST /learning-workflows/{run_id}/submit    提交整套答案，返回评分 / 点评 / 薄弱点 / 建议
- GET  /learning-workflows/{run_id}           查询运行状态与完整状态快照

本层只做鉴权、参数声明与响应组装；流程编排与状态持久化在
app/services/learning_workflow_service.py，图在 app/workflows/learning_assessment.py。
"""

from typing import Annotated

from fastapi import APIRouter, Path, status

from app.api.deps import CurrentUserDep, LearningWorkflowServiceDep
from app.schemas.learning_workflow import (
    LearningWorkflowDetailResponse,
    LearningWorkflowStartRequest,
    LearningWorkflowStartResponse,
    LearningWorkflowSubmitRequest,
    LearningWorkflowSubmitResponse,
)
from app.workflows.learning_assessment import public_quiz, public_state

router = APIRouter(prefix="/learning-workflows", tags=["learning-workflows"])

RunIdPath = Annotated[str, Path(min_length=1, max_length=64, description="工作流运行 id（start 返回的 run_id）")]


@router.post(
    "/start",
    response_model=LearningWorkflowStartResponse,
    status_code=status.HTTP_201_CREATED,
    summary="发起学习测评（生成题目）",
)
def start_learning_workflow(
    payload: LearningWorkflowStartRequest,
    current_user: CurrentUserDep,
    service: LearningWorkflowServiceDep,
) -> LearningWorkflowStartResponse:
    """按测评需求生成结构化题目，落库为 quiz_ready，返回 run_id 与题目。"""
    run = service.start(current_user, payload)
    state = run.state_json or {}
    return LearningWorkflowStartResponse(
        run_id=run.run_id,
        status=run.status,
        # 对外剥离标准答案：答案只留在服务端 state，供第二次请求阅卷
        quiz=public_quiz(state.get("quiz") or []),
    )


@router.post(
    "/{run_id}/submit",
    response_model=LearningWorkflowSubmitResponse,
    summary="提交答案（评分 / 点评 / 薄弱点 / 建议）",
)
def submit_learning_workflow(
    payload: LearningWorkflowSubmitRequest,
    current_user: CurrentUserDep,
    service: LearningWorkflowServiceDep,
    run_id: RunIdPath,
) -> LearningWorkflowSubmitResponse:
    """按 run_id 恢复题目，合并答案后一次性完成评分、逐题点评、薄弱点与学习建议。"""
    run = service.submit(current_user, run_id, payload)
    state = run.state_json or {}
    return LearningWorkflowSubmitResponse(
        run_id=run.run_id,
        status=run.status,
        score=state.get("score"),
        evaluation=state.get("evaluation") or [],
        weaknesses=state.get("weaknesses") or [],
        review=state.get("review") or "",
    )


@router.get("/{run_id}", response_model=LearningWorkflowDetailResponse, summary="查询测评运行状态")
def get_learning_workflow(
    current_user: CurrentUserDep,
    service: LearningWorkflowServiceDep,
    run_id: RunIdPath,
) -> LearningWorkflowDetailResponse:
    """返回 run_id、当前状态与完整状态快照（state 用于前端恢复页面 / 排查）。"""
    run = service.get(current_user, run_id)
    return LearningWorkflowDetailResponse(
        run_id=run.run_id,
        status=run.status,
        # 对外状态同样剥离 quiz 里的标准答案（答案只在服务端 state_json 里）
        state=public_state(run.state_json or {}),
    )
