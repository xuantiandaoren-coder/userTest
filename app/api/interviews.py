"""路由层：面试记录接口。"""

from typing import Annotated

from fastapi import APIRouter, Path

from app.api.deps import ChatServiceDep, CurrentUserDep
from app.db.models import Interview
from app.schemas.chat import InterviewPublic

router = APIRouter(prefix="/interviews", tags=["interviews"])

InterviewIdPath = Annotated[int, Path(ge=1, description="面试记录 id")]


@router.get("/{interview_id}", response_model=InterviewPublic, summary="面试详情（含 qa_object）")
def get_interview(current_user: CurrentUserDep, service: ChatServiceDep, interview_id: InterviewIdPath) -> Interview:
    """按 interview_id + 当前用户查询面试详情，越权或不存在返回 404。"""
    return service.get_interview(current_user, interview_id)
