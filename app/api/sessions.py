"""路由层：会话与聊天消息接口。

业务规则与级联删除在 app/services/chat_service.py；本层只做鉴权、参数声明与分页响应组装。
"""

from typing import Annotated

from fastapi import APIRouter, Path, Query, status
from fastapi.responses import StreamingResponse

from app.api.deps import ChatServiceDep, CurrentUserDep, StreamChatServiceDep
from app.core.sse import SSE_HEADERS, SSE_MEDIA_TYPE
from app.db.models import ChatSession
from app.schemas.chat import (
    MessagePublic,
    SessionCreate,
    SessionPublic,
    SessionUpdate,
    StreamChatRequest,
)
from app.schemas.common import Page

router = APIRouter(prefix="/sessions", tags=["sessions"])

PageQuery = Annotated[int, Query(ge=1, description="页码，从 1 开始")]
PageSizeQuery = Annotated[int, Query(ge=1, le=100, description="每页条数")]
SessionIdPath = Annotated[int, Path(ge=1, description="会话 id")]


@router.get("", response_model=Page[SessionPublic], summary="会话列表（分页，按当前用户）")
def list_sessions(
    current_user: CurrentUserDep,
    service: ChatServiceDep,
    page: PageQuery = 1,
    page_size: PageSizeQuery = 20,
) -> Page[SessionPublic]:
    """分页返回当前用户的会话，新建的在前。"""
    items, total = service.list_sessions(current_user, page=page, page_size=page_size)
    return Page(items=items, total=total, page=page, page_size=page_size)


@router.post("", response_model=SessionPublic, status_code=status.HTTP_201_CREATED, summary="创建会话主题")
def create_session(payload: SessionCreate, current_user: CurrentUserDep, service: ChatServiceDep) -> ChatSession:
    """创建会话（标题 + 会话类型）。"""
    return service.create_session(current_user, payload)


@router.put("/{session_id}", response_model=SessionPublic, summary="编辑会话标题")
def update_session(
    payload: SessionUpdate,
    current_user: CurrentUserDep,
    service: ChatServiceDep,
    session_id: SessionIdPath,
) -> ChatSession:
    """修改会话标题，只能改自己的会话。"""
    return service.rename_session(current_user, session_id, payload)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT, summary="删除会话（级联）")
async def delete_session(current_user: CurrentUserDep, service: ChatServiceDep, session_id: SessionIdPath) -> None:
    """删除会话及其消息、面试记录，并清理附件对应的对象存储文件与元数据。"""
    await service.delete_session(current_user, session_id)


@router.get("/{session_id}/messages", response_model=Page[MessagePublic], summary="会话消息（分页）")
def list_messages(
    current_user: CurrentUserDep,
    service: ChatServiceDep,
    session_id: SessionIdPath,
    page: PageQuery = 1,
    page_size: PageSizeQuery = 20,
) -> Page[MessagePublic]:
    """分页返回消息：正文 + 附件段（file/image/audio）+ status/interview_id/created_at。"""
    items, total = service.list_messages(current_user, session_id, page=page, page_size=page_size)
    return Page(items=items, total=total, page=page, page_size=page_size)


@router.post("/{session_id}/stream-chat", summary="流式聊天（SSE）")
async def stream_chat(
    payload: StreamChatRequest,
    current_user: CurrentUserDep,
    service: StreamChatServiceDep,
    session_id: SessionIdPath,
) -> StreamingResponse:
    """流式生成回答，按 SSE 下发：`meta` -> `delta`* -> `done`（失败为 `error`）。

    校验（会话归属 404 / 智能体 422 / 模板缺失 404）在返回流之前完成，
    因此这些错误仍是标准 JSON 响应；一旦开始下发，错误只能通过 `error` 事件通知前端。
    流结束后把这一轮问答写入 chat_messages（独立会话落库，不受请求生命周期影响）。
    """
    plan = await service.prepare(current_user, session_id, payload)
    return StreamingResponse(
        service.stream(plan),
        media_type=SSE_MEDIA_TYPE,
        headers=SSE_HEADERS,
    )
