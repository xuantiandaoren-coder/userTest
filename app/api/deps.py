"""路由层公共依赖：组装仓储与业务服务。"""

from collections.abc import Callable
from typing import Annotated

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core.language_models import BaseChatModel
from sqlalchemy.orm import Session

from app.core.redis_client import RedisLike, get_redis_client
from app.core.seaweedfs import SeaweedFSClient, get_seaweedfs_client
from app.core.tokens import TokenInvalidError, TokenType, decode_token
from app.db.chat_message_repository import ChatMessageRepository
from app.db.interview_repository import InterviewRepository
from app.db.models import User
from app.db.prompt_template_repository import PromptTemplateRepository
from app.db.resource_repository import ResourceRepository
from app.db.session_repository import ChatSessionRepository
from app.db.session import get_db, get_session_factory
from app.db.user_repository import UserRepository
from app.db.user_profile_repository import UserProfileRepository
from app.db.workflow_run_repository import WorkflowRunRepository
from app.llm.llm import get_chat_model
from app.prompts.prompt_template_manager import PromptTemplateManager
from app.services.chat_service import ChatService
from app.services.learning_workflow_service import LearningWorkflowService
from app.services.rag_service import RagIngestService
from app.services.stream_chat_service import StreamChatService
from app.services.upload_service_seaweedfs import SeaweedFSUploadService
from app.services.user_service import UserService


def get_storage_client() -> SeaweedFSClient:
    """依赖注入：SeaweedFS 客户端（测试可整体替换为内存实现）。"""
    return get_seaweedfs_client()


StorageClientDep = Annotated[SeaweedFSClient, Depends(get_storage_client)]


def get_user_service(db: Annotated[Session, Depends(get_db)], storage: StorageClientDep) -> UserService:
    """依赖注入：为每个请求组装一个仓储 + 对象存储客户端 + 业务服务。"""
    return UserService(UserRepository(db), storage)


ServiceDep = Annotated[UserService, Depends(get_user_service)]


def get_rag_ingest_service() -> RagIngestService:
    """依赖注入：RAG 向量化入库服务（测试可整体替换为假实现，避免连向量库）。"""
    return RagIngestService()


RagIngestServiceDep = Annotated[RagIngestService, Depends(get_rag_ingest_service)]


def get_upload_service(
    db: Annotated[Session, Depends(get_db)],
    storage: StorageClientDep,
    rag: RagIngestServiceDep,
) -> SeaweedFSUploadService:
    """依赖注入：上传服务（元数据仓储 + 用户仓储 + 对象存储客户端 + RAG 入库服务）。"""
    return SeaweedFSUploadService(ResourceRepository(db), UserRepository(db), storage, rag)


UploadServiceDep = Annotated[SeaweedFSUploadService, Depends(get_upload_service)]


def get_chat_service(db: Annotated[Session, Depends(get_db)], storage: StorageClientDep) -> ChatService:
    """依赖注入：会话 / 消息 / 面试业务服务。"""
    return ChatService(
        sessions=ChatSessionRepository(db),
        messages=ChatMessageRepository(db),
        interviews=InterviewRepository(db),
        resources=ResourceRepository(db),
        storage=storage,
    )


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]


# auto_error=False：缺少 Authorization 头时自己抛 401（而不是让 FastAPI 返回 403）
bearer_scheme = HTTPBearer(auto_error=False, description="Bearer <access token>")


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    db: Annotated[Session, Depends(get_db)],
) -> User:
    """解析 `Authorization: Bearer <access token>`，返回当前登录用户。"""
    if credentials is None or not credentials.credentials:
        raise TokenInvalidError(message="缺少访问令牌", detail="missing bearer token")

    user_id = decode_token(credentials.credentials, expected_type=TokenType.ACCESS)
    user = UserRepository(db).get(user_id)
    if user is None:  # 令牌合法但用户已被删除
        raise TokenInvalidError(detail=f"user_id={user_id}")
    return user


# 需要登录的接口：把参数声明成 CurrentUserDep 即可
CurrentUserDep = Annotated[User, Depends(get_current_user)]


# ---------------------------------------------------------------------------
# 提示词模板 / 模型 / 流式聊天依赖
# ---------------------------------------------------------------------------
def get_redis() -> RedisLike | None:
    """依赖注入：Redis 客户端（未启用返回 None，读路径自动回退数据库）。"""
    return get_redis_client()


RedisDep = Annotated[RedisLike | None, Depends(get_redis)]


def get_prompt_manager(db: Annotated[Session, Depends(get_db)], redis: RedisDep) -> PromptTemplateManager:
    """依赖注入：提示词模板管理器（版本计算 + Redis 缓存 + 公共/私有拼接 + 回滚）。"""
    return PromptTemplateManager(PromptTemplateRepository(db), redis)


PromptManagerDep = Annotated[PromptTemplateManager, Depends(get_prompt_manager)]


def get_chat_model_dep() -> BaseChatModel:
    """依赖注入：默认聊天模型（provider 由 LLM_PROVIDER 决定，默认 DeepSeek）。

    测试用 dependency_overrides 换成假模型即可，无需真实 API Key 与网络。
    """
    return get_chat_model()


ChatModelDep = Annotated[BaseChatModel, Depends(get_chat_model_dep)]


def get_persist_session_factory() -> Callable[[], Session]:
    """依赖注入：流式聊天结束后的落库会话工厂（独立于请求级会话的生命周期）。"""
    return get_session_factory()


PersistSessionFactoryDep = Annotated[Callable[[], Session], Depends(get_persist_session_factory)]


def get_stream_chat_service(
    db: Annotated[Session, Depends(get_db)],
    prompts: PromptManagerDep,
    model: ChatModelDep,
    persist_factory: PersistSessionFactoryDep,
) -> StreamChatService:
    """依赖注入：流式聊天服务（会话 + 消息 + 画像 + 提示词 + 模型 + 落库会话）。"""
    return StreamChatService(
        sessions=ChatSessionRepository(db),
        messages=ChatMessageRepository(db),
        profiles=UserProfileRepository(db),
        prompts=prompts,
        model=model,
        persist_factory=persist_factory,
        model_factory=lambda provider: get_chat_model(provider=provider),
    )


StreamChatServiceDep = Annotated[StreamChatService, Depends(get_stream_chat_service)]


# ---------------------------------------------------------------------------
# 学习测评工作流依赖
# ---------------------------------------------------------------------------
def get_workflow_model_factory() -> Callable[[str | None], BaseChatModel]:
    """依赖注入：工作流用的模型工厂（按 provider 解析，测试可整体替换为假模型）。"""
    return lambda provider=None: get_chat_model(provider=provider)


WorkflowModelFactoryDep = Annotated[Callable[[str | None], BaseChatModel], Depends(get_workflow_model_factory)]


def get_learning_workflow_service(
    db: Annotated[Session, Depends(get_db)],
    model_factory: WorkflowModelFactoryDep,
) -> LearningWorkflowService:
    """依赖注入：学习测评工作流服务（运行表 + 会话校验 + 模型工厂）。"""
    return LearningWorkflowService(
        runs=WorkflowRunRepository(db),
        sessions=ChatSessionRepository(db),
        model_factory=model_factory,
    )


LearningWorkflowServiceDep = Annotated[LearningWorkflowService, Depends(get_learning_workflow_service)]
