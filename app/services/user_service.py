"""业务层：用户增删改查逻辑与密码哈希处理。

业务规则被违反时抛出 BusinessError 子类，由全局异常处理器统一转成
含 code/message/detail 的中文错误响应。
"""

from passlib.context import CryptContext
from sqlalchemy.exc import IntegrityError

from app.core.exceptions import BusinessError
from app.core.seaweedfs import SeaweedFSClient
from app.db.models import User
from app.db.user_repository import UserRepository
from app.schemas.user import UserCreate, UserPublic, UserUpdate

# bcrypt 加盐哈希；deprecated="auto" 表示未来可平滑升级哈希算法
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class UserNotFoundError(BusinessError):
    """业务异常：目标用户不存在。"""

    code = "USER_NOT_FOUND"
    http_status = 404
    message = "用户不存在"


class InvalidCredentialsError(BusinessError):
    """业务异常：用户名或密码错误（两种情况同一提示，避免枚举用户）。"""

    code = "INVALID_CREDENTIALS"
    http_status = 401
    message = "用户名或密码错误"


class UsernameExistsError(BusinessError):
    """业务异常：用户名已被占用。"""

    code = "USERNAME_ALREADY_EXISTS"
    http_status = 409
    message = "用户名已存在"


class UserService:
    """用户业务逻辑，通过仓储层读写 MySQL 用户表。"""

    def __init__(self, repository: UserRepository, storage: SeaweedFSClient) -> None:
        self.repository = repository
        self.storage = storage

    def authenticate(self, username: str, password: str) -> User:
        """校验用户名与密码；用户不存在与密码错误返回同一个错误，避免被枚举。"""
        user = self.repository.find_by_username(username)
        if user is None:
            pwd_context.dummy_verify()  # 拉平耗时，避免通过响应时间探测用户名是否存在
            raise InvalidCredentialsError(detail=f"username={username!r}")
        if not pwd_context.verify(password, user.password):
            raise InvalidCredentialsError(detail=f"username={username!r}")
        return user

    def user_exists(self, user_id: int) -> bool:
        """判断用户是否存在（刷新令牌时校验用户仍有效）。"""
        return self.repository.get(user_id) is not None

    def create_user(self, payload: UserCreate) -> UserPublic:
        """新增用户：先查重，再写入哈希后的密码。"""
        if self.repository.find_by_username(payload.username) is not None:
            raise UsernameExistsError(detail=f"username={payload.username!r}")
        try:
            user = self.repository.create(
                user_name=payload.username,
                password=pwd_context.hash(payload.password),
            )
        except IntegrityError as exc:
            # 并发写入时兜底：唯一索引冲突同样按“用户名已存在”处理
            self.repository.session.rollback()
            raise UsernameExistsError(detail=f"username={payload.username!r}") from exc
        return self.to_public(user)

    def list_users(self) -> list[UserPublic]:
        """返回全部用户（仅公开字段）。"""
        return [self.to_public(user) for user in self.repository.list_all()]

    def get_user(self, user_id: int) -> UserPublic:
        """按 id 查询用户，不存在时抛 UserNotFoundError。"""
        user = self.repository.get(user_id)
        if user is None:
            raise UserNotFoundError(detail=f"user_id={user_id}")
        return self.to_public(user)

    def update_user(self, user_id: int, payload: UserUpdate) -> UserPublic:
        """部分更新用户名/密码；只有显式传入的字段才被修改。"""
        user = self.repository.get(user_id)
        if user is None:
            raise UserNotFoundError(detail=f"user_id={user_id}")

        changes = payload.model_dump(exclude_unset=True)
        new_username = changes.get("username")
        if new_username is not None:
            existed = self.repository.find_by_username(new_username)
            if existed is not None and existed.id != user_id:
                raise UsernameExistsError(detail=f"username={new_username!r} user_id={user_id}")
            user.user_name = new_username

        if "password" in changes:
            # 明文密码只用于生成哈希，永远不会写入数据库
            user.password = pwd_context.hash(changes["password"])

        try:
            self.repository.save(user)
        except IntegrityError as exc:
            self.repository.session.rollback()
            raise UsernameExistsError(detail=f"user_id={user_id}") from exc
        return self.to_public(user)

    def delete_user(self, user_id: int) -> None:
        """删除用户，不存在时抛 UserNotFoundError。"""
        user = self.repository.get(user_id)
        if user is None:
            raise UserNotFoundError(detail=f"user_id={user_id}")
        self.repository.delete(user)

    def to_public(self, user: User) -> UserPublic:
        """ORM 对象 -> 对外响应模型（过滤 password，并把 avatar 存储键转成可访问 URL）。"""
        return UserPublic(
            id=user.id,
            username=user.user_name,
            avatar=user.avatar,
            avatar_url=self.storage.url_for(user.avatar) if user.avatar else None,
        )
