"""校验层：健康检查响应模型。"""

from typing import Literal

from pydantic import BaseModel


class HealthReport(BaseModel):
    """健康检查结果：status 为服务状态，database 为数据库连通性。"""

    status: Literal["ok", "unhealthy"]
    database: Literal["ok", "unreachable"]
