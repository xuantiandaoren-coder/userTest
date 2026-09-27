"""统一异常体系：业务异常（可预期）与系统异常（内部故障）。"""

from http import HTTPStatus


class AppError(Exception):
    """应用异常基类：message 面向用户（中文），detail 供后端定位。"""

    code: str = "APP_ERROR"
    http_status: int = HTTPStatus.BAD_REQUEST
    message: str = "请求处理失败"
    detail: str | None = None

    def __init__(self, message: str | None = None, detail: str | None = None) -> None:
        self.message = message or self.message
        self.detail = detail
        super().__init__(self.message)


class BusinessError(AppError):
    """业务异常：违反业务规则，HTTP 4xx，可直接展示给用户。"""


class SystemError(AppError):
    """系统异常：内部故障（如存储不可用），HTTP 5xx，细节只写入日志。"""

    code = "SYSTEM_ERROR"
    http_status = HTTPStatus.INTERNAL_SERVER_ERROR
    message = "系统繁忙，请稍后重试"
