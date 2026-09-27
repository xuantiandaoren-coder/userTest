"""全局异常处理器：把各类异常统一转换为 code/message/detail 响应并分级记日志。

级别约定见 app/core/logging.py：业务异常 INFO、参数/路由问题 WARNING、系统故障 ERROR。
"""

import logging
from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.exceptions import BusinessError, SystemError
from app.schemas.common import ErrorResponse

logger = logging.getLogger("app.errors")

# 常见 HTTP 状态码 -> (统一 code, 中文提示)
_HTTP_ERRORS = {
    400: ("BAD_REQUEST", "请求参数不正确"),
    401: ("UNAUTHORIZED", "未认证或登录已过期"),
    403: ("FORBIDDEN", "没有权限执行该操作"),
    404: ("NOT_FOUND", "请求的资源不存在"),
    405: ("METHOD_NOT_ALLOWED", "请求方法不允许"),
    409: ("CONFLICT", "资源状态冲突"),
    422: ("UNPROCESSABLE_ENTITY", "请求无法被处理"),
}


def _error_response(status_code: int, code: str, message: str, detail: str | None = None) -> JSONResponse:
    """构造统一错误响应体：code/message/detail。"""
    payload = ErrorResponse(code=code, message=message, detail=detail).model_dump(exclude_none=True)
    return JSONResponse(status_code=status_code, content=payload)


def register_exception_handlers(app: FastAPI) -> FastAPI:
    """注册全局异常处理器，返回同一个 app 便于链式调用。"""

    @app.exception_handler(BusinessError)
    def handle_business_error(request: Request, exc: BusinessError) -> JSONResponse:
        """业务异常：预期内的 4xx，INFO 留痕即可，不占用告警级别。"""
        logger.info(
            "business error path=%s method=%s code=%s detail=%s",
            request.url.path,
            request.method,
            exc.code,
            exc.detail,
        )
        return _error_response(exc.http_status, exc.code, exc.message, exc.detail)

    @app.exception_handler(SystemError)
    def handle_system_error(request: Request, exc: SystemError) -> JSONResponse:
        """显式抛出的系统异常：完整堆栈进日志，响应不暴露内部细节。"""
        logger.error(
            "system error path=%s method=%s code=%s detail=%s",
            request.url.path,
            request.method,
            exc.code,
            exc.detail,
            exc_info=exc,
        )
        return _error_response(exc.http_status, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        """请求参数校验失败：字段级错误汇总到 detail。"""
        errors = []
        for error in exc.errors():
            location = ".".join(str(part) for part in error.get("loc", ()) if part != "body")
            message = error.get("msg", "invalid")
            errors.append(f"{location}: {message}" if location else message)
        detail = "; ".join(errors) or None
        logger.warning("param error path=%s method=%s errors=%s", request.url.path, request.method, detail)
        return _error_response(HTTPStatus.UNPROCESSABLE_ENTITY, "PARAMETER_ERROR", "请求参数校验失败", detail)

    @app.exception_handler(StarletteHTTPException)
    def handle_http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        """未匹配路由等框架异常：按状态码映射为中文提示。"""
        code, message = _HTTP_ERRORS.get(exc.status_code, (f"HTTP_{exc.status_code}", f"HTTP {exc.status_code} 错误"))
        detail = f"path={request.url.path} method={request.method}"
        logger.warning("http error path=%s method=%s status=%s", request.url.path, request.method, exc.status_code)
        return _error_response(exc.status_code, code, message, detail)

    @app.exception_handler(Exception)
    def handle_unhandled_error(request: Request, exc: Exception) -> JSONResponse:
        """兜底系统异常：保留原始堆栈便于定位，返回统一 500 响应。"""
        logger.error(
            "unhandled error path=%s method=%s error=%r",
            request.url.path,
            request.method,
            exc,
            exc_info=exc,
        )
        return _error_response(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "SYSTEM_ERROR",
            "系统繁忙，请稍后重试",
        )

    return app
