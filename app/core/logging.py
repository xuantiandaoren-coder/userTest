"""日志体系：控制台 + 轮转文件双通道，一次配置、不重复记录。

日志级别约定（按“需不需要人介入”分级，避免日志噪音）：

- INFO   ：访问日志（每个请求一条）、预期内的业务异常（如用户不存在、用户名重复）
- WARNING：参数校验失败、路由未命中（可能是调用方契约不一致，值得关注）
- ERROR  ：系统异常与未捕获异常，带完整堆栈，用于排障

文件日志用于留痕与排障：按大小轮转、保留固定份数，避免写满磁盘。
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from app.core.config import Settings, settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s:%(lineno)d | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 标记本模块添加的 handler，重复调用 setup_logging 时据此替换而不是叠加
_HANDLER_FLAG = "_user_api_handler"


def setup_logging(config: Settings | None = None) -> None:
    """配置根日志：控制台 + 轮转文件；重复调用不会产生重复日志。"""
    config = config or settings

    root = logging.getLogger()
    root.setLevel(config.log_level)

    # 先摘掉上一次由本模块添加的 handler，保证幂等（uvicorn --reload 会重复导入应用）
    for handler in [item for item in root.handlers if getattr(item, _HANDLER_FLAG, False)]:
        root.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    _mark(console)
    root.addHandler(console)

    # 留痕：文件与控制台各写一份，按大小轮转
    config.log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        config.log_dir / config.log_file,
        maxBytes=config.log_max_bytes,
        backupCount=config.log_backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    _mark(file_handler)
    root.addHandler(file_handler)

    _route_framework_loggers(config)


def _mark(handler: logging.Handler) -> None:
    setattr(handler, _HANDLER_FLAG, True)


def _route_framework_loggers(config: Settings) -> None:
    """把框架日志并进统一通道，并关掉重复 / 噪音日志。"""
    # uvicorn 的启动、异常日志交给根日志器统一输出（控制台 + 文件）
    for name in ("uvicorn", "uvicorn.error"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True

    # 访问日志由 app.access 中间件统一记录，关掉 uvicorn 自带的那条，避免同一请求记两遍
    access_logger = logging.getLogger("uvicorn.access")
    access_logger.handlers.clear()
    access_logger.propagate = False
    access_logger.disabled = True

    # 第三方库降噪：正常运行时不需要它们的过程日志
    logging.getLogger("passlib").setLevel(logging.WARNING)
    if not config.db_echo:
        logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
