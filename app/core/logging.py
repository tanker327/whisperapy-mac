import inspect
import logging
import sys

from loguru import logger

from app.config import Settings

_TEXT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "{extra[request_id]} | "
    "<level>{message}</level>"
)


class _InterceptHandler(logging.Handler):
    """Route stdlib logging (uvicorn, httpx, ...) through loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: int | str = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        # Walk out of the stdlib logging frames so loguru reports the real
        # caller (uvicorn, httpx) instead of logging/__init__.py.
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging(settings: Settings) -> None:
    """Configure loguru: console sink, optional rotating file, JSON option."""
    logger.remove()
    logger.configure(extra={"request_id": "no-request"})

    level = "DEBUG" if settings.debug else "INFO"
    json_mode = settings.log_format == "json"

    logger.add(
        sys.stdout,
        level=level,
        format="{message}" if json_mode else _TEXT_FORMAT,
        serialize=json_mode,
        colorize=not json_mode,
    )
    if settings.log_file is not None:
        settings.log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            settings.log_file,
            level=level,
            format="{message}" if json_mode else _TEXT_FORMAT,
            serialize=json_mode,
            rotation=settings.log_rotation,
            retention=settings.log_retention,
            enqueue=True,
        )

    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "httpx"):
        logging.getLogger(name).handlers = [_InterceptHandler()]
        logging.getLogger(name).propagate = False
