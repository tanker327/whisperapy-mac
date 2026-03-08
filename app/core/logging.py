import sys

from loguru import logger

from app.config import Settings


def setup_logging(settings: Settings) -> None:
    """Configure Loguru for structured logging."""
    logger.remove()  # Remove default handler

    log_level = "DEBUG" if settings.debug else "INFO"
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "{extra[request_id]} | "
        "<level>{message}</level>"
    )

    logger.configure(extra={"request_id": "no-request"})
    logger.add(sys.stdout, format=log_format, level=log_level, colorize=True)
