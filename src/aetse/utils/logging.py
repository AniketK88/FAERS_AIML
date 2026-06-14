"""Structured logging setup using loguru.

Provides a pre-configured logger with:
- Console output with color formatting
- File output to logs/ directory with rotation
- Structured JSON log format for production use
- Log level controlled via LOG_LEVEL env var

Usage:
    from aetse.utils.logging import logger
    logger.info("Processing review", review_id="R001")
"""

import sys

from loguru import logger

from aetse.config.settings import settings


def setup_logging() -> None:
    """Configure loguru logger for AET-SE.

    Sets up console and file sinks based on settings.
    Call this once at application startup.

    Returns:
        None
    """
    # Remove default handler
    logger.remove()

    # Console handler — human-readable
    logger.add(
        sys.stderr,
        level=settings.log.level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File handler — structured, rotated
    log_file = settings.log.dir / "aetse_{time:YYYY-MM-DD}.log"
    logger.add(
        str(log_file),
        level="DEBUG",
        format="{time:YYYY-MM-DDTHH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="10 MB",
        retention="7 days",
        compression="gz",
    )

    logger.info("Logging initialized", level=settings.log.level)


# Auto-configure on import
setup_logging()
