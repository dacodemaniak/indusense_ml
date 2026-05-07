"""Logging helpers for ingestion workflows."""

from __future__ import annotations

from pathlib import Path

from loguru import logger


def configure_logging(log_dir: Path | str = Path("logs"), filename: str = "ingestion.log"):
    """Configure project logging and return the shared Loguru logger."""
    target_dir = Path(log_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        target_dir / filename,
        rotation="10 MB",
        retention=10,
        enqueue=True,
        backtrace=True,
        diagnose=False,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    )
    logger.add(
        lambda message: print(message, end=""),
        level="INFO",
        format="{time:HH:mm:ss} | {level} | {message}",
    )
    return logger

