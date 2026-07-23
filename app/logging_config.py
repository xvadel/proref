"""
Structured logging configuration for the Personalized Prompt Refiner.

Sets up:
  - A rotating file handler → logs/proref.log  (5 MB × 3 backups)
  - A console (stderr) handler with optional color via `colorlog`
  - A consistent JSON-style format with timestamp, level, module, and message

Usage (in every module):
    from app.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("Knowledge base ingested, %d techniques loaded.", count)
    logger.error("LLM call failed", exc_info=True)

Log level is controlled by the LOG_LEVEL env variable (default: INFO).
"""

import logging
import logging.handlers
import sys
from pathlib import Path

from app.config import settings


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

_COLORED_FORMAT = (
    "%(log_color)s%(asctime)s | %(levelname)-8s%(reset)s | %(name)s | %(message)s"
)

_LOG_COLORS = {
    "DEBUG": "cyan",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bold_red",
}

# Module-level flag to ensure one-time setup
_configured = False


# ──────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────

def _setup_logging() -> None:
    """
    Configure the root logger once with file + console handlers.

    Called automatically the first time ``get_logger()`` is used.
    Idempotent — safe to call multiple times.
    """
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger()
    root.setLevel(getattr(logging, settings.log_level, logging.INFO))

    # ── Rotating file handler ──────────────────────────────────
    log_dir: Path = settings.log_dir_resolved
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "proref.log"

    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(log_file),
        maxBytes=settings.log_max_bytes,
        backupCount=settings.log_backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)  # file captures everything
    file_handler.setFormatter(
        logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)
    )
    root.addHandler(file_handler)

    # ── Console handler (colored if colorlog is available) ──────
    try:
        import colorlog  # optional dependency — graceful fallback if missing

        console_formatter = colorlog.ColoredFormatter(
            fmt=_COLORED_FORMAT,
            datefmt=_DATE_FORMAT,
            log_colors=_LOG_COLORS,
            reset=True,
            style="%",
        )
    except ImportError:
        console_formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(getattr(logging, settings.log_level, logging.INFO))
    console_handler.setFormatter(console_formatter)
    root.addHandler(console_handler)

    # Suppress overly verbose third-party loggers at WARNING and above
    for noisy_lib in (
        "chromadb",
        "sentence_transformers",
        "transformers",
        "httpx",
        "httpcore",
        "uvicorn.access",
    ):
        logging.getLogger(noisy_lib).setLevel(logging.WARNING)

    root.info(
        "Logging initialised | level=%s | file=%s",
        settings.log_level,
        log_file,
    )


# ──────────────────────────────────────────────────────────────
# Public helper
# ──────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger, triggering one-time setup on first call.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A ``logging.Logger`` instance.

    Example::

        from app.logging_config import get_logger
        logger = get_logger(__name__)
        logger.info("Server ready.")
    """
    _setup_logging()
    return logging.getLogger(name)
