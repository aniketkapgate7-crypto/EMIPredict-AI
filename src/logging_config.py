"""
Logging configuration for EMIPredict AI.
Provides unified console and file logging.
"""

import logging
import os
import sys
from pathlib import Path


def setup_logger(name: str = "EMIPredictAI", level: str | None = None) -> logging.Logger:
    """Configures and returns a standardized logger."""
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()

    numeric_level = getattr(logging, level, logging.INFO)
    logger = logging.getLogger(name)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if not logger.handlers:
        logger.setLevel(numeric_level)

        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler with utf-8 encoding support
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler in project root or logs dir
        log_dir = Path(__file__).resolve().parent.parent / "logs"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_dir / "emipredict.log", encoding="utf-8")
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception:
            pass  # Fallback gracefully if file system permissions prevent directory creation

    return logger
