"""
Logging configuration for the backend.

Provides consistent logging across all backend modules.
"""

import logging
import sys
from typing import Optional

from config import LOG_LEVEL, LOG_FORMAT


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (typically __name__).
        level: Optional level override.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
    
    log_level = level or LOG_LEVEL
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    return logger


# Pre-configured loggers for common modules
app_logger = get_logger("nyay_sathi")
rag_logger = get_logger("rag_engine")
