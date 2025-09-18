"""
Logging utilities for multi-agent collaborative learning.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union


def setup_logger(
    name: str = "multi_agent_collab_learning",
    level: Union[str, int] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger for the multi-agent collaborative learning library.

    Args:
        name: Name of the logger
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to
        format_string: Custom format string for log messages

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding multiple handlers if logger already exists
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "multi_agent_collab_learning") -> logging.Logger:
    """
    Get an existing logger or create a new one with default settings.

    Args:
        name: Name of the logger

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # If logger doesn't exist or has no handlers, set it up
    if not logger.handlers:
        return setup_logger(name)

    return logger