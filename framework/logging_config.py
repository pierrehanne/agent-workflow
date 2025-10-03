"""
Logging configuration for the LLM Workflow Framework.

This module provides centralized logging configuration with support for
environment variable-based log level control and consistent formatting
across all framework components.
"""

import logging
import os
import sys
from typing import Optional

# Default log format with timestamps and context
DEFAULT_LOG_FORMAT = "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_log_level_from_env() -> int:
    """
    Get log level from environment variable.

    Reads the LOG_LEVEL environment variable and converts it to a logging
    level constant. Defaults to INFO if not set or invalid.

    Supported values (case-insensitive):
    - DEBUG: Detailed information for debugging
    - INFO: General informational messages
    - WARNING: Warning messages for potentially problematic situations
    - ERROR: Error messages for serious problems
    - CRITICAL: Critical messages for very serious errors

    Returns:
        Logging level constant (e.g., logging.INFO)

    Example:
        >>> os.environ["LOG_LEVEL"] = "DEBUG"
        >>> level = get_log_level_from_env()
        >>> print(level == logging.DEBUG)
        True
    """
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()

    level_mapping = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    return level_mapping.get(log_level_str, logging.INFO)


def configure_logging(
    level: Optional[int] = None,
    format_string: Optional[str] = None,
    date_format: Optional[str] = None,
    handler: Optional[logging.Handler] = None
) -> None:
    """
    Configure logging for the framework.

    Sets up the root logger with the specified configuration. If no level
    is provided, reads from the LOG_LEVEL environment variable.

    Args:
        level: Logging level (e.g., logging.INFO). If None, reads from env
        format_string: Custom log format string. If None, uses default
        date_format: Custom date format string. If None, uses default
        handler: Custom handler. If None, uses StreamHandler to stdout

    Example:
        >>> # Configure with default settings
        >>> configure_logging()
        >>>
        >>> # Configure with custom level
        >>> configure_logging(level=logging.DEBUG)
        >>>
        >>> # Configure with custom format
        >>> configure_logging(format_string="%(levelname)s: %(message)s")
    """
    # Determine log level
    if level is None:
        level = get_log_level_from_env()

    # Use default format if not provided
    if format_string is None:
        format_string = DEFAULT_LOG_FORMAT

    if date_format is None:
        date_format = DEFAULT_DATE_FORMAT

    # Create formatter
    formatter = logging.Formatter(format_string, datefmt=date_format)

    # Create handler if not provided
    if handler is None:
        handler = logging.StreamHandler(sys.stdout)

    handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    for existing_handler in root_logger.handlers[:]:
        root_logger.removeHandler(existing_handler)

    # Add our handler
    root_logger.addHandler(handler)

    # Log the configuration
    logger = logging.getLogger(__name__)
    logger.debug(f"Logging configured with level: {logging.getLevelName(level)}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    This is a convenience function that returns a logger with the given name.
    The logger will inherit the configuration from the root logger.

    Args:
        name: Name for the logger (typically __name__ of the module)

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("This is an info message")
    """
    return logging.getLogger(name)


# Auto-configure logging when module is imported
# This ensures logging is set up even if configure_logging() is not called explicitly
configure_logging()
