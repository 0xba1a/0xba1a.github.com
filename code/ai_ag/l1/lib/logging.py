"""
Logging Library - Setup logging for applications
"""

import logging
import os
from typing import Optional


def setup_logging(program_name: str, log_dir: Optional[str] = None) -> logging.Logger:
    """
    Setup logging with both trace and info level logs.

    Args:
        program_name: Name of the program (used for log directory)
        log_dir: Optional custom log directory. If None, uses /tmp/<program_name>

    Returns:
        Logger instance configured for the application
    """
    # Determine log directory
    if log_dir is None:
        log_dir = f"/tmp/{program_name}"

    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Define log file paths
    trace_log = os.path.join(log_dir, "trace.log")
    info_log = os.path.join(log_dir, "info.log")

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Capture all levels

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Trace log handler (DEBUG and above)
    trace_handler = logging.FileHandler(trace_log, mode='a')
    trace_handler.setLevel(logging.DEBUG)
    trace_handler.setFormatter(detailed_formatter)
    logger.addHandler(trace_handler)

    # Info log handler (INFO and above)
    info_handler = logging.FileHandler(info_log, mode='a')
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(simple_formatter)
    logger.addHandler(info_handler)

    # Console handler (INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    # Log the initialization
    logger.info(f"\n\n{'='*60}\nStarting {program_name}\n{'='*60}")
    logger.info(f"Logging initialized for {program_name}")
    logger.debug(f"Trace log: {trace_log}")
    logger.debug(f"Info log: {info_log}")

    return logger
