"""
decologr - Decorative Logger

A logging utility with emoji decorations and structured message formatting.
"""

from .logger import (
    Decologr,
    cleanup_logging,
    setup_logging,
    set_project_name,
    get_project_name,
    log_exception,
)

# Create a Logger instance for backward compatibility
# This allows code to use "from decologr import Logger" instead of "from decologr import Decologr"
Logger = Decologr()

__all__ = ["Decologr", "Logger", "cleanup_logging", "setup_logging", "set_project_name", "get_project_name", "log_exception"]
