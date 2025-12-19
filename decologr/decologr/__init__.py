"""
decologr - Decorative Logger

A logging utility with emoji decorations and structured message formatting.
"""

from decologr.decologr.logger import (
    Logger,
    cleanup_logging,
    setup_logging,
    set_project_name,
    get_project_name,
)

__all__ = ["Logger", "cleanup_logging", "setup_logging", "set_project_name", "get_project_name"]

