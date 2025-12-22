"""
Provides tools and utilities for logging setup and message management with
additional decorations and level-specific configurations.

This module includes methods for cleaning up logging handlers, setting up a
rotating file and console-based logging setup, and managing log levels
dynamically. It also provides helper functions to manipulate and decorate log
messages, format JSON objects, and generate QC-specific emojis for logs.

Functions:
    cleanup_logging(logger): Ensures logging handlers are cleaned up properly.
    setup_logging(verbose, project_name): Sets up comprehensive logging to both file and console.
    decorate_log_message(message, level, decorate): Adds decorations to log messages.
    get_qc_tag(msg): Generates QC emojis based on message content.
"""

import json
import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

NOW = datetime.now()
DATE_STRING = NOW.strftime("%d%b%Y")
TIME_STRING = NOW.strftime("%H-%M")

LOG_PADDING_WIDTH = 40

LOGGING = True

# Default project name - can be overridden
_DEFAULT_PROJECT_NAME = "decologr"


def cleanup_logging(logger: logging.Logger) -> None:
    """Clean up logging handlers to prevent resource warnings."""
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()


def setup_logging(verbose: bool = False, project_name: str = _DEFAULT_PROJECT_NAME) -> object:
    """Set up logging configuration
    
    Args:
        verbose: Whether to enable verbose logging
        project_name: Name of the project for logging (default: "decologr")
    
    Returns:
        Logger instance
    """
    try:
        # Create logs directory in user's home directory
        _ = logging.getLogger(project_name)
        log_dir = Path.home() / f".{project_name}" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Log file path
        log_file = log_dir / f"{project_name}-{DATE_STRING}-{TIME_STRING}.log"

        # Reset root handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # Configure rotating file logging
        file_handler = RotatingFileHandler(
            str(log_file),
            maxBytes=1024 * 1024,  # 1MB per file
            backupCount=5,  # Keep 5 backup wrappers
            encoding="utf-8",
        )
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            "%(filename)-20s| %(lineno)-5s| %(levelname)-8s| %(message)-24s"
        )
        file_handler.setFormatter(file_formatter)

        # Configure console logging
        console_handler = logging.StreamHandler(
            sys.__stdout__
        )  # Use sys.__stdout__ explicitly
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            "%(filename)-20s| %(lineno)-5s| %(levelname)-8s| %(message)-24s"
        )
        console_handler.setFormatter(console_formatter)

        # Configure root logger
        logging.root.setLevel(logging.INFO)
        logging.root.addHandler(file_handler)
        logging.root.addHandler(console_handler)

        logger = logging.getLogger(project_name)
        logger.info(f"{project_name} starting up with log file {log_file}...")
        logging.getLogger("OpenGL").setLevel(logging.WARNING)
        return logger

    except Exception as ex:
        print(f"Error setting up logging: {str(ex)}")
        raise


def _restore_log_level_from_settings(project_name: str = _DEFAULT_PROJECT_NAME):
    """Restore log level from saved preferences at startup."""
    try:
        from PySide6.QtCore import QSettings

        # Load saved log level from settings
        settings = QSettings("elmo", "preferences")
        saved_level = settings.value("log_level", None, type=str)

        if saved_level:
            # Apply the saved log level using the same comprehensive method
            _apply_log_level_comprehensive(saved_level, project_name)
            print(f"🔧 Restored log level from preferences: {saved_level}")
        else:
            print("🔧 No saved log level found, using default INFO level")

    except Exception as ex:
        print(f"⚠️ Could not restore log level from settings: {ex}")


def _apply_log_level_comprehensive(level_name: str, project_name: str = _DEFAULT_PROJECT_NAME):
    """Apply the specified log level to all loggers and handlers (comprehensive version)."""
    try:
        numeric_level = getattr(logging, level_name.upper(), logging.CRITICAL)

        # Set root logger level
        root_logger = logging.getLogger()
        root_logger.setLevel(numeric_level)

        # Set all handler levels to match
        for handler in root_logger.handlers:
            handler.setLevel(numeric_level)

        # Set ALL existing loggers to the new level
        for logger_name in logging.Logger.manager.loggerDict:
            logger = logging.getLogger(logger_name)
            logger.setLevel(numeric_level)

        # Set project-specific logger level
        project_logger = logging.getLogger(project_name)
        project_logger.setLevel(numeric_level)

        # Set OpenGL logger level
        opengl_logger = logging.getLogger("OpenGL")
        opengl_logger.setLevel(numeric_level)

    except Exception as ex:
        print(f"⚠️ Error applying log level {level_name}: {ex}")


LEVEL_EMOJIS = {
    logging.DEBUG: "🔍",
    logging.INFO: "ℹ️",
    logging.WARNING: "⚠️",
    logging.ERROR: "❌",
    logging.CRITICAL: "💥",
}


def get_qc_tag(msg: str) -> str:
    """
    get QC emoji etc
    :param msg: str
    :return: str
    """
    msg = f"{msg}".lower()
    if "success rate" in msg:
        return "📊"
    if (
        "updat" in msg
        or "success" in msg
        or "passed" in msg
        or "Enabl" in msg
        or "Setting up" in msg
    ):
        return "✅"
    if "fail" in msg or "error" in msg:
        return "❌"
    return " "


def decorate_log_message(message: str, level: int, decorate: bool = False) -> str:
    """
    Adds emoji decoration to a log message based on its content and log level.

    :param message: The original log message
    :param level: The logging level
    :param decorate: Whether to decorate the message or not
    :return: Decorated log message string
    """
    if not decorate:
        return message
    if message.startswith("{") or message.startswith(
        "["
    ):  # JSON shouldn't be decorated
        return message
    level_emoji_tag = LEVEL_EMOJIS.get(level, "🔔")
    qc_tag = get_qc_tag(message)
    return f"{level_emoji_tag}{qc_tag}{message}"


# Module-level variable to store the project name
# Can be set by calling set_project_name() or accessed directly
_project_name = _DEFAULT_PROJECT_NAME


def set_project_name(project_name: str) -> None:
    """Set the project name used by Logger for logging.
    
    Args:
        project_name: Name of the project (e.g., "mxlib", "mxpandda")
    """
    global _project_name
    _project_name = project_name


def get_project_name() -> str:
    """Get the current project name used by Logger.
    
    Returns:
        Current project name
    """
    return _project_name


class Logger:
    def __init__(self):
        pass

    @staticmethod
    def error(
        message: str,
        *args,
        exception: Optional[Exception] = None,
        level: int = logging.ERROR,
        stacklevel: int = 4,
        silent: bool = False,
    ) -> None:
        """
        Log an error message, optionally with an exception, and support lazy formatting.
        """
        if exception is not None:
            # Append the exception AFTER the message but do NOT disturb printf args
            # Example:
            #   message="could not open %s"
            #   => "could not open %s (ValueError: bad)"
            message = f"{message} ({exception.__class__.__name__}: {exception})"

        Logger.message(
            message,
            *args,
            stacklevel=stacklevel,
            silent=silent,
            level=level,
        )

    exception = error

    @staticmethod
    def warning(
        message: str,
        *args,
        exception: Optional[Exception] = None,
        level: int = logging.WARNING,
        stacklevel: int = 4,
        silent: bool = False,
    ) -> None:
        """
        Log a warning message, optionally with an exception, and support lazy formatting.
        """
        if exception is not None:
            # Append the exception AFTER the message but do NOT disturb printf args
            # Example:
            #   message="could not open %s"
            #   => "could not open %s (ValueError: bad)"
            message = f"{message} ({exception.__class__.__name__}: {exception})"

        Logger.message(
            message,
            *args,
            stacklevel=stacklevel,
            silent=silent,
            level=level,
        )

    @staticmethod
    def json(data: Any, silent: bool = False) -> None:
        """
        Log a JSON object or JSON string as a single compact line.
        """
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                Logger.message(
                    "Invalid JSON string provided.", level=logging.WARNING, stacklevel=3
                )
                return

        try:
            compact_json = json.dumps(data, separators=(",", ":"))
        except (TypeError, ValueError) as e:
            Logger.error("Failed to serialize JSON", exception=e)
            return

        if not silent:
            Logger.message(compact_json, stacklevel=3)

    @staticmethod
    def message(
        format_string: str,
        *args,
        level: int = logging.INFO,
        stacklevel: int = 3,
        silent: bool = False,
    ) -> None:

        if args:
            # Only perform printf-formatting when args provided
            try:
                formatted_message = format_string % args
            except Exception as ex:
                formatted_message = f"{format_string}  [formatting failed: {ex}]"
        else:
            formatted_message = format_string

        full_message = decorate_log_message(formatted_message, level)
        if LOGGING and not silent:
            logger = logging.getLogger(_project_name)
            logger.log(level, full_message, stacklevel=stacklevel)

    debug = message
    info = message

    @staticmethod
    def parameter(
        message: str,
        parameter: Any,
        float_precision: int = 2,
        max_length: int = 300,
        level: int = logging.INFO,
        stacklevel: int = 4,
        silent: bool = False,
    ) -> None:
        """
        Log a structured message including type and a summarized value of a parameter.
        Fast for large collections, arrays, enums, and dicts.
        """

        def format_value(param: Any) -> str:
            if param is None:
                return "None"

            # Handle enums (use .name if available, fallback to value)
            try:
                import enum

                if isinstance(param, enum.Enum):
                    return param.name
            except ImportError:
                pass

            # Float formatting
            if isinstance(param, float):
                return f"{param:.{float_precision}f}"

            # List / tuple
            if isinstance(param, (list, tuple)):
                n = len(param)
                if n > 5:
                    preview = ", ".join(str(item) for item in param[:5])
                    return f"{type(param).__name__}[len={n}, preview=[{preview}, ...]]"
                return str(param)

            # Dictionary
            if isinstance(param, dict):
                items = list(param.items())
                n = len(items)
                if n > 3:
                    preview = ", ".join(f"{k}={v}" for k, v in items[:3])
                    return (
                        f"{type(param).__name__}[len={n}, preview={{ {preview}, ... }}]"
                    )
                return str(param)

            # Bytes / bytearray
            if isinstance(param, (bytes, bytearray)):
                n = len(param)
                if n > 8:
                    preview = " ".join(f"0x{b:02X}" for b in param[:8])
                    return f"{type(param).__name__}[len={n}, preview={preview} ...]"
                return " ".join(f"0x{b:02X}" for b in param)

            # NumPy arrays
            if HAS_NUMPY:
                try:
                    if isinstance(param, np.ndarray):
                        return f"ndarray(shape={param.shape}, dtype={param.dtype})"
                except ImportError:
                    pass

            # Default string with recursion protection
            try:
                return str(param)
            except RecursionError:
                return f"<{type(param).__name__} with circular reference>"

        type_name = type(parameter).__name__
        formatted_value = format_value(parameter)

        # Truncate final string if still too long
        if len(formatted_value) > max_length:
            formatted_value = formatted_value[: max_length - 3] + "..."

        padded_message = f"{message:<{LOG_PADDING_WIDTH}}"
        padded_type = f"{type_name:<12}"
        final_message = f"{padded_message} {padded_type} {formatted_value}".rstrip()

        Logger.message(final_message, silent=silent, stacklevel=stacklevel, level=level)

    @staticmethod
    def header_message(
        message: str,
        level: int = logging.INFO,
        silent: bool = False,
        stacklevel: int = 3,
    ) -> None:
        """
        Logs a visually distinct header message with separator lines and emojis.

        :param stacklevel: int
        :param silent: bool whether or not to write to the log
        :param message: The message to log.
        :param level: Logging level (default: logging.INFO).
        """
        full_separator = f"{'=' * 142}"
        separator = f"{'=' * 100}"

        Logger.message(
            f"\n{full_separator}", level=level, stacklevel=stacklevel, silent=silent
        )
        Logger.message(f"{message}", level=level, stacklevel=stacklevel, silent=silent)
        Logger.message(separator, level=level, stacklevel=stacklevel, silent=silent)

    @staticmethod
    def debug_info(successes: list, failures: list, stacklevel: int = 3) -> None:
        """
        Logs debug information about the parsed SysEx data.

        :param stacklevel: int - stacklevel
        :param successes: list – Parameters successfully decoded.
        :param failures: list – Parameters that failed decoding.
        """
        for listing in [successes, failures]:
            try:
                listing.remove("SYNTH_TONE")
            except ValueError:
                pass  # or handle the error

        total = len(successes) + len(failures)
        success_rate = (len(successes) / total * 100) if total else 0.0

        Logger.message(
            f"Successes ({len(successes)}): {successes}", stacklevel=stacklevel
        )
        Logger.message(f"Failures ({len(failures)}): {failures}", stacklevel=stacklevel)
        Logger.message(f"Success Rate: {success_rate:.1f}%", stacklevel=stacklevel)
        Logger.message("=" * 100, stacklevel=3)


def log_exception(exception: Exception, message: str, stacklevel: int = 4) -> None:
    """
    Log an exception with a descriptive message.
    
    This function provides a convenient way to log exceptions, matching the
    signature used by mxlib.core.exception.log.log_exception for compatibility.
    
    Args:
        exception: The exception to log
        message: Descriptive message about the error context
        stacklevel: Stack level for logging (default: 4)
    
    Example:
        try:
            # some code
        except Exception as ex:
            log_exception(ex, "Error initializing scheduler database")
    """
    Logger.error(message, exception=exception, stacklevel=stacklevel)

