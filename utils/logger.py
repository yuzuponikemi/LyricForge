"""
Structured Logging Utility for LyricForge

Provides a centralized logging system with support for:
- Console output (with optional Rich formatting)
- File output
- Multiple log levels
- Contextual logging (with run_id tracking)
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from rich.console import Console
    from rich.logging import RichHandler

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class LyricForgeLogger:
    """
    Custom logger for LyricForge with support for Rich console output.
    """

    def __init__(
        self,
        name: str = "lyric_forge",
        level: str = "INFO",
        log_file: Optional[str] = None,
        use_rich: bool = True,
    ):
        """
        Initialize the logger.

        Args:
            name: Logger name
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file (optional)
            use_rich: Use Rich library for colored console output
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        self.logger.handlers.clear()  # Remove any existing handlers

        # Console handler
        if use_rich and RICH_AVAILABLE:
            console_handler = RichHandler(
                rich_tracebacks=True,
                tracebacks_show_locals=True,
                show_time=True,
                show_path=False,
            )
        else:
            console_handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            console_handler.setFormatter(formatter)

        console_handler.setLevel(getattr(logging, level.upper()))
        self.logger.addHandler(console_handler)

        # File handler (if specified)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file
            self.logger.addHandler(file_handler)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        self._log(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message."""
        self._log(logging.CRITICAL, message, **kwargs)

    def exception(self, message: str, **kwargs: Any) -> None:
        """Log exception with traceback."""
        self.logger.exception(message, **kwargs)

    def _log(self, level: int, message: str, **kwargs: Any) -> None:
        """
        Internal logging method with context support.

        Args:
            level: Logging level
            message: Log message
            **kwargs: Additional context to include in the log
        """
        if kwargs:
            context_str = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
            message = f"{message} [{context_str}]"

        self.logger.log(level, message)


class ContextualLogger:
    """
    Logger wrapper that automatically includes context information (like run_id).
    """

    def __init__(self, logger: LyricForgeLogger, context: Dict[str, Any]):
        """
        Initialize contextual logger.

        Args:
            logger: Base logger instance
            context: Context dictionary to include in all logs
        """
        self.logger = logger
        self.context = context

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message with context."""
        self.logger.debug(message, **{**self.context, **kwargs})

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message with context."""
        self.logger.info(message, **{**self.context, **kwargs})

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message with context."""
        self.logger.warning(message, **{**self.context, **kwargs})

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message with context."""
        self.logger.error(message, **{**self.context, **kwargs})

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message with context."""
        self.logger.critical(message, **{**self.context, **kwargs})

    def exception(self, message: str, **kwargs: Any) -> None:
        """Log exception with context and traceback."""
        self.logger.exception(message, **{**self.context, **kwargs})


def setup_logger(config: Dict[str, Any]) -> LyricForgeLogger:
    """
    Set up logger from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Configured LyricForgeLogger instance
    """
    logging_config = config.get("logging", {})

    return LyricForgeLogger(
        name="lyric_forge",
        level=logging_config.get("level", "INFO"),
        log_file=logging_config.get("file"),
        use_rich=logging_config.get("rich_console", True),
    )


def get_contextual_logger(logger: LyricForgeLogger, run_id: str) -> ContextualLogger:
    """
    Create a contextual logger with run_id.

    Args:
        logger: Base logger instance
        run_id: Run identifier to include in logs

    Returns:
        ContextualLogger instance
    """
    return ContextualLogger(logger, {"run_id": run_id})


# Global logger instance (can be used for simple cases)
_global_logger: Optional[LyricForgeLogger] = None


def get_logger() -> LyricForgeLogger:
    """
    Get or create the global logger instance.

    Returns:
        Global LyricForgeLogger instance
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = LyricForgeLogger()
    return _global_logger
