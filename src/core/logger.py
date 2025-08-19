"""
Logging configuration for the YouTube to Twitter clipper.

This module sets up structured logging with console and file output,
proper formatting, and different log levels for development and production.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from src.core.config import config


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color coding for different log levels."""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str = "youtube_clipper",
    level: Optional[str] = None,
    log_to_file: Optional[bool] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.
    
    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_file: Log file path
        
    Returns:
        Configured logger instance
    """
    # Use config defaults if not specified
    level = level or config.LOG_LEVEL
    log_to_file = log_to_file if log_to_file is not None else config.LOG_TO_FILE
    log_file = log_file or config.LOG_FILE
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = ColoredFormatter(
        '%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if enabled)
    if log_to_file:
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(exist_ok=True)
            
            file_handler = logging.FileHandler(log_path)
            file_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Could not set up file logging: {e}")
    
    return logger


def get_logger(name: str = "youtube_clipper") -> logging.Logger:
    """
    Get a logger instance with the application's standard configuration.
    
    Args:
        name: Logger name (usually module name)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


# Create default application logger
app_logger = setup_logger()


class LoggerMixin:
    """Mixin class to add logging capability to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get a logger instance for this class."""
        return get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")


def log_function_call(func):
    """Decorator to log function calls with arguments and results."""
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {e}")
            raise
    
    return wrapper


# Test logging configuration
if __name__ == "__main__":
    logger = setup_logger()
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")
