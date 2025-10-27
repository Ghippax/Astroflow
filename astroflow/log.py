"""
Logging utilities for the Astroflow package.

This module provides a centralized logging system with configurable levels
and performance monitoring capabilities.
"""

import logging
import time
from functools import wraps
from typing import Callable, Optional


class AstroflowLogger:
    """
    Centralized logger for Astroflow with configuration support.
    
    Attributes
    ----------
    _logger : logging.Logger
        The underlying Python logger instance
    """
    
    def __init__(self, name: str = "Astroflow"):
        self._logger = logging.getLogger(name)
        self._setup_logger()
    
    def _setup_logger(self):
        """Configure the logger with settings from config."""
        # Start with INFO, can be changed via set_level()
        self._logger.setLevel(logging.INFO)

        # Only add handler if none exist
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(name)s : [%(levelname)s] %(asctime)s %(message)s")
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
    
    def debug(self, msg: str, *args, **kwargs):
        """Log a debug message."""
        self._logger.debug(msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        """Log an info message."""
        self._logger.info(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        """Log a warning message."""
        self._logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        """Log an error message."""
        self._logger.error(msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        """Log a critical message."""
        self._logger.critical(msg, *args, **kwargs)
    
    def set_level(self, level: str):
        """
        Set the logging level.
        
        Parameters
        ----------
        level : str
            Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        """
        try:
            level_val = getattr(logging, level.upper(), logging.INFO)
        except Exception as e:
            self._logger.error(f"Invalid logging level '{level}': {e}. "
                               f"Use one of DEBUG, INFO, WARNING, ERROR, CRITICAL. "
                               f"Defaulting to INFO")
            level_val = logging.INFO
        self._logger.setLevel(level_val)
        self.info(f"Logging level set to {level.upper()}")


def log_performance(func: Optional[Callable] = None, *, threshold: float = 0.0):
    """
    Decorator to log function execution time.
    
    Parameters
    ----------
    func : Callable, optional
        The function to decorate
    threshold : float, default=0.0
        Only log if execution time exceeds this threshold (in seconds)
    
    Returns
    -------
    Callable
        Decorated function
    
    Examples
    --------
    >>> @log_performance
    ... def slow_function():
    ...     time.sleep(2)
    
    >>> @log_performance(threshold=0.5)
    ... def another_function():
    ...     time.sleep(1)
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = f(*args, **kwargs)
            elapsed = time.perf_counter() - start_time
            
            if elapsed >= threshold:
                logger.info(
                    f"Function '{f.__name__}' took {elapsed:.3f}s to execute"
                )
            else:
                logger.debug(
                    f"Function '{f.__name__}' took {elapsed:.3f}s to execute"
                )
            
            return result
        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)


# Default logger instance
logger = AstroflowLogger()


def get_logger(name: Optional[str] = None) -> AstroflowLogger:
    """
    Get a logger instance.
    
    Parameters
    ----------
    name : str, optional
        Name for the logger. If None, returns the default astroflow logger.
    
    Returns
    -------
    AstroflowLogger
        Logger instance
    """
    if name is None:
        return logger
    return AstroflowLogger(name)

def set_log_level(level: str = "INFO", name: Optional[str] = None):
    """
    Set the logging level for a logger.
    
    Parameters
    ----------
    name : str, optional
        Name of the logger. If None, sets level for the default astroflow logger.
    level : str
        Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
    """
    log = get_logger(name)
    log.set_level(level)