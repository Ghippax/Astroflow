"""Logging utilities for cosmo_analysis.

This module provides structured logging capabilities including:
- Hierarchical logger configuration
- Progress tracking for long-running operations
- Performance monitoring and timing
- Debug information for troubleshooting
"""

import logging
import sys
import time
import functools
from contextlib import contextmanager
from typing import Optional, Callable, Any

# Get the root logger for the library
logger = logging.getLogger('cosmo_analysis')
logger.setLevel(logging.INFO) # Default level

# Prevent the logger from propagating to the root logger
logger.propagate = False

# Default handler to avoid "No handler found" warnings
if not logger.handlers:
    default_handler = logging.StreamHandler(sys.stdout)
    default_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(default_handler)


def set_log_level(level):
    """Sets the logging level for the library.
    
    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO)
    """
    logger.setLevel(level)


def add_file_handler(log_file, level=logging.INFO):
    """Adds a file handler to the logger.
    
    Args:
        log_file: Path to log file
        level: Logging level for the file handler
    """
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


@contextmanager
def log_progress(operation: str, detail: Optional[str] = None):
    """Context manager for tracking progress of operations.
    
    Logs the start and completion of an operation, including execution time.
    Useful for long-running analysis tasks.
    
    Args:
        operation: Name of the operation being performed
        detail: Optional additional details about the operation
    
    Example:
        >>> with log_progress("Loading snapshot", "snapshot_0100"):
        ...     # Long-running operation
        ...     pass
    """
    start_time = time.time()
    msg = f"Starting: {operation}"
    if detail:
        msg += f" ({detail})"
    logger.info(msg)
    
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        msg = f"Completed: {operation} in {elapsed:.2f}s"
        if detail:
            msg += f" ({detail})"
        logger.info(msg)


def log_performance(func: Optional[Callable] = None, *, 
                   level: int = logging.INFO,
                   include_args: bool = False) -> Callable:
    """Decorator for logging function execution time and performance.
    
    Args:
        func: Function to decorate (automatically provided when used as @log_performance)
        level: Logging level to use
        include_args: Whether to include function arguments in log
    
    Example:
        >>> @log_performance
        ... def calculate_virial_radius(sim, idx):
        ...     # Computation
        ...     pass
        
        >>> @log_performance(level=logging.DEBUG, include_args=True)
        ... def find_center(sim, snapshot_idx):
        ...     # Computation
        ...     pass
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs) -> Any:
            func_name = f"{f.__module__}.{f.__qualname__}"
            
            # Build log message
            msg_parts = [f"Executing: {func_name}"]
            if include_args:
                arg_strs = [repr(a) for a in args[:2]]  # Limit to first 2 args
                if kwargs:
                    arg_strs.extend([f"{k}={v!r}" for k, v in list(kwargs.items())[:2]])
                if arg_strs:
                    msg_parts.append(f"with args: {', '.join(arg_strs)}")
            
            logger.log(level, " ".join(msg_parts))
            
            # Execute function and time it
            start_time = time.time()
            try:
                result = f(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.log(level, f"Completed: {func_name} in {elapsed:.3f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"Failed: {func_name} after {elapsed:.3f}s - {type(e).__name__}: {e}")
                raise
        
        return wrapper
    
    # Handle both @log_performance and @log_performance()
    if func is None:
        return decorator
    else:
        return decorator(func)


@contextmanager
def log_section(title: str, level: int = logging.INFO):
    """Context manager for logging major analysis sections.
    
    Creates visual separation in logs for major workflow sections.
    
    Args:
        title: Title of the section
        level: Logging level to use
        
    Example:
        >>> with log_section("Phase Space Analysis"):
        ...     # Analysis code
        ...     pass
    """
    separator = "=" * 60
    logger.log(level, f"\n{separator}")
    logger.log(level, f"  {title}")
    logger.log(level, separator)
    try:
        yield
    finally:
        logger.log(level, f"{separator}\n")


def log_data_summary(name: str, data: Any, level: int = logging.DEBUG):
    """Log a summary of data arrays or structures.
    
    Useful for debugging data loading and processing.
    
    Args:
        name: Name/description of the data
        data: Data to summarize (numpy array, list, etc.)
        level: Logging level to use
    """
    try:
        import numpy as np
        if isinstance(data, np.ndarray):
            logger.log(level, 
                f"{name}: shape={data.shape}, dtype={data.dtype}, "
                f"min={data.min():.3e}, max={data.max():.3e}, mean={data.mean():.3e}")
        elif isinstance(data, (list, tuple)):
            logger.log(level, f"{name}: length={len(data)}, type={type(data).__name__}")
        else:
            logger.log(level, f"{name}: type={type(data).__name__}, value={data}")
    except Exception as e:
        logger.log(level, f"{name}: <unable to summarize: {e}>")


class ProgressTracker:
    """Track and log progress through a series of steps.
    
    Useful for long-running workflows with multiple stages.
    
    Example:
        >>> tracker = ProgressTracker(total_steps=5, task="Analysis Pipeline")
        >>> tracker.step("Loading data")
        >>> tracker.step("Computing centers")
        >>> tracker.complete()
    """
    
    def __init__(self, total_steps: int, task: str = "Operation"):
        """Initialize progress tracker.
        
        Args:
            total_steps: Total number of steps expected
            task: Name of the overall task
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.task = task
        self.start_time = time.time()
        logger.info(f"Starting {task} ({total_steps} steps)")
    
    def step(self, description: str = ""):
        """Log completion of a step.
        
        Args:
            description: Description of the step completed
        """
        self.current_step += 1
        elapsed = time.time() - self.start_time
        percent = (self.current_step / self.total_steps) * 100
        msg = f"[{self.current_step}/{self.total_steps}] ({percent:.0f}%) {self.task}"
        if description:
            msg += f": {description}"
        msg += f" - elapsed: {elapsed:.1f}s"
        logger.info(msg)
    
    def complete(self):
        """Log completion of all steps."""
        elapsed = time.time() - self.start_time
        logger.info(f"Completed {self.task} ({self.total_steps} steps) in {elapsed:.2f}s")
