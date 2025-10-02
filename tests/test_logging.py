"""Tests for advanced logging functionality."""

import pytest
import logging
import time
from io import StringIO
import sys

from cosmo_analysis import log
from cosmo_analysis.log import (
    set_log_level,
    log_progress,
    log_performance,
    log_section,
    log_data_summary,
    ProgressTracker
)


# Helper to capture logs from our custom logger
@pytest.fixture
def log_capture():
    """Fixture to capture logs from cosmo_analysis logger."""
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    
    # Add handler
    log.logger.addHandler(handler)
    
    yield stream
    
    # Remove handler after test
    log.logger.removeHandler(handler)


class TestLogConfiguration:
    """Test basic log configuration."""
    
    def test_set_log_level(self):
        """Test setting log level."""
        # Should not raise an error
        set_log_level(logging.DEBUG)
        assert log.logger.level == logging.DEBUG
        
        set_log_level(logging.INFO)
        assert log.logger.level == logging.INFO
    
    def test_add_file_handler(self, tmp_path):
        """Test adding file handler."""
        log_file = tmp_path / "test.log"
        
        # Remove existing handlers to test cleanly
        original_handlers = log.logger.handlers[:]
        
        log.add_file_handler(str(log_file))
        
        # Should have added one handler
        assert len(log.logger.handlers) > len(original_handlers)
        
        # Restore original handlers
        log.logger.handlers = original_handlers


class TestProgressTracking:
    """Test progress tracking utilities."""
    
    def test_log_progress_context(self, log_capture):
        """Test log_progress context manager."""
        with log_progress("Test Operation", "detail info"):
            time.sleep(0.01)  # Simulate work
        
        # Check that start and complete messages were logged
        output = log_capture.getvalue()
        assert "Starting: Test Operation" in output
        assert "Completed: Test Operation" in output
    
    def test_log_section_context(self, log_capture):
        """Test log_section context manager."""
        with log_section("Analysis Section"):
            pass
        
        output = log_capture.getvalue()
        assert "Analysis Section" in output


class TestPerformanceMonitoring:
    """Test performance monitoring decorator."""
    
    def test_log_performance_decorator(self, log_capture):
        """Test log_performance decorator without arguments."""
        @log_performance
        def test_function(x, y):
            return x + y
        
        result = test_function(2, 3)
        
        assert result == 5
        output = log_capture.getvalue()
        assert "Executing:" in output and "test_function" in output
        assert "Completed:" in output and "test_function" in output
    
    def test_log_performance_with_args(self, log_capture):
        """Test log_performance decorator with arguments."""
        # Set to DEBUG level to capture debug logs
        original_level = log.logger.level
        log.logger.setLevel(logging.DEBUG)
        
        @log_performance(level=logging.DEBUG, include_args=True)
        def test_function(x, y):
            return x * y
        
        result = test_function(4, 5)
        
        # Restore original level
        log.logger.setLevel(original_level)
        
        assert result == 20
        output = log_capture.getvalue()
        assert "test_function" in output
    
    def test_log_performance_with_exception(self, log_capture):
        """Test log_performance decorator with exception."""
        @log_performance
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            failing_function()
        
        output = log_capture.getvalue()
        assert "Failed:" in output


class TestDataSummary:
    """Test data summary logging."""
    
    def test_log_data_summary_array(self, log_capture):
        """Test logging numpy array summary."""
        import numpy as np
        
        # Set to DEBUG level to capture debug logs
        original_level = log.logger.level
        log.logger.setLevel(logging.DEBUG)
        
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        log_data_summary("test_array", data)
        
        # Restore original level
        log.logger.setLevel(original_level)
        
        output = log_capture.getvalue()
        assert "test_array" in output and "shape=" in output
    
    def test_log_data_summary_list(self, log_capture):
        """Test logging list summary."""
        # Set to DEBUG level to capture debug logs
        original_level = log.logger.level
        log.logger.setLevel(logging.DEBUG)
        
        data = [1, 2, 3, 4, 5]
        log_data_summary("test_list", data)
        
        # Restore original level
        log.logger.setLevel(original_level)
        
        output = log_capture.getvalue()
        assert "test_list" in output and "length=" in output


class TestProgressTracker:
    """Test ProgressTracker class."""
    
    def test_progress_tracker_initialization(self, log_capture):
        """Test ProgressTracker initialization."""
        tracker = ProgressTracker(total_steps=3, task="Test Task")
        
        output = log_capture.getvalue()
        assert "Starting Test Task" in output
    
    def test_progress_tracker_steps(self, log_capture):
        """Test ProgressTracker step logging."""
        tracker = ProgressTracker(total_steps=3, task="Test Task")
        tracker.step("Step 1")
        tracker.step("Step 2")
        tracker.step("Step 3")
        
        output = log_capture.getvalue()
        assert "[1/3]" in output
        assert "[2/3]" in output
        assert "[3/3]" in output
    
    def test_progress_tracker_complete(self, log_capture):
        """Test ProgressTracker completion."""
        tracker = ProgressTracker(total_steps=2, task="Test Task")
        tracker.step()
        tracker.step()
        tracker.complete()
        
        output = log_capture.getvalue()
        assert "Completed Test Task" in output


class TestIntegration:
    """Integration tests for logging features."""
    
    def test_nested_logging_contexts(self, log_capture):
        """Test nested logging contexts."""
        with log_section("Main Analysis"):
            with log_progress("Sub-operation"):
                time.sleep(0.01)
        
        # Should have logged all context messages
        output = log_capture.getvalue()
        assert "Main Analysis" in output
        assert "Sub-operation" in output
    
    def test_logging_with_multiple_trackers(self, log_capture):
        """Test multiple progress trackers."""
        tracker1 = ProgressTracker(total_steps=2, task="Task 1")
        tracker1.step("Step 1")
        
        tracker2 = ProgressTracker(total_steps=2, task="Task 2")
        tracker2.step("Step 1")
        
        tracker1.step("Step 2")
        tracker2.step("Step 2")
        
        tracker1.complete()
        tracker2.complete()
        
        output = log_capture.getvalue()
        assert "Task 1" in output
        assert "Task 2" in output


class TestBackwardCompatibility:
    """Test backward compatibility of logging module."""
    
    def test_logger_exists(self):
        """Test that logger object exists."""
        assert hasattr(log, 'logger')
        assert isinstance(log.logger, logging.Logger)
    
    def test_set_log_level_exists(self):
        """Test that set_log_level function exists."""
        assert hasattr(log, 'set_log_level')
        assert callable(log.set_log_level)
    
    def test_add_file_handler_exists(self):
        """Test that add_file_handler function exists."""
        assert hasattr(log, 'add_file_handler')
        assert callable(log.add_file_handler)
    
    def test_logger_can_log(self, log_capture):
        """Test that logger can perform basic logging."""
        log.logger.info("Test message")
        log.logger.debug("Debug message")
        log.logger.warning("Warning message")
        
        output = log_capture.getvalue()
        assert "Test message" in output
        assert "Warning message" in output
