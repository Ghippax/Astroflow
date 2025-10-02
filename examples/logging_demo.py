#!/usr/bin/env python
"""Demonstration of advanced logging features in cosmo_analysis.

This script demonstrates the structured logging capabilities including:
- Progress tracking for long-running operations
- Performance monitoring for function execution
- Contextual logging for major sections
- Data summary logging for debugging

Run with different log levels to see various output:
    python logging_demo.py --log-level INFO
    python logging_demo.py --log-level DEBUG
"""

import numpy as np
import time
import argparse
import logging

from cosmo_analysis import log
from cosmo_analysis.log import (
    set_log_level,
    log_progress,
    log_performance,
    log_section,
    log_data_summary,
    ProgressTracker,
    add_file_handler
)


@log_performance
def simulate_data_loading(size=1000):
    """Simulate loading data from disk."""
    time.sleep(0.1)  # Simulate I/O
    data = np.random.random(size)
    log.logger.debug(f"Loaded {len(data)} data points")
    return data


@log_performance(level=logging.DEBUG, include_args=True)
def simulate_computation(data, threshold=0.5):
    """Simulate a computation on data."""
    time.sleep(0.05)  # Simulate computation
    result = data[data > threshold]
    log.logger.debug(f"Computation produced {len(result)} results")
    return result


def demonstrate_basic_logging():
    """Demonstrate basic logging at different levels."""
    with log_section("Basic Logging Demo"):
        log.logger.info("This is an INFO level message")
        log.logger.debug("This is a DEBUG level message (only visible with --log-level DEBUG)")
        log.logger.warning("This is a WARNING level message")


def demonstrate_progress_tracking():
    """Demonstrate progress tracking with context manager."""
    with log_section("Progress Tracking Demo"):
        with log_progress("Loading simulation data", "snapshot_0100"):
            data = simulate_data_loading(5000)
        
        with log_progress("Processing data"):
            result = simulate_computation(data, threshold=0.7)


def demonstrate_progress_tracker():
    """Demonstrate ProgressTracker for multi-step workflows."""
    with log_section("Multi-Step Workflow Demo"):
        tracker = ProgressTracker(total_steps=5, task="Full Analysis Pipeline")
        
        tracker.step("Loading snapshot data")
        data = simulate_data_loading(3000)
        time.sleep(0.1)
        
        tracker.step("Computing centers")
        time.sleep(0.15)
        
        tracker.step("Calculating virial radius")
        time.sleep(0.12)
        
        tracker.step("Computing orientation axes")
        time.sleep(0.08)
        
        tracker.step("Generating plots")
        time.sleep(0.1)
        
        tracker.complete()


def demonstrate_data_summary():
    """Demonstrate data summary logging for debugging."""
    with log_section("Data Summary Demo"):
        # Create some example data
        positions = np.random.random((1000, 3)) * 100  # kpc
        masses = np.random.lognormal(6, 1, 1000)  # Msun
        velocities = np.random.normal(0, 50, (1000, 3))  # km/s
        
        log.logger.info("Logging data summaries for debugging:")
        log_data_summary("Positions", positions)
        log_data_summary("Masses", masses)
        log_data_summary("Velocities", velocities)


def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring with decorators."""
    with log_section("Performance Monitoring Demo"):
        log.logger.info("Functions decorated with @log_performance automatically log execution time")
        
        # These are automatically timed by the decorator
        data1 = simulate_data_loading(2000)
        result1 = simulate_computation(data1, threshold=0.6)
        
        data2 = simulate_data_loading(5000)
        result2 = simulate_computation(data2, threshold=0.8)


def demonstrate_nested_contexts():
    """Demonstrate nested logging contexts."""
    with log_section("Nested Context Demo"):
        log.logger.info("Starting analysis of multiple snapshots")
        
        for snapshot_idx in range(3):
            with log_progress(f"Analyzing snapshot {snapshot_idx}", f"snapshot_{snapshot_idx:04d}"):
                data = simulate_data_loading(1000)
                result = simulate_computation(data)
                log.logger.info(f"  Found {len(result)} particles above threshold")


def main():
    """Run all demonstrations."""
    parser = argparse.ArgumentParser(description="Demonstrate cosmo_analysis logging features")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level (default: INFO)"
    )
    parser.add_argument(
        "--log-file",
        help="Optional log file to write to (in addition to console)"
    )
    args = parser.parse_args()
    
    # Configure logging
    log_level = getattr(logging, args.log_level)
    set_log_level(log_level)
    
    if args.log_file:
        add_file_handler(args.log_file, level=log_level)
        log.logger.info(f"Logging to file: {args.log_file}")
    
    # Print header
    print("\n" + "=" * 70)
    print("  Cosmo Analysis - Advanced Logging Features Demo")
    print("=" * 70 + "\n")
    
    # Run demonstrations
    demonstrate_basic_logging()
    demonstrate_progress_tracking()
    demonstrate_progress_tracker()
    demonstrate_performance_monitoring()
    demonstrate_data_summary()
    demonstrate_nested_contexts()
    
    # Print footer
    print("\n" + "=" * 70)
    print("  Demo Complete!")
    print("=" * 70 + "\n")
    
    log.logger.info("Try running with --log-level DEBUG to see more detailed output")


if __name__ == "__main__":
    main()
