# Advanced Logging Guide

This guide explains the advanced logging features added to `cosmo_analysis` for better debugging, performance monitoring, and progress tracking.

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Progress Tracking](#progress-tracking)
- [Performance Monitoring](#performance-monitoring)
- [Contextual Logging](#contextual-logging)
- [Data Summary Logging](#data-summary-logging)
- [Multi-Step Workflows](#multi-step-workflows)
- [Configuration](#configuration)
- [Migration from constants.py](#migration-from-constantspy)

## Overview

The enhanced logging system provides:

- **Structured Logging**: Consistent, parseable log messages
- **Progress Tracking**: Real-time feedback for long-running operations
- **Performance Monitoring**: Automatic timing of function execution
- **Debug Information**: Detailed data summaries for troubleshooting
- **Backward Compatibility**: All existing code continues to work

## Quick Start

```python
from cosmo_analysis import log
from cosmo_analysis.log import log_progress, log_performance, ProgressTracker

# Basic logging
log.logger.info("Starting analysis")
log.logger.debug("Detailed debug information")

# Track progress of an operation
with log_progress("Loading simulation data", "snapshot_0100"):
    data = load_snapshot()

# Automatic performance monitoring
@log_performance
def analyze_data(data):
    # Your analysis code
    return results
```

## Progress Tracking

### Using `log_progress` Context Manager

The `log_progress` context manager automatically logs the start and completion of operations:

```python
from cosmo_analysis.log import log_progress

with log_progress("Computing centers", "snapshot detail"):
    center = findCenter7(sim, idx)
# Logs:
# INFO - Starting: Computing centers (snapshot detail)
# INFO - Completed: Computing centers in 2.35s (snapshot detail)
```

### Real-World Example from sim_prop.py

```python
def getRvir(sim, idx, method="Vir", rvirlim=500):
    """Calculate virial radius with automatic logging."""
    log.logger.info(f"Calculating virial radius for snapshot {idx} using {method} method")
    
    # Detailed progress messages at DEBUG level
    log.logger.debug(f"Target density at z={sim.snap[idx].z:.3f}: {targetDen:.3e} Msun/kpc^3")
    
    # ... computation ...
    
    log.logger.info(f"Found rvir = {rvir:.3f} kpc enclosing {mass_enclosed:.3e} Msun")
    return rvir
```

## Performance Monitoring

### Using the `@log_performance` Decorator

Automatically time function execution:

```python
from cosmo_analysis.log import log_performance

@log_performance
def expensive_calculation(sim, idx):
    # Long-running computation
    return result
# Logs:
# INFO - Executing: module.expensive_calculation
# INFO - Completed: module.expensive_calculation in 12.45s
```

### Advanced Options

```python
import logging
from cosmo_analysis.log import log_performance

# Log at DEBUG level with arguments
@log_performance(level=logging.DEBUG, include_args=True)
def findCenter(sim, snapshotN, lim=20):
    # Function implementation
    pass
# Logs:
# DEBUG - Executing: module.findCenter with args: <sim>, 100, lim=20
# DEBUG - Completed: module.findCenter in 0.85s
```

### Exception Handling

The decorator automatically logs failures:

```python
@log_performance
def risky_operation():
    raise ValueError("Something went wrong")
# Logs:
# INFO - Executing: module.risky_operation
# ERROR - Failed: module.risky_operation after 0.01s - ValueError: Something went wrong
```

## Contextual Logging

### Using `log_section` for Major Workflow Sections

Create visual separation in logs:

```python
from cosmo_analysis.log import log_section

with log_section("Galaxy Formation Analysis"):
    # Multiple operations here
    compute_centers()
    calculate_masses()
    generate_plots()
# Logs:
# INFO - ============================================================
# INFO -   Galaxy Formation Analysis
# INFO - ============================================================
# ... operation logs ...
# INFO - ============================================================
```

## Data Summary Logging

### Debugging Data Structures

Use `log_data_summary` to inspect arrays and data:

```python
from cosmo_analysis.log import log_data_summary
import numpy as np

positions = np.random.random((1000, 3)) * 100
masses = np.random.lognormal(6, 1, 1000)

log_data_summary("Particle Positions", positions)
log_data_summary("Particle Masses", masses)
# Logs (at DEBUG level):
# DEBUG - Particle Positions: shape=(1000, 3), dtype=float64, min=0.123e+00, max=9.987e+01, mean=5.012e+01
# DEBUG - Particle Masses: shape=(1000,), dtype=float64, min=1.234e+05, max=9.876e+06, mean=4.567e+06
```

## Multi-Step Workflows

### Using `ProgressTracker` for Complex Pipelines

Track progress through multi-step workflows:

```python
from cosmo_analysis.log import ProgressTracker

tracker = ProgressTracker(total_steps=5, task="Full Analysis Pipeline")

tracker.step("Loading data")
data = load_data()

tracker.step("Computing centers")
centers = compute_centers(data)

tracker.step("Calculating virial radii")
rvir = calculate_rvir(data, centers)

tracker.step("Finding orientation")
axes = find_axes(data, centers)

tracker.step("Generating plots")
create_plots(data)

tracker.complete()
# Logs:
# INFO - Starting Full Analysis Pipeline (5 steps)
# INFO - [1/5] (20%) Full Analysis Pipeline: Loading data - elapsed: 0.5s
# INFO - [2/5] (40%) Full Analysis Pipeline: Computing centers - elapsed: 2.3s
# INFO - [3/5] (60%) Full Analysis Pipeline: Calculating virial radii - elapsed: 15.7s
# INFO - [4/5] (80%) Full Analysis Pipeline: Finding orientation - elapsed: 18.2s
# INFO - [5/5] (100%) Full Analysis Pipeline: Generating plots - elapsed: 25.8s
# INFO - Completed Full Analysis Pipeline (5 steps) in 25.80s
```

## Configuration

### Setting Log Level

Control verbosity of logging:

```python
import logging
from cosmo_analysis.log import set_log_level

# Show only important messages
set_log_level(logging.INFO)

# Show detailed debug information
set_log_level(logging.DEBUG)

# Show only warnings and errors
set_log_level(logging.WARNING)
```

### Adding File Handler

Log to both console and file:

```python
from cosmo_analysis.log import add_file_handler
import logging

# Log everything to file
add_file_handler("analysis.log", level=logging.DEBUG)

# Now all log messages go to both console and file
log.logger.info("This message appears in both places")
```

### Environment-Based Configuration

```python
import os
import logging
from cosmo_analysis.log import set_log_level

# Configure based on environment variable
log_level = os.environ.get("COSMO_LOG_LEVEL", "INFO")
set_log_level(getattr(logging, log_level))
```

## Migration from constants.py

The `constants.py` module is now deprecated. Here's how to migrate your code:

### Before (deprecated):
```python
from cosmo_analysis.core.constants import verboseLevel, figSize, gasPart

if verboseLevel > 10:
    print("Debug information")

plt.figure(figsize=(figSize, figSize))
```

### After (recommended):
```python
from cosmo_analysis.config import get_config
from cosmo_analysis import log

config = get_config()

# Use structured logging instead of verboseLevel
log.logger.debug("Debug information")

# Get values from config
fig_size = config.get('plotting_defaults.figsize', [8, 8])
gas_part = config.get('simulation_parameters.particle_types.gas', 'PartType0')

plt.figure(figsize=tuple(fig_size))
```

### Deprecation Warning

When you import from `constants.py`, you'll see:
```
DeprecationWarning: The constants module is deprecated and will be removed in a future version.
Please use the configuration system from cosmo_analysis.config instead.
```

## Best Practices

### 1. Use Appropriate Log Levels

- **DEBUG**: Detailed information for diagnosing problems
- **INFO**: Confirmation that things are working as expected
- **WARNING**: Indication that something unexpected happened
- **ERROR**: A serious problem occurred

```python
log.logger.debug(f"Processing particle {particle_id}")  # Detailed info
log.logger.info(f"Loaded {len(data)} particles")        # Progress update
log.logger.warning(f"No particles found in region")     # Unexpected but handled
log.logger.error(f"Failed to load file: {filename}")    # Serious problem
```

### 2. Use Context Managers for Scoped Operations

```python
# Good: Automatic timing and cleanup
with log_progress("Complex operation"):
    do_complex_work()

# Less good: Manual logging
log.logger.info("Starting complex operation")
do_complex_work()
log.logger.info("Finished complex operation")
```

### 3. Combine Multiple Features

```python
with log_section("Snapshot Analysis"):
    with log_progress("Loading snapshot"):
        @log_performance
        def load_and_process():
            data = load_data()
            log_data_summary("Loaded data", data)
            return process(data)
        
        result = load_and_process()
```

### 4. Use log_data_summary for Debugging

```python
# Instead of print statements
log_data_summary("Centers", centers)
log_data_summary("Masses", masses) 
log_data_summary("Velocities", velocities)

# Run with --log-level DEBUG to see summaries
```

## Examples

See `examples/logging_demo.py` for a complete demonstration of all features:

```bash
# Run with INFO level (default)
python examples/logging_demo.py

# Run with DEBUG level for more detail
python examples/logging_demo.py --log-level DEBUG

# Save logs to file
python examples/logging_demo.py --log-file analysis.log
```

## Testing

The logging system includes comprehensive tests in `tests/test_logging.py`:

```bash
# Run all logging tests
pytest tests/test_logging.py -v

# Run all tests including logging
pytest tests/ -v
```

## Implementation Details

### Function Decorators in sim_prop.py

All major functions in `sim_prop.py` now include performance monitoring:

- `findCenter`, `findCenter2`, ..., `findCenter8`: Centering algorithms
- `getRvir`: Virial radius calculation
- `getAxes`: Orientation calculation

Each function logs:
- Start of operation with parameters
- Progress updates during computation
- Results summary
- Execution time

### Backward Compatibility

The logging system is fully backward compatible:
- Existing code continues to work without changes
- The base `log.logger` object is still available
- All new features are opt-in

## Troubleshooting

### Logs Not Appearing

Check the log level:
```python
import logging
from cosmo_analysis import log

print(f"Current log level: {logging.getLevelName(log.logger.level)}")
set_log_level(logging.DEBUG)  # Show all messages
```

### Too Much Output

Reduce verbosity:
```python
set_log_level(logging.WARNING)  # Only warnings and errors
```

### Performance Overhead

Logging at DEBUG level with `include_args=True` can have performance impact for functions called many times. Use INFO level in production:

```python
# Development
set_log_level(logging.DEBUG)

# Production
set_log_level(logging.INFO)
```

## Contributing

When adding new functions to cosmo_analysis:

1. Use `@log_performance` for functions that take >0.1s
2. Use `log_progress` context for multi-step operations
3. Add `log.logger.info()` for significant progress milestones
4. Use `log.logger.debug()` for detailed diagnostic information
5. Include tests for logging behavior

Example:
```python
@log_performance
def new_analysis_function(sim, idx):
    """New analysis with proper logging."""
    log.logger.info(f"Starting new analysis for snapshot {idx}")
    
    with log_progress("Computing property"):
        result = compute_something(sim, idx)
        log_data_summary("Result", result)
    
    log.logger.info(f"Analysis complete, found {len(result)} items")
    return result
```
