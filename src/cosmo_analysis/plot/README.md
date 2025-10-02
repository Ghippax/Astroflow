# Plot Module Documentation

This directory contains the refactored plotting functions for cosmo_analysis, organized into modular components.

## Module Structure

The plotting functionality is organized into specialized modules:

### `base.py` - Core Plotting Utilities

Core functions used across all plotting operations:

- **`handleFig()`** - Main figure handler for display, saving, or animation
- **`saveFrame()`** - Capture figure as numpy array for animations
- **`setLegend()`** - Create legends for simulation comparisons

**Example:**
```python
from cosmo_analysis.plot.base import handleFig
from cosmo_analysis.config import get_config

fig, ax = plt.subplots()
ax.plot(x, y)

config = get_config()
handleFig(fig, [False, False, True], "my_plot", config=config)
```

### `projection.py` - Projection Plots

Functions for creating 2D projection plots:

- **`ytMultiPanel()`** - Multi-panel projection plots for comparing simulations/fields
- **`ytProjPanel()`** - Standard projection panels with optional dual-axis views

**Example:**
```python
from cosmo_analysis.plot.projection import ytMultiPanel

# Create multi-panel projection comparing different fields
ytMultiPanel(
    sims=[sim1, sim2], 
    idx=[0, 0],
    zField=["Density", "Temperature"],
    message="multi_field_comparison"
)
```

### `utils.py` - Helper Functions

Utility functions for data processing and animations:

- **`makeMovie()`** - Create animated GIFs from frame arrays
- **`aFromT()`** - Calculate cosmological scale factor from time
- Binning functions: `binFunctionCilBins()`, `binFunctionSphBins()`, etc.

**Example:**
```python
from cosmo_analysis.plot.utils import makeMovie, aFromT

# Create animation from frames
frames = [...]  # List of numpy arrays
makeMovie(frames, interval=50, message="simulation_evolution")

# Calculate scale factor
a = aFromT(1000.0)  # Time in Myr
```

### `plots.py` - Legacy Interface

Maintains backward compatibility by re-exporting all functions. Existing code using `plots.py` continues to work without changes.

**Example:**
```python
# Old style - still works
from cosmo_analysis.plot.plots import ytMultiPanel, handleFig

# Functions work exactly as before
```

## Configuration System

All refactored functions now use the configuration system instead of hardcoded globals:

```python
from cosmo_analysis.config import Config

# Load custom config
config = Config.load_config('my_config.yaml')
Config.set_instance(config)

# Functions automatically use config settings
handleFig(fig, [False, False, True], "plot")  # Uses config for paths, DPI, etc.
```

### Key Config Parameters

Functions use these config parameters:

- `paths.output_directory` - Where to save figures
- `plotting_defaults.dpi` - Resolution for saved figures
- `plotting_defaults.save_plots` - Default save behavior
- `plotting_defaults.show_plots` - Default display behavior
- `plotting_defaults.figsize` - Default figure size
- `plotting_defaults.fontsize` - Default font size
- `simulation_parameters.particle_types` - Particle type mappings
- `analysis_options.fig_width` - Default projection width
- `analysis_options.buffer_size` - Default buffer size
- `cosmology.*` - Cosmology parameters

## Migration Guide

### For New Code

Use the specialized modules directly:

```python
from cosmo_analysis.plot.base import handleFig
from cosmo_analysis.plot.projection import ytMultiPanel
from cosmo_analysis.plot.utils import makeMovie
```

### For Existing Code

No changes needed! Continue using `plots.py`:

```python
from cosmo_analysis.plot.plots import ytMultiPanel, handleFig
```

### Removing Hardcoded Globals

**Before:**
```python
from cosmo_analysis.plot.plots import ytMultiPanel, saveAll, showAll

ytMultiPanel(sims, idx, saveFig=saveAll, showFig=showAll)
```

**After:**
```python
from cosmo_analysis.plot.projection import ytMultiPanel
from cosmo_analysis.config import get_config

config = get_config()
ytMultiPanel(sims, idx)  # Uses config automatically
```

Or explicitly:
```python
ytMultiPanel(sims, idx, saveFig=True, showFig=False)
```

## Testing

The plot module includes comprehensive tests:

```bash
# Run all plot tests
pytest tests/test_plot_*.py

# Run integration tests only
pytest tests/test_plot_integration.py -m integration

# Run visual tests
pytest tests/test_plot_integration.py -m visual

# Run visual regression tests with yt data
pytest tests/test_plot_visual_regression.py -m visual

# Skip slow tests (e.g., those downloading data)
pytest tests/test_plot_*.py -m "not slow"
```

### Visual Regression Tests

The `test_plot_visual_regression.py` module includes tests that:

- Use yt sample data (`IsolatedGalaxy`) when available
- Fall back to synthetic datasets when network is unavailable
- Establish baselines for visual output
- Compare new outputs against baselines
- Test consistency across different projections
- Verify config parameters affect output correctly

These tests ensure that changes to the code don't inadvertently break visual output.

## Development

### Adding New Plotting Functions

1. Choose the appropriate module based on functionality
2. Implement function with config support
3. Add comprehensive tests
4. Update `plots.py` to import and re-export for backward compatibility
5. Document in this README

### Config Support Pattern

```python
def my_plot_function(data, param=None, config=None):
    """My plotting function.
    
    Args:
        data: Input data
        param: Some parameter (uses config if None)
        config: Config object (uses global if None)
    """
    if config is None:
        config = get_config()
    
    if param is None:
        param = config.get('some.config.key', default_value)
    
    # Rest of function...
```

## Architecture Benefits

1. **Modularity**: Functions organized by purpose, easier to find and maintain
2. **Configurability**: No hardcoded values, everything controlled via config
3. **Testability**: Small focused modules are easier to test
4. **Backward Compatibility**: Existing code continues to work
5. **Extensibility**: Easy to add new modules or functions
6. **Documentation**: Clear module boundaries and responsibilities

## Future Enhancements

Planned additions:
- `phase.py` - Phase space plotting functions
- `profiles.py` - Profile and binned plotting functions  
- `halo.py` - Halo analysis and plotting
- `star_formation.py` - Star formation plotting
- `galaxy.py` - Galaxy property plotting

These will be extracted from the remaining functions in `plots.py` following the same pattern.

### `profiles.py` - Profile and Rotation Curve Plotting

Functions for radial profiles and velocity dispersion analysis:

- **`plotBinned()`** - General binned field profiles with error analysis
- **`plotRotDisp()`** - Velocity dispersion and rotation curves

**Example:**
```python
from cosmo_analysis.plot.profiles import plotBinned

# Plot density profile with error bars
plotBinned(
    sims=[sim1, sim2],
    idx=[0, 0],
    binFields=["particle_position_spherical_radius", "Density"],
    nBins=50,
    rLim=(0.1, 20),
    showError=1,
    message="density_profile"
)
```

### `halo.py` - Halo Finding and Analysis

Functions for halo identification and mass function analysis:

- **`findHalos()`** - Find halos using FOF or HOP methods
- **`plotClumpMassF()`** - Plot cumulative halo mass function

**Example:**
```python
from cosmo_analysis.plot.halo import findHalos, plotClumpMassF

# Find halos
halo_data = findHalos(
    simArr=[sim1, sim2],
    idxArr=[0, 0],
    partT='PartType1',
    mainPath='/path/to/output',
    haloMethod='fof'
)

# Plot mass function
plotClumpMassF(sims, idx, halo_data, message="halo_mass_function")
```

### `star_formation.py` - Star Formation Analysis

Functions for star formation rate and Kennicutt-Schmidt relations:

- **`plotSFR()`** - Star formation rate over time
- **`plotKScil()`** - Kennicutt-Schmidt relation (cylindrical binning)
- **`plotKSmock()`** - Kennicutt-Schmidt with mock observations
- **`plotSFmass()`** - Stellar mass buildup over time

**Example:**
```python
from cosmo_analysis.plot.star_formation import plotSFR, plotKScil

# Plot SFR history
plotSFR(sims, idx, tLim=(0, 1000), message="sfr_history")

# Plot Kennicutt-Schmidt relation
plotKScil(sims, idx, rLim=15, nBins=20, message="ks_relation")
```

### `galaxy.py` - Galaxy Property Correlations

Functions for galaxy-scale property relationships:

- **`plotMsMh()`** - Stellar mass vs halo mass relation

**Example:**
```python
from cosmo_analysis.plot.galaxy import plotMsMh
from cosmo_analysis.plot.halo import findHalos

# First find halos
halo_data = findHalos(simArr, idxArr, 'PartType1', mainPath)

# Plot M*-Mh relation
plotMsMh(sims, idx, halo_data, rLim=15, message="stellar_halo_mass")
```

## Refactoring Status

**✅ All modules refactored with config support (100% complete)**

All 8 modules have been successfully refactored:
1. ✅ base.py - Core utilities
2. ✅ projection.py - Projection plotting
3. ✅ phase.py - Phase space plotting
4. ✅ profiles.py - Profile and rotation curves
5. ✅ halo.py - Halo finding and analysis
6. ✅ star_formation.py - Star formation plotting
7. ✅ galaxy.py - Galaxy properties
8. ✅ utils.py - Helper functions

All functions use the configuration system for:
- Output paths and file handling
- Plotting defaults (DPI, figure sizes, save/show behavior)
- Particle types and field names
- Analysis parameters

