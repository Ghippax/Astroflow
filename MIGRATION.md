# Migration Guide

This guide helps you migrate from the old hardcoded constants to the new YAML configuration system.

## Overview

The codebase now uses a flexible YAML-based configuration system instead of hardcoded constants. This provides:
- Easy customization without modifying code
- Multiple configurations for different projects
- Better separation of configuration from code
- Type-safe configuration access

## Quick Migration

### 1. Create Configuration File

Copy the example configuration:
```bash
cp config_example.yaml config.yaml
```

Or use the setup helper:
```bash
python setup_config.py
```

### 2. Update Your Paths

Edit `config.yaml` and set your paths:
```yaml
paths:
  output_directory: "/your/output/path"
  data_files:
    bilcontour: "/your/path/bilcontour.txt"
    projection_list: "/your/path/outputlist_projection.txt"
```

### 3. Set Environment Variable (Optional)

```bash
export COSMO_CONFIG=/path/to/your/config.yaml
```

## Mapping Old Constants to New Config

### Old `constants.py` → New Config

| Old Constant | New Config Path | Notes |
|--------------|-----------------|-------|
| `savePath` | `paths.output_directory` | Now per-project configurable |
| `gasPart` | `simulation_parameters.particle_types.gas` | Still "PartType0" by default |
| `starPart` | `simulation_parameters.particle_types.star` | Still "PartType4" by default |
| `dmPart` | `simulation_parameters.particle_types.dm` | Still "PartType1" by default |
| `figSize` | `plotting_defaults.figsize[0]` | Now [width, height] array |
| `ytFigSize` | `plotting_defaults.yt_figsize` | Unchanged default |
| `fontSize` | `plotting_defaults.fontsize` | Unchanged default |
| `figWidth` | `analysis_options.fig_width` | In kpc, unchanged |
| `buffSize` | `analysis_options.buffer_size` | Unchanged default |
| `youngStarAge` | `analysis_options.young_star_age` | In Myr, unchanged |
| `lowResMock` | `analysis_options.low_res_mock` | In pc, unchanged |
| `zSolar` | `analysis_options.z_solar` | Unchanged default |

### Code Changes

#### Using Configuration in Your Scripts

**Old way:**
```python
from cosmo_analysis.core.constants import gasPart, savePath

# Had to edit constants.py to change paths
```

**New way:**
```python
from cosmo_analysis.config import load_config
from cosmo_analysis.config_utils import get_particle_type, get_output_directory

config = load_config('config.yaml')
gas_part = get_particle_type('gas')
output_dir = get_output_directory()

# Or access directly:
gas_part = config.get('simulation_parameters.particle_types.gas')
```

#### Plotting Functions

**Old way:**
```python
from cosmo_analysis.plot.plots import ytProjPanel

# Used global constants implicitly
ytProjPanel(simArr=[sim], idxArr=[0], part="PartType0")
```

**New way:**
```python
from cosmo_analysis.plot.plots import ytProjPanel
from cosmo_analysis.config_utils import get_particle_type

# Explicitly pass configuration or use helpers
part_type = get_particle_type('gas')
ytProjPanel(simArr=[sim], idxArr=[0], part=part_type)

# Or provide output path directly
ytProjPanel(simArr=[sim], idxArr=[0], 
            saveFigPath="/custom/output/path")
```

## Backward Compatibility

The old `constants.py` module still exists for backward compatibility, but:
- Colormaps are now lazily initialized to avoid import errors
- Hardcoded paths should be replaced with config values
- Direct use of constants is deprecated

## Best Practices

### For End Users

1. **Never commit `config.yaml`** - It's in `.gitignore` for a reason
2. **Use `config_template.yaml`** as reference for all options
3. **Set `COSMO_CONFIG`** environment variable for project-specific configs
4. **Keep one config per project** in the project directory

### For Developers

1. **Use `config_utils` helpers** instead of accessing config directly
2. **Provide sensible defaults** in functions using `get_plotting_default('key', default)`
3. **Document config requirements** in function docstrings
4. **Test with custom configs** using pytest fixtures

## Examples

### Example 1: Custom Analysis Script

```python
from cosmo_analysis.config import load_config
from cosmo_analysis.config_utils import get_output_directory, get_particle_type
from cosmo_analysis.io.load import load
from cosmo_analysis.plot.plots import ytProjPanel

# Load configuration
config = load_config('my_project_config.yaml')

# Get configuration values
output_dir = get_output_directory()
gas_type = get_particle_type('gas')

# Load simulation
sim = load(name="my_sim", path="/path/to/sim", centerDefs=["3", "7"])

# Create plots with configured output directory
ytProjPanel(simArr=[sim], idxArr=[0], part=gas_type, 
            saveFigPath=output_dir, message="Gas_Density")
```

### Example 2: Multiple Configurations

```python
# Analysis for Project A
import os
os.environ['COSMO_CONFIG'] = '/projects/projectA/config.yaml'
from cosmo_analysis.config import Config
Config._instance = None  # Reset singleton
# Now use config...

# Analysis for Project B
os.environ['COSMO_CONFIG'] = '/projects/projectB/config.yaml'
Config._instance = None  # Reset singleton
# Now use different config...
```

## Troubleshooting

### Config Not Found
```
FileNotFoundError: config.yaml not found
```
**Solution:** Create config.yaml from template or set `COSMO_CONFIG`

### Import Errors with Old Code
```
ValueError: 'algae' is not a valid colormap
```
**Solution:** Update to latest version where colormaps are lazily initialized

### Output Path Not Working
```
Figures saving to wrong location
```
**Solution:** Check `paths.output_directory` in config.yaml and ensure `saveFigPath` is set correctly

## Getting Help

- Check `config_template.yaml` for all available options
- See `README.md` for setup instructions  
- Review `CONTRIBUTING.md` for development guidelines
- Open an issue on GitHub for questions
