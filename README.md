# Cosmo-Analysis

Analysis routines for galaxy simulations using yt.

## Features

- Load and analyze Gadget/AREPO simulation snapshots
- Create projection and phase space plots
- Compute physical properties of galaxies
- Compare multiple simulations
- Automated analysis workflows
- Flexible YAML-based configuration system

## Installation

### Requirements

- Python 3.8+
- See `requirements.txt` for dependencies

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Ghippax/Cosmo-Analysis.git
cd Cosmo-Analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

4. Copy the configuration template and customize it:
```bash
cp config_template.yaml config.yaml
# Edit config.yaml with your paths and preferences
```

## Quick Start

```python
from cosmo_analysis.io.load import load
from cosmo_analysis.plot.plots import ytProjPanel
from cosmo_analysis.config import load_config

# Load configuration
config = load_config('config.yaml')

# Load a simulation
sim = load(name="my_sim", path="/path/to/simulation", 
           centerDefs=["3", "7"])

# Create projection plots
ytProjPanel(simArr=[sim], idxArr=[0], 
            part="PartType0", message="Gas Density")
```

## Configuration

The package uses YAML configuration files to manage paths, plotting defaults, and analysis parameters. See `config_template.yaml` for all available options.

Key configuration sections:
- `paths`: Output directories and data file locations
- `simulation_parameters`: Particle types and unit conversions
- `plotting_defaults`: Default plot settings
- `analysis_options`: Analysis parameters (box sizes, resolutions, etc.)

You can set the `COSMO_CONFIG` environment variable to point to your config file:
```bash
export COSMO_CONFIG=/path/to/your/config.yaml
```

## Testing

Run tests with pytest:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest tests/ --cov=cosmo_analysis --cov-report=html
```

## Documentation

Build the documentation:
```bash
cd docs
make html
```

View the documentation by opening `docs/_build/html/index.html` in your browser.

## Contributing

Contributions are welcome! Please see `CONTRIBUTING.md` for guidelines.

## License

MIT
