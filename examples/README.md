# Cosmo Analysis Examples

This directory contains example scripts and notebooks demonstrating how to use cosmo_analysis.

## Available Examples

### Python Scripts

#### basic_usage.py

Demonstrates the new YAML configuration system:
- Loading configuration
- Accessing configuration values
- Using configuration utilities
- Best practices for configuration

Run it:
```bash
python examples/basic_usage.py
```

### Jupyter Notebooks

#### 01_workflow_basics.ipynb

Introduction to the workflow system:
- Exploring standard workflows
- Creating custom workflows
- Validating and running workflows
- YAML workflow definitions

#### 02_centering_strategies.ipynb

Guide to centering strategies:
- Available centering methods
- Choosing the right strategy
- Legacy code compatibility
- Configuration options

## Running Examples

### From Repository Root

```bash
python examples/basic_usage.py
```

### With Custom Configuration

```bash
export COSMO_CONFIG=/path/to/your/config.yaml
python examples/basic_usage.py
```

## Prerequisites

Ensure you have installed the package:
```bash
pip install -e .
```

## Creating Your Own Scripts

Use these examples as templates for your analysis scripts:

1. Copy an example script
2. Modify for your specific needs
3. Use the configuration system for paths and parameters
4. Keep your config.yaml outside version control

## Configuration

Before running examples, you may want to create your own configuration:

```bash
# Option 1: Use the helper script
python setup_config.py

# Option 2: Copy manually
cp config_example.yaml config.yaml
# Edit config.yaml as needed
```

## Additional Resources

- `../README.md` - Installation and setup
- `../MIGRATION.md` - Migrating from old constants
- `../config_template.yaml` - Full configuration reference
- `../CONTRIBUTING.md` - Development guidelines
