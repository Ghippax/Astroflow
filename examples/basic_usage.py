#!/usr/bin/env python3
"""Basic usage example for cosmo_analysis with new configuration system.

This script demonstrates how to:
1. Load configuration
2. Access configuration values
3. Use configuration in analysis
"""

import sys
from pathlib import Path

# Add src to path if running from examples directory
src_path = Path(__file__).parent.parent / 'src'
if src_path.exists():
    sys.path.insert(0, str(src_path))

from cosmo_analysis.config import load_config, get_config
from cosmo_analysis.config_utils import (
    get_particle_type,
    get_output_directory,
    get_plotting_default,
    get_analysis_option
)


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def main():
    """Demonstrate basic configuration usage."""
    
    print_section("Loading Configuration")
    
    # Load configuration (will use config.yaml if it exists, or defaults)
    config = load_config()
    print("✓ Configuration loaded successfully")
    
    # Show which config was loaded
    print(f"Config data sections: {list(config.data.keys())}")
    
    print_section("Accessing Configuration Values")
    
    # Method 1: Using utility functions (recommended)
    print("Using utility functions:")
    print(f"  Gas particle type: {get_particle_type('gas')}")
    print(f"  DM particle type: {get_particle_type('dm')}")
    print(f"  Star particle type: {get_particle_type('star')}")
    print(f"  Output directory: {get_output_directory()}")
    print(f"  Figure DPI: {get_plotting_default('dpi', 300)}")
    print(f"  Analysis width: {get_analysis_option('fig_width', 30)} kpc")
    
    # Method 2: Using dot notation
    print("\nUsing dot notation:")
    print(f"  Save plots: {config.get('plotting_defaults.save_plots')}")
    print(f"  Show plots: {config.get('plotting_defaults.show_plots')}")
    print(f"  Font size: {config.get('plotting_defaults.fontsize')}")
    
    # Method 3: Using dictionary access
    print("\nUsing dictionary access:")
    sim_params = config['simulation_parameters']
    print(f"  Particle types: {sim_params['particle_types']}")
    
    print_section("Configuration Examples")
    
    # Example 1: Get cosmology parameters
    print("Cosmology parameters:")
    hubble = config.get('cosmology.hubble_constant', 0.7)
    omega_m = config.get('cosmology.omega_matter', 0.3)
    omega_l = config.get('cosmology.omega_lambda', 0.7)
    print(f"  H0 = {hubble * 100} km/s/Mpc")
    print(f"  Ω_m = {omega_m}")
    print(f"  Ω_Λ = {omega_l}")
    
    # Example 2: Get analysis options
    print("\nAnalysis options:")
    print(f"  Young star age threshold: {get_analysis_option('young_star_age')} Myr")
    print(f"  Mock observation resolution: {get_analysis_option('low_res_mock')} pc")
    print(f"  Solar metallicity: {get_analysis_option('z_solar')}")
    
    # Example 3: Get plotting defaults
    print("\nPlotting defaults:")
    figsize = get_plotting_default('figsize', [8, 8])
    print(f"  Figure size: {figsize[0]}x{figsize[1]} inches")
    print(f"  YT figure size: {get_plotting_default('yt_figsize', 12)}")
    print(f"  Default colormap: {get_plotting_default('default_colormap', 'viridis')}")
    
    print_section("Using Configuration in Code")
    
    print("Example code snippet:")
    print("""
    from cosmo_analysis.config_utils import get_particle_type, get_output_directory
    from cosmo_analysis.io.load import load
    
    # Get configuration values
    gas_type = get_particle_type('gas')
    output_dir = get_output_directory()
    
    # Load simulation
    sim = load(name="my_sim", path="/path/to/sim", centerDefs=["3", "7"])
    
    # Use in analysis
    # ... your analysis code here ...
    """)
    
    print_section("Configuration Tips")
    
    print("1. Copy config_example.yaml to config.yaml")
    print("2. Edit config.yaml with your paths and preferences")
    print("3. Use environment variable: export COSMO_CONFIG=/path/to/config.yaml")
    print("4. Use utility functions from config_utils for easier access")
    print("5. Check config_template.yaml for all available options")
    
    print("\n✓ Configuration demo complete!\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
