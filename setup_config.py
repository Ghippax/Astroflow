#!/usr/bin/env python3
"""Helper script to set up configuration for cosmo_analysis.

This script helps users create their initial config.yaml file from templates.
"""

import os
import shutil
import sys
from pathlib import Path


def main():
    """Set up configuration interactively."""
    print("=" * 60)
    print("Cosmo Analysis Configuration Setup")
    print("=" * 60)
    print()
    
    # Get the root directory
    root_dir = Path(__file__).parent
    template_file = root_dir / "config_template.yaml"
    example_file = root_dir / "config_example.yaml"
    config_file = root_dir / "config.yaml"
    
    # Check if config already exists
    if config_file.exists():
        print(f"Configuration file already exists: {config_file}")
        response = input("Do you want to overwrite it? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Setup cancelled.")
            return 0
    
    # Ask which template to use
    print("\nChoose a starting template:")
    print("1. Example configuration (config_example.yaml) - Ready to use with defaults")
    print("2. Full template (config_template.yaml) - All options documented")
    print()
    
    choice = input("Enter choice (1 or 2, default=1): ").strip() or "1"
    
    if choice == "1":
        source = example_file
        print(f"\nUsing example configuration from {example_file}")
    elif choice == "2":
        source = template_file
        print(f"\nUsing full template from {template_file}")
    else:
        print("Invalid choice. Using example configuration.")
        source = example_file
    
    # Copy the file
    try:
        shutil.copy(source, config_file)
        print(f"✓ Created configuration file: {config_file}")
        print()
        print("Next steps:")
        print("1. Edit config.yaml to set your paths and preferences")
        print("2. Set COSMO_CONFIG environment variable (optional):")
        print(f"   export COSMO_CONFIG={config_file}")
        print()
        print("You're ready to use cosmo_analysis!")
        return 0
    except Exception as e:
        print(f"Error creating configuration file: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
