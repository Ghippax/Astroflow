"""Tests for fields module.

Note: Some tests are minimal due to dependencies on yt and matplotlib colormaps.
Full integration tests would require mock simulation data.
"""

import pytest
import numpy as np


class TestTemperatureConversion:
    """Test cases for temperature conversion functions."""
    
    def test_calc_mu_table_local(self):
        """Test mean molecular weight calculation."""
        # Import locally to avoid issues with constants
        from cosmo_analysis.core.fields import calc_mu_table_local
        
        # Test at known temperature points
        mu_low = calc_mu_table_local(1e1)
        assert 1.0 < mu_low < 1.3  # Expected range for neutral gas
        
        mu_high = calc_mu_table_local(1e9)
        assert 0.5 < mu_high < 0.7  # Expected range for ionized gas
        
        # Test that mu decreases with temperature (ionization)
        mu_mid = calc_mu_table_local(1e5)
        assert mu_low > mu_mid > mu_high


class TestModuleStructure:
    """Test basic module structure without importing constants."""
    
    def test_fields_module_exists(self):
        """Test that fields module can be imported."""
        import cosmo_analysis.core.fields as fields_module
        assert fields_module is not None
    
    def test_calc_mu_table_local_exists(self):
        """Test that calc_mu_table_local function exists."""
        from cosmo_analysis.core.fields import calc_mu_table_local
        assert callable(calc_mu_table_local)
    
    def test_temperature_conversion_exists(self):
        """Test that temperature conversion functions exist."""
        # These are the main functions we want to ensure exist
        from cosmo_analysis.core import fields
        assert hasattr(fields, 'calc_mu_table_local')
        assert hasattr(fields, 'convert_T_over_mu_to_T')
