"""Tests for centering strategies."""

import pytest
from cosmo_analysis.core.centering import (
    CenteringRegistry,
    get_centering_registry,
    MaxDensityStrategy,
    CenterOfMassStrategy,
    AGORAIsolatedStrategy,
    OriginStrategy,
    get_strategy_name,
    LEGACY_CODE_MAPPING
)


class TestCenteringStrategies:
    """Test individual centering strategies."""
    
    def test_max_density_strategy_name(self):
        """Test MaxDensityStrategy name."""
        strategy = MaxDensityStrategy()
        assert strategy.name == "max_density"
        assert strategy.description is not None
    
    def test_center_of_mass_strategy_name(self):
        """Test CenterOfMassStrategy name."""
        strategy = CenterOfMassStrategy()
        assert strategy.name == "center_of_mass"
        assert strategy.description is not None
    
    def test_agora_isolated_strategy_name(self):
        """Test AGORAIsolatedStrategy name."""
        strategy = AGORAIsolatedStrategy()
        assert strategy.name == "agora_isolated"
        assert strategy.description is not None
    
    def test_origin_strategy_name(self):
        """Test OriginStrategy name."""
        strategy = OriginStrategy()
        assert strategy.name == "origin"
        assert strategy.description is not None


class TestCenteringRegistry:
    """Test CenteringRegistry functionality."""
    
    def test_registry_initialization(self):
        """Test that registry initializes with default strategies."""
        registry = CenteringRegistry()
        strategies = registry.list_strategies()
        assert len(strategies) > 0
    
    def test_registry_has_all_default_strategies(self):
        """Test that all default strategies are registered."""
        registry = CenteringRegistry()
        expected_strategies = [
            "max_density",
            "center_of_mass",
            "agora_isolated",
            "agora_cosmological",
            "agora_fixed",
            "origin",
            "agora_extended",
            "shrinking_sphere"
        ]
        
        for strategy_name in expected_strategies:
            strategy = registry.get(strategy_name)
            assert strategy is not None
            assert strategy.name == strategy_name
    
    def test_register_strategy(self):
        """Test registering a new strategy."""
        registry = CenteringRegistry()
        initial_count = len(registry.list_strategies())
        
        # Register origin strategy again (should overwrite)
        strategy = OriginStrategy()
        registry.register(strategy)
        
        # Count should not change (overwrite existing)
        assert len(registry.list_strategies()) == initial_count
    
    def test_get_strategy(self):
        """Test getting a strategy by name."""
        registry = CenteringRegistry()
        strategy = registry.get("origin")
        assert strategy is not None
        assert isinstance(strategy, OriginStrategy)
    
    def test_get_strategy_not_found(self):
        """Test getting a non-existent strategy."""
        registry = CenteringRegistry()
        
        with pytest.raises(KeyError, match="Centering strategy 'nonexistent' not found"):
            registry.get("nonexistent")
    
    def test_list_strategies(self):
        """Test listing all strategies."""
        registry = CenteringRegistry()
        strategies = registry.list_strategies()
        
        assert isinstance(strategies, list)
        assert len(strategies) > 0
        
        # Check structure of returned list
        for name, description in strategies:
            assert isinstance(name, str)
            assert isinstance(description, str)
            assert len(name) > 0
            assert len(description) > 0


class TestGlobalRegistry:
    """Test global registry functionality."""
    
    def test_get_centering_registry(self):
        """Test getting global registry instance."""
        registry1 = get_centering_registry()
        registry2 = get_centering_registry()
        
        # Should return same instance (singleton)
        assert registry1 is registry2
    
    def test_global_registry_has_strategies(self):
        """Test that global registry has strategies."""
        registry = get_centering_registry()
        strategies = registry.list_strategies()
        assert len(strategies) > 0


class TestLegacyCodeMapping:
    """Test legacy numeric code mapping."""
    
    def test_legacy_code_mapping_exists(self):
        """Test that legacy code mapping exists."""
        assert LEGACY_CODE_MAPPING is not None
        assert isinstance(LEGACY_CODE_MAPPING, dict)
    
    def test_legacy_code_mapping_complete(self):
        """Test that all codes 1-8 are mapped."""
        for i in range(1, 9):
            code = str(i)
            assert code in LEGACY_CODE_MAPPING
            assert LEGACY_CODE_MAPPING[code] is not None
    
    def test_get_strategy_name_legacy_code(self):
        """Test converting legacy code to strategy name."""
        assert get_strategy_name("1") == "max_density"
        assert get_strategy_name("2") == "center_of_mass"
        assert get_strategy_name("3") == "agora_isolated"
        assert get_strategy_name("6") == "origin"
    
    def test_get_strategy_name_already_strategy(self):
        """Test that strategy names pass through unchanged."""
        assert get_strategy_name("max_density") == "max_density"
        assert get_strategy_name("origin") == "origin"


class TestCenteringIntegration:
    """Integration tests for centering system."""
    
    def test_registry_calculate_center_origin(self):
        """Test calculating center using origin strategy through registry."""
        registry = get_centering_registry()
        
        # Create a mock sim object with minimal structure
        class MockSnap:
            pass
        
        class MockSim:
            def __init__(self):
                self.ytFull = [MockSnap()]
        
        sim = MockSim()
        
        # Origin strategy doesn't need actual simulation data
        cen, cen_unyt = registry.calculate_center("origin", sim, 0)
        
        assert cen is not None
        assert len(cen) == 3
        assert all(c == 0 for c in cen)
