from __future__ import annotations

from typing import Optional, Any
from functools import wraps

from ..log import get_logger

from ..core.registry import FunctionRegistry

afLogger = get_logger()

class DerivedPropRegistry(FunctionRegistry):
    def __init__(self):
        super().__init__(name="Derived prop fn")
        self._aliases: dict[str, tuple[str, dict]] = {}

    def register_alias(self, alias: str, base_name: str, preset_params: dict):
        """Register friendly name mapping to base function + preset params."""
        self._aliases[alias] = (base_name, preset_params)

    def compute(self, name: str, sim_obj, snap_id: int, **kwargs) -> Any:
        """
        Compute derived property for sim_obj at snapshot snap_id.
        
        Parameters
        ----------
        name : str
            Name of the derived property
        sim_obj : Simulation
            Simulation object
        snap_id : int
            Snapshot index
        params : dict, optional
            Parameters to pass to the computation function
            
        Returns
        -------
        Any
            Computed property value
        """
        # Resolve alias
        if name in self._aliases:
            base_name, preset = self._aliases[name]
            kwargs = {**preset, **kwargs}  # user overrides preset
            name = base_name

        fn = self.get(name)
        return fn(sim_obj, snap_id, **kwargs)

derived_fn = DerivedPropRegistry()

postpro_fn = FunctionRegistry()

def register_derived(name: str, registry: DerivedPropRegistry = derived_fn, set_config: Optional[dict] = None, config_file: Optional[str] = None, expose = True, **metadata):
    """
    Decorator to register a derived property computation function.

    Examples
    --------
    >>> @register_derived("virial_radius")
    ... def compute_rvir(sim_obj, snap_id, params):
    ...     # computation
    ...     return radius

    If expose=True, creates module-level cached wrapper:
    af.analysis.{name}(sim, snap, ..., cache=True)
    """
    def decorator(fn):
        # Standard registration
        registry.register(name, set_config=set_config, config_file=config_file, **metadata)(fn)
        
        if expose:
            @wraps(fn)
            def cached_wrapper(sim, snap_idx, cache=True, force_recompute=False, **kw):
                if cache:
                    return sim.get_derived(name, snap_idx, force_recompute=force_recompute, **kw)
                return fn(sim, snap_idx, **kw)
            
            # Inject into module namespace
            import sys
            module = sys.modules[fn.__module__]
            setattr(module, name, cached_wrapper)
        
        return fn
    return decorator

def register_alias(alias_name: str, base_name: str, registry: DerivedPropRegistry = derived_fn, **preset_params):
    """Convenience to register an alias."""
    registry.register_alias(alias_name, base_name, preset_params)

def register_postprocessing(name: str, registry: FunctionRegistry = postpro_fn, set_config: Optional[dict] = None, config_file: Optional[str] = None, **metadata):
    """
    Decorator to register a postprocessing function.

    Examples
    --------
    >>> @register_postprocessing("circ_vel")
    ... def circ_vel(mass, params):
    ...     # computation
    ...     return vel
    """
    return registry.register(name, set_config=set_config, config_file=config_file, **metadata)