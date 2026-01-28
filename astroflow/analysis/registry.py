from __future__ import annotations

from typing import Optional, Any

from ..log import get_logger

from ..core.registry import FunctionRegistry

afLogger = get_logger()

class DerivedPropRegistry(FunctionRegistry):
    def __init__(self):
        super().__init__(name="Derived prop fn")

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

        fn = self.get(name)
        return fn(sim_obj, snap_id, **kwargs)

derived_fn = DerivedPropRegistry()

postpro_fn = FunctionRegistry()

def register_derived(name: str, registry: DerivedPropRegistry = derived_fn, set_config: Optional[dict] = None, config_file: Optional[str] = None, **metadata):
    """
    Decorator to register a derived property computation function.

    Examples
    --------
    >>> @register_derived("virial_radius")
    ... def compute_rvir(sim_obj, snap_id, params):
    ...     # computation
    ...     return radius
    """
    return registry.register(name, set_config=set_config, config_file=config_file, **metadata)


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