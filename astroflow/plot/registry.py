# astroflow/plotting/low_level.py
from __future__ import annotations

from typing import Optional

from ..log import get_logger
from ..core.registry import FunctionRegistry

afLogger = get_logger()

# Default registries
data_fn = FunctionRegistry(name="Data fn")
render_fn = FunctionRegistry(name="Rendering fn")
plot_fn = FunctionRegistry(name="Plotting fn")

# Convenience functions for registering plotting functions
def register_data_fn(name: str, registry: FunctionRegistry = data_fn, set_config: Optional[dict] = None, config_file: Optional[str] = None, **metadata):
    """
    Decorator to register a data generation function.

    Examples
    --------
    >>> @register_data_fn("slice_frb")
    ... def slice_plot_frb(sim_obj, snap_id, params):
    ...     # computation
    ...     return frb
    """
    return registry.register(name, set_config=set_config, config_file=config_file, **metadata)

def register_render_fn(name: str, registry: FunctionRegistry = render_fn, set_config: Optional[dict] = None, config_file: Optional[str] = None, **metadata):
    """
    Decorator to register a rendering function.

    Examples
    --------
    >>> @register_render_fn("slice_frb")
    ... def image_from_frb(frb, params):
    ...     # computation
    ...     return fig, ax
    """
    return registry.register(name, set_config=set_config, config_file=config_file, **metadata)

def register_plot_fn(name: str, registry: FunctionRegistry = plot_fn, set_config: Optional[dict] = None, config_file: Optional[str] = None, **metadata):
    """
    Decorator to register a plotting function.

    Examples
    --------
    >>> @register_plot_fn("slice_frb")
    ... def plot_frb(sim_obj, snap_id, params):
    ...     # computation
    ...     plt.show()
    """
    return registry.register(name, set_config=set_config, config_file=config_file, **metadata)
