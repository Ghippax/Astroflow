# astroflow/plotting/low_level.py
from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from ..log import get_logger
from ..core.registry import FunctionRegistry

afLogger = get_logger()

# Default registries
data_fn = FunctionRegistry(name="Data fn")
render_fn = FunctionRegistry(name="Rendering fn")
plot_fn = FunctionRegistry(name="Plotting fn")


# Convenience functions for registering plotting functions
def register_data(
    name: str,
    registry: FunctionRegistry = data_fn,
    set_config: Optional[dict] = None,
    config_file: Optional[str] = None,
    execution_policy: Literal["cpu", "process", "yt_mpi"] = "cpu",
    n_procs: Optional[int] = None,
    resources: Optional[Dict[str, Any]] = None,
    pure: bool = False,
    **metadata,
):
    """
    Decorator to register a data generation function.

    Parameters
    ----------
    name : str
        Unique name for the data function
    registry : FunctionRegistry
        Registry to add the function to
    set_config : dict, optional
        Default configuration for this function
    config_file : str, optional
        Path to config file
    execution_policy : str, default="cpu"
        Parallel execution policy: "cpu", "process", or "yt_mpi"
    n_procs : int, optional
        Number of MPI processes for yt_mpi policy
    resources : dict, optional
        Resource hints for the scheduler (e.g., {"memory": "2GB"})
    pure : bool, default=False
        Whether the function is pure (deterministic). Usually False for yt operations.
    **metadata
        Additional metadata

    Examples
    --------
    >>> @register_data("slice_frb", execution_policy="yt_mpi", n_procs=4)
    ... def slice_plot_frb(sim_obj, snap_id, params):
    ...     # computation using yt
    ...     return frb
    """
    # Merge execution policy hints into metadata
    policy_hints = {
        "execution_policy": execution_policy,
        "n_procs": n_procs,
        "resources": resources or {},
        "pure": pure,
    }
    return registry.register(
        name,
        set_config=set_config,
        config_file=config_file,
        **policy_hints,
        **metadata,
    )


def register_render(
    name: str,
    registry: FunctionRegistry = render_fn,
    set_config: Optional[dict] = None,
    config_file: Optional[str] = None,
    execution_policy: Literal["cpu", "process", "yt_mpi"] = "cpu",
    n_procs: Optional[int] = None,
    resources: Optional[Dict[str, Any]] = None,
    pure: bool = True,
    **metadata,
):
    """
    Decorator to register a rendering function.

    Parameters
    ----------
    name : str
        Unique name for the render function
    registry : FunctionRegistry
        Registry to add the function to
    set_config : dict, optional
        Default configuration for this function
    config_file : str, optional
        Path to config file
    execution_policy : str, default="cpu"
        Parallel execution policy: "cpu", "process", or "yt_mpi"
    n_procs : int, optional
        Number of MPI processes for yt_mpi policy
    resources : dict, optional
        Resource hints for the scheduler (e.g., {"memory": "2GB"})
    pure : bool, default=True
        Whether the function is pure (deterministic). Usually True for render.
    **metadata
        Additional metadata

    Examples
    --------
    >>> @register_render("image")
    ... def image_from_frb(frb, params):
    ...     # computation
    ...     return fig, ax
    """
    policy_hints = {
        "execution_policy": execution_policy,
        "n_procs": n_procs,
        "resources": resources or {},
        "pure": pure,
    }
    return registry.register(
        name,
        set_config=set_config,
        config_file=config_file,
        **policy_hints,
        **metadata,
    )


def register_plot(
    name: str,
    registry: FunctionRegistry = plot_fn,
    set_config: Optional[dict] = None,
    config_file: Optional[str] = None,
    execution_policy: Literal["cpu", "process", "yt_mpi"] = "cpu",
    n_procs: Optional[int] = None,
    resources: Optional[Dict[str, Any]] = None,
    pure: bool = False,
    **metadata,
):
    """
    Decorator to register a plotting function.

    Parameters
    ----------
    name : str
        Unique name for the plot function
    registry : FunctionRegistry
        Registry to add the function to
    set_config : dict, optional
        Default configuration for this function
    config_file : str, optional
        Path to config file
    execution_policy : str, default="cpu"
        Parallel execution policy: "cpu", "process", or "yt_mpi"
    n_procs : int, optional
        Number of MPI processes for yt_mpi policy
    resources : dict, optional
        Resource hints for the scheduler (e.g., {"memory": "2GB"})
    pure : bool, default=False
        Whether the function is pure (deterministic). Usually False for plots.
    **metadata
        Additional metadata

    Examples
    --------
    >>> @register_plot("projection", execution_policy="yt_mpi")
    ... def plot_frb(sim_obj, snap_id, params):
    ...     # computation
    ...     plt.show()
    """
    policy_hints = {
        "execution_policy": execution_policy,
        "n_procs": n_procs,
        "resources": resources or {},
        "pure": pure,
    }
    return registry.register(
        name,
        set_config=set_config,
        config_file=config_file,
        **policy_hints,
        **metadata,
    )
