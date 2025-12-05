from __future__ import annotations

from typing import Any, Dict, Literal, Optional

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

    def get_execution_policy(self, name: str) -> Dict[str, Any]:
        """
        Get execution policy hints for a derived property.

        Parameters
        ----------
        name : str
            Name of the derived property

        Returns
        -------
        dict
            Execution policy hints (execution_policy, n_procs, resources, pure)
        """
        metadata = self.get_metadata(name)
        return {
            "execution_policy": metadata.get("execution_policy", "cpu"),
            "n_procs": metadata.get("n_procs"),
            "resources": metadata.get("resources", {}),
            "pure": metadata.get("pure", False),
        }


derived_fn = DerivedPropRegistry()


def register_derived(
    name: str,
    registry: DerivedPropRegistry = derived_fn,
    set_config: Optional[dict] = None,
    config_file: Optional[str] = None,
    execution_policy: Literal["cpu", "process", "yt_mpi"] = "cpu",
    n_procs: Optional[int] = None,
    resources: Optional[Dict[str, Any]] = None,
    pure: bool = False,
    **metadata,
):
    """
    Decorator to register a derived property computation function.

    Parameters
    ----------
    name : str
        Unique name for the derived property
    registry : DerivedPropRegistry
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
    >>> @register_derived("virial_radius", execution_policy="yt_mpi", n_procs=4)
    ... def compute_rvir(sim_obj, snap_id, params):
    ...     # computation using yt
    ...     return radius

    >>> @register_derived("center_of_mass", pure=True)
    ... def compute_com(sim_obj, snap_id, params):
    ...     # pure computation
    ...     return center
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
