from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from .. import config
from ..analysis.registry import derived_fn
from .registry import workflow_fn
from ..plot.registry import data_fn, plot_fn, render_fn

from ..core.registry import FunctionRegistry


# Per run config (setup when you run a workflow), different workflow runs can have the same runtime config
class WorkflowRuntimeConfig(BaseModel):
    """
    Runtime configuration for workflow execution.

    Attributes
    ----------
    parallel : bool
        Enable parallel execution via Dask scheduler adapter
    max_workers : int, optional
        Maximum parallel workers
    scheduler_address : str, optional
        Dask scheduler address. If None, creates a LocalCluster.
    threads_per_worker : int
        Threads per worker for LocalCluster (default: 1)
    yt_mpi_procs : int
        Default number of MPI processes for yt_mpi execution policy
    eager_small : bool
        If True, execute small tasks immediately without scheduling overhead
    cache_results : bool
        Cache intermediate results
    dry_run : bool
        If True, log planned execution without running tasks
    continue_on_error : bool
        Continue workflow even if a task fails
    output_root : str, optional
        Root directory for outputs (accessible via $runtime.output_root)
    metadata_root : str, optional
        Root directory for metadata files
    log_level : str, optional
        Override log level for this run
    plot_registry : FunctionRegistry
        Registry for plot functions
    data_registry : FunctionRegistry
        Registry for data functions
    render_registry : FunctionRegistry
        Registry for render functions
    derived_registry : FunctionRegistry
        Registry for derived properties
    fn_registry : FunctionRegistry
        Registry for arbitrary Python functions
    scheduler_adapter : Any
        Dask scheduler adapter instance (created automatically when parallel=True)
    """

    parallel: bool = Field(
        default_factory=lambda: config.get("workflow/parallel", False)
    )
    max_workers: Optional[int] = Field(
        default_factory=lambda: config.get("workflow/max_workers")
    )
    scheduler_address: Optional[str] = Field(
        default_factory=lambda: config.get("workflow/scheduler_address")
    )
    threads_per_worker: int = Field(
        default_factory=lambda: config.get("workflow/threads_per_worker", 1)
    )
    yt_mpi_procs: int = Field(
        default_factory=lambda: config.get("workflow/yt_mpi_procs", 4)
    )
    eager_small: bool = Field(
        default_factory=lambda: config.get("workflow/eager_small", False)
    )
    cache_results: bool = Field(
        default_factory=lambda: config.get("workflow/cache_results", True)
    )
    dry_run: bool = Field(default_factory=lambda: config.get("workflow/dry_run", False))
    continue_on_error: bool = Field(
        default_factory=lambda: config.get("workflow/continue_on_error", True)
    )
    output_root: Optional[str] = Field(
        default_factory=lambda: config.get("workflow/output_root")
    )
    metadata_root: Optional[str] = Field(
        default_factory=lambda: config.get("workflow/metadata_root")
    )
    log_level: Optional[str] = Field(
        default_factory=lambda: config.get("workflow/log_level")
    )
    plot_registry: Optional[FunctionRegistry] = plot_fn
    data_registry: Optional[FunctionRegistry] = data_fn
    render_registry: Optional[FunctionRegistry] = render_fn
    derived_registry: Optional[FunctionRegistry] = derived_fn
    fn_registry: Optional[FunctionRegistry] = workflow_fn
    # Internal: scheduler adapter instance (not serialized)
    scheduler_adapter: Optional[Any] = Field(default=None, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_scheduler_adapter(self) -> Any:
        """
        Get or create the Dask scheduler adapter.

        Returns
        -------
        DaskSchedulerAdapter
            The scheduler adapter for parallel task submission.
        """
        if not self.parallel:
            return None

        if self.scheduler_adapter is None:
            from ..parallel import DaskSchedulerAdapter, auto

            # Initialize Dask client
            auto(
                scheduler_address=self.scheduler_address,
                n_workers=self.max_workers,
                threads_per_worker=self.threads_per_worker,
            )
            self.scheduler_adapter = DaskSchedulerAdapter()
        return self.scheduler_adapter
