from __future__ import annotations

from typing import Optional

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
        Enable parallel execution (future: yt.piter support)
    max_workers : int, optional
        Maximum parallel workers
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
    """
    parallel: bool = Field(default_factory=lambda: config.get("workflow/parallel"))
    max_workers: Optional[int] = Field(default_factory=lambda: config.get("workflow/max_workers"))
    cache_results: bool = Field(default_factory=lambda: config.get("workflow/cache_results"))
    dry_run: bool = Field(default_factory=lambda: config.get("workflow/dry_run"))
    continue_on_error: bool = Field(default_factory=lambda: config.get("workflow/continue_on_error"))
    output_root: Optional[str] = Field(default_factory=lambda: config.get("workflow/output_root"))
    metadata_root: Optional[str] = Field(default_factory=lambda: config.get("workflow/metadata_root"))
    log_level: Optional[str] = Field(default_factory=lambda: config.get("workflow/log_level"))
    plot_registry: Optional[FunctionRegistry] = plot_fn
    data_registry: Optional[FunctionRegistry] = data_fn
    render_registry: Optional[FunctionRegistry] = render_fn
    derived_registry: Optional[FunctionRegistry] = derived_fn
    fn_registry: Optional[FunctionRegistry] = workflow_fn

    model_config = ConfigDict(arbitrary_types_allowed=True)