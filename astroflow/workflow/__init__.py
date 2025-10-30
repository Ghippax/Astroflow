"""
Workflow module for defining and executing analysis pipelines.

Provides:
- Workflow/Task specifications (serializable to/from YAML)
- Workflow registry (register, load, export workflows)
- Token-based parameter expansion ($params.field, $sim_name, etc.)
- Execution strategies (global, per_simulation, per_snapshot)
- Nested workflow support
"""

from .workflow import Workflow, Task, WorkflowContext, run
from .settings import WorkflowRuntimeConfig
from .registry import (
    export,
    get,
    list_all,
    load,
    register,
    unregister,
    validate,
    workflow_fn,
    workflow_registry,
)

__all__ = [
    "Workflow",
    "Task",
    "WorkflowContext",
    "WorkflowRuntimeConfig",
    "run",
    "get",
    "register",
    "unregister",
    "list_all",
    "load",
    "export",
    "validate",
    "workflow_registry",
    "workflow_fn",
]