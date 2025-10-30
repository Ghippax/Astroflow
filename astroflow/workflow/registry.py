from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Union, TYPE_CHECKING

import yaml

from ..log import get_logger
from ..io_utils import atomic_save_yaml
from ..core.registry import FunctionRegistry

if TYPE_CHECKING:
    from .workflow import Workflow

afLogger = get_logger()

workflow_registry: Dict[str, Workflow] = {}
workflow_fn = FunctionRegistry(name="Arbitrary fn")

def validate(workflow: Workflow) -> None:
    """
    Validate workflow structure.
    
    Checks:
    - No duplicate task IDs
    - All task dependencies exist
    
    Raises
    ------
    ValueError
        If validation fails
    """
    ids = [task.id for task in workflow.tasks]
    duplicates = {tid for tid in ids if ids.count(tid) > 1}
    if duplicates:
        raise ValueError(f"Workflow '{workflow.name}' has duplicated task ids: {sorted(duplicates)}")
    
    known = set(ids)
    missing: Dict[str, set[str]] = {}
    for task in workflow.tasks:
        missing_deps = set(task.requires) - known
        if missing_deps:
            missing[task.id] = missing_deps
    if missing:
        details = ", ".join(f"{tid}: {sorted(deps)}" for tid, deps in missing.items())
        raise ValueError(f"Workflow '{workflow.name}' references unknown dependencies -> {details}")

def register(workflow: Workflow, override: bool = False, registry: Dict[str, Workflow] = workflow_registry) -> Workflow:
    """
    Register a workflow in the global registry.
    
    Parameters
    ----------
    workflow : Workflow
        Workflow to register
    override : bool, default=False
        If True, replace existing workflow with same name
    registry : dict, optional
        Target registry (defaults to workflow_registry)
    
    Returns
    -------
    Workflow
        The registered workflow
    """
    validate(workflow)
    if workflow.name in registry and not override:
        raise ValueError(f"Workflow '{workflow.name}' already registered. Use override=True to replace.")
    registry[workflow.name] = workflow
    afLogger.debug(f"Registered workflow '{workflow.name}' with {len(workflow.tasks)} tasks.")
    return workflow

def unregister(name: str, registry: Dict[str, Workflow] = workflow_registry) -> None:
    """Unregister a workflow from the global registry."""
    registry.pop(name, None)
    afLogger.debug(f"Unregistered workflow '{name}'.")

def list_all(registry: Dict[str, Workflow] = workflow_registry) -> Iterable[str]:
    """List all registered workflow names."""
    return tuple(registry.keys())

def get(name: str, default: Optional[Workflow] = None, registry: Dict[str, Workflow] = workflow_registry) -> Workflow:
    """
    Get a registered workflow by name.
    
    Parameters
    ----------
    name : str
        Workflow name
    default : Workflow, optional
        Fallback if not found
    registry : dict, optional
        Target registry
    
    Returns
    -------
    Workflow
        The requested workflow
    
    Raises
    ------
    KeyError
        If workflow not found and no default provided
    """
    if name not in registry:
        if default is not None:
            afLogger.debug(f"Workflow '{name}' not found, returning default.")
            return default
        raise KeyError(f"Workflow '{name}' not found. Registered: {list_all()}")
    return registry[name]

def load(path: str, auto_register: bool = True, override: bool = False) -> Workflow:
    """
    Load a workflow from a YAML file.
    
    Parameters
    ----------
    path : str
        Path to YAML file
    auto_register : bool, default=True
        If True, register workflow after loading
    override : bool, default=False
        If True, replace existing workflow with same name
    
    Returns
    -------
    Workflow
        The loaded workflow
    """
    file_path = Path(path)
    with file_path.open("r") as file:
        data = yaml.safe_load(file) or {}
    workflow = Workflow(**data)
    if auto_register:
        register(workflow, override=override)
    return workflow

def export(workflow: Union[str, Workflow], path: str, include_none: bool = False) -> Path:
    """
    Export a workflow to a YAML file.
    
    Parameters
    ----------
    workflow : str or Workflow
        Workflow name or instance
    path : str
        Target file path
    include_none : bool, default=False
        If True, include fields with None values
    
    Returns
    -------
    Path
        Path to exported file
    """
    spec = get(workflow) if isinstance(workflow, str) else workflow
    payload = spec.model_dump(exclude_none=(not include_none))
    target = Path(path)
    atomic_save_yaml(payload, target)
    afLogger.debug(f"Exported workflow '{spec.name}' to {target}")
    return target