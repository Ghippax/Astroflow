from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Union, List, Optional, Sequence, Literal, Iterable

from pydantic import BaseModel, ConfigDict, Field

from .settings import WorkflowRuntimeConfig
from .registry import validate, workflow_registry
from ..log import get_logger
from ..io_utils import deep_merge
from ..core.simulation import Simulation

afLogger = get_logger()


# Per task instance
class Task(BaseModel):
    """
    Individual task specification within a workflow.

    Attributes
    ----------
    id : str
        Unique identifier for this task
    target : str
        Name of the function/workflow to execute
    kind : str
        Task type: "plot", "derived", "data", "python", "workflow"
    params : dict
        Parameters passed to the target function (supports $token expansion)
    requires : list of str
        Task IDs that must complete before this task
    persist : bool
        If True, store result in context.results
    execution : str
        Execution scope: "global", "per_simulation", "per_snapshot"
    snapshots : list of int, optional
        Specific snapshot indices (for per_snapshot execution)
    outputs : dict
        Output specifications (e.g., file paths)
    metadata : dict
        Additional metadata for this task
    timeout : float, optional
        Timeout in seconds (future use)
    execution_policy : str
        Parallel execution policy: "cpu" (Dask threads), "process" (Dask processes),
        or "yt_mpi" (MPI-based yt execution). Default is "cpu".
    n_procs : int, optional
        Number of MPI processes for yt_mpi policy. Default is 4.
    resources : dict
        Resource hints for the scheduler (e.g., {"memory": "2GB"})
    pure : bool
        Whether the function is pure (deterministic). Set to False for yt tasks.
    """

    id: str
    target: str
    kind: Literal["plot", "derived", "data", "python", "workflow"]
    params: Dict[str, Any] = Field(default_factory=dict)
    requires: List[str] = Field(default_factory=list)
    persist: bool = True
    execution: Literal["global", "per_simulation", "per_snapshot"] = "global"
    snapshots: Optional[List[int]] = None
    outputs: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timeout: Optional[float] = None
    # Parallel execution policy hints
    execution_policy: Literal["cpu", "process", "yt_mpi"] = "cpu"
    n_procs: Optional[int] = None
    resources: Dict[str, Any] = Field(default_factory=dict)
    pure: bool = True

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


@dataclass
class WorkflowContext:
    """
    Context for a workflow run, holding state and results.

    Attributes
    ----------
    workflow : Workflow
        The workflow being executed
    simulations : sequence of Simulation
        Simulations to process
    params : dict
        Merged workflow parameters. Library config not included here.
    runtime : WorkflowRuntimeConfig
        Runtime configuration
    results : dict
        Task results keyed by task.id
    artifacts : dict
        Mutable store for intermediate artifacts
    metadata : dict
        Mutable store for run metadata (e.g., errors)
    """

    workflow: Workflow
    simulations: Sequence[Simulation]
    params: Dict[str, Any]
    runtime: WorkflowRuntimeConfig
    results: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def store_result(self, task: Task, value: Any) -> Any:
        """Store task result in context.results."""
        self.results[task.id] = value
        return value

    def lookup(self, token: str) -> Any:
        """
        Resolve a dot-separated token path.

        Supports:
        - params.* → workflow parameters
        - runtime.* → runtime config
        - workflow.* → workflow spec
        - simulations.* → simulation dict
        - artifacts.* → artifact store
        - metadata.* → metadata store
        - <task_id> → previous task result

        Returns None if path is invalid.
        """
        if not token:
            return None
        parts = token.split(".")
        root = parts[0]
        if root == "params":
            value = self.params
        elif root == "runtime":
            value = self.runtime.model_dump()
        elif root == "workflow":
            value = self.workflow.model_dump()
        elif root == "artifacts":
            value = self.artifacts
        elif root == "metadata":
            value = self.metadata
        elif root == "simulations":
            value = {sim.name: sim for sim in self.simulations}
        else:
            value = self.results.get(root)
        for part in parts[1:]:
            if value is None:
                return None
            if isinstance(value, dict):
                value = value.get(part)
            else:
                value = getattr(value, part, None)
        return value


# TODO: Read Pydantic docs to ensure below, keep in mind .registry.py load and export should work with this
# Per workflow instance, this should be serializable to/from YAML and runnable
class Workflow(BaseModel):
    """
    Workflow specification, serializable to/from YAML.

    Attributes
    ----------
    name : str
        Unique workflow identifier
    tasks : list of Task
        Tasks to execute
    default_params : dict
        Default parameters
    runtime : WorkflowRuntimeConfig
        Default runtime configuration
    metadata : dict
        Workflow metadata (version, authors, etc.)
    """

    name: str
    tasks: List[Task]
    default_params: Dict[str, Any] = Field(default_factory=dict)
    runtime: WorkflowRuntimeConfig = Field(default_factory=WorkflowRuntimeConfig)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    def task_dict(self) -> Dict[str, Task]:
        """Return dict mapping task.id → Task."""
        return {task.id: task for task in self.tasks}

    def _resolve_task_order(self) -> List[str]:
        """
        Resolve task execution order using topological sort.

        Raises
        ------
        ValueError
            If workflow contains cyclic dependencies
        """
        graph = {}
        indegree = {}
        for task in self.tasks:
            deps = set(task.requires)
            graph[task.id] = deps
            indegree[task.id] = len(deps)

        queue = [tid for tid, deg in indegree.items() if deg == 0]
        order = []

        while queue:
            current = queue.pop(0)
            order.append(current)
            for task in self.tasks:
                if current in graph.get(task.id, set()):
                    graph[task.id].remove(current)
                    indegree[task.id] -= 1
                    if indegree[task.id] == 0:
                        queue.append(task.id)

        if len(order) != len(self.tasks):
            raise ValueError(f"Workflow '{self.name}' contains cyclic dependencies.")
        return order

    def run(
        self,
        simulations: Sequence["Simulation"],
        params: Optional[Dict[str, Any]] = None,
        runtime_config: Union[WorkflowRuntimeConfig, Dict[str, Any], None] = None,
    ) -> WorkflowContext:
        """
        Execute the workflow on given simulations.

        Parameters
        ----------
        simulations : sequence of Simulation
            Simulations to process
        params : dict, optional
            Parameters to merge with default_params
        runtime_config : WorkflowRuntimeConfig or dict, optional
            Runtime configuration overrides

        Returns
        -------
        WorkflowContext
            Final context with results, artifacts, metadata
        """
        validate(self)
        # Merge default params with provided params
        merged_params = deep_merge(self.default_params, params or {})
        if runtime_config is None:
            runtime_cfg = self.runtime.model_copy(deep=True)
        elif isinstance(runtime_config, WorkflowRuntimeConfig):
            runtime_cfg = runtime_config.model_copy(deep=True)
        else:
            runtime_cfg = self.runtime.model_copy(deep=True, update=runtime_config)
        ctx = WorkflowContext(self, tuple(simulations), merged_params, runtime_cfg)

        task_order = self._resolve_task_order()
        afLogger.info(f"Executing workflow '{self.name}' with {len(task_order)} tasks.")
        task_map = self.task_dict()

        # TODO: Should be parallel! But need to handle dependencies properly
        for task_id in task_order:
            task = task_map[task_id]
            resolved_params = self._expand_placeholders(task.params, ctx)

            if ctx.runtime.dry_run:
                afLogger.info(
                    f"[DRY-RUN] {task.id}: {task.kind}:{task.target} {resolved_params}"
                )
                if task.persist:
                    ctx.store_result(
                        task, {"status": "dry-run", "params": resolved_params}
                    )
                continue

            try:
                result = self._dispatch_task(task, ctx, resolved_params)
            except Exception as e:
                afLogger.error(f"Task '{task.id}' failed: {e}")
                if not ctx.runtime.continue_on_error:
                    raise
                ctx.metadata.setdefault("errors", {})[task.id] = repr(e)
                result = {"error": repr(e)}

            if task.persist:
                ctx.store_result(task, result)

        return ctx

    def _dispatch_task(
        self,
        task: Task,
        ctx: WorkflowContext,
        params: Dict[str, Any],
    ) -> Any:
        """
        Dispatch task based on execution strategy.

        - global: run once
        - per_simulation: run once per sim, inject sim/sim_name
        - per_snapshot: run once per (sim, snapshot), inject sim/snapshot/idx

        Returns
        -------
        Any
            Raw result (global) or dict keyed by sim_name (per_sim/per_snap)
        """
        if task.execution == "global":
            return self._invoke(task, ctx, params)

        if task.execution == "per_simulation":
            results = {}
            # TODO: Should be parallel! easily done with yt .piter()
            for sim in ctx.simulations:
                overrides = {
                    "sim": sim,
                    "simulation": sim,
                    "sim_name": getattr(sim, "name", None),
                }
                call_params = self._expand_placeholders(
                    copy.deepcopy(params), ctx, overrides
                )
                result = self._invoke(task, ctx, call_params)
                results[getattr(sim, "name", str(sim))] = result
            return results

        if task.execution == "per_snapshot":
            results = {}
            base_params = copy.deepcopy(
                params
            )  # Avoid mutation (_expand_placeholders mutates params)
            snaps = base_params.pop(
                "snapshots", None
            )  # Always remove snapshots param if present
            snapshots = task.snapshots or snaps
            # TODO: Should be parallel, as in prev if
            for sim in ctx.simulations:
                sim_results = {}
                iterable = self._resolve_snapshots(sim, snapshots)
                # TODO: Can also be parallel with .piter()
                for snap in iterable:
                    overrides = {
                        "sim": sim,
                        "simulation": sim,
                        "sim_name": getattr(sim, "name", None),
                        "snapshot": snap,
                        "idx": snap,
                    }
                    call_params = self._expand_placeholders(base_params, ctx, overrides)
                    call_params.setdefault("sim", sim)
                    call_params.setdefault("idx", snap)
                    call_params.setdefault("snapshot", snap)
                    sim_results[snap] = self._invoke(task, ctx, call_params)
                results[getattr(sim, "name", str(sim))] = sim_results
            return results
        raise ValueError(
            f"Unknown execution strategy '{task.execution}' for task '{task.id}'"
        )

    def _invoke(self, task: Task, ctx: WorkflowContext, params: Dict[str, Any]) -> Any:
        """
        Invoke the target function based on task.kind.

        - plot/data/render: lookup in registry, call with **params
        - derived: extract sim/snapshot, call derived_registry.compute
        - python: call fn_registry, pass context as first arg
        - workflow: recursively run nested workflow

        When parallel execution is enabled (ctx.runtime.parallel=True), tasks are
        submitted through the DaskSchedulerAdapter using the task's execution_policy.
        Registry-defined execution hints are used as defaults when task doesn't specify them.

        Returns
        -------
        Any
            Result of the invoked function
        """
        # Get the target function and registry hints based on task kind
        fn = None
        fn_params = params
        registry_hints = {}

        if task.kind == "plot":
            fn = ctx.runtime.plot_registry.get(task.target)
            registry_hints = ctx.runtime.plot_registry.get_metadata(task.target)
        elif task.kind == "data":
            fn = ctx.runtime.data_registry.get(task.target)
            registry_hints = ctx.runtime.data_registry.get_metadata(task.target)
        elif task.kind == "render":
            fn = ctx.runtime.render_registry.get(task.target)
            registry_hints = ctx.runtime.render_registry.get_metadata(task.target)
        elif task.kind == "derived":
            if "sim" not in params or (
                "snapshot" not in params and "idx" not in params
            ):
                raise ValueError(
                    f"Derived task '{task.id}' requires 'sim' and 'snapshot/idx' parameters."
                )
            # Get registry hints for derived
            registry_hints = ctx.runtime.derived_registry.get_metadata(task.target)
            # For derived, we wrap the call since it needs special handling
            sim = params.pop("sim")
            snapshot = params.pop("snapshot", params.pop("idx", None))
            return self._execute_with_policy(
                task,
                ctx,
                lambda: ctx.runtime.derived_registry.compute(
                    task.target, sim, snapshot, **params
                ),
                registry_hints=registry_hints,
            )
        elif task.kind == "python":
            fn = ctx.runtime.fn_registry.get(task.target)
            fn_params = {"context": ctx, **params}
            registry_hints = ctx.runtime.fn_registry.get_metadata(task.target)
        elif task.kind == "workflow":
            nested = params.get("workflow") or task.target
            nested_params = params.get("params")
            nested_runtime = params.get("runtime")

            nested_workflow = workflow_registry.get(nested)
            if nested_workflow is self:
                raise ValueError(
                    f"Workflow '{self.name}' cannot invoke itself recursively."
                )

            return nested_workflow.run(
                ctx.simulations, params=nested_params, runtime_config=nested_runtime
            ).results
        else:
            raise ValueError(f"Unsupported task kind '{task.kind}'")

        if fn is None:
            raise ValueError(f"Could not resolve function for task '{task.id}'")

        # Execute the function with parallel policy if enabled
        return self._execute_with_policy(
            task, ctx, lambda: fn(**fn_params), registry_hints=registry_hints
        )

    def _execute_with_policy(
        self,
        task: Task,
        ctx: WorkflowContext,
        fn_call: Any,
        registry_hints: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Execute a function call respecting the task's execution policy.

        When parallel execution is enabled, submits through DaskSchedulerAdapter.
        Otherwise, executes directly.

        Execution policy is determined by:
        1. Task-level attributes (highest priority)
        2. Registry-defined hints from function registration
        3. Runtime defaults (lowest priority)

        Parameters
        ----------
        task : Task
            The task being executed
        ctx : WorkflowContext
            Workflow context
        fn_call : callable
            Zero-argument callable that performs the actual work
        registry_hints : dict, optional
            Execution policy hints from the function registry

        Returns
        -------
        Any
            Result of the function call
        """
        # If parallel execution is not enabled, execute directly
        if not ctx.runtime.parallel:
            return fn_call()

        # Get or create the scheduler adapter
        adapter = ctx.runtime.get_scheduler_adapter()
        if adapter is None:
            return fn_call()

        # Build task policy from task attributes, falling back to registry hints
        from ..parallel import TaskPolicy

        hints = registry_hints or {}

        # Task attributes override registry hints which override defaults
        # For execution_policy: task > registry > default("cpu")
        execution_policy = task.execution_policy
        if execution_policy == "cpu" and hints.get("execution_policy"):
            execution_policy = hints.get("execution_policy", "cpu")

        # For n_procs: task > registry > runtime config
        n_procs = task.n_procs
        if n_procs is None:
            n_procs = hints.get("n_procs") or ctx.runtime.yt_mpi_procs

        # For resources: merge registry hints with task resources (task wins)
        resources = {**hints.get("resources", {}), **task.resources}

        # For pure: task > registry > default(True)
        pure = task.pure
        if pure is True and hints.get("pure") is not None:
            pure = hints.get("pure", True)

        policy = TaskPolicy(
            policy=execution_policy,
            resources=resources,
            pure=pure,
            n_procs=n_procs,
            eager_small=ctx.runtime.eager_small,
        )

        afLogger.debug(
            f"Submitting task '{task.id}' with policy: "
            f"{policy.policy}, pure={policy.pure}, n_procs={policy.n_procs}"
        )

        # Submit through the adapter
        # Note: For yt_mpi and eager_small, submit returns result directly
        # For cpu/process, it returns a Future
        result = adapter.submit(fn_call, policy=policy)

        # If it's a Future (from Dask), wait for the result
        if hasattr(result, "result") and callable(result.result):
            return result.result()
        return result

    def _expand_placeholders(
        self,
        obj: Any,
        ctx: WorkflowContext,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Recursively replace $tokens with live values.

        Supports:
        - Strings: "$params.field" → ctx.params["field"]
        - Dicts: {k: "$v"} → {k: resolved_v}
        - Lists: ["$v"] → [resolved_v]
        - Tuples: ("$v",) → (resolved_v,)

        Parameters
        ----------
        obj : Any
            Object to expand (string, dict, list, tuple, or passthrough)
        ctx : WorkflowContext
            Context for token resolution
        extra : dict, optional
            Override dict for per-sim/per-snap injected vars

        Returns
        -------
        Any
            Expanded object with $tokens replaced
        """
        if isinstance(obj, str) and obj.startswith("$"):
            token = obj[1:]
            return self._resolve_token(token, ctx, extra)
        if isinstance(obj, dict):
            return {k: self._expand_placeholders(v, ctx, extra) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._expand_placeholders(v, ctx, extra) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._expand_placeholders(v, ctx, extra) for v in obj)
        return obj

    @staticmethod
    def _resolve_snapshots(
        sim: Simulation, snapshots: Optional[Iterable[int]]
    ) -> Iterable[int]:
        """
        Resolve snapshot indices for a simulation.

        If snapshots is None, returns range(len(sim)).
        Otherwise returns snapshots as-is.
        """
        if snapshots is None:
            return range(len(sim))
        return snapshots

    def _resolve_token(
        self,
        token: str,
        ctx: WorkflowContext,
        extra: Optional[Dict[str, Any]],
    ) -> Any:
        """
        Resolve a single token path.

        Resolution order:
        1. Check `extra` dict (per-sim/per-snap overrides)
        2. Check `extra` nested paths (e.g., sim.name)
        3. Delegate to ctx.lookup for workflow-level paths

        Parameters
        ----------
        token : str
            Dot-separated path (e.g., "params.field", "sim_name")
        ctx : WorkflowContext
            Context for lookup
        extra : dict, optional
            Override dict

        Returns
        -------
        Any
            Resolved value or None if not found
        """
        if extra and token in extra:
            return extra[token]
        if extra:
            root, *rest = token.split(".")
            if root in extra:
                value = extra[root]
                for part in rest:
                    if isinstance(value, dict):
                        value = value.get(part)
                    else:
                        value = getattr(value, part, None)
                return value
        return ctx.lookup(token)


# Rebuild models to resolve forward references
Task.model_rebuild()
Workflow.model_rebuild()


# For registry workflows
def run(
    workflow: Union[str, Workflow],
    simulations: Sequence[Simulation],
    params: Optional[Dict[str, Any]] = None,
    runtime_config: Optional[Dict[str, Any]] = None,
    registry: Dict = workflow_registry,
) -> WorkflowContext:
    """
    Run a workflow by name or instance.

    Parameters
    ----------
    workflow : str or Workflow
        Workflow name (looked up in registry) or Workflow instance
    simulations : sequence of Simulation
        Simulations to process
    params : dict, optional
        Parameters to merge with workflow defaults
    runtime_config : WorkflowRuntimeConfig or dict, optional
        Runtime configuration overrides
    registry : dict, optional
        Workflow registry (defaults to workflow_registry)

    Returns
    -------
    WorkflowContext
        Final context object with results
    """
    flow = registry.get(workflow) if isinstance(workflow, str) else workflow
    return flow.run(simulations, params=params, runtime_config=runtime_config)
