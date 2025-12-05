"""
Parallel execution utilities for Astroflow.

Provides Dask-based scheduling with optional MPI support for yt operations.
This module implements:
- enable_parallelism() / disable_parallelism() for global parallel mode
- Client management (auto, connect, set_client, get_client, dashboard_link)
- DaskSchedulerAdapter for task submission with execution policies
- run_yt_parallel for MPI-based parallel yt execution
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Literal, Optional, TypeVar, Union

from .log import get_logger

afLogger = get_logger()

# Type variable for generic result types
T = TypeVar("T")

# Execution policy types
ExecutionPolicy = Literal["cpu", "process", "yt_mpi"]

# Module-level state
_active_client: Optional[Any] = None
_parallel_enabled: bool = False
_parallel_config: Dict[str, Any] = {
    "suppress_io": True,
    "n_procs": 4,
}


def enable_parallelism(
    suppress_io: bool = True,
    scheduler_address: Optional[str] = None,
    n_workers: Optional[int] = None,
    threads_per_worker: int = 1,
    n_procs: int = 4,
    **kwargs: Any,
) -> Any:
    """
    Enable parallel execution for Astroflow operations.

    This function initializes the parallel execution environment, similar to
    yt.enable_parallelism(). Once enabled, yt operations like projections and
    slices will automatically use parallel execution.

    Parameters
    ----------
    suppress_io : bool, default=True
        If True, suppress output from worker processes (only rank 0 outputs).
    scheduler_address : str, optional
        Address of an existing Dask scheduler. If None, creates a LocalCluster.
    n_workers : int, optional
        Number of workers for LocalCluster. Defaults to system CPU count.
    threads_per_worker : int, default=1
        Threads per worker for LocalCluster.
    n_procs : int, default=4
        Number of MPI processes for yt_mpi operations.
    **kwargs
        Additional arguments passed to the Dask client/cluster.

    Returns
    -------
    distributed.Client
        The active Dask client.

    Examples
    --------
    >>> import astroflow as af
    >>> af.enable_parallelism()  # Enable with defaults
    >>> # Now all yt operations run in parallel
    >>> af.plot.proj(sim, 0, ("gas", "density"))

    >>> # Or connect to existing cluster
    >>> af.enable_parallelism(scheduler_address="tcp://scheduler:8786")

    >>> # Enable yt's native parallelism as well
    >>> import yt
    >>> yt.enable_parallelism()
    >>> af.enable_parallelism()
    """
    global _parallel_enabled, _parallel_config

    # Store configuration
    _parallel_config["suppress_io"] = suppress_io
    _parallel_config["n_procs"] = n_procs

    # Initialize Dask client
    client = auto(
        scheduler_address=scheduler_address,
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        **kwargs,
    )

    _parallel_enabled = True

    afLogger.info(
        f"Parallelism enabled. Dashboard: {dashboard_link()}. "
        f"Use disable_parallelism() to turn off."
    )

    return client


def disable_parallelism() -> None:
    """
    Disable parallel execution and close the Dask client.

    Examples
    --------
    >>> import astroflow as af
    >>> af.enable_parallelism()
    >>> # ... do parallel work ...
    >>> af.disable_parallelism()
    """
    global _parallel_enabled

    close()
    _parallel_enabled = False
    afLogger.info("Parallelism disabled.")


def is_parallel_enabled() -> bool:
    """
    Check if parallel execution is enabled.

    Returns
    -------
    bool
        True if parallelism is enabled, False otherwise.

    Examples
    --------
    >>> import astroflow as af
    >>> af.is_parallel_enabled()
    False
    >>> af.enable_parallelism()
    >>> af.is_parallel_enabled()
    True
    """
    return _parallel_enabled


def get_parallel_config() -> Dict[str, Any]:
    """
    Get the current parallel configuration.

    Returns
    -------
    dict
        Current parallel configuration including n_procs, suppress_io, etc.
    """
    return dict(_parallel_config)


def run_parallel(
    func: Callable[..., T],
    *args: Any,
    policy: Optional[ExecutionPolicy] = None,
    pure: bool = False,
    **kwargs: Any,
) -> T:
    """
    Execute a function using the parallel backend if parallelism is enabled.

    This is a convenience function that automatically uses the active Dask
    client when parallelism is enabled. If parallelism is disabled, the
    function is executed directly.

    Parameters
    ----------
    func : Callable
        The function to execute.
    *args
        Positional arguments for the function.
    policy : str, optional
        Execution policy: "cpu", "process", or "yt_mpi".
        If None, uses "cpu" by default.
    pure : bool, default=False
        Whether the function is pure (deterministic).
        Set to False for yt operations.
    **kwargs
        Keyword arguments for the function.

    Returns
    -------
    Any
        The result of the function.

    Examples
    --------
    >>> import astroflow as af
    >>> af.enable_parallelism()
    >>> # Run a yt operation in parallel
    >>> result = af.parallel.run_parallel(my_yt_func, ds, policy="yt_mpi")
    """
    if not _parallel_enabled:
        return func(*args, **kwargs)

    # Get or create adapter
    adapter = DaskSchedulerAdapter()

    # Build policy
    task_policy = TaskPolicy(
        policy=policy or "cpu",
        pure=pure,
        n_procs=_parallel_config.get("n_procs", 4),
    )

    # Submit and wait for result
    result = adapter.submit(func, *args, policy=task_policy, **kwargs)

    # If it's a Future, wait for result
    if hasattr(result, "result") and callable(result.result):
        return result.result()
    return result


def piter(iterable: Any, **kwargs: Any) -> Any:
    """
    Parallel iterator that works with yt's piter when parallelism is enabled.

    When parallelism is enabled, this wraps yt.parallel_objects or similar
    functionality. When disabled, returns the iterable unchanged.

    Parameters
    ----------
    iterable : iterable
        The iterable to parallelize.
    **kwargs
        Additional arguments passed to the parallel iterator.

    Returns
    -------
    iterable
        A parallel iterator if parallelism is enabled, otherwise the original.

    Examples
    --------
    >>> import astroflow as af
    >>> af.enable_parallelism()
    >>> for ds in af.parallel.piter(simulation):
    ...     # Process each snapshot in parallel
    ...     process(ds)
    """
    if not _parallel_enabled:
        return iterable

    # Try to use yt's parallel_objects if available
    try:
        import yt

        return yt.parallel_objects(iterable, **kwargs)
    except (ImportError, AttributeError):
        pass

    # Fallback: return iterable (serial execution)
    afLogger.debug("yt.parallel_objects not available, using serial iteration")
    return iterable


def _import_dask_distributed() -> Any:
    """Import dask.distributed, raising ImportError with guidance if unavailable."""
    try:
        import dask.distributed

        return dask.distributed
    except ImportError as e:
        raise ImportError(
            "Dask distributed is required for parallel execution. "
            "Install it with: pip install 'astroflow[parallel]' or pip install dask distributed"
        ) from e


def auto(
    scheduler_address: Optional[str] = None,
    n_workers: Optional[int] = None,
    threads_per_worker: int = 1,
    **kwargs: Any,
) -> Any:
    """
    Automatically configure and return a Dask client.

    If a scheduler_address is provided, connects to an existing cluster.
    Otherwise, creates a LocalCluster with the specified parameters.

    Parameters
    ----------
    scheduler_address : str, optional
        Address of an existing Dask scheduler to connect to.
        If None, creates a new LocalCluster.
    n_workers : int, optional
        Number of workers for LocalCluster. Defaults to system CPU count.
    threads_per_worker : int, default=1
        Threads per worker for LocalCluster.
    **kwargs
        Additional arguments passed to LocalCluster or Client.

    Returns
    -------
    distributed.Client
        The active Dask client.

    Examples
    --------
    >>> client = auto()  # Creates local cluster
    >>> client = auto("tcp://scheduler:8786")  # Connects to existing
    """
    global _active_client

    if scheduler_address:
        _active_client = connect(scheduler_address, **kwargs)
    else:
        distributed = _import_dask_distributed()
        cluster_kwargs = {"threads_per_worker": threads_per_worker}
        if n_workers is not None:
            cluster_kwargs["n_workers"] = n_workers

        # Extract client-specific kwargs
        client_kwargs = {}
        for key in list(kwargs.keys()):
            if key in ("timeout", "set_as_default", "direct_to_workers"):
                client_kwargs[key] = kwargs.pop(key)

        cluster_kwargs.update(kwargs)
        cluster = distributed.LocalCluster(**cluster_kwargs)
        _active_client = distributed.Client(cluster, **client_kwargs)
        afLogger.info(
            f"Created LocalCluster with {cluster.n_workers} workers. "
            f"Dashboard: {_active_client.dashboard_link}"
        )

    return _active_client


def connect(scheduler_address: str, **kwargs: Any) -> Any:
    """
    Connect to an existing Dask scheduler.

    Parameters
    ----------
    scheduler_address : str
        Address of the Dask scheduler (e.g., "tcp://scheduler:8786").
    **kwargs
        Additional arguments passed to distributed.Client.

    Returns
    -------
    distributed.Client
        The connected Dask client.

    Examples
    --------
    >>> client = connect("tcp://localhost:8786")
    """
    global _active_client
    distributed = _import_dask_distributed()
    _active_client = distributed.Client(scheduler_address, **kwargs)
    afLogger.info(
        f"Connected to scheduler at {scheduler_address}. "
        f"Dashboard: {_active_client.dashboard_link}"
    )
    return _active_client


def set_client(client: Any) -> None:
    """
    Set an externally created Dask client as the active client.

    Parameters
    ----------
    client : distributed.Client
        The Dask client to set as active.

    Examples
    --------
    >>> from dask.distributed import Client
    >>> my_client = Client()
    >>> set_client(my_client)
    """
    global _active_client
    _active_client = client
    afLogger.info(f"Set active client: {client}")


def get_client() -> Optional[Any]:
    """
    Get the currently active Dask client.

    Returns
    -------
    distributed.Client or None
        The active Dask client, or None if no client is set.

    Examples
    --------
    >>> client = get_client()
    >>> if client is None:
    ...     client = auto()
    """
    return _active_client


def dashboard_link() -> Optional[str]:
    """
    Get the dashboard URL for the active Dask client.

    Returns
    -------
    str or None
        The dashboard URL, or None if no client is active.

    Examples
    --------
    >>> link = dashboard_link()
    >>> print(f"Dashboard: {link}")
    """
    if _active_client is None:
        return None
    return getattr(_active_client, "dashboard_link", None)


def close() -> None:
    """
    Close the active Dask client and release resources.

    Examples
    --------
    >>> close()
    """
    global _active_client
    if _active_client is not None:
        _active_client.close()
        afLogger.info("Closed Dask client")
        _active_client = None


@dataclass
class TaskPolicy:
    """
    Metadata for task execution policy.

    Attributes
    ----------
    policy : ExecutionPolicy
        Execution policy: "cpu" (Dask threads), "process" (Dask processes),
        or "yt_mpi" (MPI-based yt execution).
    resources : dict
        Resource hints for the scheduler (e.g., {"memory": "2GB"}).
    pure : bool
        Whether the function is pure (deterministic). Set to False for yt tasks.
    priority : int
        Task priority (higher = more important).
    n_procs : int
        Number of MPI processes for yt_mpi policy.
    eager_small : bool
        If True, execute small tasks immediately without scheduling overhead.
    """

    policy: ExecutionPolicy = "cpu"
    resources: Dict[str, Any] = field(default_factory=dict)
    pure: bool = True
    priority: int = 0
    n_procs: int = 4
    eager_small: bool = False


def run_yt_parallel(
    func: Callable[..., T],
    *args: Any,
    n_procs: int = 4,
    **kwargs: Any,
) -> T:
    """
    Run a function in parallel using MPI, preferring mpi4py.futures.

    This function first attempts to use mpi4py.futures.MPIPoolExecutor for
    in-process MPI execution. If mpi4py is not available, it falls back to
    spawning an external mpirun process.

    Parameters
    ----------
    func : Callable
        The function to execute in parallel.
    *args
        Positional arguments passed to the function.
    n_procs : int, default=4
        Number of MPI processes to use.
    **kwargs
        Keyword arguments passed to the function.

    Returns
    -------
    Any
        The result of the function execution.

    Raises
    ------
    RuntimeError
        If both mpi4py and mpirun are unavailable, or if execution fails.

    Examples
    --------
    >>> def process_data(data):
    ...     return data * 2
    >>> result = run_yt_parallel(process_data, my_data, n_procs=8)
    """
    # Try mpi4py.futures first
    try:
        from mpi4py.futures import MPIPoolExecutor

        afLogger.debug(f"Using mpi4py.futures with {n_procs} processes")
        with MPIPoolExecutor(max_workers=n_procs) as executor:
            future = executor.submit(func, *args, **kwargs)
            return future.result()
    except ImportError:
        afLogger.debug(
            "mpi4py.futures not available, falling back to mpirun subprocess"
        )

    # Fallback to mpirun subprocess
    return _run_mpirun_fallback(func, *args, n_procs=n_procs, **kwargs)


def _run_mpirun_fallback(
    func: Callable[..., T],
    *args: Any,
    n_procs: int = 4,
    **kwargs: Any,
) -> T:
    """
    Execute function via mpirun subprocess.

    This is a fallback when mpi4py is not available. It serializes the
    function and arguments, then executes them via mpirun.
    """
    import os
    import pickle
    import tempfile

    # Validate n_procs to prevent command injection
    if not isinstance(n_procs, int) or n_procs < 1:
        raise ValueError(f"n_procs must be a positive integer, got {n_procs}")

    # Create wrapper script
    wrapper_code = """
import pickle
import sys

if __name__ == "__main__":
    data_file = sys.argv[1]
    with open(data_file, "rb") as f:
        func, args, kwargs = pickle.load(f)
    result = func(*args, **kwargs)
    with open(data_file + ".result", "wb") as f:
        pickle.dump(result, f)
"""

    # Serialize function and arguments
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".pkl", delete=False
    ) as data_file:
        pickle.dump((func, args, kwargs), data_file)
        data_path = data_file.name

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as script_file:
        script_file.write(wrapper_code)
        script_path = script_file.name

    result_path = data_path + ".result"
    try:
        # Run via mpirun
        cmd = ["mpirun", "-n", str(n_procs), sys.executable, script_path, data_path]
        afLogger.debug(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Load result
        if not os.path.exists(result_path):
            raise RuntimeError(
                "MPI execution completed but result file not found. "
                "The function may have failed silently."
            )
        with open(result_path, "rb") as f:
            return pickle.load(f)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"mpirun failed with exit code {e.returncode}: {e.stderr}"
        ) from e
    except FileNotFoundError:
        raise RuntimeError(
            "mpirun not found. Install MPI (e.g., OpenMPI) or mpi4py."
        ) from None
    except (pickle.UnpicklingError, EOFError) as e:
        raise RuntimeError(f"Failed to read result from MPI execution: {e}") from e
    finally:
        # Cleanup temp files
        for path in [data_path, script_path, result_path]:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass  # File may not exist, which is fine
            except OSError as e:
                afLogger.debug(f"Failed to cleanup temp file {path}: {e}")


class DaskSchedulerAdapter:
    """
    Adapter for submitting tasks to Dask with execution policy support.

    This adapter handles different execution policies:
    - "cpu": Execute via Dask with thread-based workers
    - "process": Execute via Dask with process-based workers
    - "yt_mpi": Execute via run_yt_parallel with MPI

    Parameters
    ----------
    client : distributed.Client, optional
        Dask client to use. If None, uses the module-level active client.

    Examples
    --------
    >>> adapter = DaskSchedulerAdapter()
    >>> policy = TaskPolicy(policy="cpu", pure=False)
    >>> future = adapter.submit(my_func, arg1, arg2, policy=policy)
    >>> result = future.result()
    """

    def __init__(self, client: Optional[Any] = None):
        self._client = client

    @property
    def client(self) -> Any:
        """Get the Dask client, creating one if necessary."""
        if self._client is not None:
            return self._client

        global _active_client
        if _active_client is None:
            _active_client = auto()
        return _active_client

    def submit(
        self,
        func: Callable[..., T],
        *args: Any,
        policy: Optional[Union[TaskPolicy, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Submit a task for execution based on the execution policy.

        Parameters
        ----------
        func : Callable
            The function to execute.
        *args
            Positional arguments for the function.
        policy : TaskPolicy or dict, optional
            Execution policy. If dict, converted to TaskPolicy.
            If None, uses default TaskPolicy (cpu, pure=True).
        **kwargs
            Keyword arguments for the function.

        Returns
        -------
        Future or result
            For Dask execution, returns a Future.
            For eager_small or yt_mpi, returns the result directly.

        Examples
        --------
        >>> future = adapter.submit(compute_halo, data, policy={"policy": "cpu"})
        >>> result = future.result()

        >>> # For yt operations, use pure=False
        >>> policy = TaskPolicy(policy="cpu", pure=False)
        >>> future = adapter.submit(yt_operation, ds, policy=policy)
        """
        # Convert dict to TaskPolicy if needed
        if policy is None:
            task_policy = TaskPolicy()
        elif isinstance(policy, dict):
            task_policy = TaskPolicy(**policy)
        else:
            task_policy = policy

        # Handle eager_small fast path
        if task_policy.eager_small:
            afLogger.debug(f"Executing {func.__name__} eagerly (eager_small=True)")
            return func(*args, **kwargs)

        # Handle yt_mpi policy
        if task_policy.policy == "yt_mpi":
            afLogger.debug(
                f"Executing {func.__name__} via MPI with {task_policy.n_procs} procs"
            )
            return run_yt_parallel(func, *args, n_procs=task_policy.n_procs, **kwargs)

        # Handle cpu/process policies via Dask
        # Filter out keys that conflict with our explicit policy settings
        filtered_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ("pure", "priority", "resources")
        }

        submit_kwargs: Dict[str, Any] = {
            "pure": task_policy.pure,
            "priority": task_policy.priority,
        }

        if task_policy.resources:
            submit_kwargs["resources"] = task_policy.resources

        afLogger.debug(
            f"Submitting {func.__name__} to Dask "
            f"(policy={task_policy.policy}, pure={task_policy.pure})"
        )

        return self.client.submit(func, *args, **filtered_kwargs, **submit_kwargs)

    def map(
        self,
        func: Callable[..., T],
        *iterables: Any,
        policy: Optional[Union[TaskPolicy, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Map a function over iterables using Dask.

        Parameters
        ----------
        func : Callable
            The function to map.
        *iterables
            Iterables to map over.
        policy : TaskPolicy or dict, optional
            Execution policy for all tasks.
        **kwargs
            Additional arguments passed to client.map.

        Returns
        -------
        list of Future
            Futures for each mapped task.

        Examples
        --------
        >>> futures = adapter.map(process_snapshot, snapshots)
        >>> results = [f.result() for f in futures]
        """
        if policy is None:
            task_policy = TaskPolicy()
        elif isinstance(policy, dict):
            task_policy = TaskPolicy(**policy)
        else:
            task_policy = policy

        # For yt_mpi, process sequentially
        if task_policy.policy == "yt_mpi":
            results = []
            for items in zip(*iterables):
                result = run_yt_parallel(
                    func, *items, n_procs=task_policy.n_procs, **kwargs
                )
                results.append(result)
            return results

        # Filter out keys that conflict with our explicit policy settings
        filtered_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ("pure", "priority", "resources")
        }

        map_kwargs: Dict[str, Any] = {
            "pure": task_policy.pure,
        }

        if task_policy.resources:
            map_kwargs["resources"] = task_policy.resources

        return self.client.map(func, *iterables, **filtered_kwargs, **map_kwargs)

    def gather(self, futures: Any) -> Any:
        """
        Gather results from futures.

        Parameters
        ----------
        futures : list of Future or Future
            Futures to gather.

        Returns
        -------
        list or single result
            The gathered results.
        """
        return self.client.gather(futures)
