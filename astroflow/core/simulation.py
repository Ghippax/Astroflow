from datetime import datetime
from typing import Any, Callable, Literal

from yt.data_objects.time_series import DatasetSeries
import numpy as np
from unyt import unyt_quantity

from .. import config
from ..utils import serialize_units, deserialize_units, param_hash
from .registry import sim_metadata, FunctionRegistry, SimulationMetadata
from ..analysis.registry import derived_fn
from ..log import get_logger

afLogger = get_logger()

# TODO: Eventually we need to abstract over yt entirely and just use it a backend for loading datasets. GPU accelerated out-of-memory arrays are ideal, but for this we essentially need to build an universal data format. SWIFT/GADGET/GIZMO/AREPO (maybe more) share very similar data structures, so starting with a dm/gas/star particle categorization with some standard fields (position, velocity, mass, id, maybe potential energy (also allow calculation of this) and smoothing length if applicable) would be a good start. However handling different code geometries (AREPO's voronoi for example) would be a challenge... Otherwise, yt's abstractions are excellent (geometric data objects, field system, etc). Pynbody does also very well with it's halo system (and Tangos/Caesar would be interesting to look at too), but it's not as flexible as yt in terms of loading different formats and defining custom fields, although the heuristics for particle codes loading may be even better. Scida is very important to look at as well. Some parts of our structure are very good (caching, render/data/plot system, registries, etc) but others are terrible (config system and the registry wrapper is a bit messy)

class Simulation:
    """
    Wrapper around a yt DatasetSeries providing #TODO

    Parameters
    ----------
    ts : yt.DatasetSeries
        The time series dataset to wrap
    path : str or Path
        Path to the simulation directory
    name : str
        Unique identifier for this simulation
    code_name : str
        Name of the simulation code (e.g., 'gadget', 'arepo')
    metadata_file : SimMetadata, optional
        Metadata file handler (default: global sim_metadata)

    Attributes
    ----------
    ts : yt.DatasetSeries
        The wrapped time series
    meta : dict
        Simulation metadata dictionary
    """

    def __init__(
        self,
        ts: DatasetSeries,
        path: str,
        name: str,
        code_name: str,
        metadata_file: SimulationMetadata = sim_metadata,
        derived_registry: FunctionRegistry = derived_fn
    ):
        self.ts = ts
        self.path = path
        self.name = name
        self.code_name = code_name

        self.derived_registry = derived_registry
        self.metadata_file = metadata_file
        self.meta = self.metadata_file.get(name) if name else {}
        self._metadata_dirty = False

        # Cache for loaded datasets
        self._ds_cache: dict = {}
        # Handles arbitrary setup hooks
        self._setup_hooks: list[dict[str, Any]] = []
        # Handles z/a/t mapping to index and snapshot access patterns
        self._timeline: list[dict[str, float]] | None = None
        self.snap = _SnapshotAccessor(self)
        

    def register_setup_hook(
        self,
        fn: Callable,
        *,
        scope: Literal["sim", "snapshot"] = "snapshot",
        stage: Literal["post_fields", "post_setup"] = "post_fields",
        name: str | None = None,
    ) -> None:
        self._setup_hooks.append(
            {"fn": fn, "scope": scope, "stage": stage, "name": name or fn.__name__}
        )

    def run_setup_hooks(
        self,
        *,
        stage: Literal["post_fields", "post_setup"],
        snapshot: int | Literal["all"] | None = "all",
    ) -> None:
        """Run registered setup hooks for a given stage and snapshot(s)."""
        hooks = [h for h in self._setup_hooks if h["stage"] == stage]
        if not hooks:
            return

        # sim-scoped hooks
        for h in hooks:
            if h["scope"] == "sim":
                h["fn"](self)

        # snapshot-scoped hooks
        if snapshot is None:
            return
        idxs = range(len(self.ts)) if snapshot == "all" else [snapshot]
        for i in idxs:
            ds = self[i]
            for h in hooks:
                if h["scope"] == "snapshot":
                    h["fn"](self, i, ds)

    def build_timeline(self, force: bool = False) -> list[dict[str, float]]:
        """Build internal snapshot table with idx/z/a/t_gyr."""
        if self._timeline is not None and not force:
            return self._timeline

        timeline: list[dict[str, float]] = []
        snaps_meta = self.meta.setdefault("snapshots", {})

        for i in range(len(self.ts)):
            ds = self[i]
            smeta = snaps_meta.setdefault(i, {})

            # Prefer internal derived values; fallback to yt only if unavailable.
            # z
            try:
                z = float(self.get_derived("redshift", i, force_recompute=False))
            except Exception:
                afLogger.warning(f"Redshift not found nor able to be calculated in derived properties for snapshot {i}, falling back to yt dataset attribute.")
                try:
                    z = float(getattr(ds, "current_redshift"))
                except Exception:
                    afLogger.warning(f"yt dataset attribute 'current_redshift' not found for snapshot {i}, defaulting to z=0.")
                    z = 0.0
            # a
            try:
                a = float(self.get_derived("scale_factor", i, force_recompute=False))
            except Exception:
                afLogger.warning(f"Scale factor not found nor able to be calculated in derived properties for snapshot {i}, falling back to yt dataset attribute.")
                if np.isfinite(z):
                    a = (1.0 / (1.0 + z)) 
                else:
                    afLogger.warning(f"Redshift is not finite for snapshot {i}, defaulting to a=1.")
                    a = 1.0
            # t
            try:
                t_gyr = float(self.get_derived("time", i, force_recompute=False).to("Gyr").to_value())
            except Exception:
                afLogger.warning(f"Time not found nor able to be calculated in derived properties for snapshot {i}, falling back to yt dataset attribute.")
                try:
                    t_gyr = float(ds.current_time.to("Gyr").value)
                except Exception:
                    afLogger.warning(f"yt dataset attribute 'current_time' not found for snapshot {i}, defaulting to t=0.")
                    t_gyr = 0

            row = {"idx": i, "z": z, "a": a, "t_gyr": t_gyr}
            timeline.append(row)
            smeta["timeline"] = row

        self._timeline = timeline
        self._metadata_dirty = True
        return timeline
    
    def snapshot_index(self) -> list[dict[str, float]]:
        return self.build_timeline()
    
    def sel(
        self,
        *,
        idx: int | None = None,
        z: float | None = None,
        a: float | None = None,
        t: float | None = None,
        unit: str = "Gyr",
        tol: float | None = None,
        return_idx: bool = False,
    ):
        """Select one snapshot by idx or nearest z/a/t."""
        provided = [idx is not None, z is not None, a is not None, t is not None]
        if sum(provided) != 1:
            raise ValueError("Provide exactly one of: idx, z, a, t")

        if idx is not None:
            if not (0 <= idx < len(self.ts)):
                raise ValueError(f"idx out of range: {idx}")
            return idx if return_idx else self.at(idx)

        tl = self.build_timeline()

        if z is not None:
            key, target = "z", float(z)
        elif a is not None:
            key, target = "a", float(a)
        else:
            key, target = "t_gyr", _to_gyr_value(t, unit)

        vals = np.array([row[key] for row in tl], dtype=float)
        good = np.isfinite(vals)
        if not good.any():
            raise RuntimeError(f"No finite timeline values for key '{key}'")

        good_idx = np.where(good)[0]
        j_local = int(np.argmin(np.abs(vals[good_idx] - target)))
        j = int(good_idx[j_local])
        if tol is not None and abs(vals[j] - target) > tol:
            raise ValueError(
                f"No snapshot within tol={tol} for {key}={target}. Closest idx={j}, value={vals[j]}"
            )

        return j if return_idx else self.at(j)
    
    def sel_range(
        self,
        *,
        idx: tuple[int, int] | None = None,
        z: tuple[float, float] | None = None,
        a: tuple[float, float] | None = None,
        t: tuple[float, float] | None = None,
        unit: str = "Gyr",
        step: int = 1,
        tol: float | None = None,
        return_idx: bool = False,
    ):
        """
        Select a slice/range of snapshots.
        A tuple of idx/z/a or t must be provided.
        - idx: step is index stride (int)
        - z/a/t: step is value stride in that coordinate
        """
        provided = [idx is not None, z is not None, a is not None, t is not None]
        if sum(provided) != 1:
            raise ValueError("Provide exactly one of: idx, z, a, t")

        if idx is not None:
            if not isinstance(step, int) or step <= 0:
                raise ValueError("For idx ranges, step must be a positive int")
            i0, i1 = int(idx[0]), int(idx[1])
            lo, hi = sorted((i0, i1))
            sel_idx = list(range(lo, hi + 1, step))
            return sel_idx if return_idx else [self.at(i) for i in sel_idx]

        tl = self.build_timeline()

        if z is not None:
            key, v0, v1 = "z", float(z[0]), float(z[1])
        elif a is not None:
            key, v0, v1 = "a", float(a[0]), float(a[1])
        else:
            key = "t_gyr"
            v0 = self._to_gyr_value(t[0], unit)
            v1 = self._to_gyr_value(t[1], unit)
    
        # Create the range with stepping and tolerance
        vals = np.asarray([row[key] for row in tl], dtype=float)
        finite_idx = np.flatnonzero(np.isfinite(vals))
        if finite_idx.size == 0:
            raise RuntimeError(f"No finite timeline values for key '{key}'")
        step_abs = float(step)
        if step_abs <= 0:
            raise ValueError("For z/a/t ranges, step must be > 0")

        direction = 1.0 if v1 >= v0 else -1.0
        targets = np.arange(
            v0,
            v1 + direction * (0.5 * step_abs),
            direction * step_abs,
            dtype=float,
        )
        if targets.size == 0 or not np.isclose(targets[-1], v1):
            targets = np.append(targets, v1)

        eff_tol = float(tol) if tol is not None else (0.5 * step_abs)

        sel_idx: list[int] = []
        seen: set[int] = set()

        for tv in targets:
            j = int(finite_idx[np.argmin(np.abs(vals[finite_idx] - tv))])
            delta = abs(vals[j] - tv)
            if delta <= eff_tol:
                snap_idx = int(tl[j]["idx"])
                if snap_idx not in seen:
                    sel_idx.append(snap_idx)
                    seen.add(snap_idx)
            else:
                afLogger.warning(
                    f"sel_range: no snapshot within tol={eff_tol:.4g} for "
                    f"{key} target={tv:.4g}. Closest idx={int(tl[j]['idx'])}, value={vals[j]:.4g}"
                )

        if not sel_idx:
            afLogger.warning("sel_range: no snapshots selected with requested range/step/tolerance")

        return sel_idx if return_idx else [self.at(i) for i in sel_idx]



    def get_derived(
        self,
        prop_name: str,
        snapshot: int,
        force_recompute: bool = False,
        auto_save: bool = True,
        label: str | None = None,
        **kwargs
    ) -> Any:
        """
        Get or compute a derived property for a specific snapshot.

        Parameters
        ----------
        prop_name : str
            Name of the derived property to retrieve
        snapshot : int
            Snapshot index
        params : dict, optional
            Parameters to pass to the computation function
        force : bool, default=False
            If True, recompute even if cached value exists
        auto_save : bool, default=True
            If True, save metadata immediately. If False, defer until flush().

        Returns
        -------
        unyt_array or Any
            The computed derived property with units if applicable

        Raises
        ------
        ValueError
            If prop_name is not registered
        IndexError
            If snapshot is out of range
        """
        if not isinstance(snapshot, int) or snapshot < 0 or snapshot >= len(self.ts):
            raise ValueError(
                f"Invalid snapshot index: {snapshot}. "
                f"Must be integer in range [0, {len(self.ts)})"
            )

        snaps = self.meta.setdefault("snapshots", {})
        target = snaps.setdefault(snapshot, {})
        derived = target.setdefault("derived_properties", {})

        # Parameter-aware storage: derived[prop_name][param_hash] = {value, params, ...}
        prop_group = derived.setdefault(prop_name, {})
        phash = param_hash(kwargs)
        prop_dict = prop_group.setdefault(phash, {})

        # If not forcing, return cached value if it exists
        if "value" in prop_dict and not force_recompute:
            afLogger.debug(f"Using cached value for '{prop_name}' [{phash}] at snapshot {snapshot}")
            if label:
                named = target.setdefault("named_derived", {})
                named[label] = {"prop": prop_name, "hash": phash}
                self._metadata_dirty = True
                if auto_save:
                    self.flush()
            return deserialize_units(prop_dict)

        # Perform calculation using the derived properties registry
        try:
            result = self.derived_registry.compute(prop_name, self, snapshot, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to compute {prop_name} for snapshot {snapshot}: {e}")

        # Separate units from value with unyt
        serialized = serialize_units(result)

        if isinstance(serialized, dict) and "value" in serialized and "unit" in serialized:
            prop_dict.update(serialized)
        else:
            # For non-unyt values, store as-is
            prop_dict["value"] = serialized
            prop_dict["unit"] = None

        prop_dict["computed_at"] = datetime.now().isoformat()
        # Serialize kwargs before storing
        serialized_params = {}
        for k, v in kwargs.items():
            serialized_params[k] = serialize_units(v)
        prop_dict["params"] = serialized_params
        
        # Store label mapping if label is provided
        if label:
            named = target.setdefault("named_derived", {})
            named[label] = {"prop": prop_name, "hash": phash}

        self._metadata_dirty = True
        
        if auto_save:
            self.flush()

        return result

    def setup_snapshots(self, force_recompute: bool = False):
        # read default derived list from config
        derived_list = config.get("derived/load_list", [])
        save_one_by_one = config.get("derived/save_in_setup", False)

        for i, ds in enumerate(self.ts.piter()):
            for prop_name in derived_list:
                self.get_derived(prop_name, i,force_recompute=force_recompute, auto_save=save_one_by_one)

        # Single save at the end if specified
        if not save_one_by_one:
            self.flush()

    def get_named(self, snapshot: int, label: str) -> Any:
        snaps = self.meta.setdefault("snapshots", {})
        target = snaps.setdefault(snapshot, {})
        named = target.get("named_derived", {})
        if label not in named:
            raise KeyError(f"Named derived '{label}' not found at snapshot {snapshot}")
        ref = named[label]
        derived = target.get("derived_properties", {})
        return deserialize_units(derived[ref["prop"]][ref["hash"]])

    def list_named(self, snapshot: int) -> list[str]:
        snaps = self.meta.get("snapshots", {})
        target = snaps.get(snapshot, {})
        return sorted((target.get("named_derived") or {}).keys())

    def at(self, snapshot: int):
        return _SnapshotView(self, snapshot)
    
    def flush(self):
        """Save metadata if any changes were made."""
        if self._metadata_dirty:
            self.metadata_file.save()
            self._metadata_dirty = False

    def list_derived_properties(self):
        """List all registered derived property names."""
        return list(self.derived_registry._reg.keys())

    def purge_metadata(self):
        """Remove this simulation's metadata from its metadata file. Can be used to reset cached derived properties or correct corrupt computations."""
        self.metadata_file.purge_sim(self.name)

    def add_fields(self):
        # Placeholder for adding additional fields
        pass

    def __len__(self):
        return len(self.ts)

    def __getitem__(self, index):
        # We try to cache datasets to ensure consistent identity
        if index not in self._ds_cache:
            self._ds_cache[index] = self.ts[index]
        return self._ds_cache[index]

    def __repr__(self):
        return f"<Simulation obj: name={self.name} frontend={self.code_name} path={self.path}>"

class _DerivedProxy:
    def __init__(self, sim: Simulation, idx: int):
        self._sim = sim
        self._idx = idx

    def __getattr__(self, prop_name: str):
        def _call(**kwargs):
            return self._sim.get_derived(prop_name, self._idx, **kwargs)
        return _call

class _NamedProxy:
    def __init__(self, sim: Simulation, idx: int):
        self._sim = sim
        self._idx = idx

    def __getattr__(self, label: str):
        return self._sim.get_named(self._idx, label)
class _SnapshotView:
    def __init__(self, sim: Simulation, idx: int):
        self.sim = sim
        self.idx = idx
        self.d = _DerivedProxy(sim, idx)  # computed access
        self.v = _NamedProxy(sim, idx)    # named access

    @property
    def ds(self):
        return self.sim[self.idx]
        
class _SnapshotAccessor:
    """Compact snapshot API for flexible access"""
    def __init__(self, sim: Simulation):
        self._sim = sim

    def __getitem__(self, idx: int):
        return self._sim.at(idx)

    def __call__(self, **selector_kwargs):
        return self._sim.sel(**selector_kwargs)

    def index(self):
        return self._sim.snapshot_index()

    def range(self, **kwargs):
        return self._sim.sel_range(**kwargs)

    def slice(self, **kwargs):
        return self._sim.sel_range(**kwargs)
    
    def yt(self, idx: int | None = None, **selector_kwargs):
        if idx is not None:
            return self._sim[idx]
        j = self._sim.sel(return_idx=True, **selector_kwargs)
        return self._sim[j]
    
def _to_gyr_value(self, t, unit: str = "Gyr") -> float:
    """Accept float or quantity-like time and return value in Gyr."""
    if hasattr(t, "to"):
        return float(t.to("Gyr").to_value())
    return float(unyt_quantity(t, unit).to("Gyr").value)