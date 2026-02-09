from datetime import datetime
from typing import Any

from yt.data_objects.time_series import DatasetSeries

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

    def get_derived(
        self,
        prop_name: str,
        snapshot: int,
        force_recompute: bool = False,
        auto_save: bool = True,
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
    
    def flush(self):
        """Save metadata if any changes were made."""
        if self._metadata_dirty:
            self.metadata_file.save()
            self._metadata_dirty = False

    def list_derived_properties(self):
        """List all registered derived property names."""
        return list(self.derived_registry._reg.keys())

    def purge_metadata(self):
        """Remove this simulation's metadata from its metadata file. May be used to reset cached derived properties or correct corrupt computations."""
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
