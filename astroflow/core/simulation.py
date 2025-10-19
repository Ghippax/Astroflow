from datetime import datetime
from typing import Any, Optional

from yt.data_objects.time_series import DatasetSeries

from .. import conf, utils
from .registries import derived_registry, sim_metadata
from ..logging import get_logger

afLogger = get_logger()

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
        metadata_file=None,
    ):
        self.ts = ts
        self.path = path
        self.name = name
        self.code_name = code_name

        self.metadata_file = metadata_file or sim_metadata
        self.meta = self.metadata_file.get(name) if name else {}
        self._metadata_dirty = False

    def get_derived(
        self,
        prop_name: str,
        snapshot: int,
        params: Optional[dict] = None,
        force_recompute: bool = False,
        auto_save: bool = True,
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
        prop_dict = derived.setdefault(prop_name, {})

        # If not forcing, return cached value if it exists
        if "value" in prop_dict and not force_recompute:
            afLogger.debug(f"Using cached value for '{prop_name}' at snapshot {snapshot}")
            return utils.deserialize_units(prop_dict)

        # Perform calculation using the derived properties registry
        try:
            result = derived_registry.compute(prop_name, self, snapshot, params)
        except Exception as e:
            raise RuntimeError(f"Failed to compute {prop_name} for snapshot {snapshot}: {e}")

        # Separate units from value with unyt
        serialized = utils.serialize_units(result)
        if isinstance(serialized, dict) and "value" in serialized and "unit" in serialized:
            prop_dict.update(serialized)
        else:
            # For non-unyt values, store as-is
            prop_dict["value"] = serialized
            prop_dict["unit"] = None

        prop_dict["computed_at"] = datetime.now().isoformat()
        prop_dict["params"] = params or {}
        
        self._metadata_dirty = True
        
        if auto_save:
            self.flush()

        return result

    def setup_snapshots(self, force_recompute: bool = False):
        # read default derived list and params from config
        derived_list = conf.get("derived/load_list", [])
        derived_params = conf.get("derived/params", {})
        save_one_by_one = conf.get("derived/save_in_setup", False)

        for i, ds in enumerate(self.ts.piter()):
            for prop_name in derived_list:
                params = derived_params.get(prop_name, {})
                self.get_derived(prop_name, i, params=params, force_recompute=force_recompute, auto_save=save_one_by_one)

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
        return list(derived_registry._reg.keys())

    def add_fields(self):
        # Placeholder for adding additional fields
        pass

    def __len__(self):
        return len(self.ts)

    def __getitem__(self, index):
        return self.ts[index]

    def __repr__(self):
        return f"<Simulation obj: name={self.name} frontend={self.code_name} path={self.path}>"
