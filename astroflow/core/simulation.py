# Wrapper around a yt DatasetSeries
from .registries import sim_metadata
from .registries import derived_registry
from .. import utils
from datetime import datetime
from unyt import unyt_array


class Simulation:
    def __init__(self, ts, path, name, code_name, metadata_file=sim_metadata):
        self.ts = ts
        self.path = path
        self.name = name
        self.code_name = code_name

        self.metadata_file = metadata_file
        self.meta = self.metadata_file.get(name) if name else {}

    def get_derived(self, prop_name, snapshot, params=None, force=False):
        """
        Returns cached derived property if exists, else computes it.
        Parameters specified as a dict.
        """
        # Navigate the metadata structure
        snaps = self.meta.setdefault("snapshots", {})
        target = snaps.setdefault(snapshot, {})

        derived = target.setdefault("derived_properties", {})
        prop_dict = derived.setdefault(prop_name, {})

        # If not forcing, return cached value if it exists
        if "value" in prop_dict and not force:
            return utils.deserialize_units(prop_dict)

        # Perform calculation using the derived properties registry
        result = derived_registry.compute(prop_name, self, snapshot, params)

        # Write metadata to file and return value with units
        prop_dict["value"] = utils.serialize_units(result)
        prop_dict["unit"] = (
            str(result.units) if isinstance(result, unyt_array) else None
        )
        prop_dict["computed_at"] = datetime.now().isoformat()
        prop_dict["params"] = params
        self.metadata_file.save()
        return result

    def setup_snapshots(self, force_comp = False):
        for i, ds in enumerate(self.ts.piter()):
            self.get_derived("redshift", i, force=force_comp)

    def add_fields(self):
        # Placeholder for adding additional fields
        pass

    def __len__(self):
        return len(self.ts)

    def __getitem__(self, index):
        return self.ts[index]

    def __repr__(self):
        return f"<Simulation obj: name={self.name} frontend={self.code_name} path={self.path}>"
