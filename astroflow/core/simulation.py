# Wrapper around a yt DatasetSeries
from .sim_metadata import sim_metadata

class Simulation:
    def __init__(self, ts, path, name, code_name, metadata_file = sim_metadata):
        self.ts = ts
        self.path = path
        self.name = name
        self.code_name = code_name

        self.metadata_file = metadata_file
        self.meta = self.metadata_file.get(name) if name else {}

    def get_derived(self, snapshot, prop_name, method="default"):
        """
        Returns cached derived property if exists, else computes it.
        """
        target = self.meta[snapshot]

        derived = target.setdefault("derived_properties", {})
        prop_dict = derived.setdefault(prop_name, {})

        if method in prop_dict:
            return prop_dict[method]
        else:
            # Perform calculation
            result = self._compute_derived(prop_name, snapshot, method)
            prop_dict[method] = result
            self.metadata_file.save()
            return result

    def _compute_derived(self, prop_name, snapshot, method):
        pass    
    
    def setup_snapshots(self):
        for i, ds in enumerate(self.ts.piter()):
            self.get_derived(i, "redshift")

    def add_fields(self):
        # Placeholder for adding additional fields
        pass

    def __len__(self):
        return len(self.ts)

    def __getitem__(self, index):
        return self.ts[index]

    def __repr__(self):
        return f"<Simulation obj: name={self.name} frontend={self.code_name} path={self.path}>"


