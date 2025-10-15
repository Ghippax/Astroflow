from datetime import datetime
from pathlib import Path
from functools import wraps
from typing import Callable, Dict
import yaml

"""Metadata management for simulations."""


class SimulationMetadata:
    def __init__(self, path="~/.astroflow/simulation_metadata.yaml"):
        self.path = Path(path).expanduser()
        self.data = self._load()

    def _load(self):
        if self.path.exists():
            return yaml.safe_load(self.path.open()) or {}
        return {}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w") as f:
            yaml.safe_dump(self.data, f)

    def register_sim(self, path, name, code_name, force_reg = False):
        # Generate a name if not provided
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            base_name = f"{code_name}-{timestamp}"
            # Ensure uniqueness
            counter = 1
            name = base_name
            while name in self.data:
                name = f"{base_name}-{counter}"
                counter += 1

        if name in self.data and not force_reg:
            raise ValueError(f"Simulation '{name}' already exists in {self.path}. Consider using force_reg = True to overwrite")
        entry = {
            "path": str(path),
            "type": code_name,
            "created_at": datetime.now().isoformat(),
            "snapshots": {},
        }
        self.data[name] = entry
        self.save()
        return name

    def get(self, name):
        if name not in self.data:
            raise KeyError(f"Simulation '{name}' not found in {self.path}")
        return self.data[name]

    def list(self):
        return list(self.data.keys())

"""Registry for derived property calculation methods."""


class DerivedPropRegistry:
    def __init__(self):
        self._reg: Dict[str, Dict] = {}

    def register(self, name: str):
        def _dec(fn: Callable):
            self._reg[name] = {"fn": fn}

            @wraps(fn)
            def wrapper(*a, **kw):
                return fn(*a, **kw)

            return wrapper

        return _dec

    def all(self):
        return {k: v for k, v in self._reg.items()}

    def compute(self, name, sim_obj, snap_id: int, params: dict | None = None):
        """
        Run method `name` for sim_obj and snapshot snap_id.
        Responsible only for executing the function; caller handles metadata writes.
        """
        if name not in self._reg:
            raise KeyError(f"Derived method '{name}' not registered")
        fn = self._reg[name]["fn"]
        return fn(sim_obj, snap_id, params or {})


# Default registries

sim_metadata = SimulationMetadata()

derived_registry = DerivedPropRegistry()
