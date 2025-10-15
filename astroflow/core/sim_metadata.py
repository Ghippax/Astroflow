from datetime import datetime
from pathlib import Path

from functools import wraps
from typing import Callable, Dict, Tuple, Any

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

    def register_sim(self, sim):
        # Generate a name if not provided
        if sim.name is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            base_name = f"{sim.code_name}-{timestamp}"
            # Ensure uniqueness
            counter = 1
            sim.name = base_name
            while sim.name in self.data:
                sim.name = f"{base_name}-{counter}"
                counter += 1

        if sim.name in self.data:
            raise ValueError(f"Simulation '{sim.name}' already exists in {self.path}")
        entry = {
            "path": str(sim.path),
            "type": sim.code_name,
            "created_at": datetime.now().isoformat(),
            "snapshots": {},
        }
        self.data[sim.name] = entry
        self.save()
        return sim.name

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

    def compute(
        self,
        name: str,
        sim_obj,
        snap_id: str,
        params: dict | None = None,
        force: bool = False,
    ) -> Tuple[Any, dict]:
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

derived_prop = DerivedPropRegistry()
