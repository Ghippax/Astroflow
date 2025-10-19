from __future__ import annotations

from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Callable, Dict, Optional

import yaml
import copy

from ..logging import get_logger
from ..utils import atomic_save_yaml

afLogger = get_logger()

"""Metadata management for simulations."""


class SimulationMetadata:
    def __init__(self, path="~/.astroflow/simulation_metadata.yaml"):
        self.path = Path(path).expanduser()
        self.data = self._load()

    def _load(self):
        """Load metadata from file with backup recovery."""
        if self.path.exists():
            try:
                with self.path.open("r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    return data if isinstance(data, dict) else {}
            except yaml.YAMLError as e:
                afLogger.error(f"Corrupted metadata file: {e}")
                # Try backup
                backup = self.path.with_suffix(".yaml.bak")
                if backup.exists():
                    afLogger.warning(f"Loading from backup: {backup}")
                    try:
                        with backup.open("r", encoding="utf-8") as f:
                            data = yaml.safe_load(f)
                            return data if isinstance(data, dict) else {}
                    except Exception as e2:
                        afLogger.error(f"Backup also corrupted: {e2}")
                raise
        return {}

    def save(self):
        """
        Save metadata to file.

        Raises
        ------
        OSError
            If save operation fails
        """
        atomic_save_yaml(self.data, self.path)

    def register_sim(self, path, name, code_name, force_registration: bool = False):
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

        if name in self.data and not force_registration:
            raise ValueError(
                f"Simulation '{name}' already exists in {self.path}. "
                f"Consider using force_registration = True to overwrite"
            )
        
        # Preserve existing snapshots if forcing registration
        existing_snapshots = {}
        if force_registration and name in self.data:
            existing_snapshots = copy.deepcopy(self.data[name].get("snapshots", {}))
            afLogger.info(f"Preserving {len(existing_snapshots)} existing snapshots")
        
        entry = {
            "path": str(path),
            "type": code_name,
            "created_at": datetime.now().isoformat(),
            "snapshots": existing_snapshots,
        }
        self.data[name] = entry
        self.save()
        return name

    def get(self, name):
        if name not in self.data:
            afLogger.warning(
                f"Simulation '{name}' not found in {self.path} \n Returning empty dict "
            )
        return self.data.get(name, {})

    def list(self):
        return list(self.data.keys())

    def __contains__(self, name):
        return name in self.data


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

    def compute(self, name, sim_obj, snap_id: int, params: Optional[dict] = None):
        """
        Run method `name` for sim_obj and snapshot snap_id.
        """
        if name not in self._reg:
            available = ", ".join(self._reg.keys())
            raise KeyError(
                f"Derived method '{name}' not registered. Available: {available}"
            )
        fn = self._reg[name]["fn"]
        return fn(sim_obj, snap_id, params or {})


# Default registries
sim_metadata = SimulationMetadata()
derived_registry = DerivedPropRegistry()


# Convenience function for registering derived properties
def register_derived(name: str):
    """
    Decorator to register a derived property computation function.

    Examples
    --------
    >>> @register_derived("virial_radius")
    ... def compute_rvir(sim_obj, snap_id, params):
    ...     # computation
    ...     return radius
    """
    return derived_registry.register(name)
