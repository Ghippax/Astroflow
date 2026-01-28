from __future__ import annotations

from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Callable, Dict, Optional, Any, get_type_hints, get_origin, get_args, Union

import yaml
import copy
import inspect

from .. import config

from pydantic import BaseModel, TypeAdapter

from ..log import get_logger
from ..io_utils import atomic_save_yaml

afLogger = get_logger()


class SimulationMetadata:
    """Metadata management for simulations."""

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
        """Register a simulation in the metadata."""
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
            raise ValueError(f"Simulation '{name}' already exists in {self.path}. "
                             f"Consider using force_registration = True to overwrite")

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
        """Get simulation metadata by name."""
        if name not in self.data:
            afLogger.warning(f"Simulation '{name}' not found in {self.path}. "
                             f"Returning empty dict")
        return self.data.get(name, {})

    def list(self):
        """List all registered simulation names."""
        return list(self.data.keys())
    
    def all(self):
        """Return all metadata."""
        return dict(self.data)

    def __contains__(self, name):
        return name in self.data


def conditional_update(dst: dict, src: dict, provided_set: set = set()):
    """
    Update `dst` with keys from `src` only if the value in `src` is not None
    or if the key is in `provided_set`.
    """
    for k, v in src.items():
        if v is not None or k in provided_set:
            dst[k] = v

class FunctionRegistry():
    """
    Generic registry for callable functions.
    
    Parameters
    ----------
    name : str
        Registry name for error messages
    """
    def __init__(self, name: str = "Generic"):
        self._reg: Dict[str, Any] = {}
        self._name = name

    def register(self, key: str, config_key: Optional[str] = None, set_config: Optional[dict] = None, config_file: Optional[str] = None, **metadata):
        """
        Decorator to register a function.
        
        Parameters
        ----------
        key : str
            Unique identifier for the function
        set_config : dict, optional
            This dict will be set in the user config file (if config_file is not provided), as the function default parameters
        config_file : str, optional
            Path to config file. If None, uses default user config.
        **metadata
            Additional metadata (e.g., description, params)
        """
    
        def decorator(fn: Callable) -> Callable:
            computed_config_key = self._name.lower().replace(" ", "_") + "/" + key if config_key is None else config_key
            config_param_key = computed_config_key + "/params"
            
            if set_config:
                config.set(config_param_key, set_config)
                config.save(config_file)
            
            # Find BaseModel parameters from type hints
            try:
                type_hints = get_type_hints(fn, globalns=fn.__globals__, localns=None)
            except Exception as e:
                afLogger.warning(f"Could not get type hints for function {fn.__name__}: {e}. Considering using explicit type hints in your function definition if you want Pydantic validation and coercion. "
                f"Config overloading may also break if not handled properly. Continuing without type hints")
                type_hints = {}
            
            sig = inspect.signature(fn)
            basemodels = {}
            adapters = {}
            
            # Identify BaseModel parameters in the signature
            for name, value in sig.parameters.items():
                model_cls = type_hints.get(name)
                
                # Handle Optional[BaseModel]
                if get_origin(model_cls) is Union:
                    args = get_args(model_cls)
                    model_cls = next((a for a in args if a is not type(None)), None)
                
                # Check if type hint is a BaseModel subclass
                if model_cls and isinstance(model_cls, type) and issubclass(model_cls, BaseModel):
                    basemodels[name] = model_cls
                    adapters[name] = TypeAdapter(model_cls)
            
            @wraps(fn)
            def wrapper(*args, **kwargs):
                config_defaults = config.get(config_param_key)

                if config_defaults is None:
                    afLogger.warning(
                        f"No configuration found for function '{fn.__name__}' at path '{config_param_key}'. "
                        f"Consider using the set_config parameter when registering the function or manually populate the config file with this path. "
                        f"Continuing with empty defaults")
                    config_defaults = {}

                bound = sig.bind_partial(*args, **kwargs)

                afLogger.debug(f"Adapters: {adapters}")

                # Process each BaseModel parameter
                for pname, model_cls in basemodels.items():
                    # Get field names for this model
                    ta = adapters[pname]
                    model_field_names = set(model_cls.model_fields.keys())

                    # Collect the config defaults of parameters inside this basemodel, even if not explicit in this call
                    basemodel_explicit_conf = {}
                    for sig_param_name, _ in sig.parameters.items():
                        if sig_param_name in model_field_names and sig_param_name in config_defaults:
                            basemodel_explicit_conf[sig_param_name] = config_defaults.get(sig_param_name)
                    # Collect the basemodel config defaults
                    afLogger.debug(f"Config key: {config_param_key} Dict: {config_defaults}")
                    basemodel_conf = config_defaults.get(pname,None) or {}

                    # Collect explicit arguments matching model fields
                    explicit = {}
                    for field_name in bound.arguments.keys():
                        if field_name in model_field_names and field_name != pname:
                            explicit[field_name] = bound.arguments.get(field_name)

                    # Get basemodel param value if provided
                    raw = bound.arguments.get(pname, None)
                    afLogger.debug(f"Raw       ({pname}) user input : {raw}")

                    # coerce/validate user value first
                    param_value = None
                    if raw is not None:
                        param_value = ta.validate_python(raw)
                    
                    # Merge layers: config (basemodel) < config (explicit) < basemodel args < explicit args
                    afLogger.debug(f"Validated ({pname}) basemodel : {param_value}")
                    if param_value is not None:
                        param_dict = param_value.model_dump(exclude_unset=True)

                        afLogger.debug(f"Config    ({pname}) basemodel : {basemodel_conf}")
                        afLogger.debug(f"Config    ({pname}) explicit  : {basemodel_explicit_conf}")
                        afLogger.debug(f"Basemodel ({pname}) user-set  : {param_dict}")
                        afLogger.debug(f"Explicit  (all): {explicit}")
                        merged = {}
                        conditional_update(merged, basemodel_conf)
                        conditional_update(merged, basemodel_explicit_conf)
                        conditional_update(merged, param_dict, set(param_dict.keys()))
                        conditional_update(merged, explicit, set(explicit.keys()))

                        afLogger.debug(f"Full      ({pname}) merged   : {merged}")
                        bound.arguments[pname] = model_cls(**merged)
                    else:
                        afLogger.debug(f"Config    ({pname}) basemodel : {basemodel_conf}")
                        afLogger.debug(f"Config    ({pname}) explicit  : {basemodel_explicit_conf}")
                        afLogger.debug(f"Explicit  (all): {explicit}")
                        merged = {}
                        conditional_update(merged, basemodel_conf)
                        conditional_update(merged, basemodel_explicit_conf)
                        conditional_update(merged, explicit, set(explicit.keys()))

                        afLogger.debug(f"Full      ({pname}) merged   : {merged}")
                        bound.arguments[pname] = model_cls(**merged)

                # Apply config defaults for non-BaseModel parameters
                for pname, _ in sig.parameters.items():
                    if pname not in bound.arguments and pname in config_defaults:
                            bound.arguments[pname] = config_defaults[pname]

                return fn(*bound.args, **bound.kwargs)

            self._reg[key] = {"fn": wrapper, "config_path": computed_config_key, **metadata}
            return wrapper
        return decorator

    def get(self, key: str) -> Callable:
        """Get registered function by key."""
        if key not in self._reg:
            available = ", ".join(self.keys())
            raise KeyError(f"{self._name} '{key}' not registered. Available: {available}")
        return self._reg[key]["fn"]
    
    def get_config_path(self, key: str) -> str:
        """Get config path for a key."""
        if key not in self._reg:
            available = ", ".join(self.keys())
            raise KeyError(f"{self._name} '{key}' not registered. Available: {available}")
        return self._reg[key]["config_path"]

    def get_metadata(self, key: str) -> Dict[str, Any]:
        """Get all metadata for a key."""
        if key not in self._reg:
            available = ", ".join(self.keys())
            raise KeyError(f"{self._name} '{key}' not registered. Available: {available}")
        return {k: v for k, v in self._reg[key].items() if k != "fn"}
    
    def unregister(self, key: str) -> None:
        """Unregister a function by key."""
        if key in self._reg:
            self._reg.pop(key)
        else:
            raise KeyError(f"{self._name} '{key}' not registered")

    def all(self) -> Dict[str, Dict[str, Any]]:
        """Return all registered items."""
        return dict(self._reg)
    
    def keys(self):
        """Return all registered keys."""
        return list(self._reg.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._reg

# Default registries
sim_metadata = SimulationMetadata()