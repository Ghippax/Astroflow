from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


from .log import get_logger
from .io_utils import atomic_save_yaml, deep_merge

afLogger = get_logger()

class ConfigError(Exception):
    pass

# TODO: I'm regretting the configuration being different from default values in the functions, it would be better to just use the defaults values in functions and then override with config values. Exactly how this would work with data/style/io args defaults is not clear yet, needs reworking


class Config:
    """
    Configuration manager.
    
    Attributes
    ----------
    _data : dict
        The configuration dictionary
    """
    def __init__(self, user_path: Optional[str] = None):
        self.user_path = Path(user_path).expanduser() if user_path else Path("~/.astroflow/user_config.yaml").expanduser()
        self._pkg_default_path = Path(__file__).parent / "default_config.yaml"
        self._data: Dict[str, Any] = {}
        self.load()

    def _load_pkg_default(self) -> Dict[str, Any]:
        if not self._pkg_default_path.exists():
            afLogger.error("Default config file not found, returning empty dict. Functions will break")
            return {}
        try:
            with self._pkg_default_path.open() as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            afLogger.error(f"Failed to load default config: {e}")
            return {}

    def _load_user(self) -> Dict[str, Any]:
        if not self.user_path.exists():
            return {}
        try:
            with self.user_path.open() as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            afLogger.error(f"Corrupted user config file: {e}")
            # Try backup
            backup = self.user_path.with_suffix('.yaml.bak')
            if backup.exists():
                afLogger.warning(f"Loading from backup: {backup}")
                try:
                    with backup.open() as f:
                        return yaml.safe_load(f) or {}
                except Exception as e2:
                    afLogger.error(f"Backup also corrupted: {e2}")
            return {}
        except Exception as e:
            afLogger.error(f"Failed to load user config: {e}")
            return {}

    def load(self) -> None:
        pkg = self._load_pkg_default()
        user = self._load_user()
        # deep merge: user keys override package defaults
        self._data = deep_merge(pkg, user)

    def get(self, key_path: str, default: Any = None) -> Any:
        """Slash-separated key path, e.g. 'derived/default_list'"""
        parts = key_path.split("/") if key_path else []
        cur = self._data
        for p in parts:
            if not isinstance(cur, dict) or p not in cur:
                return default
            cur = cur[p]
        return cur

    def set(self, key_path: str, value: Any, save_user: bool = True) -> None:
        parts = key_path.split("/")
        cur = self._data
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
            if not isinstance(cur, dict):
                raise ConfigError("Invalid config path")
        cur[parts[-1]] = value
        if save_user:
            self.save()

    def save(self, path: Optional[str] = None) -> None:
        """
        Save configuration to file.
        
        Parameters
        ----------
        path : str, optional
            Path to save to. If None, saves to user_path.

        Raises
        ------
        ConfigError
            If save operation fails
        """
        p = Path(path) if path else self.user_path
        try:
            atomic_save_yaml(self._data, p)
        except OSError as e:
            raise ConfigError(f"Failed to save config: {e}") from e
        
    def all(self):
        return dict(self._data)



# Module-level convenience instance and helpers
def load(user_path: Optional[str] = None) -> Config:
    return Config(user_path)

default_config = load()

def reload(config: Config = default_config) -> None:
    config.load()

def get(key: str, default: Any = None, config: Config = default_config) -> Any:
    return config.get(key, default)


def set(key: str, value: Any, save_user: bool = False, config: Config = default_config) -> None:
    config.set(key, value, save_user=save_user)


def save(path: Optional[str] = None, config: Config = default_config) -> None:
    config.save(path)
