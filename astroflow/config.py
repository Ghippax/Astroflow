from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
import contextlib
import copy
import time
import os
import tempfile

from .log import get_logger

afLogger = get_logger()

# A few utility functions (in utils too) to avoid circular imports
def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Return a new dict with `b` merged into `a` recursively.
    Values in `b` override `a`. Lists are replaced.
    """
    result = copy.deepcopy(a)

    def _merge(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
        for k, v in src.items():
            if k in dst and isinstance(dst[k], dict) and isinstance(v, dict):
                _merge(dst[k], v)
            else:
                dst[k] = copy.deepcopy(v)

    _merge(result, b)
    return result

@contextlib.contextmanager
def file_lock(path, timeout: Optional[float] = 10):
    if path.suffix:
        lock_file = path.with_suffix(path.suffix + ".lock")
    else:
        lock_file = path.with_suffix(".lock")
    start_time = time.time()

    while True:
        try:
            # Try to create lock file exclusively
            fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            afLogger.debug(f"Acquired lock: {lock_file}")
            break
        except FileExistsError:
            # Lock exists, wait and retry
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Could not acquire lock on {path} after {timeout}s")
            time.sleep(0.1)

    try:
        yield
    finally:
        # Release lock
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                lock_file.unlink()
                afLogger.debug(f"Released lock: {lock_file}")
                break
            except (FileNotFoundError, PermissionError) as e:
                if attempt == max_attempts - 1:
                    afLogger.warning(f"Failed to remove lock file: {e}")
                else:
                    time.sleep(0.05)

def atomic_save_yaml(data: dict, path: Path, max_retries: Optional[int] = 5):
    """
    Atomically save YAML data to file with backup and retry logic.
    
    This function:
    1. Creates a backup of the existing file
    2. Writes to a temporary file
    3. Atomically replaces the target file
    
    Parameters
    ----------
    data : dict
        Data to serialize to YAML
    path : Path
        Target file path
    max_retries : int, default=5
        Maximum number of retry attempts
        
    Raises
    ------
    OSError
        If save fails after all retries
    """
    import shutil
    
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Use file lock to prevent concurrent writes
    with file_lock(path):
        # Create backup of existing file
        if path.exists():
            backup_path = path.with_suffix(".yaml.bak")
            try:
                # Remove backup first
                if backup_path.exists():
                    backup_path.unlink()
                # Copy current to backup
                shutil.copy2(path, backup_path)
                afLogger.debug(f"Created backup: {backup_path}")
            except Exception as e:
                afLogger.warning(f"Failed to create backup: {e}")

        # Write to temporary file in same directory
        fd, tmp_path = tempfile.mkstemp(
            dir=path.parent, 
            prefix=".tmp_yaml_", 
            suffix=".yaml", 
            text=True
        )

        try:
            # Write data to temp file
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f)

            # Windows-specific: retry atomic replace with exponential backoff
            for attempt in range(max_retries):
                try:
                    os.replace(tmp_path, path)
                    afLogger.debug(f"Saved YAML to {path}")
                    return  # Success!
                except OSError as e:
                    if attempt == max_retries - 1:
                        raise OSError(
                            f"Failed to save {path} after {max_retries} attempts: {e}"
                        ) from e
                    # Wait with exponential backoff
                    wait_time = 0.1 * (2 ** attempt)
                    afLogger.debug(
                        f"Retry {attempt + 1}/{max_retries} after {wait_time:.2f}s: {e}"
                    )
                    time.sleep(wait_time)

        except Exception as e:
            # Clean up temp file on error
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception as e2:
                afLogger.warning(f"Failed to remove temp file: {e2}")
            raise OSError(f"Failed to save {path}: {e}") from e

class ConfigError(Exception):
    pass


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
        self._load()

    def _load_pkg_default(self) -> Dict[str, Any]:
        if not self._pkg_default_path.exists():
            afLogger.error("Default config file not found, returning empty dict")
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

    def _load(self) -> None:
        pkg = self._load_pkg_default()
        user = self._load_user()
        # deep merge: user keys override package defaults
        self._data = deep_merge(pkg, user)

    def get_value(self, key_path: str, default: Any = None) -> Any:
        """Slash-separated key path, e.g. 'derived/default_list'"""
        parts = key_path.split("/") if key_path else []
        cur = self._data
        for p in parts:
            if not isinstance(cur, dict) or p not in cur:
                return default
            cur = cur[p]
        return cur

    def set_value(self, key_path: str, value: Any, save_user: bool = True) -> None:
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


def get(key: str, default: Any = None) -> Any:
    return default_config.get_value(key, default)


def set(key: str, value: Any, save_user: bool = False) -> None:
    default_config.set_value(key, value, save_user=save_user)


def save(path: Optional[str] = None) -> None:
    default_config.save(path)
