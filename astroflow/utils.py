import copy
import contextlib
import time
import os
import tempfile
import yaml

from typing import Any, Dict, Optional
from pathlib import Path

import numpy as np
from unyt import unyt_array, unyt_quantity

from .logging import get_logger

afLogger = get_logger()


# Return unyt arrays to their original form when loading from metadata
def deserialize_units(d):
    if isinstance(d, dict) and "value" in d and "unit" in d:
        if isinstance(d["value"], (list, tuple, np.ndarray)):
            return unyt_array(np.array(d["value"]), d["unit"])
        elif isinstance(d["value"], (float, int)):
            return unyt_quantity(d["value"], d["unit"])
    return d


# Convert unyt arrays to a serializable form for metadata storage
def serialize_units(val):
    if isinstance(val, unyt_array):
        return {"value": val.value.tolist(), "unit": str(val.units)}
    elif isinstance(val, unyt_quantity):
        return {"value": val.value, "unit": str(val.units)}
    return val


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