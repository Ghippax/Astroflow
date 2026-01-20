"""
Simulation loading utilities.

Provides functions to load simulation data from various formats.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from yt import DatasetSeries

from .registry import sim_metadata
from .simulation import Simulation
from ..log import get_logger

from datetime import datetime

afLogger = get_logger()

def load(
    path: Optional[str] = None,
    name: Optional[str] = None,
    force_registration: bool = False,
    force_recompute: bool = False,
    **kwargs,
) -> Simulation:
    """Return a :class:`Simulation` instance for the requested dataset.

    Parameters
    ----------
    path
        Filesystem path. If present in the metadata file, this will override the path
        provided here. If a glob pattern is provided, all matching datasets will be loaded.
    unique_name
        Optional unique name for the simulation. Matches against a .yaml file and allows future loading by name. Raises an error if the name is already present. Defaults to None.

    Returns
    -------
    :class:`Simulation` is returned.
    """
    # Determine unique name
    if name is None:
        name = Path(path).stem
    # Check the metadata file for existing simulations, and get path if present
    in_registry = name in sim_metadata.data
    if in_registry:
        afLogger.info(f"Simulation '{name}' found in registry. Loading metadata")
        sim_data = sim_metadata.get(name)
        # Update path if different
        if path is not None and path != sim_data.get("path"):
            afLogger.info(f"Path given for '{name}' is different from registry, updating from {sim_data['path']} to {path}")
            sim_data["path"] = path
            sim_metadata.save()
        path = sim_data.get("path")

    # Load the timeseries using yt
    if isinstance(path, str) and "*" not in path:
        ts = DatasetSeries([path], **kwargs)
    else:
        ts = DatasetSeries(path, **kwargs)

    # Figure out name and register (we register first to check for duplicates)
    dataset_name = type(ts[0]).__name__

    if not in_registry or force_registration:
        name = sim_metadata.register_sim(path, name, dataset_name, force_registration=force_registration)

    # Create the final Simulation object
    sim = Simulation(ts, path, name, dataset_name)

    # Load additional fields
    sim.setup_snapshots(force_recompute=force_recompute)
    sim.add_fields()

    return sim
