"""Helpers for loading simulation data into :class:`Simulation` objects."""

import yt

from .sim_metadata import sim_metadata
from .simulation import Simulation


def load(
    unique_name: str | None = None, path: str | None = None, **kwargs
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

    # Check the metadata file for existing simulations, and get path if present
    in_registry = unique_name in sim_metadata.data
    if in_registry:
        sim_data = sim_metadata.get(unique_name)
        # Update path if different
        if path != sim_data["path"]:
            sim_data["path"] = path
            sim_metadata.save()
        path = sim_data["path"]

    # Load the timeseries using yt
    if isinstance(path, str) and "*" not in path:
        ts = yt.DatasetSeries([path], **kwargs)
    else:
        ts = yt.DatasetSeries(path, **kwargs)

    # Figure out name and register (we register first to check for duplicates)
    dataset_name = type(ts[0]).__name__

    if not in_registry:
        sim_metadata.register_sim(Simulation(ts, path, unique_name, dataset_name))

    # Create the final Simulation object
    sim = Simulation(ts, path, unique_name, dataset_name)

    # Load additional fields
    sim.setup_snapshots()
    sim.add_fields()

    return sim
