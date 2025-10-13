"""Helpers for loading simulation data into :class:`Simulation` objects."""

import yt
from .simulation import Simulation


def load(path: str) -> Simulation:
    """Return a :class:`Simulation` instance for the requested dataset.

    Parameters
    ----------
    path
        Filesystem path.
        :class:`Simulation` is returned.
    """

    # Load the timeseries using yt
    ts = yt.DatasetSeries([path])

    # Create the simulation object
    sim = Simulation(ts)
    # Load additional fields
    sim.add_fields()

    return sim