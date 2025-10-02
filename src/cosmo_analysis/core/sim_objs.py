"""Data structures for simulation objects.

This module defines dataclasses for storing simulation data, snapshots,
and particle information. These structures provide a consistent interface
for accessing simulation data across the codebase.
"""

import numpy as np
import yt
from dataclasses import dataclass, field


@dataclass
class particleList:
    """Container for particle data arrays.
    
    Stores position, velocity, and physical properties for particles.
    
    Attributes:
        x: X positions (pc)
        y: Y positions (pc)
        z: Z positions (pc)
        m: Masses (Msun)
        vx: X velocities (km/s)
        vy: Y velocities (km/s)
        vz: Z velocities (km/s)
        d: Gas density (Msun/pc^3)
        h: Smoothing length (pc)
        t: Temperature (K)
        mt: Metallicity
    """
    # Global properties
    x:  np.ndarray = None
    y:  np.ndarray = None
    z:  np.ndarray = None
    m:  np.ndarray = None
    vx: np.ndarray = None
    vy: np.ndarray = None
    vz: np.ndarray = None
    # Gas-specific properties
    d:  np.ndarray = None
    h:  np.ndarray = None
    t:  np.ndarray = None
    mt: np.ndarray = None


@dataclass
class snapshot:
    """Container for snapshot metadata and data.
    
    Stores information about a single simulation snapshot including
    time, redshift, center, and particle data.
    
    Attributes:
        idx: Snapshot index in analysis
        ytIdx: True file index
        time: Time in Myr
        z: Redshift (for cosmological simulations)
        a: Scale factor (for cosmological simulations)
        p: List of particleList objects
        center: Center coordinates (pc)
        ytcen: Center as YT array
        rvir: Virial radius
        cosmo: Whether this is a cosmological simulation (0/1)
        pType: List of particle types present
        face_on: Face-on orientation vector
        edge_on: Edge-on orientation vector
    """
    idx    : int
    ytIdx  : int
    time   : float
    z      : float
    a      : float

    p      : list             = field(default_factory=list)
    center : np.ndarray       = None
    ytcen  : yt.units.YTArray = None
    rvir   : float            = None
    cosmo  : int              = None
    pType  : list             = field(default_factory=list)
    face_on: np.ndarray       = None
    edge_on: np.ndarray       = None


@dataclass
class simul:
    """Container for simulation data.
    
    Top-level object for a simulation, containing multiple snapshots
    and YT dataset objects.
    
    Attributes:
        name: Simulation name
        cosmo: Whether this is a cosmological simulation (0/1)
        ytFile: YT time series object
        ytFull: List of YT dataset objects for each snapshot
        snap: List of snapshot objects
    """
    name  : str
    cosmo : int  = None
    ytFile: ...  = None
    ytFull: list = field(default_factory=list)
    snap  : list = field(default_factory=list)