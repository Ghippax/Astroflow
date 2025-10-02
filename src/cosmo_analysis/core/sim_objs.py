"""Data structures for simulation objects.

This module defines dataclasses for storing simulation data, snapshots,
and particle information. These structures provide a consistent interface
for accessing simulation data across the codebase.

The dataclasses are designed to be lightweight and efficient for storing
analysis results. For logging object creation and modifications, use the
logging utilities from cosmo_analysis.log module.
"""

import numpy as np
import yt
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class particleList:
    """Container for particle data arrays.
    
    Stores position, velocity, and physical properties for particles.
    All arrays should have the same length (number of particles).
    
    Attributes:
        x: X positions (pc)
        y: Y positions (pc)
        z: Z positions (pc)
        m: Masses (Msun)
        vx: X velocities (km/s)
        vy: Y velocities (km/s)
        vz: Z velocities (km/s)
        d: Gas density (Msun/pc^3) - gas particles only
        h: Smoothing length (pc) - gas particles only
        t: Temperature (K) - gas particles only
        mt: Metallicity - gas/star particles only
        
    Example:
        >>> particles = particleList()
        >>> particles.x = np.array([1.0, 2.0, 3.0])
        >>> particles.m = np.array([1e6, 1e6, 1e6])
    """
    # Global properties (all particle types)
    x:  Optional[np.ndarray] = None
    y:  Optional[np.ndarray] = None
    z:  Optional[np.ndarray] = None
    m:  Optional[np.ndarray] = None
    vx: Optional[np.ndarray] = None
    vy: Optional[np.ndarray] = None
    vz: Optional[np.ndarray] = None
    # Gas-specific properties
    d:  Optional[np.ndarray] = None
    h:  Optional[np.ndarray] = None
    t:  Optional[np.ndarray] = None
    mt: Optional[np.ndarray] = None


@dataclass
class snapshot:
    """Container for snapshot metadata and data.
    
    Stores information about a single simulation snapshot including
    time, redshift, center, and particle data. This is the primary
    data structure for snapshot-level analysis.
    
    Attributes:
        idx: Snapshot index in analysis sequence
        ytIdx: True file index in the simulation output
        time: Time since simulation start (Myr)
        z: Redshift (for cosmological simulations)
        a: Scale factor (for cosmological simulations, a = 1/(1+z))
        p: List of particleList objects for each particle type
        center: Center coordinates in parsecs (np.ndarray of shape (3,))
        ytcen: Center as YT array with units
        rvir: Virial radius in kpc
        cosmo: Cosmological flag (0 for isolated, 1 for cosmological)
        pType: List of particle types present (e.g., ['PartType0', 'PartType1'])
        face_on: Face-on orientation vector (np.ndarray of shape (3,))
        edge_on: Edge-on orientation vector (np.ndarray of shape (3,))
        
    Example:
        >>> snap = snapshot(idx=0, ytIdx=100, time=500.0, z=0.0, a=1.0)
        >>> snap.center = np.array([0.0, 0.0, 0.0])
        >>> snap.rvir = 150.0
    """
    idx    : int
    ytIdx  : int
    time   : float
    z      : float
    a      : float

    p      : List[particleList]   = field(default_factory=list)
    center : Optional[np.ndarray] = None
    ytcen  : Optional[yt.units.YTArray] = None
    rvir   : Optional[float]      = None
    cosmo  : Optional[int]        = None
    pType  : List[str]            = field(default_factory=list)
    face_on: Optional[np.ndarray] = None
    edge_on: Optional[np.ndarray] = None


@dataclass
class simul:
    """Container for simulation data.
    
    Top-level object for a simulation, containing multiple snapshots
    and YT dataset objects. This is the primary data structure for
    multi-snapshot analysis and comparison.
    
    Attributes:
        name: Simulation name/identifier
        cosmo: Cosmological flag (0 for isolated, 1 for cosmological)
        ytFile: YT time series object for the simulation
        ytFull: List of YT dataset objects, one for each loaded snapshot
        snap: List of snapshot objects with analyzed properties
        
    Example:
        >>> sim = simul(name="my_simulation", cosmo=1)
        >>> # Load snapshots into sim.ytFull
        >>> # Create snapshot objects in sim.snap
        >>> for idx, ds in enumerate(sim.ytFull):
        ...     snap = snapshot(idx=idx, ytIdx=idx, time=ds.current_time.to('Myr').d, 
        ...                    z=ds.current_redshift, a=ds.scale_factor)
        ...     sim.snap.append(snap)
    """
    name  : str
    cosmo : Optional[int] = None
    ytFile: Optional[object] = None  # yt.DatasetSeries
    ytFull: List[object] = field(default_factory=list)  # List of yt.Dataset
    snap  : List[snapshot] = field(default_factory=list)