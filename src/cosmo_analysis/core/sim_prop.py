"""Simulation property calculations.

This module provides functions for calculating physical properties of simulations,
including various centering algorithms, virial radius calculations, and other
snapshot-level properties.

All functions include structured logging for progress tracking and debugging.
"""

from .. import log
from ..log import log_performance, log_progress
from ..core import utils
from ..io   import load
import numpy as np
import unyt
import yt

# Calculate the actual center of a simulation snapshot
@log_performance(level=log.logging.DEBUG)
def findCenter(sim, snapshotN, lim=20):
    """Calculate center by finding maximum density particle.
    
    Args:
        sim: Simulation object
        snapshotN: Snapshot index
        lim: Radius limit in kpc for initial sphere (default: 20)
    
    Returns:
        tuple: (center_array in pc, center_unyt_array in kpc)
    """
    log.logger.debug(f"Finding center using max density method for snapshot {snapshotN} with lim={lim} kpc")
    snap   = sim.ytFull[snapshotN]
    cutOff = snap.sphere("center",(lim,"kpc"))
    den    = np.array(cutOff["PartType0", "Density"].to("Msun/pc**3"))
    x      = np.array(cutOff["PartType0", "x"].to("pc"))
    y      = np.array(cutOff["PartType0", "y"].to("pc"))
    z      = np.array(cutOff["PartType0", "z"].to("pc"))
    cenIdx = utils.maxIdx(den)
    cen    = np.array([x[cenIdx],y[cenIdx],z[cenIdx]])
    log.logger.debug(f"Found center at ({cen[0]:.2f}, {cen[1]:.2f}, {cen[2]:.2f}) pc")
    return (cen,cen/1e3*unyt.kpc)

# Centering calc2 (center of mass)
@log_performance(level=log.logging.DEBUG)
def findCenter2(sim, snapshotN, lim=20):
    """Calculate center using gas center of mass.
    
    Args:
        sim: Simulation object
        snapshotN: Snapshot index
        lim: Radius limit in kpc for initial sphere (default: 20)
    
    Returns:
        tuple: (center_array in pc, center_unyt_array in kpc)
    """
    log.logger.debug(f"Finding center using center of mass method for snapshot {snapshotN} with lim={lim} kpc")
    snap   = sim.ytFull[snapshotN]
    cutOff = snap.sphere("center",(lim,"kpc"))
    cen    = cutOff.quantities.center_of_mass(use_gas=True, use_particles=False).in_units("pc")
    log.logger.debug(f"Found center at ({cen.d[0]:.2f}, {cen.d[1]:.2f}, {cen.d[2]:.2f}) pc")
    return (cen.d,cen.d/1e3*unyt.kpc)

# Centering calc3 (like AGORA Paper)
@log_performance(level=log.logging.DEBUG)
def findCenter3(sim, snapshotN):
    """Calculate center using AGORA method for isolated simulations.
    
    Finds maximum density, refines with center of mass, then finds max density again.
    
    Args:
        sim: Simulation object
        snapshotN: Snapshot index
    
    Returns:
        tuple: (center_array in pc, center_unyt_array in kpc)
    """
    log.logger.debug(f"Finding center using AGORA isolated method for snapshot {snapshotN}")
    snap      = sim.ytFull[snapshotN]

    v, cen1   = snap.find_max(("gas", "density"))
    log.logger.debug(f"Initial max density at {cen1}")
    bigCutOff = snap.sphere(cen1, (30.0, "kpc"))
    cen2      = bigCutOff.quantities.center_of_mass(use_gas=True, use_particles=False).in_units("kpc")
    log.logger.debug(f"Refined to center of mass at {cen2}")
    cutOff    = snap.sphere(cen2,(1.0,"kpc"))
    cen       = cutOff.quantities.max_location(("gas", "density"))
    center    = np.array([cen[1].d,cen[2].d,cen[3].d])*1e3
    log.logger.debug(f"Final center at ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}) pc")

    return (center,center/1e3*unyt.kpc)

# Centering calc4 (like AGORA Paper, for CosmoRun)
@log_performance(level=log.logging.DEBUG)
def findCenter4(sim, idx, projPath=None):
    """Calculate center using AGORA method for cosmological runs.
    
    Args:
        sim: Simulation object
        idx: Snapshot index
        projPath: Path to projection file. If None, uses config value.
    
    Returns:
        tuple: (center_array, center_unyt_array)
    """
    from ..config import get_config
    
    log.logger.debug(f"Finding center using AGORA cosmological method for snapshot {idx}")
    snap = sim.ytFull[idx]
    
    # Get projection path from config if not provided
    if projPath is None:
        config = get_config()
        projPath = config.get('paths.data_files.projection_list', 'outputlist_projection.txt')
    
    log.logger.debug(f"Loading projection list from: {projPath}")
    f        = np.loadtxt(projPath,skiprows=4)
    idx0     = utils.getClosestIdx(f[:,0],0.99999) # Finds end of first projection
    tIdx     = utils.getClosestIdx(f[0:idx0,1],sim.snap[idx].a)

    cen1     = np.array([f[tIdx,3],f[tIdx,4],f[tIdx,5]]) # In Mpc
    log.logger.debug(f"Initial center from projection: ({cen1[0]:.4f}, {cen1[1]:.4f}, {cen1[2]:.4f}) Mpc")
    center   = snap.arr([cen1[0], cen1[1], cen1[2]],'code_length')
    
    # Iterative refinement with shrinking spheres
    log.logger.debug("Refining center with iterative shrinking spheres: 2.0, 1.0, 0.5 kpc")
    sp       = snap.sphere(center, (2,'kpc'))
    center   = sp.quantities.center_of_mass(use_gas=True,use_particles=False)
    sp       = snap.sphere(center, (2*0.5,'kpc'))
    center   = sp.quantities.center_of_mass(use_gas=True,use_particles=False)
    sp       = snap.sphere(center, (2*0.25,'kpc'))
    center   = sp.quantities.center_of_mass(use_gas=True,use_particles=False)
    center   = center.to("code_length")

    cen1     = np.array([center[0].d,center[1].d,center[2].d]) 
    log.logger.debug(f"Final center: ({cen1[0]:.6e}, {cen1[1]:.6e}, {cen1[2]:.6e}) code_length")

    return   (cen1,center)

# Centering calc5 (AGORA code shortcut, for isolated)
@log_performance(level=log.logging.DEBUG)
def findCenter5(sim, snapshotN):
    """Calculate center using hardcoded AGORA coordinates.
    
    Uses a specific center position from AGORA simulations.
    
    Args:
        sim: Simulation object
        snapshotN: Snapshot index
    
    Returns:
        tuple: (center_array in pc, center_unyt_array in kpc)
    """
    log.logger.debug(f"Using hardcoded AGORA center for snapshot {snapshotN}")
    snap   = sim.ytFull[snapshotN]
    cen    = snap.arr([6.184520935812296e+21, 4.972678132728082e+21, 6.559067311284074e+21], 'cm')
    cen    = cen.to("pc")
    cent   = np.array([cen[0].d,cen[1].d,cen[2].d])
    log.logger.debug(f"Hardcoded center: ({cent[0]:.2f}, {cent[1]:.2f}, {cent[2]:.2f}) pc")
    return (cen,cen/1e3*unyt.kpc)

# Centering calc6 (just 0 0 0)
@log_performance(level=log.logging.DEBUG)
def findCenter6(sim, snapshotN):
    """Calculate center at origin (0,0,0).
    
    Args:
        sim: Simulation object
        snapshotN: Snapshot index
    
    Returns:
        tuple: (center_array in pc, center_unyt_array in kpc)
    """
    log.logger.debug(f"Using origin (0,0,0) as center for snapshot {snapshotN}")
    cen    = np.array([0,0,0])
    return (cen,cen/1e3*unyt.kpc)

# Centering calc7 (like 4, but expanded in scope)
@log_performance(level=log.logging.DEBUG)
def findCenter7(sim, idx, projPath=None):
    """Calculate center using expanded AGORA method.
    
    Args:
        sim: Simulation object
        idx: Snapshot index
        projPath: Path to projection file. If None, uses config value.
    
    Returns:
        tuple: (center_array, center_unyt_array)
    """
    from ..config import get_config
    
    log.logger.debug(f"Finding center using expanded AGORA method for snapshot {idx}")
    snap = sim.ytFull[idx]
    
    # Get projection path from config if not provided
    if projPath is None:
        config = get_config()
        projPath = config.get('paths.data_files.projection_list', 'outputlist_projection.txt')
    
    log.logger.debug("Loading projection file from: %s", projPath)

    f        = np.loadtxt(projPath,skiprows=4)
    idx0     = utils.getClosestIdx(f[:,0],0.99999) # Finds end of first projection
    tIdx     = utils.getClosestIdx(f[0:idx0,1],sim.snap[idx].a)

    cen1     = np.array([f[tIdx,3],f[tIdx,4],f[tIdx,5]]) # In Mpc
    log.logger.debug(f"Initial center from projection: ({cen1[0]:.4f}, {cen1[1]:.4f}, {cen1[2]:.4f}) Mpc")
    center   = snap.arr([cen1[0], cen1[1], cen1[2]],'code_length')
    
    # Extended iterative refinement with larger initial sphere
    log.logger.debug("Refining center with iterative shrinking spheres: 40, 10, 2.0, 1.0, 0.5 kpc")
    sp       = snap.sphere(center, (40,'kpc'))
    center   = sp.quantities.center_of_mass(use_gas=True,use_particles=False)
    sp       = snap.sphere(center, (10,'kpc'))
    center   = sp.quantities.center_of_mass(use_gas=True,use_particles=False)
    sp       = snap.sphere(center, (2,'kpc'))
    center   = sp.quantities.center_of_mass(use_gas=True,use_particles=False)
    sp       = snap.sphere(center, (2*0.5,'kpc'))
    center   = sp.quantities.center_of_mass(use_gas=True,use_particles=False)
    sp       = snap.sphere(center, (2*0.25,'kpc'))
    center   = sp.quantities.center_of_mass(use_gas=True,use_particles=False)
    center   = center.to("code_length")

    cen1     = np.array([center[0].d,center[1].d,center[2].d]) 
    log.logger.debug(f"Final center: ({cen1[0]:.6e}, {cen1[1]:.6e}, {cen1[2]:.6e}) code_length")

    return   (cen1,center)

# Centering calc8 (like 4, for any box)
@log_performance(level=log.logging.DEBUG)
def findCenter8(sim, idx):
    """Calculate center using iterative shrinking sphere method.
    
    Starts from box center and iteratively refines using particle center of mass
    with progressively smaller spheres.
    
    Args:
        sim: Simulation object
        idx: Snapshot index
    
    Returns:
        tuple: (center_array in code units, center_unyt_array in code_length)
    """
    log.logger.debug(f"Finding center using iterative shrinking sphere method for snapshot {idx}")
    snap     = sim.ytFull[idx]

    boxSize = snap.parameters['BoxSize']
    
    cen1     = np.array([boxSize/2,boxSize/2,boxSize/2]) # In code units
    log.logger.debug(f"Starting from box center: ({cen1[0]:.6e}, {cen1[1]:.6e}, {cen1[2]:.6e}) code_length")
    center   = snap.arr([cen1[0], cen1[1], cen1[2]],'code_length')

    itNumber   = 8
    sizeSphere = np.logspace(np.log10(boxSize),np.log10(boxSize*0.0001),itNumber)
    log.logger.debug(f"Iterating {itNumber} times with sphere sizes from {sizeSphere[0]:.3e} to {sizeSphere[-1]:.3e}")

    for i in range(itNumber):
        sp = snap.sphere(center, (sizeSphere[i],'code_length'))
        center   = sp.quantities.center_of_mass(use_gas=False,use_particles=True)
        log.logger.debug(f"  Iteration {i+1}/{itNumber}: sphere size={sizeSphere[i]:.3e}, center={center.d}")

    center   = center.to("code_length")
    cen1     = np.array([center[0].d,center[1].d,center[2].d])
    log.logger.debug(f"Final center: ({cen1[0]:.6e}, {cen1[1]:.6e}, {cen1[2]:.6e}) code_length")

    return   (cen1,center)

# Calculates the critical density based on cosmology and redshift
def getCritRho(co,z):
    return co.critical_density(z).to("Msun/kpc**3")
def getCritRho200(co,z):
    return getCritRho(co,z)*200
# Calculates the mean density based on cosmology and redshift
def getMeanRho(co,z):
    rho_crit0 = getCritRho(co,0)
    rho_mean0 = co.omega_matter * rho_crit0
    return rho_mean0 * (1 + z)**3
def getMeanRho200(co,z):
    return getMeanRho(co,z)*200
# Calculates the density at virial radius according to Paper IV (based on the factor by Bryan & Norman (1998))
def getVirRho(co,z):
    Hz = co.hubble_parameter(z)
    H0 = co.hubble_parameter(0.0)
    Ez = (Hz / H0)
    # Matter fraction at z
    Omega_z = (co.omega_matter * (1 + z)**3 / Ez**2)
    x = Omega_z - 1.0
    # Δc fits from the literature:
    #   • Flat‐curvature (ΩR = 0):   Δc = 18π² + 82x – 39x²
    #   • No‐Λ universe  (ΩΛ = 0):   Δc = 18π² + 60x – 32x²
    Delta_c     = 18*np.pi**2 + 82*x - 39*x**2
    #Delta_c     = 18*np.pi**2 + 60*x - 32*x**2
    return Delta_c*getCritRho(co,z)

# Calculates the virial radius at present redshift via multiple methods
@log_performance
def getRvir(sim,idx,method="Vir",rvirlim=500):
    """Calculate virial radius at current redshift.
    
    Args:
        sim: Simulation object
        idx: Snapshot index
        method: Density method to use ('Vir', 'Crit', or 'Mean')
        rvirlim: Maximum radius limit in kpc
    
    Returns:
        float: Virial radius in kpc
    """
    log.logger.info(f"Calculating virial radius for snapshot {idx} using {method} method (limit: {rvirlim} kpc)")
    
    co = yt.utilities.cosmology.Cosmology(hubble_constant=0.702, omega_matter=0.272,omega_lambda=0.728, omega_curvature=0.0)
    methodDict = {"Crit":getCritRho200,"Mean":getMeanRho200,"Vir":getVirRho}
    snapshot   = sim.ytFull[idx]
    targetDen  = float(methodDict[method](co,sim.snap[idx].z))
    log.logger.debug(f"Target density at z={sim.snap[idx].z:.3f}: {targetDen:.3e} Msun/kpc^3")
    
    log.logger.debug("Loading particle data within 500 kpc sphere")
    sp         = snapshot.sphere(sim.snap[idx].ytcen,(500,"kpc"))
    allMass    = load.getData(sp, "particle_mass", sim.snap[idx].pType, units = "Msun")
    allR       = load.getData(sp, "particle_position_spherical_radius", sim.snap[idx].pType, units = "kpc")
    log.logger.debug(f"Loaded {len(allMass)} particles")
    
    log.logger.debug("Sorting particles by radius and computing cumulative mass profile")
    idx_sort = np.argsort(allR)
    mSort = np.array(allMass)[idx_sort]
    rSort = np.array(allR)[idx_sort]
    cumM  = np.cumsum(mSort)
    denR  = cumM/(4/3*np.pi*rSort**3) 

    idxAtVir = np.argmin(np.abs(denR-targetDen))
    rvir = rSort[idxAtVir]
    mass_enclosed = cumM[idxAtVir]
    mass_predicted = targetDen*(4/3*np.pi*rvir**3)
    
    log.logger.info(f"Found rvir = {rvir:.3f} kpc enclosing {mass_enclosed:.3e} Msun (predicted: {mass_predicted:.3e} Msun)")
    return rvir

# Calculates edge_on and face_on vectors from the total angular momentum
# TODO: Experimental, need to check if faces detected are good
# TODO: Do full halo recognition, requiring integration with Rockstar or Haskap Pie
@log_performance
def getAxes(sim,idx,rvirRatio=0.15):
    """Calculate face-on and edge-on orientation vectors.
    
    Computes galaxy orientation from angular momentum within a sphere.
    
    Args:
        sim: Simulation object
        idx: Snapshot index
        rvirRatio: Fraction of virial radius to use (default: 0.15)
    
    Returns:
        tuple: (face_on_vector, edge_on_vector) as numpy arrays
    """
    sphere_radius = rvirRatio * sim.snap[idx].rvir
    log.logger.info(f"Calculating orientation axes for snapshot {idx} using sphere radius {sphere_radius:.2f} kpc ({rvirRatio:.2f} * rvir)")
    
    # Selects a well centered sphere upto rvirRatio*rvir (defaults to the 0.15 used in AGORA Paper VIII)
    sp = sim.ytFull[idx].sphere(sim.snap[idx].ytcen,(sphere_radius,"kpc"))
    
    # Gets the total angular momentum from the gas or star particles
    part = "PartType0" if not "PartType4" in sim.snap[idx].pType else "PartType4"
    log.logger.debug(f"Computing angular momentum from {part} particles")
    
    lx = sp[(part,"particle_angular_momentum_x")].sum()
    ly = sp[(part,"particle_angular_momentum_y")].sum()
    lz = sp[(part,"particle_angular_momentum_z")].sum()
    lMom = np.array([lx,ly,lz])
    lMom_magnitude = np.linalg.norm(lMom)
    log.logger.debug(f"Total angular momentum: ({lx:.3e}, {ly:.3e}, {lz:.3e}), magnitude: {lMom_magnitude:.3e}")
    
    face_on = lMom/lMom_magnitude
    
    # Pick an arbitrary vector for edge_on calc
    z0 = np.array([0,0,1.0])
    if abs(np.dot(face_on, z0)) > 0.9:
        log.logger.debug("Face-on nearly aligned with z-axis, using x-axis for edge-on calculation")
        z0 = np.array([1.0,0,0])
    
    # Calculate cross to get edge_on vector
    edge_on = np.cross(face_on, z0)
    edge_on /= np.linalg.norm(edge_on)
    
    log.logger.info(f"Calculated face-on = ({face_on[0]:.3f}, {face_on[1]:.3f}, {face_on[2]:.3f})")
    log.logger.info(f"Calculated edge-on = ({edge_on[0]:.3f}, {edge_on[1]:.3f}, {edge_on[2]:.3f})")

    return (face_on,edge_on)