"""Utility functions for plotting and data processing.

This module contains helper functions used across different plotting modules,
including animation creation, binning functions, and cosmological calculations.
"""

import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from matplotlib import rc_context
import yt

from .. import log
from ..config import get_config


def makeMovie(frames, interval=50, verbose=None, saveFigPath=None, message=None, config=None):
    """Create an animated GIF from a list of image frames.
    
    Args:
        frames: list of numpy arrays representing image frames
        interval: delay between frames in milliseconds
        verbose: verbosity level (optional, for backward compatibility)
        saveFigPath: directory path to save animation (optional, uses config if not provided)
        message: filename for saved animation (without extension)
        config: Config object (optional, will use global if not provided)
    
    Returns:
        matplotlib.animation.FuncAnimation: The animation object
    """
    if config is None:
        config = get_config()
    
    if message: 
        log.logger.info(f"\n{message}")
    
    # Create an animation figure using the first frame
    fig_anim, ax_anim = plt.subplots()
    im = ax_anim.imshow(frames[0], animated=True)
    ax_anim.axis('off') 
    fig_anim.tight_layout()
    fig_anim.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    
    # Create animation by setting frames
    def update_frame(i):
        im.set_array(frames[i])
        return [im]
    
    anime = ani.FuncAnimation(fig_anim, update_frame, frames=len(frames), interval=interval, blit=True)

    # Determine save path
    if saveFigPath is not None and saveFigPath != 0:
        base_dir = saveFigPath
    else:
        base_dir = config.get('paths.output_directory', './output')
    
    # Construct full path
    if message and message != 0:
        fullPath = os.path.join(base_dir, message.replace(" ", "_") + ".gif")
    else:
        fullPath = os.path.join(base_dir, "placeholder.gif")
        log.logger.warning("TITLE NOT SPECIFIED FOR THIS ANIMATION, PLEASE SPECIFY A TITLE")

    log.logger.info(f"  Saving animation to {fullPath}")
    
    # Ensure directory exists
    os.makedirs(base_dir, exist_ok=True)
    
    # Get DPI from config
    dpi = config.get('plotting_defaults.dpi', 300)

    with rc_context({"mathtext.fontset": "stix"}):
        anime.save(fullPath, dpi=dpi)
    plt.close(fig_anim)
    return anime


def binFunctionCilBins(cil, bin):
    """Convert binned data to surface density (cylindrical bins).
    
    Args:
        cil: bin center positions in kpc
        bin: binned values (e.g., mass, particle count)
    
    Returns:
        list: Surface density normalized by cylindrical annulus area
    """
    dr = 0.5 * (cil[1] - cil[0])
    newBin = []
    for i in range(len(cil)):
        newBin.append(bin[i] / (np.pi * (((cil[i] + dr) * 1e3)**2 - ((cil[i] - dr) * 1e3)**2)))
    return newBin


def binFunctionSphBins(cil, bin):
    """Convert binned data to volume density (spherical bins).
    
    Args:
        cil: bin center positions in kpc
        bin: binned values (e.g., mass, particle count)
    
    Returns:
        list: Volume density normalized by spherical shell volume
    """
    dr = 0.5 * (cil[1] - cil[0])
    newBin = []
    for i in range(len(cil)):
        newBin.append(bin[i] / (4/3 * np.pi * (((cil[i] + dr) * 1e3)**3 - ((cil[i] - dr) * 1e3)**3)))
    return newBin


def binFunctionCilBinsSFR(cil, bin, youngStarAge=None, config=None):
    """Convert binned star mass to SFR surface density.
    
    Args:
        cil: bin center positions in kpc
        bin: binned star masses
        youngStarAge: age threshold for young stars in Myr (optional, uses config if not provided)
        config: Config object (optional, will use global if not provided)
    
    Returns:
        list: SFR surface density in cylindrical bins
    """
    if config is None:
        config = get_config()
    
    if youngStarAge is None:
        youngStarAge = config.get('analysis_options.young_star_age', 20)
    
    dr = 0.5 * (cil[1] - cil[0])
    bin = np.array(bin) / youngStarAge
    newBin = []
    for i in range(len(cil)):
        newBin.append(bin[i] / (np.pi * (((cil[i] + dr) * 1e3)**2 - ((cil[i] - dr) * 1e3)**2)))
    return newBin


def makeZbinFun(rlimit):
    """Create a binning function for vertical (z-axis) distributions.
    
    Args:
        rlimit: radial limit for cylindrical volume calculation in kpc
    
    Returns:
        function: Binning function that normalizes by cylindrical volume
    """
    def binFunctionZBins(zData, bin, rLim=rlimit):
        dh = (zData[1] - zData[0])
        newBin = []
        for i in range(len(zData)):
            newBin.append(bin[i] / (4 * dh * 1e3 * rLim * 1e3))
        return newBin
    return binFunctionZBins


def binFunctionSphVol(cil, bin):
    """Normalize binned data by spherical shell volume.
    
    This is an alternative formulation using bin edges.
    
    Args:
        cil: bin object with x attribute containing bin edges
        bin: binned values
    
    Returns:
        array: Volume-normalized bin values
    """
    binVal = bin.x
    vol = (4/3) * np.pi * (binVal[1:]**3 - binVal[:-1]**3)
    return bin.d / vol


def aFromT(time, eps=0.1, config=None):
    """Calculate scale factor from cosmic time.
    
    Args:
        time: cosmic time in Myr
        eps: minimum time threshold to avoid issues at t=0
        config: Config object (optional, will use global if not provided)
    
    Returns:
        float: Scale factor a(t), or 0 if time < eps
    """
    if config is None:
        config = get_config()
    
    # Get cosmology parameters from config
    hubble = config.get('cosmology.hubble_constant', 0.702)
    omega_m = config.get('cosmology.omega_matter', 0.272)
    omega_l = config.get('cosmology.omega_lambda', 0.728)
    omega_k = config.get('cosmology.omega_curvature', 0.0)
    
    co = yt.utilities.cosmology.Cosmology(
        hubble_constant=hubble,
        omega_matter=omega_m,
        omega_lambda=omega_l,
        omega_curvature=omega_k
    )
    
    if time < eps:
        return 0
    return co.a_from_t(co.quan(time, "Myr"))
