from typing import Optional

from .registry import register_postprocessing
from . import settings

import numpy as np
from unyt import unyt_array, G

from ..log import get_logger
afLogger = get_logger()

@register_postprocessing("circ_velocity", label = "Circular Velocity")
def compute_circ_vel(profile, field):
    mass = profile[field]
    radius = profile.x
    return np.sqrt((G*mass/(radius)))

@register_postprocessing("spherical_shell", label = "Density")
def sphere_shell(profile,field):
    mass = profile[field]

    edges = profile.x_bins
    vol = (4.0 / 3.0) * np.pi * (edges[1:]**3 - edges[:-1]**3)

    return mass/vol


@register_postprocessing("circular_surface", label = "Surface Density")
def circ_surface(profile,field):
    mass = profile[field]

    edges = profile.x_bins
    area = np.pi * (edges[1:]**2 - edges[:-1]**2)

    return mass/area

@register_postprocessing("full_circular_surface", label = "Surface Density")
def full_circ_surface(profile,field):
    mass = profile[field]

    edges = profile.x_bins
    area = np.pi * edges[1:]**2

    return mass/area

# TODO: This postprocessing is out of place (no profile usage). Need a way to chain postprocessings (for modularity). This probably involves altering the yt profile object to update after each postprocess and then enabling postprocessing lists in plot.profile
@register_postprocessing("log_spline_derivative", label = "Slope")
def log_spline_derivative(x,y,weights = None, s = None):
    """
    Takes the log of x and y, fits a spline, and returns the derivative of the spline (i.e. the local slope in log-log space)
    """
    from scipy.interpolate import UnivariateSpline
    valid = (x > 0) & (y > 0)
    log_x = np.log10(x[valid])
    log_y = np.log10(y[valid])
    if len(log_x) < 3:
        raise ValueError("Insufficient valid bins for derivative calculation")
    # Fit spline in log-log space
    if weights is None:
        weights = np.ones_like(x[valid])
    else:
        weights = weights[valid]

    spline = UnivariateSpline(log_x, log_y, k = 3, s = s, w = weights)
    alpha = spline.derivative()

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(log_x, log_y, "bo")
    plt.plot(log_x, spline(log_x), "r-")
    plt.plot(np.log10(x), alpha(np.log10(x)))
    
    return alpha(np.log10(x))
