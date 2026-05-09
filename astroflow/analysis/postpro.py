from typing import Optional

from .registry import register_postprocessing
from . import settings

import numpy as np
import yt
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

@register_postprocessing("sfh", label="SFR")
def sfh(profile, field, cosmology=None, time_unit="Myr", as_lookback=True):
    """
    Convert a profile of stellar mass binned in scale factor into a SFH.

    Notes
    -----
    - Expects ``profile`` to be built with ``bin_field=('PartType4', 'StellarFormationTime')``
      and ``field=('PartType4', 'Masses')``.
    - Cosmology integrals are evaluated only at bin edges (not particle-by-particle).
    - Updates ``profile.x`` to the returned time coordinate so ``plot.profile`` uses
      the correct x-axis labels.
    """
    if cosmology is None:
        ds = profile.ds
        cosmology = [
            float(getattr(ds, "hubble_constant", 0.702)),
            float(getattr(ds, "omega_matter", 0.272)),
            float(getattr(ds, "omega_lambda", 0.728)),
            float(getattr(ds, "omega_curvature", 0.0)),
        ]

    co = yt.utilities.cosmology.Cosmology(
        hubble_constant=cosmology[0],
        omega_matter=cosmology[1],
        omega_lambda=cosmology[2],
        omega_curvature=cosmology[3],
    )

    # Use bin edges for cosmology conversion to avoid expensive per-particle work.
    z_edges   = (1.0 / np.clip(np.asarray(profile.x_bins, dtype=float), 1e-8, None)) - 1.0
    z_centers = (1.0 / np.clip(np.asarray(profile.x, dtype=float), 1e-8, None)) - 1.0
    t_edges   = unyt_array([co.t_from_z(float(z)).to(time_unit) for z in z_edges],time_unit)
    t_centers = unyt_array([co.t_from_z(float(z)).to(time_unit) for z in z_centers], time_unit)

    mass_per_bin = profile[field].to("Msun")
    dt_yr = np.abs(np.diff(t_edges)).to("yr")
    sfr = (mass_per_bin / dt_yr)

    if as_lookback:
        current_t = profile.ds.current_time.to(time_unit)
        xvals = current_t - t_centers
    else:
        xvals = t_centers

    order = np.argsort(xvals)
    profile.x = unyt_array(xvals[order], time_unit)
    return sfr[order]

# TODO: This postprocessing is out of place (no profile usage). Need a way to chain postprocessings (for modularity). This probably involves altering the yt profile object to update after each postprocess and then enabling postprocessing lists in plot.profile
@register_postprocessing("log_spline_derivative", label = "Slope")
def log_spline_derivative(x,y,weights = None, s = None, see_fit = False):
    """
    Takes the log of x and y, fits a spline, and returns the derivative of the spline (i.e. the local slope in log-log space)
    """
    from scipy.interpolate import UnivariateSpline
    valid = (x > 1e-2) & (y > 0)
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

    if see_fit:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(log_x, log_y, "bo", label = "Data filtered")
        plt.plot(log_x, spline(log_x), "r-", label = "Spline Fit")
        plt.plot(np.log10(x), alpha(np.log10(x)), "g--", label = "Derivative")
        plt.title("Log-Log Derivative using Scipy's UnivariateSpline")
        plt.xlabel("log10(x)")
        plt.ylabel("Slope dlog(y)/dlog(x) and log10(y)")
        plt.legend()
        plt.show()

    return alpha(np.log10(x))

@register_postprocessing("log_numpy_derivative", label = "Slope")
def log_numpy_derivative(x,y, weights = None, s = None, see_fit = False):
    valid = (x > 0) & (y > 0)
    log_x = np.log10(x[valid])
    log_y = np.log10(y[valid])
    if len(log_x) < 3:
        raise ValueError("Insufficient valid bins for derivative calculation")

    alpha_valid = np.gradient(log_y, log_x, edge_order = 2)

    # return full-length output, NaN where invalid
    alpha = np.full_like(x, np.nan, dtype=float)
    alpha[valid] = alpha_valid

    if see_fit:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(log_x, log_y, "bo", label = "Data filtered")
        plt.plot(log_x, alpha_valid, "g--", label = "Derivative")
        plt.xlabel("log10(x)")
        plt.ylabel("Slope dlog(y)/dlog(x) and log10(y)")
        plt.title("Log-Log Derivative using NumPy's Gradient")
        plt.legend()
        plt.show()

    return alpha

@register_postprocessing("log_spline_derivative_fn", label = "Slope")
def log_spline_derivative_func(x,y,weights = None, s = None, see_fit = False):
    """
    Takes the log of x and y, fits a spline, and returns the derivative of the spline (i.e. the local slope in log-log space)
    """
    from scipy.interpolate import UnivariateSpline
    valid = (x > 1e-12) & (y > 0)
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
    dspline = spline.derivative()

    if see_fit:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(log_x, log_y, "bo", label = "Data filtered")
        plt.plot(log_x, spline(log_x), "r-", label = "Spline Fit")
        plt.plot(log_x, dspline(log_x), "g--", label = "Derivative")
        plt.title("Log-Log Derivative using Scipy's UnivariateSpline")
        plt.xlabel("log10(x)")
        plt.ylabel("Slope dlog(y)/dlog(x) and log10(y)")
        plt.legend()
        plt.show()

    return dspline

@register_postprocessing("log_spline_fn", label = "Slope")
def log_spline_func(x,y,weights = None, s = None):
    """
    Takes the log of x and y, fits a spline, and returns the spline function (i.e. the smoothed log-log relation)
    """
    from scipy.interpolate import UnivariateSpline
    valid = (x > 1e-12) & (y > 0)
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

    return spline