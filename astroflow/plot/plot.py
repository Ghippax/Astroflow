from __future__ import annotations

from typing import Optional, Union

import matplotlib.pyplot as plt
from .registry import register_plot
from . import data, render, settings

from ..log import get_logger

afLogger = get_logger()

"""
Plotting API: They output/save final plots or animations, generated with plot_data and rendering with plot_render modules
"""

# TODO: Add functionality for saving data outputs (frbs and profiles) to disk using yt (just add a new io_args argument with path, format, overwrite, etc)

# TODO: Add functionality for loading data outputs (frbs and profiles) from disk using yt. 

# TODO: Automatic caching should be a config option. This could be seen as an extension of the simulation metadata system, but now for plots. However plots (and derived data) can be of multiple simulations, so we need to think carefully about the design. Maybe a global plot metadata registry? This could also include automatic loading of cached plots/data if available.

@register_plot("slice")
def slice(
    sim,
    idx,
    field,
    # Shortcuts
    center: Optional[str] = None,
    width: Union[float, tuple, None] = None,
    save: Optional[bool] = None,
    show: Optional[bool] = None,
    path: Optional[str] = None,
    # Full arguments
    data_args: Optional[settings.DataConfig] = None,
    style_args: Optional[settings.StyleImageConfig] = None,
    io_args: Optional[settings.IOConfig] = None,
):
    """
    Plot a slice of a simulation field.
    """
    # Prepare data
    data_args.set_defaults(sim, idx)
    frb = data.slice_frb(sim[idx], field, center=data_args.center, data_args=data_args)
    style_args.set_defaults(frb, field, data_args)
    frb_arr = frb[field[0], field[1]]

    # Plot and save figure
    fig, ax = render.image(frb_arr, style_args=style_args)
    io_args.set_defaults("slice", field)
    
    return output(fig, ax, io_args)

@register_plot("proj")
def proj(
    sim,
    idx,
    field,
    # Shortcuts
    axis: Optional[Union[int, list, tuple, str]] = None,
    center: Optional[str] = None,
    width: Union[float, tuple, None] = None,
    save: Optional[bool] = None,
    show: Optional[bool] = None,
    path: Optional[str] = None,
    density: Optional[bool] = None,
    # Full arguments
    data_args: Optional[settings.DataConfig] = None,
    style_args: Optional[settings.StyleImageConfig] = None,
    io_args: Optional[settings.IOConfig] = None,
):
    """
    Plot a projection of a simulation field.
    """

    # Prepare data
    data_args.set_defaults(sim, idx)
    frb = data.proj_frb(sim[idx], field, center=data_args.center, data_args=data_args)
    style_args.set_defaults(frb, field, data_args)
    frb_arr = frb[field[0], field[1]]

    # Plot and save figure
    fig, ax = render.image(frb_arr, style_args=style_args)
    io_args.set_defaults("proj", field)
    
    return output(fig, ax, io_args)

# TODO: sim,idx syntax is not very elegant, need a way to enable time/redshift sim slicing (could be a method of the sim object that returns dataset, issue is plot functions do need the sim object for other things)
@register_plot("profile")
def profile(
    sim,
    idx,
    bin_fields,
    field,
    # Shortcuts
    n_bins: Union[int, list, None] = None,
    save: Optional[bool] = None,
    show: Optional[bool] = None,
    path: Optional[str] = None,
    # Full arguments
    data_args: Optional[settings.DataConfig] = None,
    style_args: Optional[settings.StyleConfig] = None, # Should be line for 1D and image for 2D
    io_args: Optional[settings.IOConfig] = None,
):
    """
    Plot a profile of a simulation field.
    """

    # Prepare data
    dim = len(bin_fields) if isinstance(bin_fields, list) else 1
    profile = data.profile(sim[idx], bin_fields, field, data_args=data_args)
    profiled_data = profile[0][field].in_units(data_args.unit).v

    # TODO: POSTPROCESSING!!!

    # Plot
    if dim == 1:
        style_args.set_defaults(profile, [*bin_fields,field], data_args) # TODO: look into this
        xbin = profile[0].x.in_units(data_args.x_unit).v
        fig, ax = render.line(xbin, profiled_data, style_args=style_args)
    elif dim == 2:
        style_args.set_defaults(profile, [*bin_fields,field], data_args) # TODO: look into this
        fig, ax = render.image(profiled_data, style_args=style_args)
        pass
    else:
        afLogger.error(f"Profile plotting for {dim}D profiles is not supported.")
        fig, ax = plt.subplots()

    # Output
    io_args.set_defaults("profile", field)
    return output(fig, ax, io_args)

@register_plot("output")
def output(fig, ax, io_args: settings.IOConfig):
    """Handle showing and saving of figures based on IOConfig."""
    if io_args.show:
        plt.show()

    if io_args.save:
        fig.savefig(io_args.path)
        plt.close(fig)

    if io_args.return_fig:
        return (fig, ax)