from __future__ import annotations

from typing import Optional, Union

import matplotlib.pyplot as plt
from .registry import register_plot_fn
from . import data, render, settings

from ..log import get_logger

afLogger = get_logger()

"""
Plotting API: They output/save final plots or animations, generated with plot_data and rendering with plot_render modules
"""

# TODO: Port helper functions, animations, saving, etc

@register_plot_fn("slice")
def slice(
    sim,
    idx,
    field,
    # Shortcuts
    center: Optional[str] = None,
    width: Union[float, tuple, None] = None,
    save: Optional[bool] = None,
    show: Optional[bool] = None,
    path_to_save: Optional[str] = None,
    # Full arguments
    data_args: Optional[settings.DataParams] = None,
    style_args: Optional[settings.StyleParams] = None,
    io_args: Optional[settings.IOParams] = None,
):
    """
    Plot a slice of a simulation field.
    """

    data_args.set_defaults(sim, idx)

    frb = data.slice_frb(sim[idx], field, center=data_args.center, data_args=data_args)

    style_args.set_defaults(frb, field, data_args)

    frb_arr = frb[field[0], field[1]] # Assuming field is a tuple like ('gas', 'density')

    fig, ax = render.image(frb_arr, style_args=style_args)

    if io_args.show:
        plt.show()

    if io_args.save:
        fig.savefig(io_args.path_to_save)
        plt.close(fig)

@register_plot_fn("proj")
def proj(
    sim,
    idx,
    field,
    # Shortcuts
    center: Optional[str] = None,
    width: Union[float, tuple, None] = None,
    save: Optional[bool] = None,
    show: Optional[bool] = None,
    path_to_save: Optional[str] = None,
    # Full arguments
    data_args: Optional[settings.DataParams] = None,
    style_args: Optional[settings.StyleParams] = None,
    io_args: Optional[settings.IOParams] = None,
):
    """
    Plot a projection of a simulation field.
    """

    data_args.set_defaults(sim, idx)

    frb = data.proj_frb(sim[idx], field, center=data_args.center, data_args=data_args)

    style_args.set_defaults(frb, field, data_args)

    frb_arr = frb[field[0], field[1]] # Assuming field is a tuple like ('gas', 'density')
    
    fig, ax = render.image(frb_arr, style_args=style_args)

    if io_args.show:
        plt.show()

    if io_args.save:
        fig.savefig(io_args.path_to_save)
        plt.close(fig)
    