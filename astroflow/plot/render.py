from __future__ import annotations
from typing import Optional, Tuple, Union
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from pathlib import Path
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .registry import register_render
from . import settings

from ..log import get_logger

afLogger = get_logger()

"""
Rendering functions: They create intermediate representations (e.g., figures, axes) from data, that plotting functions can use
"""

here = Path(__file__).resolve().parent.parent
plt.style.use(['science', str(here / "default.mplstyle")])

@register_render("image")
def image(
    image: np.ndarray,
    cmap: Optional[str] = None,
    norm: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorbar: Optional[bool] = None,
    figsize: Iterable[int, int] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    style_args: settings.StyleImageConfig = None,
) -> Tuple[Figure, Axes]:
    """
    Wrapper around matplotlib imshow to create a figure and axes from an image array.
    """
    if style_args.ax is not None:
        ax = style_args.ax
        fig = ax.get_figure()
    elif style_args.figsize is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(figsize=style_args.figsize)

    im = ax.imshow(image, cmap=style_args.cmap, norm=style_args.norm, vmin=style_args.vmin, vmax=style_args.vmax, extent=style_args.extent)

    if style_args.colorbar:
        # TODO: improve colorbar placement
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = fig.colorbar(im, cax=cax)
        if style_args.cbar_label:
            cb.set_label(style_args.cbar_label)

    if style_args.xlabel:
        ax.set_xlabel(style_args.xlabel)
    if style_args.ylabel:
        ax.set_ylabel(style_args.ylabel)
    if style_args.title:
        ax.set_title(style_args.title)
    if style_args.grid:
        ax.grid(style_args.grid)

    return fig, ax

@register_render("line")
def line(
    x: np.ndarray,
    y: np.ndarray,
    style_args: settings.StyleLineConfig = None,
) -> Tuple[Figure, Axes]:
    """Render a 1D line plot."""
    if style_args.ax is not None:
        ax = style_args.ax
        fig = ax.get_figure()
    elif style_args.figsize is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(figsize=style_args.figsize)
        
    # Handle log scales defined in style_args
    if style_args.xlog: ax.set_xscale("log")
    if style_args.ylog: ax.set_yscale("log")

    ax.plot(x, y, style_args.style, label=style_args.label, color=style_args.color, linewidth=style_args.linewidth, markersize=style_args.markersize, alpha=style_args.alpha, **(style_args.line_kwargs or {}))

    if style_args.xlabel:
        ax.set_xlabel(style_args.xlabel)
    if style_args.ylabel:
        ax.set_ylabel(style_args.ylabel)
    if style_args.title:
        ax.set_title(style_args.title)
    if style_args.grid:
        ax.grid(style_args.grid)
    
    return fig, ax

@register_render("mosaic")
def mosaic(
    layout: Union[list, tuple, str],
    axes: Union[None, list, tuple] = None,
    **mosaic_args,
) -> Tuple[Figure, Axes]:
    """
    General mosaic plot function that wraps around matplotlib's subplot_mosaic capabilities

    Parameters
    ----------

    Returns
    -------
    fig : Figure
        The matplotlib figure object
    ax_dict : dict
        The dictionary of matplotlib axes objects
    """
    fig, ax_dict = plt.subplot_mosaic(layout, **mosaic_args)

    if axes is not None:
        for key, user_ax in zip(ax_dict.keys(), axes):
            ax_dict[key] = user_ax
    
    return fig, ax_dict