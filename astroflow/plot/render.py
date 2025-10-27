from __future__ import annotations
from typing import Optional, Tuple
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from .registry import register_render_fn
from . import settings

from ..log import get_logger

afLogger = get_logger()

"""
Rendering functions: They create intermediate representations (e.g., figures, axes) from data, that plotting functions can use
"""


@register_render_fn("image")
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
    style_args: settings.StyleParams = None,
) -> Tuple[Figure, Axes]:
    """
    Wrapper around matplotlib imshow to create a figure and axes from an image array.
    """
    if style_args.figsize is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(figsize=style_args.figsize)
    im = ax.imshow(image, cmap=style_args.cmap, norm=style_args.norm, vmin=style_args.vmin, vmax=style_args.vmax, extent=style_args.extent)

    if style_args.colorbar:
        fig.colorbar(im, ax=ax)
        if style_args.cbar_label:
            im.colorbar.set_label(style_args.cbar_label)
    if style_args.xlabel:
        ax.set_xlabel(style_args.xlabel)
    if style_args.ylabel:
        ax.set_ylabel(style_args.ylabel)
    if style_args.title:
        ax.set_title(style_args.title)

    return fig, ax