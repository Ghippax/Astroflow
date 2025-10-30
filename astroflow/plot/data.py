#from typing import Tuple, Optional, Any

from typing import Optional
import yt
from . import settings
from .registry import register_data
from ..log import get_logger
from yt.data_objects.static_output import Dataset as YTDataset


afLogger = get_logger()


"""
Functions for getting data from simulations: They can output profiles, frb, np arrays, etc
"""

@register_data("slice_frb")
def slice_frb(
    ds: YTDataset,
    field,
    center = None,
    width = None,
    resolution = None,
    axis = None,
    data_args: Optional[settings.DataConfig] = None,
) -> yt.visualization.fixed_resolution.FixedResolutionBuffer:
    """
    Create and return a FRB from a slice plot of the given dataset.

    Returns
    -------
    yt.visualization.fixed_resolution.FixedResolutionBuffer
    """
    _ = ds.index # Force index creation for metadata access
    plot = yt.SlicePlot(ds, data_args.axis, field, center=data_args.center, width=data_args.width)
    frb = plot.data_source.to_frb(data_args.width, data_args.resolution)

    if data_args.field_units is not None:
        frb.set_unit(field, data_args.field_units)

    return frb

@register_data("proj_frb")
def proj_frb(
    ds: YTDataset,
    field,
    center = None,
    width = None,
    resolution = None,
    axis = None,
    data_args: Optional[settings.DataConfig] = None,
) -> yt.visualization.fixed_resolution.FixedResolutionBuffer:
    """
    Create and return a FRB from a projection plot of the given dataset.

    Returns
    -------
    yt.visualization.fixed_resolution.FixedResolutionBuffer
    """
    _ = ds.index # Force index creation for metadata access
    plot = yt.ProjectionPlot(ds, data_args.axis, field, center=data_args.center, width=data_args.width)
    frb = plot.data_source.to_frb(data_args.width, data_args.resolution)

    if data_args.field_units is not None:
        frb.set_unit(field, data_args.field_units)

    return frb
