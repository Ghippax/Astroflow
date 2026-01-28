#from typing import Tuple, Optional, Any

from typing import Optional, Callable, Union
import yt
from . import settings
from .registry import register_data
from ..log import get_logger
from ..utils import is_particle
from yt.data_objects.static_output import Dataset as YTDataset
from yt.data_objects.profiles import create_profile

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
    is_particle_field = is_particle(ds, field)
    if is_particle_field:
        afLogger.warning(f"Field {field} is a particle field; slice plot cannot be created")

    plot = yt.SlicePlot(ds, data_args.axis, field, center=data_args.center, width=data_args.width, data_source=data_args.data_source)
    frb = plot.data_source.to_frb(data_args.width, data_args.resolution)

    if data_args.unit is not None:
        frb.set_unit(field, data_args.unit)

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
    is_particle_field = is_particle(ds, field)
    
    if is_particle_field:
        plot = yt.ParticleProjectionPlot(ds, data_args.axis, field, center=data_args.center, width=data_args.width, data_source=data_args.data_source, density = data_args.density, deposition = data_args.deposition, depth = data_args.depth, north_vector = data_args.up_vector, weight_field=data_args.weight_field)
    else:
        if isinstance(data_args.axis, (str, int)):
            afLogger.warning(f"North vector is ignored for axis aligned projection plots.")
            plot = yt.ProjectionPlot(ds, data_args.axis, field, center=data_args.center, width=data_args.width, data_source=data_args.data_source, weight_field=data_args.weight_field)
        else:
            plot = yt.ProjectionPlot(ds, data_args.axis, field, center=data_args.center, width=data_args.width, data_source=data_args.data_source, north_vector = data_args.up_vector, weight_field=data_args.weight_field)

    plot.set_buff_size(data_args.resolution)
    frb = plot.frb

    if data_args.unit is not None:
        frb.set_unit(field, data_args.unit)

    return frb

def _map_per_bin(val, bin_fields):
    """
    Helper: convert scalar/list/dict into {bin_field: value} mapping (or None)"""
    if val is None:
        return None
    if isinstance(val, dict):
        return val
    if isinstance(val, list):
        dim = len(bin_fields) if isinstance(bin_fields, list) else 1
        if len(val) != dim:
            raise ValueError(f"Expected {dim} entries for this argument, got {len(val)}")
        if dim == 1:
            return {bin_fields: val[0]}
        return {bf: v for bf, v in zip(bin_fields, val)}
    # scalar -> apply to all bin fields
    return {bf: val for bf in bin_fields}

@register_data("profile")
def profile(
    ds,
    bin_fields,
    field,
    data_args: Optional[settings.DataConfig] = None,
) -> yt.data_objects.profiles.ProfileND:
    """
    Create and return a 1D, 2D or 3D Profile from the given dataset and data source.

    weigh_field: Dict of (ftype, fname) : field

    Returns
    -------
    yt.data_objects.profiles.ProfileND
    """
    if isinstance(ds, YTDataset):
        ds = ds.all_data()

    _ = ds.index # Force index creation for metadata access

    logs = _map_per_bin(data_args.log, bin_fields)
    extrema = _map_per_bin(data_args.bin_extrema, bin_fields)    # (min,max) per bin field or dict
    override_bins = _map_per_bin(data_args.set_bins, bin_fields) # explicit bin edges per field or dict

    dim = len(bin_fields) if isinstance(bin_fields, list) else 1
    axis_unit_vals = [data_args.x_unit, data_args.y_unit, data_args.z_unit][:dim]
    axis_units = _map_per_bin(axis_unit_vals, bin_fields)
        
    field_unit = {field: data_args.unit} if data_args.postprocess is None else {}
    units = (field_unit) | axis_units
    units = {k: v for k, v in units.items() if v is not None}
    if not units:
        units = None

    profile = create_profile(
        ds,
        bin_fields,
        field,
        n_bins=data_args.n_bins,
        weight_field=data_args.weight_field,
        accumulation=data_args.accumulate,
        fractional=data_args.pdf,
        deposition=data_args.deposition,
        logs=logs,
        extrema=extrema,
        override_bins=override_bins,
        units=units,
    )

    return profile