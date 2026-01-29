from pydantic import BaseModel, ConfigDict
from typing import Optional, Union, Any
from unyt import unyt_array, unyt_quantity, Unit
import numpy as np

from datetime import datetime
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import yt

from ..log import get_logger
from ..analysis.registry import postpro_fn

afLogger = get_logger()

"""Validation logic and structure for plotting parameters
"""

class IOConfig(BaseModel):
    save: Optional[bool] = None
    show: Optional[bool] = None
    return_fig: Optional[bool] = None
    path: Optional[str] = None

    def set_defaults(self, plot_type, field):
        if self.path is None and self.save:
            self.path = f"{plot_type}_{field[0]}_{field[1]}_{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.png"

class StyleConfig(BaseModel):
    # General
    figsize: Optional[tuple] = None
    ax: Optional[Axes] = None
    title: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    grid: Optional[bool] = None

    # Image specific
    cmap: Optional[str] = None
    norm: Optional[str] = None
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    colorbar: Optional[bool] = None
    cbar_label: Optional[str] = None
    extent: Union[tuple, list, None] = None
    aspect: Optional[str] = None

    # Line plot specific
    label: Optional[str] = None
    xlog: Optional[bool] = None
    ylog: Optional[bool] = None
    color: Optional[str] = None
    style: Optional[str] = None
    linewidth: Optional[float] = None
    markersize: Optional[float] = None
    alpha: Optional[float] = None
    line_kwargs: Optional[dict] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def set_defaults_frb(self, data_obj, field, data_args):
        if self.xlabel is None or self.ylabel is None:
            # Derive axis-aware labels when aligned with dataset axes
            ds = data_obj.ds
            axis = getattr(data_obj.data_source, "axis", None)
            if axis in (0, 1, 2):
                xname = ds.coordinates.axis_name[ds.coordinates.x_axis[axis]]
                yname = ds.coordinates.axis_name[ds.coordinates.y_axis[axis]]
            else:
                # off-axis or not provided: generic image plane axes
                xname, yname = "x", "y"

            if self.xlabel is None:
                unit = data_args.x_unit if data_args.x_unit is not None else "code_length"
                self.xlabel = f"{xname} ({unit})"
            if self.ylabel is None:
                unit = data_args.y_unit if data_args.y_unit is not None else "code_length"
                self.ylabel = f"{yname} ({unit})"

        if self.extent is None:
            # Parse width to get extent in valid units
            wh_unit = "code_length"
            # single float
            if isinstance(data_args.width, (float, int)): 
                w_val = data_args.width
                h_val = w_val
            elif isinstance(data_args.width, (tuple)):
                # (float, float)
                if isinstance(data_args.width[0], (float, int)) and isinstance(data_args.width[1], (float, int)): 
                    w_val = data_args.width[0]
                    h_val = data_args.width[1]
                # (float, str)
                elif isinstance(data_args.width[0], (float, int)) and isinstance(data_args.width[1], str): 
                    w_val = data_args.width[0]
                    h_val = w_val
                    wh_unit = data_args.width[1]
                # ((float, str), (float, str))
                else: 
                    w_val = data_args.width[0][0]
                    h_val = unyt_quantity(data_args.width[1][0],data_args.width[1][1]).to(data_args.width[0][1]).value
                    wh_unit = data_args.width[0][1]
            xpart = data_obj.ds.arr([-w_val/2, w_val/2],wh_unit).to(data_args.x_unit).v
            ypart = data_obj.ds.arr([-h_val/2, h_val/2],wh_unit).to(data_args.y_unit).v

            self.extent = [xpart[0], xpart[1], ypart[0], ypart[1]]

        if self.cbar_label is None:
            # Figure out field unit
            if data_args.unit is not None:
                unit = f"${Unit(data_args.unit).latex_repr}$"
            else:
                unit = "code_units"

            # Determine field name
            cbar_field = [field[0],field[1]]

            self.cbar_label = f"{cbar_field[0].capitalize()} {cbar_field[1].capitalize()} ({unit})"

    def set_defaults_profile(self, data_obj, field, data_args):
        if self.xlabel is None or self.ylabel is None:
            if len(field) == 2:
                yunit = data_args.unit if data_args.unit is not None else "code_units"
            elif len(field) == 3:
                yunit = data_args.y_unit if data_args.y_unit is not None else "code_units"
            else:
                afLogger.warning("StyleConfig: Unable to set default xlabel/ylabel for profile with more than 2 bin fields.")

            if self.xlabel is None:
                unit = data_args.x_unit if data_args.x_unit is not None else "code_units"
                self.xlabel = f"{field[0][0].capitalize()} {field[0][1].capitalize()} ({unit})"
            if self.ylabel is None:
                unit = yunit
                
                field_label = field[1][1].capitalize() if data_args.postprocess is None else postpro_fn.get_metadata(data_args.postprocess).get("label")
                    
                self.ylabel = f"{field[1][0].capitalize()} {field_label} ({unit})"

        if self.extent is None and len(field) == 3:
            self.extent = [
                data_obj.x.in_units(data_args.x_unit).v[0]  if data_args.x_unit is not None else data_obj.x.v[0],
                data_obj.x.in_units(data_args.x_unit).v[-1] if data_args.x_unit is not None else data_obj.x.v[-1],
                data_obj.y.in_units(data_args.y_unit).v[0]  if data_args.y_unit is not None else data_obj.y.v[0],
                data_obj.y.in_units(data_args.y_unit).v[-1] if data_args.y_unit is not None else data_obj.y.v[-1]]
    
        if self.cbar_label is None:
            # Figure out field unit
            if data_args.unit is not None:
                unit = f"${Unit(data_args.unit).latex_repr}$"
            else:
                unit = "code_units"

            # For profiles, profiled field is passed as last field
            cbar_field = [field[-1][0],field[-1][1]]
            
            self.cbar_label = f"{cbar_field[0].capitalize()} {cbar_field[1].capitalize() if data_args.postprocess is None else postpro_fn.get_metadata(data_args.postprocess).get('label')} ({unit})"


class StyleImageConfig(StyleConfig):
    # Image specific
    cmap: Optional[str] = None
    norm: Optional[str] = None
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    colorbar: Optional[bool] = None
    cbar_label: Optional[str] = None
    extent: Union[tuple, list, None] = None

    def set_defaults(self, data_obj, field, data_args):
        if self.xlabel is None or self.ylabel is None:
            if isinstance(data_obj, yt.visualization.fixed_resolution.FixedResolutionBuffer):
                # Derive axis-aware labels when aligned with dataset axes
                ds = data_obj.ds
                axis = getattr(data_obj.data_source, "axis", None)
                if axis in (0, 1, 2):
                    xname = ds.coordinates.axis_name[ds.coordinates.x_axis[axis]]
                    yname = ds.coordinates.axis_name[ds.coordinates.y_axis[axis]]
                else:
                    # off-axis or not provided: generic image plane axes
                    xname, yname = "x", "y"

                if self.xlabel is None:
                    unit = data_args.x_unit if data_args.x_unit is not None else "code_length"
                    self.xlabel = f"{xname} ({unit})"
                if self.ylabel is None:
                    unit = data_args.y_unit if data_args.y_unit is not None else "code_length"
                    self.ylabel = f"{yname} ({unit})"
                
            elif isinstance(data_obj, yt.data_objects.profiles.ProfileND):
                if len(field) == 2:
                    yunit = data_args.unit if data_args.unit is not None else "code_units"
                elif len(field) == 3:
                    yunit = data_args.y_unit if data_args.y_unit is not None else "code_units"
                else:
                    afLogger.warning("StyleImageConfig: Unable to set default xlabel/ylabel for profile with more than 2 bin fields.")

                if self.xlabel is None:
                    unit = data_args.x_unit if data_args.x_unit is not None else "code_units"
                    self.xlabel = f"{field[0][0].capitalize()} {field[0][1].capitalize()} ({unit})"
                if self.ylabel is None:
                    unit = yunit
                    self.ylabel = f"{field[1][0].capitalize()} {field[1][1].capitalize()} ({unit})"

        if self.extent is None:
            if isinstance(data_obj, yt.visualization.fixed_resolution.FixedResolutionBuffer):
                # Parse width to get extent in valid units
                wh_unit = "code_length"
                if isinstance(data_args.width, (float, int)): # single float
                    w_val = data_args.width
                    h_val = w_val
                elif isinstance(data_args.width, (tuple)):
                    if isinstance(data_args.width[0], (float, int)) and isinstance(data_args.width[1], (float, int)): # (float, float)
                        w_val = data_args.width[0]
                        h_val = data_args.width[1]
                    elif isinstance(data_args.width[0], (float, int)) and isinstance(data_args.width[1], str): # (float, str)
                        w_val = data_args.width[0]
                        h_val = w_val
                        wh_unit = data_args.width[1]
                    else: # ((float, str), (float, str))
                        w_val = data_args.width[0][0]
                        h_val = unyt_quantity(data_args.width[1][0],data_args.width[1][1]).to(data_args.width[0][1]).value
                        wh_unit = data_args.width[0][1]
                
                xpart = data_obj.ds.arr([-w_val/2, w_val/2],wh_unit).to(data_args.x_unit).v
                ypart = data_obj.ds.arr([-h_val/2, h_val/2],wh_unit).to(data_args.y_unit).v

                self.extent = [xpart[0], xpart[1], ypart[0], ypart[1]]
            elif isinstance(data_obj, yt.data_objects.profiles.ProfileND):
                self.extent = [
                    data_obj[0].x.in_units(data_args.x_unit).v[0],
                    data_obj[0].x.in_units(data_args.x_unit).v[-1],
                    data_obj[0].y.in_units(data_args.y_unit).v[0],
                    data_obj[0].y.in_units(data_args.y_unit).v[-1]]
        
        if self.cbar_label is None:
            # Figure out field unit
            if data_args.unit is not None:
                unit = f"${Unit(data_args.unit).latex_repr}$"
            else:
                unit = "code_units"

            # Determine field name
            cbar_field = [field[0],field[1]]
            if isinstance(data_obj, yt.data_objects.profiles.ProfileND):
                # For profiles, profiled field is passed as last field
                cbar_field = [field[-1][0],field[-1][1]]

            self.cbar_label = f"{cbar_field[0].capitalize()} {cbar_field[1].capitalize()} ({unit})"

class StyleLineConfig(StyleConfig):
    # Line plot specific
    label: Optional[str] = None
    xlog: Optional[bool] = None
    ylog: Optional[bool] = None
    color: Optional[str] = None
    style: Optional[str] = None
    linewidth: Optional[float] = None
    markersize: Optional[float] = None
    alpha: Optional[float] = None
    line_kwargs: Optional[dict] = None


class DataConfig(BaseModel):
    center: Union[str, None, unyt_array, tuple, list] = None
    width: Union[float, None, tuple] = None
    axis: Union[int, str, tuple, list, None] = None
    x_unit: Optional[str] = None
    y_unit: Optional[str] = None
    z_unit: Optional[str] = None
    unit: Optional[str] = None
    postprocess: Optional[str] = None

    weight_field: Optional[tuple] = None
    resolution: Optional[int] = None
    data_source: Optional[Any] = None
    density: Optional[bool] = None
    deposition: Optional[str] = None
    depth: Union[float, None, tuple] = None
    up_vector: Union[int, str, tuple, list, None] = None

    n_bins: Union[list, int, None] = None
    accumulate: Union[bool, list, None] = None
    pdf: Optional[bool] = None
    bin_extrema: Union[None, list, tuple] = None
    set_bins: Union[None, list, tuple] = None
    log: Optional[bool] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def set_defaults(self, sim, idx):
        if isinstance(self.center, str):
            self.center = sim.get_derived(self.center, idx)

        if isinstance(self.axis, str) and self.axis not in ["x","y","z"]:
            self.axis = sim.get_derived(self.axis, idx)

        if isinstance(self.up_vector, str) and self.up_vector not in ["x","y","z"]:
            self.up_vector = np.array(sim.get_derived(self.up_vector, idx))
        elif self.up_vector in ["x","y","z"]:
            axis_map = {"x":[1,0,0],"y":[0,1,0],"z":[0,0,1]}
            self.up_vector = axis_map[self.up_vector]
        elif isinstance(self.up_vector, int) :
            axis_map = {0:[1,0,0],1:[0,1,0],2:[0,0,1]}
            self.up_vector = axis_map[self.up_vector]

        if self.width is None:
            self.width = sim[idx].domain_width.to("Mpc").min()
            afLogger.debug(f"DataConfig: No width in config or user-input. Setting default value to domain_width: {self.width}")

        if self.depth is None:
            self.depth = self.width
