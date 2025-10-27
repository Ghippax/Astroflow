from pydantic import BaseModel, ConfigDict
from typing import Optional, Union
from unyt import unyt_array

"""Validation logic and structure for plotting parameters
"""

class IOParams(BaseModel):
    save: Optional[bool] = None
    show: Optional[bool] = None
    path_to_save: Optional[str] = None

class StyleParams(BaseModel):
    cmap: Optional[str] = None
    norm: Optional[str] = None
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    colorbar: Optional[bool] = True
    figsize: Optional[tuple] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    cbar_label: Optional[str] = None
    extent: Union[tuple, list, None] = None
    title: Optional[str] = None

    def set_defaults(self, frb, field, data_args):
        if self.xlabel is None or self.ylabel is None:
            # Derive axis-aware labels when aligned with dataset axes
            ds = frb.ds
            axis = getattr(frb.data_source, "axis", None)
            if axis in (0, 1, 2):
                xname = ds.coordinates.axis_name[ds.coordinates.x_axis[axis]]
                yname = ds.coordinates.axis_name[ds.coordinates.y_axis[axis]]
            else:
                # off-axis or not provided: generic image plane axes
                xname, yname = "x", "y"

        if self.extent is None:
            self.extent = unyt_array(frb.bounds).to(data_args.xy_units).v
        if self.xlabel is None:
            unit = data_args.xy_units if data_args.xy_units is not None else "code_length"
            self.xlabel = f"{xname} [{unit}]"
        if self.ylabel is None:
            unit = data_args.xy_units if data_args.xy_units is not None else "code_length"
            self.ylabel = f"{yname} [{unit}]"
        if self.cbar_label is None:
            unit = data_args.field_units if data_args.field_units is not None else "code_units"
            self.cbar_label = f"{field[0]} {field[1]} [{unit}]"

class DataParams(BaseModel):
    center: Union[str, None, unyt_array, tuple, list] = None
    width: Union[float, None, tuple] = None
    axis: Union[int, str, tuple, list, None] = None
    resolution: Optional[int] = None
    xy_units: Optional[str] = None
    field_units: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def set_defaults(self, sim, idx):
        if isinstance(self.center, str):
            self.center = sim.get_derived(self.center, idx)
