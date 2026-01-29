import numpy as np
from unyt import unyt_array, unyt_quantity

from .log import get_logger

from collections.abc import Iterable

afLogger = get_logger()


# Return unyt arrays to their original form when loading from metadata. In case of None unit, unyt correctly returns a dimensionless object
def deserialize_units(d):
    if isinstance(d, dict) and "value" in d and "unit" in d:
        if isinstance(d["value"], (list, tuple, np.ndarray)):
            return unyt_array(np.array(d["value"]), d["unit"])
        elif isinstance(d["value"], (float, int)):
            return unyt_quantity(d["value"], d["unit"])
    return d

# Convert unyt arrays to a serializable form for metadata storage
def serialize_units(val):
    if isinstance(val, unyt_array):
        return {"value": val.value.tolist(), "unit": str(val.units)}
    elif isinstance(val, unyt_quantity):
        return {"value": val.value, "unit": str(val.units)}
    elif not isinstance(val, Iterable):
        return {"value": float(val), "unit": "dimensionless"}
    return val

def is_particle(ds, field):
    _ = ds.index # Force index creation for metadata access
    if field not in ds.field_info:
        return False
    return True if ds.field_info[field].sampling_type == "particle" else False
