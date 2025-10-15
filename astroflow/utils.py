from unyt import unyt_array
import numpy as np

# Return unyt arrays to their original form when loading from metadata
def deserialize_units(d):
    if isinstance(d, dict) and "value" in d and "unit" in d:
        arr = unyt_array(np.array(d["value"]), d["unit"])
        return arr
    # TODO handle scalar unyt quantities
    return d

# Convert unyt arrays to a serializable form for metadata storage
def serialize_units(val):
    if isinstance(val, unyt_array):
        return {"value": val.value.tolist(), "unit": str(val.units)}
    # TODO handle scalar unyt quantities
    return val
