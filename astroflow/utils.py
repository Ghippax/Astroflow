import numpy as np
import hashlib
import json
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
    elif val is None: # TODO: Check, may break things
        return None
    elif not isinstance(val, Iterable):
        return {"value": float(val), "unit": "dimensionless"}
    return val

def is_particle(ds, field):
    _ = ds.index # Force index creation for metadata access
    if field not in ds.field_info:
        return False
    return True if ds.field_info[field].sampling_type == "particle" else False

def param_hash(params: dict) -> str:
    """Generate stable 8-char hash from parameters."""
    if not params:
        return "default"
    serialized = json.dumps(params, sort_keys=True, default=lambda x: str(x))
    return hashlib.md5(serialized.encode()).hexdigest()[:8]

def getCritRho200(co,z):
    """Calculates 200x the critical density based on cosmology and redshift"""
    return 200 * co.critical_density(z).to("Msun/kpc**3")

def getMeanRho200(co,z):
    """Calculates 200x the mean density based on cosmology and redshift"""
    rho_crit0 = getCritRho200(co,0)
    rho_mean0 = co.omega_matter * rho_crit0
    return rho_mean0 * (1 + z)**3

def getVirRho(co,z):
    """Calculates the density at virial radius based on the factor by Bryan & Norman (1998)"""
    Hz = co.hubble_parameter(z)
    H0 = co.hubble_parameter(0.0)
    Ez = (Hz / H0)
    # Matter fraction at z
    Omega_z = (co.omega_matter * (1 + z)**3 / Ez**2)
    x = Omega_z - 1.0
    # dc fits from the literature:
    # - Flat‐curvature      (OmegaR = 0)     :   dc = 18pi^2 + 82x – 39x^2
    # - No‐Lambda universe  (OmegaLambda = 0):   dc = 18pi^2 + 60x – 32x^2
    Delta_c     = 18*np.pi**2 + 82*x - 39*x**2
    #Delta_c     = 18*np.pi**2 + 60*x - 32*x**2
    return Delta_c*getCritRho200(co,z)/200