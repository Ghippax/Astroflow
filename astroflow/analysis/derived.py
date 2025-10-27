from typing import Optional
from ..core.registry import register_derived
from . import settings


@register_derived("center_default")
def compute_center(sim_obj, snap_id: int, params: Optional[settings.PropParams] = None):
    ds = sim_obj[snap_id]
    # tol = float(params.get("tol", 1e-3))
    center = ds.domain_center.to("Mpc")
    return center


@register_derived("redshift")
def compute_redshift(sim_obj, snap_id: int):
    ds = sim_obj[snap_id]
    redshift = ds.current_redshift
    return redshift
