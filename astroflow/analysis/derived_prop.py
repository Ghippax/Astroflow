from ..core.registries import derived_registry

@derived_registry.register("center_default")
def compute_center(sim_obj, snap_id: int, params: dict):
    ds = sim_obj[snap_id]
    # TODO: On some methods, default config file should be loaded here for defaults
    # tol = float(params.get("tol", 1e-3))
    center = ds.domain_center
    return center

@derived_registry.register("redshift")
def compute_redshift(sim_obj, snap_id: int, params: dict):
    ds = sim_obj[snap_id]
    redshift = ds.current_redshift
    return redshift