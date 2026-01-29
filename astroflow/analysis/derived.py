from typing import Optional
from .registry import register_derived
from . import settings

import numpy as np
from unyt import unyt_array

from ..log import get_logger
from ..plot import plot, settings
afLogger = get_logger()

# TODO: You should be able to call these functions to register with arbitrary names and parameter combinations (also would have typehints!) and then get_derived is just a query that optionally computes (good internally, but user API should be simpler, so accessing this module is better (faceon = af.analysis.faceon(sim, snap_idx, ..., cache=True, name="my_faceon")). Put logic in derived quantity registry to handle caching and naming!

@register_derived("center_default")
def compute_center_def(sim, snap_idx: int):
    ds = sim[snap_idx]
    center = ds.domain_center.to("Mpc")
    return center

@register_derived("redshift")
def compute_redshift(sim, snap_idx: int):
    ds = sim[snap_idx]
    redshift = ds.current_redshift
    return redshift

# TODO: The see option here should give you a zoom-in plot (like recent Hopkins's BH super-refiniment plots) of the centers up to a bit more than the radius used (and indicate the radius with a circle)
@register_derived("center_com_it",set_config={"center1": None,"iterations":10,"bounds":None,"see": False,"com_kwargs":{"use_gas":False,"use_particles":True}})
def compute_center_it(sim, snap_idx: int, center = None, iterations=None, bounds = None, see = None, com_kwargs={}):
    ds = sim[snap_idx]

    centerTemp = center if center is not None else sim.get_derived("center_default", snap_idx)
    if bounds is None:
        bounds = [ds.domain_width.to("Mpc").v.min(), 0.001] # 1 kpc as heuristic radius with at least 1 particle inside

    sizeSphere = np.logspace(np.log10(bounds[0]),np.log10(bounds[1]),iterations)

    for i in range(iterations):
        afLogger.debug(f"Iteration {i+1}/{iterations}: Computing center of mass within sphere of radius {sizeSphere[i]:.4f} Mpc around {centerTemp.to('Mpc')}")
        sp     = ds.sphere(centerTemp, (sizeSphere[i],"Mpc"))
        if see:
            afLogger.info(f"Iteration {i+1}: Plotting density projection...")
            plot.proj(sim, snap_idx, center=centerTemp, width=(sizeSphere[i],"Mpc"), field=("gas","density"), save = False, show = True, data_args=settings.DataConfig(data_source=sp))
        centerTemp = sp.quantities.center_of_mass(**com_kwargs)
        
    return unyt_array(centerTemp).to("Mpc")

@register_derived("center_max",set_config={"center": None,"radius":5, "see": False, "field":("gas","density")})
def compute_center_max(sim, snap_idx, center = None, radius=5, see = None, field=("gas","density")):
    ds = sim[snap_idx]
    centerTemp = center if center is not None else sim.get_derived("center_default", snap_idx)

    sp = ds.sphere(centerTemp, (radius,"Mpc"))
    _, x, y, z = sp.quantities.max_location(field)

    center = unyt_array([x,y,z],"Mpc").to("Mpc")

    if see:
        afLogger.info(f"Plotting density projection...")
        plot.proj(sim, snap_idx, center=center, width=(radius,"Mpc"), field=("gas","density"), save = False, show = True, data_args=settings.DataConfig(data_source=sp))
    
    return center

# TODO: Maybe make all radius be float with def units or (float,str) with units?
@register_derived("faceon",set_config={"center": "center_default","particle":"all", "gas": True, "use_particle": False, "radius": 10, "temp":1e4})
def faceon(sim, snap_idx, center = None, particle = None, gas = None, radius = None, use_particle = None, temp = None):
    afLogger.debug(f"Calculating face-on axis for snapshot {snap_idx} limited by sphere of radius {radius} kpc with T < {temp} K gas only")

    # Selects a well centered sphere with rvir radius
    ds = sim[snap_idx]
    if isinstance(center, str):
        center = sim.get_derived(center, snap_idx)
    sph = ds.sphere(center,(radius,"kpc"))
    if temp is not None:
        sp = sph.include_below(("gas", "temperature"), temp, "K") #TODO: Fix field
    else:
        sp = sph

    # Get angular momentum vector and normalize
    lmom = sp.quantities.angular_momentum_vector(use_gas=gas, use_particles=use_particle, particle_type=particle)
    face_on = lmom/np.linalg.norm(lmom)

    afLogger.debug(f"Calculated face-on axis {face_on}")

    return face_on

@register_derived("edgeon",set_config={"center": "center_default","particle":"all", "gas": True, "use_particle": False, "radius": 10, "temp":1e4})
def edgeon(sim, snap_idx, center = None, particle = None, gas = None, radius = None, use_particle = None, temp = None):
    afLogger.debug(f"Calculating edge-on axis for snapshot {snap_idx} limited by sphere of radius {radius} kpc with T < {temp} K gas only")
    faceon = sim.get_derived("faceon", snap_idx, center=center, particle=particle, gas=gas, radius=radius, use_particle=use_particle, temp=temp)

    z0 = np.array([0,0,1.0])
    if abs(np.dot(faceon, z0)) > 0.9:
        z0 = np.array([1.0,0,0])

    edge_on = np.cross(faceon, z0)
    edge_on /= np.linalg.norm(edge_on)

    afLogger.debug(f"Calculated edge-on axis {edge_on}")

    return edge_on