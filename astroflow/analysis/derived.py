from typing import Optional
from .registry import register_derived, register_alias
from . import settings

import numpy as np
import yt
import unyt
from unyt import unyt_array
import h5py
import matplotlib.pyplot as plt

from scipy.interpolate import UnivariateSpline
from .registry import postpro_fn
from ..log import get_logger
from ..plot import plot, settings, data
from ..utils import getCritRho200, getMeanRho200, getVirRho
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

@register_derived("scale_factor")
def compute_scale_factor(sim, snap_idx: int):
    ds = sim[snap_idx]
    redshift = ds.current_redshift
    return 1.0 / (1.0 + redshift)

@register_derived("time")
def compute_time(sim, snap_idx: int):
    ds = sim[snap_idx]
    time = ds.current_time.to("Gyr")
    return time

# TODO: The see option here should give you a zoom-in plot (like recent Hopkins's BH super-refiniment plots) of the centers up to a bit more than the radius used (and indicate the radius with a circle)
@register_derived("center_com_it",set_config={"center": None,"iterations":10,"bounds":None,"see": False,"com_kwargs":{"use_gas":False,"use_particles":True}})
def compute_center_it(sim, snap_idx: int, center = None, iterations=None, bounds = None, see = None, com_kwargs={}):
    ds = sim[snap_idx]

    centerTemp = center if center is not None else sim.get_derived("center_default", snap_idx)
    if bounds is None:
        bounds = [ds.domain_width.to("Mpc").v.min(), 0.001] # 1 kpc as heuristic radius with at least 1 particle inside

    sizeSphere = np.logspace(np.log10(bounds[0]),np.log10(bounds[1]),iterations)

    for i in range(iterations):
        afLogger.info(f"Iteration {i+1}/{iterations}: Computing center of mass within sphere of radius {sizeSphere[i]:.4f} Mpc around {centerTemp.to('Mpc')}")
        sp     = ds.sphere(centerTemp, (sizeSphere[i],"Mpc"))
        if see:
            afLogger.info(f"Iteration {i+1}: Plotting density projection...")
            plot.proj(sim.snap[snap_idx], center=centerTemp, width=(sizeSphere[i],"Mpc"), field=("gas","density"), save = False, show = True)
        centerTemp = sp.quantities.center_of_mass(**com_kwargs)
        
    return unyt_array(centerTemp).to("Mpc")

@register_derived("bulk_v",set_config={"center": "center_default","radius":10,"bv_kwargs":{"use_gas":True,"use_particles":False}})
def bulk_velocity(sim, snap_idx, center = None, radius=None, bv_kwargs={}):
    ds = sim[snap_idx]
    if isinstance(center, str):
        center = sim.get_derived(center, snap_idx)

    sp = ds.sphere(center, (radius,"kpc"))
    bv = sp.quantities.bulk_velocity(**bv_kwargs)
        
    return bv.to("km/s")

@register_derived("bulk_v_weighted",set_config={"center": "center_default","radius":10,"particle":"PartType0", "weight_field": "HI_mass"})
def bulk_velocity_weighted(sim, snap_idx, center = None, radius=None, particle = None, weight_field = None):
    ds = sim[snap_idx]
    if isinstance(center, str):
        center = sim.get_derived(center, snap_idx)

    sp = ds.sphere(center, (radius,"kpc"))
    bv = np.array([sp.quantities.weighted_average_quantity((particle,f"particle_velocity_{ax}"), weight = (particle, weight_field)).to("km/s").to_value() for ax in ['x', 'y', 'z']])
        
    return bv*unyt.km/unyt.s

@register_derived("center_max",set_config={"center": None,"radius":5, "see": False, "field":("gas","density")})
def compute_center_max(sim, snap_idx, center = None, radius=5, see = None, field=("gas","density")):
    ds = sim[snap_idx]
    centerTemp = center if center is not None else sim.get_derived("center_default", snap_idx)

    sp = ds.sphere(centerTemp, (radius,"Mpc"))
    _, x, y, z = sp.quantities.max_location(field)

    center = unyt_array([x,y,z],"Mpc").to("Mpc")

    if see:
        afLogger.info("Plotting density projection...")
        plot.proj(sim.snap[snap_idx], center=center, width=(radius,"Mpc"), field=("gas","density"), save = False, show = True)
    
    return center

# TODO: Maybe make all radius be float with def units or (float,str) with units?
@register_derived("faceon",set_config={"center": "center_default","particle":"all", "gas": True, "use_particle": False, "radius": 10, "temp":1e4})
def faceon(sim, snap_idx, center = None, particle = None, gas = None, radius = None, use_particle = None, temp = None):
    afLogger.info(f"Calculating face-on axis for snapshot {snap_idx} limited by sphere of radius {radius} kpc with T < {temp} K gas only")

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

    return face_on.v.tolist()

@register_derived("faceon_weighted",set_config={"center": "center_default","particle":"PartType0", "weight":"HI_mass", "radius": 10})
def faceon_weighted(sim, snap_idx, center = None, particle = None, radius = None, weight = None):
    afLogger.info(f"Calculating face-on axis for snapshot {snap_idx} limited by sphere of radius {radius} kpc using weighted angular momentum with weight {weight}")
    # Selects a well centered sphere with rvir radius
    ds = sim[snap_idx]
    if isinstance(center, str):
        center = sim.get_derived(center, snap_idx)
    sp = ds.sphere(center,(radius,"kpc"))

    # Get angular momentum vector and normalize
    lmom = np.array([sp.quantities.weighted_average_quantity((particle, f"specific_angular_momentum_{ax}"), (particle, weight)).to_value() for ax in ['x', 'y', 'z']])
    face_on = lmom/np.linalg.norm(lmom)

    afLogger.debug(f"Calculated face-on axis {face_on}")

    return face_on.tolist()

@register_derived("edgeon_weighted",set_config={"center": "center_default","particle":"PartType0", "weight":"HI_mass", "radius": 10})
def edgeon_weighted(sim, snap_idx, center = None, particle = None, radius = None, weight = None):
    afLogger.info(f"Calculating edge-on axis for snapshot {snap_idx} limited by sphere of radius {radius} kpc using weighted angular momentum with weight {weight}")
    faceon = sim.get_derived("faceon_weighted", snap_idx, center=center, particle=particle, radius=radius, weight=weight)

    z0 = np.array([0,0,1.0])
    if abs(np.dot(faceon, z0)) > 0.9:
        z0 = np.array([1.0,0,0])

    edge_on = np.cross(faceon, z0)
    edge_on /= np.linalg.norm(edge_on)

    afLogger.debug(f"Calculated edge-on axis {edge_on}")

    return edge_on.tolist()

@register_derived("ang_mom_weighted",set_config={"center": "center_default","particle":"PartType0", "weight":"HI_mass", "radius": 10})
def ang_mom_weighted(sim, snap_idx, center = None, particle = None, radius = None, weight = None):
    afLogger.info(f"Calculating angular momentum for snapshot {snap_idx} limited by sphere of radius {radius} kpc using weighted angular momentum with weight {weight}")
    ds = sim[snap_idx]
    if isinstance(center, str):
        center = sim.get_derived(center, snap_idx)
    sp = ds.sphere(center,(radius,"kpc"))

    # Get angular momentum vector and normalize
    offset = "particle_" if particle in ["PartType4","PartType1"] else ""
    lmom = unyt_array([sp.quantities.weighted_average_quantity((particle, f"{offset}specific_angular_momentum_{ax}"), (particle, weight)) for ax in ['x', 'y', 'z']])

    afLogger.debug(f"Calculated specific angular momentum {lmom}")

    return lmom.to("km**2/s")

@register_derived("ang_mom",set_config={"center": "center_default","particle":"all", "gas": True, "use_particle": False, "radius": 10, "temp":None})
def ang_mom(sim, snap_idx, center = None, particle = None, gas = None, radius = None, use_particle = None, temp = None):
    afLogger.info(f"Calculating angular momentum for snapshot {snap_idx} limited by sphere of radius {radius} kpc with T < {temp} K gas only")

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
    afLogger.debug(f"Calculated angular momentum {lmom}")

    return lmom.to("km**2/s")

@register_derived("edgeon",set_config={"center": "center_default","particle":"all", "gas": True, "use_particle": False, "radius": 10, "temp":1e4})
def edgeon(sim, snap_idx, center = None, particle = None, gas = None, radius = None, use_particle = None, temp = None):
    afLogger.info(f"Calculating edge-on axis for snapshot {snap_idx} limited by sphere of radius {radius} kpc with T < {temp} K gas only")
    faceon = sim.get_derived("faceon", snap_idx, center=center, particle=particle, gas=gas, radius=radius, use_particle=use_particle, temp=temp)

    z0 = np.array([0,0,1.0])
    if abs(np.dot(faceon, z0)) > 0.9:
        z0 = np.array([1.0,0,0])

    edge_on = np.cross(faceon, z0)
    edge_on /= np.linalg.norm(edge_on)

    afLogger.debug(f"Calculated edge-on axis {edge_on}")

    return edge_on.tolist()

@register_derived("virial_radius", set_config={"method": "crit", "center": "center_default", "radius": 500, "cosmology": [0.702,0.272,0.728,0.0]})
def virial_radius(sim, snap_idx, method = None, center = None, radius = None, cosmology = None):
    # Setup center, cosmology and method
    if isinstance(center, str):
        center = sim.get_derived(center, snap_idx)
    co = yt.utilities.cosmology.Cosmology(hubble_constant=cosmology[0], omega_matter=cosmology[1],omega_lambda=cosmology[2], omega_curvature=cosmology[3])
    methodDict = {"crit":getCritRho200,"mean":getMeanRho200,"vir":getVirRho}
    snapshot   = sim[snap_idx]
    afLogger.info(f"Calculating virial radius for snapshot {snap_idx} using method {method} and limited by {radius} kpc")

    # Calculate target density and get sphere
    targetDen  = float(methodDict[method](co,sim.get_derived("redshift", snap_idx)))
    sp         = snapshot.sphere(center,(radius,"kpc"))
    allMass    = sp[("all","particle_mass")].in_units("Msun")
    allR       = sp[("all","particle_position_spherical_radius")].in_units("kpc")
    idx   = np.argsort(allR)
    mSort = np.array(allMass)[idx]
    rSort = np.array(allR)[idx]
    cumM  = np.cumsum(mSort)
    denR  = cumM/(4/3*np.pi*rSort**3) 
    # TODO: Time this and optimize: binary search? yt histogram and interpolation? both? etc

    # Find radius where density matches target density
    idxAtVir = np.argmin(np.abs(denR-targetDen))

    if (denR[idxAtVir]-targetDen)/targetDen > 0.1:
        afLogger.warning(f"Virial radius determination may be inaccurate: density at r_vir is {denR[idxAtVir]:.3E} Msun/kpc^3 vs target {targetDen:.3E} Msun/kpc^3")
    afLogger.debug(f"Found rvir: {rSort[idxAtVir]:.3f} kpc, enclosing {cumM[idxAtVir]:.3E} Msun, with predicted {targetDen*(4/3*np.pi*rSort[idxAtVir]**3):.3E} Msun")

    return rSort[idxAtVir]*unyt.kpc

register_alias("virial_radius_crit", "virial_radius", method="crit")
register_alias("virial_radius_mean", "virial_radius", method="mean")
register_alias("virial_radius_BN", "virial_radius", method="vir")

def total_in_obj(data, field):
    total = data.quantities.total_quantity(field)
    return total

@register_derived("total_in_sphere", set_config={"center": "center_default", "radius": 10, "field": ("all","particle_ones")})
def total_in_sphere(sim, snap_idx, center = None, radius = None, field = None):
    ds = sim[snap_idx]
    if isinstance(center, str):
        center = sim.get_derived(center, snap_idx)
    if isinstance(radius, str):
        radius = sim.get_derived(radius, snap_idx, center=center).to("kpc").to_value()
    sp = ds.sphere(center, (radius,"kpc"))
    return total_in_obj(sp, field)

@register_derived("half_mass_radius", set_config={"center": "center_default", "particle": "PartType4", "radius": None, "mass_field":"Masses"})
def half_radius(sim, snap_idx, center = None, particle = None, radius = None, mass_field = None):
    # Setup sphere and calculate target half mass
    ds = sim[snap_idx]
    if isinstance(center, str):
        center = sim.get_derived(center, snap_idx)
    if radius is None:
        radius = sim.get_derived("virial_radius", snap_idx, center=center).to_value()
    sp = ds.sphere(center, (radius,"kpc"))
    afLogger.debug(f"Calculating half-mass radius for snapshot {snap_idx} using particle type {particle} within radius {radius}")
    total_mass = total_in_obj(sp, (particle, mass_field)).to("Msun").v
    half_mass = total_mass / 2.0

    allMass    = sp[(particle, mass_field)].in_units("Msun")
    allR       = sp[(particle,"particle_position_spherical_radius")].to("kpc")
    idx   = np.argsort(allR)
    mSort = np.array(allMass)[idx]
    rSort = np.array(allR)[idx]
    cumM  = np.cumsum(mSort)

    idxAtHalf = np.argmin(np.abs(cumM - half_mass))
    if (cumM[idxAtHalf]-half_mass)/half_mass > 0.1:
        afLogger.warning(f"Half-mass radius determination may be inaccurate: enclosed mass at r_half is {cumM[idxAtHalf]:.3E} Msun vs target {half_mass:.3E} Msun")

    afLogger.debug(f"Found half-mass radius: {rSort[idxAtHalf]:.3f} kpc, enclosing {cumM[idxAtHalf]:.3E} Msun, with target {half_mass:.3E} Msun")
    return rSort[idxAtHalf]*unyt.kpc

@register_derived("percent_mass_radius", set_config={"center": "center_default", "particle": "PartType4", "radius": None, "percent": 0.9, "mass_field":"Masses"})
def percent_radius(sim, snap_idx, center = None, particle = None, radius = None, percent = None, mass_field = None):
    # Setup sphere and calculate target half mass
    ds = sim[snap_idx]
    if isinstance(center, str):
        center = sim.get_derived(center, snap_idx)
    if radius is None:
        radius = sim.get_derived("virial_radius", snap_idx, center=center).to_value()
    sp = ds.sphere(center, (radius,"kpc"))
    afLogger.debug(f"Calculating percent-mass radius for snapshot {snap_idx} using particle type {particle} within radius {radius}")
    total_mass = total_in_obj(sp, (particle, mass_field)).to("Msun").v
    part_mass = total_mass * percent

    allMass    = sp[(particle, mass_field)].in_units("Msun")
    allR       = sp[(particle,"particle_position_spherical_radius")].to("kpc")
    idx   = np.argsort(allR)
    mSort = np.array(allMass)[idx]
    rSort = np.array(allR)[idx]
    cumM  = np.cumsum(mSort)

    idxAtHalf = np.argmin(np.abs(cumM - part_mass))
    if (cumM[idxAtHalf]-part_mass)/part_mass > 0.1:
        afLogger.warning(f"Percent-mass radius determination may be inaccurate: enclosed mass at r_{percent*100:.0f} is {cumM[idxAtHalf]:.3E} Msun vs target {part_mass:.3E} Msun")

    afLogger.info(f"Found percent-mass radius: {rSort[idxAtHalf]:.3f} kpc, enclosing {cumM[idxAtHalf]:.3E} Msun, with target {part_mass:.3E} Msun")
    return rSort[idxAtHalf]*unyt.kpc

@register_derived("percent_mass_height", set_config={"center": "center_default", "particle": "PartType4", "radius": None, "percent": 0.5, "mass_field":"Masses", "axis": "faceon", "height": None})
def percent_height(sim, snap_idx, center = None, particle = None, radius = None, percent = None, mass_field = None, axis = None, height = None):
    # Setup disk and calculate target half mass
    ds = sim[snap_idx]
    if isinstance(center, str):
        center = sim.get_derived(center, snap_idx)
    if radius is None:
        radius = sim.get_derived("virial_radius", snap_idx, center=center).to_value()
    if height is None:
        height = sim.get_derived("virial_radius", snap_idx, center=center).to_value()
    if axis is None:
        axis = sim.get_derived("faceon", snap_idx, center=center, particle=particle, radius=radius, use_particle=False, temp=None)  
        
    sp = ds.disk(center, axis, (radius,"kpc"), (height, "kpc"))
    afLogger.debug(f"Calculating percent-mass height for snapshot {snap_idx} using particle type {particle} with disk with radius {radius} and within height {height}")
    total_mass = total_in_obj(sp, (particle, mass_field)).to("Msun").v
    part_mass = total_mass * percent

    allMass    = sp[(particle, mass_field)].in_units("Msun")
    allZ       = np.abs(sp[(particle,"particle_position_cylindrical_z")].to("kpc").v)
    idx   = np.argsort(allZ)
    mSort = np.array(allMass)[idx]
    zSort = np.array(allZ)[idx]
    cumM  = np.cumsum(mSort)

    idxAtHalf = np.argmin(np.abs(cumM - part_mass))
    if (cumM[idxAtHalf]-part_mass)/part_mass > 0.1:
        afLogger.warning(f"Percent-mass height determination may be inaccurate: enclosed mass at z_{percent*100:.0f} is {cumM[idxAtHalf]:.3E} Msun vs target {part_mass:.3E} Msun, difference of {(cumM[idxAtHalf]-part_mass)/part_mass:.2%}")

    afLogger.info(f"Found percent-mass height: {zSort[idxAtHalf]:.3f} kpc, enclosing {cumM[idxAtHalf]:.3E} Msun, with target {part_mass:.3E} Msun")
    return zSort[idxAtHalf]*unyt.kpc

register_alias("height_e", "percent_mass_height", percent = 0.5)

register_alias("radius_e_star", "half_mass_radius", particle="PartType4")
register_alias("radius_e_gas", "half_mass_radius", particle="PartType0")
register_alias("radius_e_dm", "half_mass_radius", particle="PartType1")

@register_derived("total_mass", set_config={"particle": "all", "mass_field": "Masses"})
def total_mass(sim, snap_idx, particle=None, mass_field=None):
    ds = sim[snap_idx].all_data()
    return total_in_obj(ds, (particle, mass_field)).to("Msun")

register_alias("total_mass_gas", "total_mass", particle="PartType0")
register_alias("total_mass_stars", "total_mass", particle="PartType4")
register_alias("total_mass_dm", "total_mass", particle="PartType1")
register_alias("total_mass_all", "total_mass", particle="all")

@register_derived("mass_in_sphere", set_config={"center": "center_default", "radius": 10, "particle": "all", "mass_field": "Masses"})
def mass_in_sphere(sim, snap_idx, center = None, radius = None, particle=None, mass_field=None):
    ds = sim[snap_idx]
    if isinstance(center, str):
        center = sim.get_derived(center, snap_idx)
    if isinstance(radius, str):
        radius = sim.get_derived(radius, snap_idx, center=center).to("kpc").to_value()
    sp = ds.sphere(center, (radius,"kpc"))
    return total_in_obj(sp, (particle, mass_field)).to("Msun")

@register_derived("mass_in_los", set_config={"center": "center_default", "radius": 10, "particle": "all", "mass_field": "Masses", "height": 10, "axis": [0,0,1]})
def mass_in_los(sim, snap_idx, center = None, radius = None, particle=None, mass_field=None, height = None, axis = None):
    ds = sim[snap_idx]
    if isinstance(center, str):
        center = sim.get_derived(center, snap_idx)
    if isinstance(radius, str):
        radius = sim.get_derived(radius, snap_idx, center=center).to("kpc").to_value()
    sp = ds.disk(center, axis, (radius,"kpc"), (height,"kpc"))
    return total_in_obj(sp, (particle, mass_field)).to("Msun")

register_alias("mass_200", "mass_in_sphere", radius="virial_radius")
register_alias("mass_re", "mass_in_sphere", radius="half_mass_radius")

@register_derived("v_max", set_config={"center": "center_default", "radius": 10, "particle": "all","bins":40})
def v_max(sim, snap_idx, center = None, radius = None, particle=None, bins=None):
    ds = sim[snap_idx]
    if isinstance(center, str):
        center = sim.get_derived(center, snap_idx)
    if isinstance(radius, str):
        radius = sim.get_derived(radius, snap_idx, center=center).to("kpc").to_value()

    sp = ds.sphere(center, (radius,"kpc"))

    profile = data.profile(sp, (particle,"particle_position_spherical_radius"), (particle,"Masses"), data_args=settings.DataConfig(n_bins=bins,x_unit="kpc",unit="Msun", bin_extrema=[(0.01,10)], log = True, accumulate=True))
    vcirc = (postpro_fn.get("circ_velocity")(profile,(particle,"Masses"))).in_units("km/s")

    return np.max(vcirc)
    
@register_derived("v_fid", set_config={"center": "center_default", "radius": 2, "particle": "all"})
def v_fid(sim, snap_idx, center = None, radius = None, particle=None):
    ds = sim[snap_idx]
    if isinstance(center, str):
        center = sim.get_derived(center, snap_idx)
    if isinstance(radius, str):
        radius = sim.get_derived(radius, snap_idx, center=center).to("kpc").to_value()

    sp = ds.sphere(center, (radius,"kpc"))

    mass = total_in_obj(sp, (particle,"Masses")).to("Msun")
    return np.sqrt( unyt.G * mass / (radius * unyt.kpc) ).to("km/s")

@register_derived("v_phi", set_config={"center": "center_default", "radius": 2, "particle":"PartType0","bins":40,"axis":"faceon", "bulk_v":"bulk_v", "temp":None, "mass_field":"Masses"})
def v_phi(sim, snap_idx, center = None, radius = None, particle=None, bins=None, axis=None, bulk_v=None, temp=None, mass_field=None):
    ds = sim[snap_idx]
    if isinstance(center, str):
        center = sim.get_derived(center, snap_idx)
    if isinstance(radius, str):
        radius = sim.get_derived(radius, snap_idx, center=center).to("kpc").to_value()
    if isinstance(axis, str):
        axis = sim.get_derived(axis, snap_idx, center=center)
    if isinstance(bulk_v, str):
        bulk_v = sim.get_derived(bulk_v, snap_idx, center=center, radius=radius)

    sp = ds.sphere(center, (radius,"kpc"))
    if temp is not None:
        sp.include_below((particle, "temperature"), temp, "K")
    sp.set_field_parameter("normal", axis)
    sp.set_field_parameter("bulk_velocity", bulk_v)
    # TODO: Fix 0.2 (should be few times epsilon)
    field = (particle,"particle_velocity_cylindrical_theta")
    profile = data.profile(sp, (particle,"particle_position_cylindrical_radius"), field, data_args=settings.DataConfig(n_bins=bins,x_unit="kpc",unit="km/s", bin_extrema=[(0.2,radius)], log = True, accumulate=False, weight_field=(particle,mass_field)))

    return profile[field].in_units("km/s")[-1]

@register_derived("v_disp", set_config={"center": "center_default", "radius": 2, "particle":"PartType0","bins":40,"axis":"faceon", "bulk_v":"bulk_v", "temp":None, "mass_field":"Masses"})
def v_disp(sim, snap_idx, center = None, radius = None, particle=None, bins=None, axis=None, bulk_v=None, temp=None, mass_field=None):
    ds = sim[snap_idx]
    if isinstance(center, str):
        center = sim.get_derived(center, snap_idx)
    if isinstance(radius, str):
        radius = sim.get_derived(radius, snap_idx, center=center).to("kpc").to_value()
    if isinstance(axis, str):
        axis = sim.get_derived(axis, snap_idx, center=center)
    if isinstance(bulk_v, str):
        bulk_v = sim.get_derived(bulk_v, snap_idx, center=center, radius=radius)

    sp = ds.sphere(center, (radius,"kpc"))
    if temp is not None:
        sp.include_below((particle, "temperature"), temp, "K")
    sp.set_field_parameter("normal", axis)
    sp.set_field_parameter("bulk_velocity", bulk_v)
    # TODO: Fix 0.2 (should be few times epsilon)
    field = (particle,"particle_velocity_cylindrical_theta")
    profile = data.profile(sp, (particle,"particle_position_cylindrical_radius"), field, data_args=settings.DataConfig(n_bins=bins,x_unit="kpc",unit="km/s", bin_extrema=[(0.2,radius)], log = True, accumulate=False, weight_field=(particle,mass_field)))
    return profile.standard_deviation[field].in_units("km/s")[-1]

@register_derived("min_radius", set_config={"center": "center_default", "particle":"PartType1", "N":1000, "tol":1})
def min_particle_radius(sim, snap_idx, N=None, center=None, particle=None, tol=None):
    """
    Find radius (kpc) where number of `particle` enclosed is ~ N within `tol`.
    Uses doubling to find an upper bound and then binary search.
    Returns a radius in kpc.
    """
    ds = sim[snap_idx]
    if isinstance(center, str):
        center = sim.get_derived(center, snap_idx)
    def _count_at(r_kpc):
        sp = ds.sphere(center, (r_kpc, "kpc"))
        return sp[(particle, "particle_ones")].sum().to_value()

    # initial bounds (kpc)
    lo = 0.0
    hi = 1.0 
    # expand hi until we enclose >= N or hit a sane cap
    max_cap = 1e4  # kpc 
    iters = 0
    while True:
        cnt = _count_at(hi)
        afLogger.debug(f"Counting particles within {hi:.2f} kpc: {cnt} found, target {N}")
        if cnt >= N or hi >= max_cap or iters > 60:
            break
        hi *= 2.0
        iters += 1
    # if even at max cap we don't reach N, return hi
    if cnt < N and hi >= max_cap:
        return hi*unyt.kpc

    # binary search until count within tolerance or radius precision reached
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        cnt_mid = _count_at(mid)
        afLogger.debug(f"Binary search: lo={lo:.4f} kpc hi={hi:.4f} kpc, mid={mid:.4f} kpc (cnt={cnt_mid}), target {N}, tol {tol}")
        if abs(cnt_mid - N) <= tol:
            return mid*unyt.kpc
        if cnt_mid < N:
            lo = mid
        else:
            hi = mid
        # stop if radius change is negligible (< pc level)
        if (hi - lo) < 1e-4:
            break

    # final best estimate
    return 0.5 * (lo + hi) * unyt.kpc


@register_derived("cuspyness", set_config={"center": "center_default", "radius": 2, "particle":"PartType1","bins":30})
def cuspyness(sim, snap_idx, center = None, radius = None, particle=None, bins=None):
    ds = sim[snap_idx]
    if isinstance(center, str):
        center = sim.get_derived(center, snap_idx)
    if isinstance(radius, str):
        radius = sim.get_derived(radius, snap_idx, center=center).to("kpc").to_value()

    sp = ds.sphere(center, (2*radius,"kpc"))
    
    min_rad = sim.get_derived("min_radius", snap_idx, center=center, particle="all", N=1000, tol=10).to_value() # kpc

    field = (particle,"Masses")
    profile = data.profile(sp, (particle,"particle_position_spherical_radius"), field, data_args=settings.DataConfig(n_bins=bins,x_unit="kpc",unit="Msun/kpc**3", bin_extrema=[(min(min_rad,radius - radius*0.5),radius + radius*0.5)], log = False, accumulate=False, postprocess="spherical_shell"))

    rho = postpro_fn.get("spherical_shell")(profile,field).in_units("Msun/kpc**3").v
    r = profile.x.in_units("kpc").v
    valid = (rho > 0) & (r > 0)
    log_rho = np.log10(rho[valid])
    log_r = np.log10(r[valid])
    log_fid = np.log10(radius)
    
    if len(log_r) < 3:
        raise ValueError("Insufficient valid bins for derivative calculation")
    
    # Fit spline in log-log space
    spline = UnivariateSpline(log_r, log_rho, k = 3)
    slope = float(spline.derivative()(log_fid))

    return slope

def adaptive_bin_merging(edges, mid, counts, min_particles=50):
    """Generalized agglomerative binning for any 1D coordinate."""
    mid_adapt, edges_adapt, counts_adapt = [], [edges[0]], []
    curr_cnt, curr_sum = 0, 0.0
    
    for i in range(len(mid)):
        curr_cnt += counts[i]
        curr_sum += mid[i] * counts[i]
        if curr_cnt >= min_particles or i == len(mid) - 1:
            mid_adapt.append(curr_sum / curr_cnt if curr_cnt > 0 else mid[i])
            edges_adapt.append(edges[i+1])
            counts_adapt.append(curr_cnt)
            curr_cnt, curr_sum = 0, 0.0
    # Warn if min particle numbers is not reached 
    if counts_adapt[-1] < min_particles:
        afLogger.warning(f"Outermost bin is starved! Out of desired {min_particles}, only {counts_adapt[-1]} were reached")
        afLogger.warning(f"Full particle count per bin is: {counts_adapt}")
    return np.array(edges_adapt), np.array(mid_adapt), np.array(counts_adapt)

def build_bins(sp, ptype, field, nbins, extrema = None, log_x = False, min_particles = 50, unit = None):
    profile = data.profile(sp, (ptype,field), (ptype,"particle_ones"), data_args=settings.DataConfig(n_bins=nbins, bin_extrema=extrema, log = log_x, accumulate=False, weight_field=None, x_unit = unit))
    amount_of_p = profile[(ptype,"particle_ones")].v
    edge, mid, _ = adaptive_bin_merging(profile.x_bins.v, profile.x.v, amount_of_p, min_particles = min_particles)
    return {"set_bins": [edge], "x_data": unyt_array(mid, profile.x.units)}

@register_derived("R_den", set_config={"center": "center_default", "radius": "virial_radius", "density_thresh": 1e1, "particle":"PartType0","bins":40,"axis":"faceon","min_N_radius":1000, "mass_field":"Masses", "adaptive": True, "min_particles": 75})
def R_den(sim, snap_idx, center = None, density_thresh = None, radius = None, particle=None, bins=None, axis=None, min_N_radius=None, mass_field = None, adaptive = None, min_particles = None):
    ds = sim[snap_idx]
    if isinstance(center, str):
        center = sim.get_derived(center, snap_idx)
    if isinstance(radius, str):
        radius = sim.get_derived(radius, snap_idx, center=center).to("kpc").to_value()
    if isinstance(axis, str):
        axis = sim.get_derived(axis, snap_idx, center=center)

    sp = ds.sphere(center, (radius,"kpc"))
    sp.set_field_parameter("normal", axis)

    min_rad = sim.get_derived("min_radius", snap_idx, center=center, particle="all", N=min_N_radius, tol=50).to_value() # kpc

    extra_kwargs = {}
    if adaptive:
        extra_kwargs = build_bins(sp, particle, "particle_position_cylindrical_radius", bins, extrema=[(min_rad,radius)], log_x=True, min_particles=min_particles, unit="kpc")

    profile = data.profile(sp, bin_fields=(particle,"particle_position_cylindrical_radius"), field=(particle,mass_field), data_args=settings.DataConfig(n_bins=bins,x_unit="kpc",unit="Msun/pc**2",  bin_extrema=[(min_rad,radius)], accumulate=False, postprocess="circular_surface", log = True, **extra_kwargs))
    
    sigma = postpro_fn.get("circular_surface")(profile,(particle,mass_field)).in_units("Msun/pc**2").v
    r = profile.x.in_units("kpc").v

    valid = (sigma > 0) & (r > 0)
    sigma_v = sigma[valid]
    r_v = r[valid]
    # Fit a spline
    if len(r_v) < 4:
        afLogger.warning("Not enough valid bins to fit spline for R_den determination, returning 0")
        return 0
    spline = UnivariateSpline(r_v, sigma_v - density_thresh, k = 3, s = 0)
    solutions = spline.roots()

    afLogger.info(f"Finding R_den {particle}: fitted spline to sigma(r) - {density_thresh:.2e} Msun/pc^2, found roots at {solutions} kpc")
    afLogger.info(f"Using outermost root as R_den, between {r_v.min():.2f} and {r_v.max():.2f} kpc")
    valid_solutions = solutions[(solutions > r_v.min()) & (solutions < r_v.max())]
    afLogger.info(f"Valid roots within data range: {valid_solutions} kpc")
    if len(valid_solutions) == 0:
        fig, ax = plt.subplots()
        spline_rv = UnivariateSpline(r_v, sigma_v, k = 3, s = 0)
        ax.loglog(r_v, spline_rv(r_v), label="Spline fit")
        ax.loglog(r_v, sigma_v, "go", label="Sigma(r)")
        ax.axhline(density_thresh, color="red", linestyle="--", label=f"({density_thresh:.2e} Msun/pc**2)")
        ax.set_xlabel("Radius (kpc)")
        ax.set_ylabel("Surface Density (Msun/pc**2)")
        ax.set_title(f"R_den determination for snap {snap_idx} with particle {particle}")
        ax.legend()
        plt.show()
        afLogger.warning("No valid roots found for R_den within data range, returning 0")
        return 0
    return float(valid_solutions.max())*unyt.kpc

@register_derived("principal_axes", set_config={"center": "center_default", "radius": "virial_radius", "particle":"PartType0", "initial_q": 1, "initial_s": 1, "max_it": 20, "tol": 1e-3, "mass_field":"Masses"})
def principal_axes(sim, snap_idx, center = None, density_thresh = None, radius = None, particle=None, initial_q = None, initial_s = None, max_it = None, tol = None, mass_field = None):
    ds = sim[snap_idx]
    if isinstance(center, str):
        center = sim.get_derived(center, snap_idx)
    if isinstance(radius, str):
        radius = sim.get_derived(radius, snap_idx, center=center).to("kpc").to_value()

    sp = ds.sphere(center, (radius,"kpc"))
    # Initial guess
    q, s = initial_q, initial_s

    for it in range(max_it):
        sp.set_field_parameter("q", q)
        sp.set_field_parameter("s", s)
        
        w = (particle, mass_field)
        Sxx = sp.quantities.weighted_average_quantity((particle, "s_xx"), weight=w).to_value()
        Syy = sp.quantities.weighted_average_quantity((particle, "s_yy"), weight=w).to_value()
        Szz = sp.quantities.weighted_average_quantity((particle, "s_zz"), weight=w).to_value()
        Sxy = sp.quantities.weighted_average_quantity((particle, "s_xy"), weight=w).to_value()
        Sxz = sp.quantities.weighted_average_quantity((particle, "s_xz"), weight=w).to_value()
        Syz = sp.quantities.weighted_average_quantity((particle, "s_yz"), weight=w).to_value()

        M = np.array([[Sxx, Sxy, Sxz],
                    [Sxy, Syy, Syz],
                    [Sxz, Syz, Szz]], dtype=float)

        # eigvals sorted
        lam, eigv = np.linalg.eigh(M)
        sorted_idx = np.argsort(lam)[::-1]
        lam = lam[sorted_idx]
        eigv = eigv[:, sorted_idx]
        A, B, C = np.sqrt(lam)   # proportional axis lengths
        q_new = B / A
        s_new = C / A
        afLogger.info(f"Particle {particle}, Iter {it}: q={q_new:.4f}, s={s_new:.4f}, A = {A}, B = {B}, C = {C}")

        if abs(q_new - q) < tol and abs(s_new - s) < tol:
            q, s = q_new, s_new
            afLogger.info(f"Converged at iter {it}: q={q:.4f}, s={s:.4f}")
            break
        q, s = q_new, s_new

    return (A,B,C, eigv[:,0].tolist(), eigv[:,1].tolist(), eigv[:,2].tolist())

def _inertia_tensor_spectra(pos, mass):
    """Calculates eigenvalues and eigenvectors for the inertia tensor for a given list of masses and positions"""
    # S_ij = sum_n m_n * x_{n,i} * x_{n,j}
    s = np.einsum("ni,nj,n->ij", pos, pos, mass, optimize=True)
    tr = np.trace(s)
    I_tensor = tr * np.eye(3, dtype=s.dtype) - s # I = tr(S) * identity - S

    return np.linalg.eigh(I_tensor)

@register_derived("shape_ceverino", set_config={"radius": 2.0, "particle": "PartType4", "n_neighbors": 80, "max_iter": 10, "shell_dr":0.1, "mass_field":"Masses"})
def shape_ceverino(sim, snap_idx, center=None, radius=None, particle=None, n_neighbors=None, max_iter=None, shell_dr = None, mass_field=None):
    from scipy.spatial import cKDTree
    """
    Implements the Ceverino et al. (2015) 3D Isodensity Shape Fitting algorithm.
    """
    ds = sim[snap_idx]
    if isinstance(center, str):
        center = sim.get_derived(center, snap_idx)
    if isinstance(radius, str):
        radius = sim.get_derived(radius, snap_idx, center=center).to("kpc").to_value()

    # We grab a sphere 2x the target radius to ensure we capture all the particles (and neighbours)
    sp = ds.sphere(center, (radius * 2, "kpc"))
    pos = sp[(particle, "relative_particle_position")].to("kpc").v
    masses = sp[(particle, mass_field)].to("Msun").v

    if len(pos) < n_neighbors * 2:
        afLogger.warning(f"Not enough particles to compute shape at r={radius} kpc. N = {len(pos)}, required at least {n_neighbors*2}. Returning default spherical shape.")
        return (1, 1, 1, [1,0,0], [0,1,0], [0,0,1])

    # Local Density Calculation (80 nearest neighbors)
    tree = cKDTree(pos)
    dists, neighbor_idx = tree.query(pos, k=n_neighbors)
    r_80 = dists[:, -1]
    
    # Density = Mass of 80 neighbors / Volume
    m_80 = np.sum(masses[neighbor_idx], axis=1)
    vols = (4.0 / 3.0) * np.pi * (r_80**3)
    local_rho = m_80 / (vols + 1e-20)

    # Initial Guess: Standard inertia tensor of a thin spherical shell at radius r
    dr = shell_dr # kpc
    particle_rad = np.linalg.norm(pos, axis=1)
    shell_mask = (particle_rad > radius - dr) & (particle_rad < radius + dr)
    if np.sum(shell_mask) < 10:
        afLogger.warning(f"Initial shell has too few particles: N = {np.sum(shell_mask)}, required at least 10")
    p_shell = pos[shell_mask]
    m_shell = masses[shell_mask]
    evals, evecs = _inertia_tensor_spectra(p_shell, m_shell)
    # Major axis is eigenvector of smallest eigenvalue for classical inertia tensor
    sort_idx = np.argsort(evals)
    major_axis = evecs[:, sort_idx[0]]
    e1, e2, e3 = evals[sort_idx]
    a = np.sqrt(np.abs(1.5 * (e3 + e2 - e1)))
    b = np.sqrt(np.abs(1.5 * (e3 + e1 - e2)))
    c = np.sqrt(np.abs(1.5 * (e2 + e1 - e3)))
    v = evecs
    afLogger.info(f"Shape at iteration 0: b/a={b/a:.3f}, c/a={c/a:.3f}, a = {a:.3f}, b = {b:.3f}, c = {c:.3f}")
    # Iterative Isodensity Fitting
    for iteration in range(max_iter):
        # Find points where major axis intersects the sphere of radius r
        pole_1 = major_axis * radius
        pole_2 = -major_axis * radius
        # Find the single closest particle to each pole to get the target density
        _, idx_1 = tree.query(pole_1, k=1)
        _, idx_2 = tree.query(pole_2, k=1)
        rho_1 = local_rho[idx_1]
        rho_2 = local_rho[idx_2]
        rho_s = (rho_1 + rho_2) / 2.0
        sigma_s = max(np.abs(rho_1 - rho_2) / 2.0,0)# 0.1 * rho_s) # Use at least 10 percent if sigma is tiny
        afLogger.info(f"Iteration {iteration+1}: rho_1={rho_1:.3e}, rho_2={rho_2:.3e}, rho_s={rho_s:.3e}, sigma_s={sigma_s:.3e}, true_sigma={np.abs(rho_1 - rho_2) / 2.0:.3e}")
        # Isolate the isodensity shell
        iso_mask = (local_rho > (rho_s - sigma_s)) & (local_rho < (rho_s + sigma_s))
        if np.sum(iso_mask) < 10:
            afLogger.warning(f"Isodensity shell empty at iteration {iteration+1}. Stopping.")
            break
        p_iso = pos[iso_mask]
        m_iso = masses[iso_mask]
        # Calculate Classical Inertia Tensor of the isodensity shell
        e, v = _inertia_tensor_spectra(p_iso, m_iso)
        sort_idx = np.argsort(e) # Sort eigenvalues: e[0] < e[1] < e[2]
        e1, e2, e3 = e[sort_idx]
        # Calculate Routh (2013) axes lengths for a thin ellipsoidal shell. Use abs() to prevent sqrt of negative numbers due to precision errors on very spherical halos
        a = np.sqrt(np.abs(1.5 * (e3 + e2 - e1)))
        b = np.sqrt(np.abs(1.5 * (e3 + e1 - e2)))
        c = np.sqrt(np.abs(1.5 * (e2 + e1 - e3)))
        afLogger.info(f"Shape at iteration {iteration+1}: b/a={b/a:.3f}, c/a={c/a:.3f}, a = {a:.3f}, b = {b:.3f}, c = {c:.3f}")   
        # Update major axis
        new_major_axis = v[:, sort_idx[0]]        
        if np.abs(np.dot(major_axis, new_major_axis)) > 0.99:
            major_axis = new_major_axis
            break
            
        major_axis = new_major_axis

    afLogger.info(f"Shape at r={radius:.2f} kpc: b/a={b/a:.3f}, c/a={c/a:.3f}")    
    return (a, b, c, v[:, sort_idx[0]].tolist(), v[:, sort_idx[1]].tolist(), v[:, sort_idx[2]].tolist())

@register_derived("sfr_young_star", set_config={"center": "center_default", "radius": "virial_radius", "max_age": 20, "cosmology": [0.702,0.272,0.728,0.0]})
def sfr_young_star(sim, snap_idx, center=None, radius=None, max_age=None, cosmology=None):
    ds = sim[snap_idx]
    if isinstance(center, str):
        center = sim.get_derived(center, snap_idx)
    if isinstance(radius, str):
        radius = sim.get_derived(radius, snap_idx, center=center).to("kpc").to_value()
    # Prepare sphere and cosmology
    sp = ds.sphere(center, (radius,"kpc"))
    co = yt.utilities.cosmology.Cosmology(hubble_constant=cosmology[0], omega_matter=cosmology[1],omega_lambda=cosmology[2], omega_curvature=cosmology[3])

    allStarScale = np.array(sp["PartType4", "StellarFormationTime"])
    allStarMass = np.array(sp["PartType4", "Masses"].to("Msun"))
    # Convert scale factor to age
    cur_t = ds.current_time.to("Myr").to_value()
    min_a = float(1/(1 + co.z_from_t(co.quan(cur_t - max_age, "Myr"))))
    # Select stars formed after min_a (younger than max_age)
    youngMask = allStarScale > min_a
    youngMass = np.sum(allStarMass[youngMask])
    sfr = youngMass /(max_age * 1e6)
    return (sfr * unyt.Msun / unyt.yr).to("Msun/yr")

@register_derived("instant_sfr", set_config={"center": "center_default", "radius": "virial_radius", "particle":"PartType0"})
def instant_sfr(sim, snap_idx, center=None, radius=None, particle=None):
    ds = sim[snap_idx]
    if isinstance(center, str):
        center = sim.get_derived(center, snap_idx)
    if isinstance(radius, str):
        radius = sim.get_derived(radius, snap_idx, center=center).to("kpc").to_value()
    # Prepare sphere and cosmology
    sp = ds.sphere(center, (radius,"kpc"))
    sfr = sp[(particle, "StarFormationRate")].to("Msun/yr").sum()
    return sfr

# TODO: Everything halo related should be reworked at some point to an specific halo analysis module, these are just quick workarounds to get some basic halo properties for the TFG, but ideally you would have a more general halo finding and analysis module that can be used for different halo finders and has more properties (also ideally with some caching and ability to use pre-computed halo catalogs from disk, etc) 

# TO BE DEPRECATED
@register_derived("halo_file")
def halo_file(sim, snap_idx, path = None):
    if path is None:
        afLogger.error("Halo file path not provided nor registered for this simulation, returning None")
        return None
    return path

@register_derived("most_bound_pos",set_config={"path": None, "idx": None, "particle_type": 4})
def most_bound_pos(sim, snap_idx, path = None, idx = None, particle_type = None):
    with h5py.File(path,"r") as f:
        if idx is None:
            # TODO: This only works for gadget/arepo Subfind catalogs
            idx = np.argmax(f["Subhalo/SubhaloMassType"][:,particle_type])
            afLogger.info(f'Using most massive subhalo in terms of particle type {particle_type}, with mass {(f["Subhalo/SubhaloMassType"][idx,particle_type]* unyt.Msun * 1e10).to("Msun").v}')

        idbound = f["Subhalo/SubhaloIDMostbound"][idx]
        ad = sim[snap_idx].all_data()
        mask = ad[("all","ParticleIDs")] == idbound
        return ad[("aFl","particle_position")][mask][0].to("Mpc")