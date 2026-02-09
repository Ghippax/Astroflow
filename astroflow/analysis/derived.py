from typing import Optional
from .registry import register_derived, register_alias
from . import settings

import numpy as np
import yt
import unyt
from unyt import unyt_array
import h5py

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
            plot.proj(sim, snap_idx, center=centerTemp, width=(sizeSphere[i],"Mpc"), field=("gas","density"), save = False, show = True)
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

@register_derived("center_max",set_config={"center": None,"radius":5, "see": False, "field":("gas","density")})
def compute_center_max(sim, snap_idx, center = None, radius=5, see = None, field=("gas","density")):
    ds = sim[snap_idx]
    centerTemp = center if center is not None else sim.get_derived("center_default", snap_idx)

    sp = ds.sphere(centerTemp, (radius,"Mpc"))
    _, x, y, z = sp.quantities.max_location(field)

    center = unyt_array([x,y,z],"Mpc").to("Mpc")

    if see:
        afLogger.info(f"Plotting density projection...")
        plot.proj(sim, snap_idx, center=center, width=(radius,"Mpc"), field=("gas","density"), save = False, show = True)
    
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

@register_derived("half_mass_radius", set_config={"center": "center_default", "particle": "PartType4", "radius": None})
def half_radius(sim, snap_idx, center = None, particle = None, radius = None):
    # Setup sphere and calculate target half mass
    ds = sim[snap_idx]
    if isinstance(center, str):
        center = sim.get_derived(center, snap_idx)
    if radius is None:
        radius = sim.get_derived("virial_radius", snap_idx, center=center).to_value()
    sp = ds.sphere(center, (radius,"kpc"))
    afLogger.debug(f"Calculating half-mass radius for snapshot {snap_idx} using particle type {particle} within radius {radius}")
    total_mass = total_in_obj(sp, (particle,"Masses")).to("Msun").v
    half_mass = total_mass / 2.0

    allMass    = sp[(particle,"Masses")].in_units("Msun")
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

register_alias("radius_e_star", "half_mass_radius", particle="PartType4")
register_alias("radius_e_gas", "half_mass_radius", particle="PartType0")
register_alias("radius_e_dm", "half_mass_radius", particle="PartType1")

@register_derived("total_mass", set_config={"particle": "all"})
def total_mass(sim, snap_idx, particle=None):
    ds = sim[snap_idx].all_data()
    return total_in_obj(ds, (particle,"Masses")).to("Msun")

register_alias("total_mass_gas", "total_mass", particle="PartType0")
register_alias("total_mass_stars", "total_mass", particle="PartType4")
register_alias("total_mass_dm", "total_mass", particle="PartType1")
register_alias("total_mass_all", "total_mass", particle="all")

@register_derived("mass_in_sphere", set_config={"center": "center_default", "radius": 10, "particle": "all"})
def mass_in_sphere(sim, snap_idx, center = None, radius = None, particle=None):
    ds = sim[snap_idx]
    if isinstance(center, str):
        center = sim.get_derived(center, snap_idx)
    if isinstance(radius, str):
        radius = sim.get_derived(radius, snap_idx, center=center).to("kpc").to_value()
    sp = ds.sphere(center, (radius,"kpc"))
    return total_in_obj(sp, (particle,"Masses")).to("Msun")

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

@register_derived("v_phi", set_config={"center": "center_default", "radius": 2, "particle":"PartType0","bins":40,"axis":"faceon", "bulk_v":"bulk_v", "temp":None})
def v_phi(sim, snap_idx, center = None, radius = None, particle=None, bins=None, axis=None, bulk_v=None, temp=None):
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
    profile = data.profile(sp, (particle,"particle_position_cylindrical_radius"), field, data_args=settings.DataConfig(n_bins=bins,x_unit="kpc",unit="km/s", bin_extrema=[(0.2,radius)], log = True, accumulate=False, weight_field=(particle,"mass")))

    return profile[field].in_units("km/s")[-1]

@register_derived("v_disp", set_config={"center": "center_default", "radius": 2, "particle":"PartType0","bins":40,"axis":"faceon", "bulk_v":"bulk_v", "temp":None})
def v_disp(sim, snap_idx, center = None, radius = None, particle=None, bins=None, axis=None, bulk_v=None, temp=None):
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
    profile = data.profile(sp, (particle,"particle_position_cylindrical_radius"), field, data_args=settings.DataConfig(n_bins=bins,x_unit="kpc",unit="km/s", bin_extrema=[(0.2,radius)], log = True, accumulate=False, weight_field=(particle,"mass")))
    return profile.standard_deviation[field].in_units("km/s")[-1]

@register_derived("cuspyness", set_config={"center": "center_default", "radius": 2, "particle":"PartType0","bins":30})
def cuspyness(sim, snap_idx, center = None, radius = None, particle=None, bins=None):
    ds = sim[snap_idx]
    if isinstance(center, str):
        center = sim.get_derived(center, snap_idx)
    if isinstance(radius, str):
        radius = sim.get_derived(radius, snap_idx, center=center).to("kpc").to_value()

    sp = ds.sphere(center, (radius,"kpc"))
    # TODO: Fix 0.01 (should be few times epsilon) also radius*1.1
    field = (particle,"Masses")
    profile = data.profile(sp, (particle,"particle_position_spherical_radius"), field, data_args=settings.DataConfig(n_bins=bins,x_unit="kpc",unit="Msun/kpc**3", bin_extrema=[(0.01,radius*1.1)], log = True, accumulate=True, postprocess="spherical_shell"))

    rho = postpro_fn.get("spherical_shell")(profile,field).in_units("Msun/kpc**3").v
    r = profile.x.in_units("kpc").v
    valid = (rho > 0) & (r > 0)
    log_rho = np.log10(rho[valid])
    log_r = np.log10(r[valid])
    
    if len(log_r) < 3:
        raise ValueError("Insufficient valid bins for derivative calculation")
    
    # Fit spline in log-log space
    spline = UnivariateSpline(log_r, log_rho)
    log_fid = np.log10(radius)

    return float(spline.derivative()(log_fid))

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
        return ad[("all","particle_position")][mask][0].to("Mpc")