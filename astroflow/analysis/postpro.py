from typing import Optional

from .registry import register_postprocessing
from . import settings

import numpy as np
from unyt import unyt_array, G

from ..log import get_logger
afLogger = get_logger()

@register_postprocessing("circ_velocity", label = "Circular Velocity")
def compute_circ_vel(profile, field):
    mass = profile[field]
    radius = profile.x
    return np.sqrt((G*mass/(radius)))

@register_postprocessing("spherical_shell", label = "Density")
def binFunctionSphBins(profile,field):
    mass = profile[field]
    radius = profile.x
    dr = 0.5*(radius[1]-radius[0])
    density = mass/(4/3 * np.pi * (((radius+dr))**3-((radius-dr))**3) )
    
    return density