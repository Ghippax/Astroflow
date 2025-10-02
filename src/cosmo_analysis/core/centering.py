"""Centering strategies for finding galaxy centers in simulations.

This module implements a strategy pattern for different centering algorithms,
making it easier to add new methods and select them programmatically.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np
import unyt
from .. import log
from . import utils


class CenteringStrategy(ABC):
    """Abstract base class for centering strategies."""
    
    @abstractmethod
    def calculate_center(self, sim, idx, **kwargs) -> Tuple[np.ndarray, unyt.unyt_array]:
        """Calculate the center for a snapshot.
        
        Args:
            sim: Simulation object
            idx: Snapshot index
            **kwargs: Additional parameters specific to the strategy
        
        Returns:
            tuple: (center_array, center_unyt_array)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this centering strategy."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Return a description of this centering strategy."""
        pass


class MaxDensityStrategy(CenteringStrategy):
    """Find center by locating maximum density particle."""
    
    def calculate_center(self, sim, idx, lim=20, **kwargs) -> Tuple[np.ndarray, unyt.unyt_array]:
        """Calculate center by finding maximum density particle.
        
        Args:
            sim: Simulation object
            idx: Snapshot index
            lim: Radius limit in kpc for initial sphere (default: 20)
        
        Returns:
            tuple: (center_array in pc, center_unyt_array in kpc)
        """
        snap   = sim.ytFull[idx]
        cutOff = snap.sphere("center",(lim,"kpc"))
        den    = np.array(cutOff["PartType0", "Density"].to("Msun/pc**3"))
        x      = np.array(cutOff["PartType0", "x"].to("pc"))
        y      = np.array(cutOff["PartType0", "y"].to("pc"))
        z      = np.array(cutOff["PartType0", "z"].to("pc"))
        cenIdx = utils.maxIdx(den)
        cen    = np.array([x[cenIdx],y[cenIdx],z[cenIdx]])
        return (cen,cen/1e3*unyt.kpc)
    
    @property
    def name(self) -> str:
        return "max_density"
    
    @property
    def description(self) -> str:
        return "Find center by locating maximum density particle"


class CenterOfMassStrategy(CenteringStrategy):
    """Find center using gas center of mass."""
    
    def calculate_center(self, sim, idx, lim=20, **kwargs) -> Tuple[np.ndarray, unyt.unyt_array]:
        """Calculate center using gas center of mass.
        
        Args:
            sim: Simulation object
            idx: Snapshot index
            lim: Radius limit in kpc for initial sphere (default: 20)
        
        Returns:
            tuple: (center_array in pc, center_unyt_array in kpc)
        """
        snap   = sim.ytFull[idx]
        cutOff = snap.sphere("center",(lim,"kpc"))
        cen    = cutOff.quantities.center_of_mass(use_gas=True, use_particles=False).in_units("pc")
        return (cen.d,cen.d/1e3*unyt.kpc)
    
    @property
    def name(self) -> str:
        return "center_of_mass"
    
    @property
    def description(self) -> str:
        return "Calculate center using gas center of mass"


class AGORAIsolatedStrategy(CenteringStrategy):
    """AGORA method for isolated simulations."""
    
    def calculate_center(self, sim, idx, **kwargs) -> Tuple[np.ndarray, unyt.unyt_array]:
        """Calculate center using AGORA method for isolated simulations.
        
        Finds maximum density, refines with center of mass, then finds max density again.
        
        Args:
            sim: Simulation object
            idx: Snapshot index
        
        Returns:
            tuple: (center_array in pc, center_unyt_array in kpc)
        """
        snap      = sim.ytFull[idx]

        v, cen1   = snap.find_max(("gas", "density"))
        bigCutOff = snap.sphere(cen1, (30.0, "kpc"))
        cen2      = bigCutOff.quantities.center_of_mass(use_gas=True, use_particles=False).in_units("kpc")
        cutOff    = snap.sphere(cen2,(1.0,"kpc"))
        cen       = cutOff.quantities.max_location(("gas", "density"))
        center    = np.array([cen[1].d,cen[2].d,cen[3].d])*1e3

        return (center,center/1e3*unyt.kpc)
    
    @property
    def name(self) -> str:
        return "agora_isolated"
    
    @property
    def description(self) -> str:
        return "AGORA method for isolated simulations"


class AGORACosmologicalStrategy(CenteringStrategy):
    """AGORA method for cosmological simulations using projection file."""
    
    def calculate_center(self, sim, idx, projPath=None, **kwargs) -> Tuple[np.ndarray, unyt.unyt_array]:
        """Calculate center using AGORA method for cosmological runs.
        
        Args:
            sim: Simulation object
            idx: Snapshot index
            projPath: Path to projection file. If None, uses config value.
        
        Returns:
            tuple: (center_array, center_unyt_array)
        """
        from ..config import get_config
        
        snap = sim.ytFull[idx]
        
        # Get projection path from config if not provided
        if projPath is None:
            config = get_config()
            projPath = config.get('paths.data_files.projection_list', 'outputlist_projection.txt')
        
        f        = np.loadtxt(projPath,skiprows=4)
        idx0     = utils.getClosestIdx(f[:,0],0.99999)
        tIdx     = utils.getClosestIdx(f[0:idx0,1],sim.snap[idx].a)

        cen1     = np.array([f[tIdx,3],f[tIdx,4],f[tIdx,5]])
        center   = snap.arr([cen1[0], cen1[1], cen1[2]],'code_length')
        sp       = snap.sphere(center, (2,'kpc'))
        center   = sp.quantities.center_of_mass(use_gas=True,use_particles=False)
        sp       = snap.sphere(center, (2*0.5,'kpc'))
        center   = sp.quantities.center_of_mass(use_gas=True,use_particles=False)
        sp       = snap.sphere(center, (2*0.25,'kpc'))
        center   = sp.quantities.center_of_mass(use_gas=True,use_particles=False)
        center   = center.to("code_length")

        cen1     = np.array([center[0].d,center[1].d,center[2].d])

        return   (cen1,center)
    
    @property
    def name(self) -> str:
        return "agora_cosmological"
    
    @property
    def description(self) -> str:
        return "AGORA method for cosmological simulations"


class AGORAFixedStrategy(CenteringStrategy):
    """AGORA method with hardcoded center coordinates."""
    
    def calculate_center(self, sim, idx, **kwargs) -> Tuple[np.ndarray, unyt.unyt_array]:
        """Calculate center using hardcoded AGORA coordinates.
        
        Args:
            sim: Simulation object
            idx: Snapshot index
        
        Returns:
            tuple: (center_array in pc, center_unyt_array in kpc)
        """
        snap   = sim.ytFull[idx]
        cen    = snap.arr([6.184520935812296e+21, 4.972678132728082e+21, 6.559067311284074e+21], 'cm')
        cen    = cen.to("pc")
        cent   = np.array([cen[0].d,cen[1].d,cen[2].d])
        return (cen,cen/1e3*unyt.kpc)
    
    @property
    def name(self) -> str:
        return "agora_fixed"
    
    @property
    def description(self) -> str:
        return "AGORA method with hardcoded coordinates"


class OriginStrategy(CenteringStrategy):
    """Use origin (0,0,0) as center."""
    
    def calculate_center(self, sim, idx, **kwargs) -> Tuple[np.ndarray, unyt.unyt_array]:
        """Calculate center at origin (0,0,0).
        
        Args:
            sim: Simulation object
            idx: Snapshot index
        
        Returns:
            tuple: (center_array in pc, center_unyt_array in kpc)
        """
        cen = np.array([0,0,0])
        return (cen,cen/1e3*unyt.kpc)
    
    @property
    def name(self) -> str:
        return "origin"
    
    @property
    def description(self) -> str:
        return "Use origin (0,0,0) as center"


class AGORAExtendedStrategy(CenteringStrategy):
    """Extended AGORA method with more refinement steps."""
    
    def calculate_center(self, sim, idx, projPath=None, **kwargs) -> Tuple[np.ndarray, unyt.unyt_array]:
        """Calculate center using expanded AGORA method.
        
        Args:
            sim: Simulation object
            idx: Snapshot index
            projPath: Path to projection file. If None, uses config value.
        
        Returns:
            tuple: (center_array, center_unyt_array)
        """
        from ..config import get_config
        
        snap = sim.ytFull[idx]
        
        # Get projection path from config if not provided
        if projPath is None:
            config = get_config()
            projPath = config.get('paths.data_files.projection_list', 'outputlist_projection.txt')
        
        log.logger.debug("Loading projection file from: %s", projPath)

        f        = np.loadtxt(projPath,skiprows=4)
        idx0     = utils.getClosestIdx(f[:,0],0.99999)
        tIdx     = utils.getClosestIdx(f[0:idx0,1],sim.snap[idx].a)

        cen1     = np.array([f[tIdx,3],f[tIdx,4],f[tIdx,5]])
        center   = snap.arr([cen1[0], cen1[1], cen1[2]],'code_length')
        
        # Multiple refinement steps
        for radius in [40, 10, 2, 1, 0.5, 0.25]:
            sp = snap.sphere(center, (radius,'kpc'))
            center = sp.quantities.center_of_mass(use_gas=True,use_particles=False)
        
        center   = center.to("code_length")
        cen1     = np.array([center[0].d,center[1].d,center[2].d])

        return   (cen1,center)
    
    @property
    def name(self) -> str:
        return "agora_extended"
    
    @property
    def description(self) -> str:
        return "Extended AGORA method with more refinement"


class ShrinkingSphereStrategy(CenteringStrategy):
    """Iterative shrinking sphere method using particles."""
    
    def calculate_center(self, sim, idx, iterations=8, **kwargs) -> Tuple[np.ndarray, unyt.unyt_array]:
        """Calculate center using iterative shrinking sphere method.
        
        Starts from box center and iteratively refines using particle center of mass
        with progressively smaller spheres.
        
        Args:
            sim: Simulation object
            idx: Snapshot index
            iterations: Number of refinement iterations (default: 8)
        
        Returns:
            tuple: (center_array in code units, center_unyt_array in code_length)
        """
        snap = sim.ytFull[idx]
        boxSize = snap.parameters['BoxSize']
        
        cen1 = np.array([boxSize/2,boxSize/2,boxSize/2])
        center = snap.arr([cen1[0], cen1[1], cen1[2]],'code_length')

        sizeSphere = np.logspace(np.log10(boxSize),np.log10(boxSize*0.0001),iterations)

        for i in range(iterations):
            sp = snap.sphere(center, (sizeSphere[i],'code_length'))
            center = sp.quantities.center_of_mass(use_gas=False,use_particles=True)

        center = center.to("code_length")
        cen1 = np.array([center[0].d,center[1].d,center[2].d])

        return (cen1,center)
    
    @property
    def name(self) -> str:
        return "shrinking_sphere"
    
    @property
    def description(self) -> str:
        return "Iterative shrinking sphere using particles"


class CenteringRegistry:
    """Registry for managing available centering strategies."""
    
    def __init__(self):
        """Initialize the centering registry with default strategies."""
        self._strategies = {}
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register default centering strategies."""
        strategies = [
            MaxDensityStrategy(),
            CenterOfMassStrategy(),
            AGORAIsolatedStrategy(),
            AGORACosmologicalStrategy(),
            AGORAFixedStrategy(),
            OriginStrategy(),
            AGORAExtendedStrategy(),
            ShrinkingSphereStrategy()
        ]
        
        for strategy in strategies:
            self.register(strategy)
    
    def register(self, strategy: CenteringStrategy):
        """Register a new centering strategy.
        
        Args:
            strategy: CenteringStrategy instance to register
        """
        self._strategies[strategy.name] = strategy
        log.logger.debug(f"Registered centering strategy: {strategy.name}")
    
    def get(self, name: str) -> CenteringStrategy:
        """Get a centering strategy by name.
        
        Args:
            name: Name of the strategy
        
        Returns:
            CenteringStrategy: The requested strategy
        
        Raises:
            KeyError: If strategy not found
        """
        if name not in self._strategies:
            raise KeyError(f"Centering strategy '{name}' not found. "
                          f"Available strategies: {list(self._strategies.keys())}")
        return self._strategies[name]
    
    def list_strategies(self) -> list:
        """List all available strategies.
        
        Returns:
            list: List of tuples (name, description)
        """
        return [(name, strategy.description) 
                for name, strategy in self._strategies.items()]
    
    def calculate_center(self, strategy_name: str, sim, idx, **kwargs) -> Tuple[np.ndarray, unyt.unyt_array]:
        """Calculate center using specified strategy.
        
        Args:
            strategy_name: Name of the strategy to use
            sim: Simulation object
            idx: Snapshot index
            **kwargs: Additional parameters for the strategy
        
        Returns:
            tuple: (center_array, center_unyt_array)
        """
        strategy = self.get(strategy_name)
        return strategy.calculate_center(sim, idx, **kwargs)


# Global registry instance
_registry = None


def get_centering_registry() -> CenteringRegistry:
    """Get the global centering registry instance.
    
    Returns:
        CenteringRegistry: The global registry
    """
    global _registry
    if _registry is None:
        _registry = CenteringRegistry()
    return _registry


# Mapping from old numeric codes to new strategy names
LEGACY_CODE_MAPPING = {
    "1": "max_density",
    "2": "center_of_mass",
    "3": "agora_isolated",
    "4": "agora_cosmological",
    "5": "agora_fixed",
    "6": "origin",
    "7": "agora_extended",
    "8": "shrinking_sphere"
}


def get_strategy_name(code: str) -> str:
    """Convert legacy numeric code to strategy name.
    
    Args:
        code: Legacy numeric code or strategy name
    
    Returns:
        str: Strategy name
    """
    return LEGACY_CODE_MAPPING.get(code, code)
