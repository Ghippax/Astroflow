"""Base plotting utilities for cosmo_analysis.

This module contains core plotting utilities used across all plotting functions,
including figure handling, legend creation, and frame capture for animations.
"""

import io
import os.path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from .. import log
from ..config import get_config


def saveFrame(figure, verbose=None, config=None):
    """Capture matplotlib figure as numpy array for animation.
    
    Args:
        figure: matplotlib figure object
        verbose: verbosity level for logging (optional, for backward compatibility)
        config: Config object (optional, will use global if not provided)
    
    Returns:
        numpy.ndarray: Image data as array
    """
    if config is None:
        config = get_config()
    
    dpi = config.get('plotting_defaults.dpi', 300)
    
    buf = io.BytesIO()
    figure.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=dpi)
    buf.seek(0)
    img = np.array(Image.open(buf))
    log.logger.debug("Figure saved to buffer")
    buf.close()
    plt.close(figure)
    return img


def setLegend(ax, sims, idx):
    """Set legend for simulation comparison plots.
    
    Creates legend entries showing simulation names with time or redshift info.
    
    Args:
        ax: matplotlib axes object
        sims: list of Simulation objects
        idx: list of snapshot indices for each simulation
    """
    timeInfo = [None] * len(sims)
    for i in range(len(sims)):
        timeInfo[i] = " t=" + str(round(sims[i].snap[idx[i]].time)) + " Myr"
        if sims[i].cosmo:
            timeInfo[i] = " z=" + str(round(sims[i].snap[idx[i]].z, 2))
    ax.legend([sims[i].name + timeInfo[i] for i in range(len(sims))])


def handleFig(figure, switches, message, saveFigPath=None, verbose=None, config=None):
    """Handle figure display, saving, or animation frame capture.
    
    This is the main figure output handler used by all plotting functions.
    It can show the figure, save it to disk, or capture it as animation frame.
    
    Args:
        figure: matplotlib figure object
        switches: tuple of (show, animate, save) boolean flags
        message: filename or message for saved figure
        saveFigPath: directory path to save figure (overrides config, optional)
        verbose: verbosity level for logging (optional, for backward compatibility)
        config: Config object (optional, will use global if not provided)
    
    Returns:
        numpy.ndarray: Image array if animate flag is True, None otherwise
    """
    if config is None:
        config = get_config()
    
    # Shows figure
    if switches[0]:
        log.logger.debug("Showing to screen")
        plt.show()

    # Return the frame for an animation
    if switches[1]:     
        return saveFrame(figure, verbose=verbose, config=config)
    
    # Saves figure with message path as title
    if switches[2]:
        # Determine base directory for saving
        if saveFigPath is not None and saveFigPath != 0:
            # Use provided path
            base_dir = saveFigPath
        else:
            # Get from config
            base_dir = config.get('paths.output_directory', './output')
        
        # Construct full path
        if message and message != 0:
            fullPath = os.path.join(base_dir, message.replace(" ", "_") + ".png")
        else:
            fullPath = os.path.join(base_dir, "placeholder.png")
            log.logger.warning("TITLE NOT SPECIFIED FOR THIS FIGURE, PLEASE SPECIFY A TITLE")

        log.logger.info(f"  Saving figure to {fullPath}")
        
        # Get DPI from config
        dpi = config.get('plotting_defaults.dpi', 300)
            
        # Ensure directory exists
        os.makedirs(base_dir, exist_ok=True)
        
        figure.savefig(fullPath, bbox_inches='tight', pad_inches=0.03, dpi=dpi)
        plt.close(figure)
    
    return None
