# TODO: Description of file for documentation

# TODO: Description of functions for documentation

import copy
import matplotlib.pyplot as plt

## OPTIONS
# Figure options
figSize   = 8  # Fig size for plt plots
ytFigSize = 12 # Window size for the yt plot
fontSize  = 12 # Fontsize in yt plots

# TODO: Fix this path
savePath = "/sqfs/work/hp240141/z6b616/analysis"

# Modified color maps
starCmap = copy.copy(plt.get_cmap('autumn'))
starCmap.set_bad(color='k') # (log(1e4) - log(1e1)) / (log(1e6) - log(1e1)) = 0.6
starCmap.set_under(color='k')
mColorMap = copy.copy(plt.get_cmap('algae'))
mColorMap.set_bad(mColorMap(0.347)) # (0.02041 - 0.01) / (0.04 - 0.01) = 0.347
mColorMap.set_under(mColorMap(0))
mColorMap2 = copy.copy(plt.get_cmap('algae'))
mColorMap2.set_bad(mColorMap2(0))
mColorMap2.set_under(mColorMap2(0))

# Scaffolding options (Global parameters for how the plotting functions below should operate)
verboseLevel    = 15                    # How detailed is the console log, higher is more detailed
showErrorGlobal = 0                     # If the dispersion is plotted or not
errorLimGlobal  = (-1,1)                # Y limits on the dispersion plot
showAll         = False                  # If the plots are showed in the notebook or not
saveAll         = True                 # If the plots are saved to disk or not

# Analysis options
figWidth       = 30          # Kpc (Big box)
buffSize       = 800         # N   (Default bin number for coordinate-type histograms) 
youngStarAge   = 20          # Myr (Age to be considered for SFR density calculations)
lowResMock     = 750         # pc  (Resolution of the mock observations)
starFigSize    = 80          # kpc (Size of star map)
starBufferSize = 400         # N   (Buffer size of star map)
gasPart        = "PartType0" # Field for gas SPH particles 
starPart       = "PartType4" # Field for star particles
dmPart         = "PartType1" # Field for dark matter particles
zSolar         = 0.0204      # Solar metallicity