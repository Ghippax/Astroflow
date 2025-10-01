Cosmo Analysis Documentation
============================

Welcome to the documentation for **Cosmo Analysis**, a Python package for analyzing galaxy simulations using yt.

Overview
--------

Cosmo Analysis provides tools for:

* Loading and analyzing Gadget/AREPO simulation snapshots
* Creating projection and phase space plots
* Computing physical properties of galaxies
* Comparing multiple simulations
* Automated analysis workflows

Installation
------------

To install Cosmo Analysis, clone the repository and install with pip:

.. code-block:: bash

   git clone https://github.com/Ghippax/Cosmo-Analysis.git
   cd Cosmo-Analysis
   pip install -r requirements.txt
   pip install -e .

Quick Start
-----------

Basic usage example:

.. code-block:: python

   from cosmo_analysis.io.load import load
   from cosmo_analysis.plot.plots import ytProjPanel
   from cosmo_analysis.config import load_config

   # Load configuration
   config = load_config('config.yaml')
   
   # Load a simulation
   sim = load(name="my_sim", path="/path/to/simulation", 
              centerDefs=["3", "7"])
   
   # Create projection plots
   ytProjPanel(simArr=[sim], idxArr=[0], 
               part="PartType0", message="Gas Density")

Configuration
-------------

Cosmo Analysis uses YAML configuration files. Copy ``config_template.yaml`` to ``config.yaml`` and customize:

.. code-block:: yaml

   paths:
     output_directory: "/path/to/output"
   
   plotting_defaults:
     save_plots: true
     show_plots: false

API Reference
-------------

.. toctree::
   :maxdepth: 2
   :caption: Modules:

   api/config
   api/io
   api/core
   api/plot

Core Modules
~~~~~~~~~~~~

Configuration
^^^^^^^^^^^^^

.. automodule:: cosmo_analysis.config
   :members:
   :undoc-members:
   :show-inheritance:

Input/Output
^^^^^^^^^^^^

.. automodule:: cosmo_analysis.io.load
   :members:
   :undoc-members:
   :show-inheritance:

Core Functions
^^^^^^^^^^^^^^

.. automodule:: cosmo_analysis.core.fields
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: cosmo_analysis.core.sim_prop
   :members:
   :undoc-members:
   :show-inheritance:

Plotting
^^^^^^^^

.. automodule:: cosmo_analysis.plot.plots
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: cosmo_analysis.plot.workflows
   :members:
   :undoc-members:
   :show-inheritance:

Contributing
------------

Contributions are welcome! Please see ``CONTRIBUTING.md`` for guidelines.

Testing
~~~~~~~

Run tests with pytest:

.. code-block:: bash

   pytest tests/

Build documentation:

.. code-block:: bash

   cd docs
   make html

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
