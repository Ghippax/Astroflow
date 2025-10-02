"""Workflow execution engine.

This module implements the WorkflowEngine class which loads workflow definitions
from YAML files and executes them on simulation data.
"""

import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from .. import log


class WorkflowEngine:
    """Engine for executing analysis workflows defined in YAML files.
    
    The WorkflowEngine reads workflow configurations from YAML files and executes
    the specified plots on given simulations and snapshots.
    
    Attributes:
        config (dict): Workflow configuration loaded from YAML
        plot_registry (dict): Mapping of plot names to plot functions
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the workflow engine.
        
        Args:
            config_path: Path to workflow configuration file. If None, loads
                        default workflows.
        """
        self.config = self._load_workflow_config(config_path)
        self.plot_registry = self._build_plot_registry()
    
    def _load_workflow_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load workflow configuration from YAML file.
        
        Args:
            config_path: Path to workflow configuration file
            
        Returns:
            dict: Workflow configuration
        """
        if config_path is None:
            # Try to find workflow_template.yaml in standard locations
            candidates = [
                'workflow_template.yaml',
                'workflows.yaml',
                Path(__file__).parent.parent.parent.parent / 'workflow_template.yaml'
            ]
            
            for candidate in candidates:
                if Path(candidate).exists():
                    config_path = candidate
                    break
            
            if config_path is None:
                log.logger.info("No workflow config found, using empty config")
                return {'workflows': {}}
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config or {'workflows': {}}
    
    def _build_plot_registry(self) -> Dict[str, Callable]:
        """Build registry of available plot functions.
        
        Returns:
            dict: Mapping of plot type names to functions
        """
        # Import plotting functions - we'll expand this as we create plotlib
        from ..plot import plots
        
        # Registry maps plot types to their implementations
        registry = {
            # Add plot functions as they become available
            # 'projection': plotlib.projection_panel,
            # 'phase': plotlib.phase_panel,
            # 'profile': plotlib.profile_panel,
        }
        
        return registry
    
    def list_workflows(self) -> List[str]:
        """List available workflow names.
        
        Returns:
            list: Names of available workflows
        """
        return list(self.config.get('workflows', {}).keys())
    
    def get_workflow(self, workflow_name: str) -> Dict[str, Any]:
        """Get workflow configuration by name.
        
        Args:
            workflow_name: Name of the workflow
            
        Returns:
            dict: Workflow configuration
            
        Raises:
            KeyError: If workflow not found
        """
        workflows = self.config.get('workflows', {})
        if workflow_name not in workflows:
            raise KeyError(f"Workflow '{workflow_name}' not found. "
                          f"Available workflows: {list(workflows.keys())}")
        
        return workflows[workflow_name]
    
    def run_workflow(
        self,
        workflow_name: str,
        simulations: List,
        snapshots: List[int],
        output_dir: Optional[str] = None
    ):
        """Execute a predefined workflow on simulations.
        
        Args:
            workflow_name: Name of workflow to execute
            simulations: List of simulation objects
            snapshots: List of snapshot indices to analyze
            output_dir: Directory for output files. If None, uses config default.
            
        Raises:
            KeyError: If workflow not found
            ValueError: If workflow configuration is invalid
        """
        workflow = self.get_workflow(workflow_name)
        
        log.logger.info(f"Starting workflow: {workflow_name}")
        
        if 'description' in workflow:
            log.logger.info(f"Description: {workflow['description']}")
        
        plots = workflow.get('plots', [])
        
        if not plots:
            log.logger.warning(f"Workflow '{workflow_name}' has no plots defined")
            return
        
        # Execute each plot in the workflow
        for i, plot_config in enumerate(plots):
            plot_type = plot_config.get('type')
            
            if not plot_type:
                log.logger.error(f"Plot {i} missing 'type' field, skipping")
                continue
            
            log.logger.info(f"  Executing plot {i+1}/{len(plots)}: {plot_type}")
            
            # Get plot function from registry
            if plot_type in self.plot_registry:
                plot_func = self.plot_registry[plot_type]
                
                # Extract plot-specific parameters
                plot_params = {k: v for k, v in plot_config.items() if k != 'type'}
                
                try:
                    plot_func(
                        simulations=simulations,
                        snapshots=snapshots,
                        output_dir=output_dir,
                        **plot_params
                    )
                except Exception as e:
                    log.logger.error(f"Error executing plot {plot_type}: {e}")
            else:
                log.logger.warning(
                    f"Plot type '{plot_type}' not implemented yet. "
                    f"Available types: {list(self.plot_registry.keys())}"
                )
        
        log.logger.info(f"Completed workflow: {workflow_name}")
    
    def validate_workflow(self, workflow_name: str) -> bool:
        """Validate that a workflow is properly configured.
        
        Args:
            workflow_name: Name of workflow to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            workflow = self.get_workflow(workflow_name)
        except KeyError:
            log.logger.error(f"Workflow '{workflow_name}' not found")
            return False
        
        if 'plots' not in workflow:
            log.logger.error(f"Workflow '{workflow_name}' missing 'plots' section")
            return False
        
        plots = workflow['plots']
        if not isinstance(plots, list):
            log.logger.error(f"Workflow '{workflow_name}' 'plots' must be a list")
            return False
        
        # Validate each plot
        for i, plot in enumerate(plots):
            if not isinstance(plot, dict):
                log.logger.error(f"Plot {i} must be a dictionary")
                return False
            
            if 'type' not in plot:
                log.logger.error(f"Plot {i} missing required 'type' field")
                return False
        
        log.logger.info(f"Workflow '{workflow_name}' is valid")
        return True
