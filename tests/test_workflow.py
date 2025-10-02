"""Tests for workflow engine and standard workflows."""

import pytest
import tempfile
import yaml
from pathlib import Path
from cosmo_analysis.workflow import WorkflowEngine, STANDARD_WORKFLOWS


class TestWorkflowEngine:
    """Test WorkflowEngine class."""
    
    def test_init_without_config(self):
        """Test engine initialization without config file."""
        engine = WorkflowEngine()
        assert engine.config is not None
        assert 'workflows' in engine.config
    
    def test_init_with_config_file(self):
        """Test engine initialization with config file."""
        # Create temporary config file
        config_data = {
            'workflows': {
                'test_workflow': {
                    'description': 'Test workflow',
                    'plots': [
                        {'type': 'projection', 'field': 'Density'}
                    ]
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            engine = WorkflowEngine(config_path)
            assert 'test_workflow' in engine.config['workflows']
        finally:
            Path(config_path).unlink()
    
    def test_list_workflows(self):
        """Test listing available workflows."""
        config_data = {
            'workflows': {
                'workflow1': {'plots': []},
                'workflow2': {'plots': []}
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            engine = WorkflowEngine(config_path)
            workflows = engine.list_workflows()
            assert 'workflow1' in workflows
            assert 'workflow2' in workflows
        finally:
            Path(config_path).unlink()
    
    def test_get_workflow(self):
        """Test getting a specific workflow."""
        config_data = {
            'workflows': {
                'test_workflow': {
                    'description': 'Test',
                    'plots': [{'type': 'projection'}]
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            engine = WorkflowEngine(config_path)
            workflow = engine.get_workflow('test_workflow')
            assert workflow['description'] == 'Test'
            assert len(workflow['plots']) == 1
        finally:
            Path(config_path).unlink()
    
    def test_get_workflow_not_found(self):
        """Test getting a non-existent workflow."""
        engine = WorkflowEngine()
        
        with pytest.raises(KeyError, match="Workflow 'nonexistent' not found"):
            engine.get_workflow('nonexistent')
    
    def test_validate_workflow_valid(self):
        """Test validation of a valid workflow."""
        config_data = {
            'workflows': {
                'valid_workflow': {
                    'plots': [
                        {'type': 'projection', 'field': 'Density'},
                        {'type': 'phase', 'x_field': 'Density', 'y_field': 'Temperature'}
                    ]
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            engine = WorkflowEngine(config_path)
            assert engine.validate_workflow('valid_workflow') is True
        finally:
            Path(config_path).unlink()
    
    def test_validate_workflow_missing_plots(self):
        """Test validation of workflow without plots section."""
        config_data = {
            'workflows': {
                'invalid_workflow': {
                    'description': 'Missing plots'
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            engine = WorkflowEngine(config_path)
            assert engine.validate_workflow('invalid_workflow') is False
        finally:
            Path(config_path).unlink()
    
    def test_validate_workflow_plot_missing_type(self):
        """Test validation of plot without type field."""
        config_data = {
            'workflows': {
                'invalid_workflow': {
                    'plots': [
                        {'field': 'Density'}  # Missing 'type'
                    ]
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            engine = WorkflowEngine(config_path)
            assert engine.validate_workflow('invalid_workflow') is False
        finally:
            Path(config_path).unlink()


class TestStandardWorkflows:
    """Test standard workflow definitions."""
    
    def test_standard_workflows_exist(self):
        """Test that standard workflows are defined."""
        assert STANDARD_WORKFLOWS is not None
        assert isinstance(STANDARD_WORKFLOWS, dict)
        assert len(STANDARD_WORKFLOWS) > 0
    
    def test_standard_workflows_structure(self):
        """Test that all standard workflows have proper structure."""
        for name, workflow in STANDARD_WORKFLOWS.items():
            assert 'plots' in workflow, f"Workflow {name} missing 'plots'"
            assert isinstance(workflow['plots'], list), f"Workflow {name} 'plots' not a list"
            
            for i, plot in enumerate(workflow['plots']):
                assert isinstance(plot, dict), f"Workflow {name} plot {i} not a dict"
                assert 'type' in plot, f"Workflow {name} plot {i} missing 'type'"
    
    def test_gas_analysis_workflow(self):
        """Test gas_analysis workflow structure."""
        assert 'gas_analysis' in STANDARD_WORKFLOWS
        workflow = STANDARD_WORKFLOWS['gas_analysis']
        assert 'description' in workflow
        assert len(workflow['plots']) > 0
    
    def test_star_formation_workflow(self):
        """Test star_formation workflow structure."""
        assert 'star_formation' in STANDARD_WORKFLOWS
        workflow = STANDARD_WORKFLOWS['star_formation']
        assert 'description' in workflow
        assert len(workflow['plots']) > 0
    
    def test_nsff_analysis_workflow(self):
        """Test nsff_analysis workflow structure."""
        assert 'nsff_analysis' in STANDARD_WORKFLOWS
        workflow = STANDARD_WORKFLOWS['nsff_analysis']
        assert 'description' in workflow
        assert len(workflow['plots']) > 0


class TestWorkflowIntegration:
    """Integration tests for workflow system."""
    
    def test_load_standard_workflows_in_engine(self):
        """Test that standard workflows can be loaded in engine."""
        # Create a config with standard workflows
        config_data = {'workflows': STANDARD_WORKFLOWS}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            engine = WorkflowEngine(config_path)
            
            # Test all standard workflows are loadable
            for workflow_name in STANDARD_WORKFLOWS.keys():
                workflow = engine.get_workflow(workflow_name)
                assert workflow is not None
                assert engine.validate_workflow(workflow_name)
        finally:
            Path(config_path).unlink()
