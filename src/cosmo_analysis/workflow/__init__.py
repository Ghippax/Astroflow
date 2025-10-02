"""Workflow management system for automated analysis pipelines.

This module provides classes and functions for defining, loading, and executing
analysis workflows from YAML configuration files.
"""

from .engine import WorkflowEngine
from .standard import STANDARD_WORKFLOWS

__all__ = ['WorkflowEngine', 'STANDARD_WORKFLOWS']
