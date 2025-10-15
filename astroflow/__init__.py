"""Astroflow public API."""

from .analysis import derived_prop # Gets the registry populated
from .core.load import load

__all__ = ["load", "derived_prop"]