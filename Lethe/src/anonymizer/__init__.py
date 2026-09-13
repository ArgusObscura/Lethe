"""Lethe: Vehicle Camera Video Anonymization Tool

A comprehensive tool for anonymizing sensitive objects (faces, license plates)
in vehicle camera videos for autonomous driving R&D.
"""

from .core import AnonymizationPipeline
from .models.config import AnonymizationConfig, DetectionConfig

__version__ = "0.1.0"
__author__ = "ArgusObscura"

__all__ = [
    "AnonymizationPipeline",
    "AnonymizationConfig",
    "DetectionConfig",
]
