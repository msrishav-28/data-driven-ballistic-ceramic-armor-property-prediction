"""
ML module for ceramic armor property prediction.

This module provides model loading, caching, and prediction services
for mechanical and ballistic properties of ceramic armor materials.
"""

from .model_loader import (
    ModelLoader,
    ModelCache,
    ModelMetadata,
    get_model_loader,
    initialize_models
)
from .predictor import (
    CeramicArmorPredictor,
    get_predictor
)
from .startup import (
    ModelStartupManager,
    get_startup_manager,
    startup_event,
    shutdown_event,
    health_check_event
)

__all__ = [
    'ModelLoader',
    'ModelCache', 
    'ModelMetadata',
    'get_model_loader',
    'initialize_models',
    'CeramicArmorPredictor',
    'get_predictor',
    'ModelStartupManager',
    'get_startup_manager',
    'startup_event',
    'shutdown_event',
    'health_check_event'
]