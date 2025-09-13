"""
Pydantic models for the Ceramic Armor ML API.

This package contains all request and response models, along with
custom exceptions and validation utilities.
"""

from .request_models import (
    CompositionModel,
    ProcessingModel,
    MicrostructureModel,
    PredictionRequest,
    BatchPredictionRequest,
    PhaseDistribution,
    InterfaceQuality
)

from .response_models import (
    PropertyPrediction,
    MechanicalPredictions,
    BallisticPredictions,
    FeatureImportance,
    ModelInfo,
    PredictionResponse,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    PredictionStatus
)

from .exceptions import (
    CeramicArmorMLException,
    ValidationException,
    ModelException,
    PredictionException,
    DataProcessingException,
    FileProcessingException,
    RateLimitException,
    create_validation_error_response,
    create_http_exception,
    log_exception,
    validate_material_composition,
    validate_processing_parameters
)

__all__ = [
    # Request models
    "CompositionModel",
    "ProcessingModel", 
    "MicrostructureModel",
    "PredictionRequest",
    "BatchPredictionRequest",
    "PhaseDistribution",
    "InterfaceQuality",
    
    # Response models
    "PropertyPrediction",
    "MechanicalPredictions",
    "BallisticPredictions", 
    "FeatureImportance",
    "ModelInfo",
    "PredictionResponse",
    "BatchPredictionResponse",
    "ErrorResponse",
    "HealthResponse",
    "PredictionStatus",
    
    # Exceptions and utilities
    "CeramicArmorMLException",
    "ValidationException",
    "ModelException", 
    "PredictionException",
    "DataProcessingException",
    "FileProcessingException",
    "RateLimitException",
    "create_validation_error_response",
    "create_http_exception",
    "log_exception",
    "validate_material_composition",
    "validate_processing_parameters"
]