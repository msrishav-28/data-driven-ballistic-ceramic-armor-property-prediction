"""
Pydantic response models for the Ceramic Armor ML API.

This module defines the data models for API responses, including mechanical and
ballistic property predictions with uncertainty quantification and metadata.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union, Any
from datetime import datetime
from enum import Enum


class PredictionStatus(str, Enum):
    """Enumeration for prediction status."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    WARNING = "warning"


class PropertyPrediction(BaseModel):
    """
    Individual property prediction with uncertainty quantification.
    
    Contains the predicted value, units, confidence intervals, and uncertainty metrics.
    """
    value: float = Field(..., description="Predicted property value")
    unit: str = Field(..., description="Unit of measurement")
    confidence_interval: List[float] = Field(
        ..., 
        min_length=2, 
        max_length=2,
        description="95% confidence interval [lower, upper]"
    )
    uncertainty: float = Field(
        ..., 
        ge=0, 
        le=1,
        description="Relative uncertainty (0-1)"
    )
    prediction_quality: str = Field(
        ..., 
        pattern="^(excellent|good|fair|poor)$",
        description="Quality assessment of the prediction"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "value": 4.6,
                "unit": "MPa·m^0.5",
                "confidence_interval": [4.2, 5.0],
                "uncertainty": 0.15,
                "prediction_quality": "good"
            }
        }


class MechanicalPredictions(BaseModel):
    """
    Mechanical property predictions for ceramic armor materials.
    
    Includes fracture toughness, hardness, density, and elastic properties.
    """
    fracture_toughness: PropertyPrediction = Field(
        ..., 
        description="Fracture toughness prediction"
    )
    vickers_hardness: PropertyPrediction = Field(
        ..., 
        description="Vickers hardness prediction"
    )
    density: PropertyPrediction = Field(
        ..., 
        description="Material density prediction"
    )
    elastic_modulus: PropertyPrediction = Field(
        ..., 
        description="Elastic modulus prediction"
    )
    compressive_strength: Optional[PropertyPrediction] = Field(
        None, 
        description="Compressive strength prediction"
    )
    flexural_strength: Optional[PropertyPrediction] = Field(
        None, 
        description="Flexural strength prediction"
    )
    poisson_ratio: Optional[PropertyPrediction] = Field(
        None, 
        description="Poisson's ratio prediction"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "fracture_toughness": {
                    "value": 4.6,
                    "unit": "MPa·m^0.5",
                    "confidence_interval": [4.2, 5.0],
                    "uncertainty": 0.15,
                    "prediction_quality": "good"
                },
                "vickers_hardness": {
                    "value": 2800,
                    "unit": "HV",
                    "confidence_interval": [2650, 2950],
                    "uncertainty": 0.12,
                    "prediction_quality": "excellent"
                },
                "density": {
                    "value": 3.21,
                    "unit": "g/cm³",
                    "confidence_interval": [3.18, 3.24],
                    "uncertainty": 0.08,
                    "prediction_quality": "excellent"
                },
                "elastic_modulus": {
                    "value": 410,
                    "unit": "GPa",
                    "confidence_interval": [395, 425],
                    "uncertainty": 0.10,
                    "prediction_quality": "good"
                }
            }
        }


class BallisticPredictions(BaseModel):
    """
    Ballistic property predictions for ceramic armor materials.
    
    Includes V50 velocity, penetration resistance, and armor performance metrics.
    """
    v50_velocity: PropertyPrediction = Field(
        ..., 
        description="V50 ballistic limit velocity"
    )
    penetration_resistance: PropertyPrediction = Field(
        ..., 
        description="Penetration resistance index"
    )
    back_face_deformation: PropertyPrediction = Field(
        ..., 
        description="Back face deformation under impact"
    )
    multi_hit_capability: PropertyPrediction = Field(
        ..., 
        description="Multi-hit capability score"
    )
    energy_absorption: Optional[PropertyPrediction] = Field(
        None, 
        description="Energy absorption capacity"
    )
    damage_tolerance: Optional[PropertyPrediction] = Field(
        None, 
        description="Damage tolerance index"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "v50_velocity": {
                    "value": 850,
                    "unit": "m/s",
                    "confidence_interval": [820, 880],
                    "uncertainty": 0.18,
                    "prediction_quality": "good"
                },
                "penetration_resistance": {
                    "value": 7.2,
                    "unit": "dimensionless",
                    "confidence_interval": [6.8, 7.6],
                    "uncertainty": 0.22,
                    "prediction_quality": "fair"
                },
                "back_face_deformation": {
                    "value": 12.5,
                    "unit": "mm",
                    "confidence_interval": [11.2, 13.8],
                    "uncertainty": 0.25,
                    "prediction_quality": "fair"
                },
                "multi_hit_capability": {
                    "value": 0.75,
                    "unit": "probability",
                    "confidence_interval": [0.65, 0.85],
                    "uncertainty": 0.30,
                    "prediction_quality": "poor"
                }
            }
        }


class FeatureImportance(BaseModel):
    """
    Feature importance information for model interpretability.
    
    Provides insights into which material properties drive predictions.
    """
    name: str = Field(..., description="Feature name")
    importance: float = Field(
        ..., 
        ge=0, 
        le=1,
        description="Normalized importance score (0-1)"
    )
    category: str = Field(
        ..., 
        pattern="^(composition|processing|microstructure|derived)$",
        description="Feature category"
    )
    description: Optional[str] = Field(
        None, 
        description="Human-readable feature description"
    )
    shap_value: Optional[float] = Field(
        None, 
        description="SHAP value for this prediction"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "name": "SiC_content",
                "importance": 0.35,
                "category": "composition",
                "description": "Silicon Carbide content in material composition",
                "shap_value": 0.12
            }
        }


class ModelInfo(BaseModel):
    """
    Information about the ML model used for predictions.
    
    Includes version, performance metrics, and prediction metadata.
    """
    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "example": {
                "model_version": "v1.2.0",
                "model_type": "XGBoost Ensemble",
                "training_r2": 0.87,
                "validation_r2": 0.82,
                "prediction_time_ms": 45,
                "feature_count": 156,
                "training_samples": 2847,
                "last_updated": "2024-01-15T10:30:00Z"
            }
        }
    }
    
    model_version: str = Field(..., description="Model version identifier")
    model_type: str = Field(..., description="Type of ML model used")
    training_r2: float = Field(
        ..., 
        ge=0, 
        le=1,
        description="R² score on training data"
    )
    validation_r2: float = Field(
        ..., 
        ge=0, 
        le=1,
        description="R² score on validation data"
    )
    prediction_time_ms: int = Field(
        ..., 
        ge=0,
        description="Prediction time in milliseconds"
    )
    feature_count: int = Field(
        ..., 
        ge=1,
        description="Number of features used by the model"
    )
    training_samples: Optional[int] = Field(
        None, 
        ge=1,
        description="Number of training samples"
    )
    last_updated: Optional[datetime] = Field(
        None, 
        description="Model last update timestamp"
    )


class PredictionResponse(BaseModel):
    """
    Complete prediction response containing all prediction results and metadata.
    
    This is the main response model for individual predictions.
    """
    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "example": {
                "status": "success",
                "predictions": {
                    "fracture_toughness": {
                        "value": 4.6,
                        "unit": "MPa·m^0.5",
                        "confidence_interval": [4.2, 5.0],
                        "uncertainty": 0.15,
                        "prediction_quality": "good"
                    }
                },
                "feature_importance": [
                    {
                        "name": "SiC_content",
                        "importance": 0.35,
                        "category": "composition",
                        "description": "Silicon Carbide content"
                    }
                ],
                "model_info": {
                    "model_version": "v1.2.0",
                    "model_type": "XGBoost Ensemble",
                    "training_r2": 0.87,
                    "validation_r2": 0.82,
                    "prediction_time_ms": 45,
                    "feature_count": 156
                },
                "request_id": "req_123456789",
                "timestamp": "2024-01-15T14:30:00Z",
                "warnings": [],
                "processing_notes": []
            }
        }
    }
    
    status: PredictionStatus = Field(..., description="Prediction status")
    predictions: Optional[Union[MechanicalPredictions, BallisticPredictions, Dict[str, Any]]] = Field(
        None, 
        description="Property predictions"
    )
    feature_importance: Optional[List[FeatureImportance]] = Field(
        None, 
        description="Feature importance analysis"
    )
    model_info: ModelInfo = Field(..., description="Model information")
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    warnings: Optional[List[str]] = Field(
        None, 
        description="Any warnings generated during prediction"
    )
    processing_notes: Optional[List[str]] = Field(
        None, 
        description="Notes about data processing or assumptions"
    )


class BatchPredictionResponse(BaseModel):
    """
    Response model for batch prediction requests.
    
    Contains summary statistics and download information for batch results.
    """
    status: PredictionStatus = Field(..., description="Overall batch status")
    total_processed: int = Field(..., ge=0, description="Total number of materials processed")
    successful_predictions: int = Field(..., ge=0, description="Number of successful predictions")
    failed_predictions: int = Field(..., ge=0, description="Number of failed predictions")
    processing_time_seconds: float = Field(..., ge=0, description="Total processing time")
    download_url: Optional[str] = Field(None, description="URL to download results")
    file_size_mb: Optional[float] = Field(None, ge=0, description="Result file size in MB")
    expires_at: Optional[datetime] = Field(None, description="Download link expiration")
    summary_statistics: Optional[Dict[str, Any]] = Field(
        None, 
        description="Summary statistics of predictions"
    )
    request_id: str = Field(..., description="Unique batch request identifier")
    timestamp: datetime = Field(..., description="Batch completion timestamp")
    warnings: Optional[List[str]] = Field(None, description="Batch processing warnings")

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "status": "success",
                "total_processed": 150,
                "successful_predictions": 147,
                "failed_predictions": 3,
                "processing_time_seconds": 12.5,
                "download_url": "https://api.example.com/download/batch_results_123.csv",
                "file_size_mb": 2.3,
                "expires_at": "2024-01-16T14:30:00Z",
                "summary_statistics": {
                    "avg_fracture_toughness": 4.2,
                    "std_fracture_toughness": 0.8
                },
                "request_id": "batch_123456789",
                "timestamp": "2024-01-15T14:30:00Z",
                "warnings": ["3 materials had incomplete composition data"]
            }
        }


class ErrorResponse(BaseModel):
    """
    Error response model for API errors.
    
    Provides detailed error information for debugging and user feedback.
    """
    error: str = Field(..., description="Error type or category")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(
        None, 
        description="Additional error details"
    )
    field_errors: Optional[Dict[str, List[str]]] = Field(
        None, 
        description="Field-specific validation errors"
    )
    request_id: str = Field(..., description="Request identifier for tracking")
    timestamp: datetime = Field(..., description="Error timestamp")
    suggestion: Optional[str] = Field(
        None, 
        description="Suggested action to resolve the error"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid material composition data",
                "details": {
                    "total_composition": 1.15,
                    "max_allowed": 1.0
                },
                "field_errors": {
                    "composition": ["Total composition exceeds 100%"]
                },
                "request_id": "req_123456789",
                "timestamp": "2024-01-15T14:30:00Z",
                "suggestion": "Ensure all composition fractions sum to 1.0 or less"
            }
        }


class HealthResponse(BaseModel):
    """
    Health check response model.
    
    Provides system status and health information.
    """
    status: str = Field(..., pattern="^(healthy|degraded|unhealthy)$")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")
    uptime_seconds: float = Field(..., ge=0, description="System uptime in seconds")
    models_loaded: int = Field(..., ge=0, description="Number of ML models loaded")
    memory_usage_mb: Optional[float] = Field(None, ge=0, description="Memory usage in MB")
    cpu_usage_percent: Optional[float] = Field(None, ge=0, le=100, description="CPU usage percentage")
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T14:30:00Z",
                "version": "1.0.0",
                "uptime_seconds": 3600.5,
                "models_loaded": 4,
                "memory_usage_mb": 512.3,
                "cpu_usage_percent": 15.2
            }
        }
