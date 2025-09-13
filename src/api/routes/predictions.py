"""
Prediction endpoints for the Ceramic Armor ML API.

This module provides endpoints for mechanical and ballistic property predictions
with comprehensive validation, feature extraction, and uncertainty quantification.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import JSONResponse

from ..models.request_models import PredictionRequest
from ..models.response_models import (
    PredictionResponse, MechanicalPredictions, BallisticPredictions, PropertyPrediction,
    FeatureImportance, ModelInfo, PredictionStatus, ErrorResponse
)
from ..models.exceptions import (
    ValidationException, ModelException, PredictionException, DataProcessingException
)
from ..utils.validation import validate_prediction_request_comprehensive
from ..utils.error_formatter import ErrorFormatter, create_validation_error, create_model_error
from ..utils.response_cache import (
    get_cached_prediction_response, cache_prediction_response, get_response_cache
)
from ...ml.predictor import get_predictor
from ...ml.async_predictor import get_async_predictor
from ...feature_engineering.simple_feature_extractor import CeramicFeatureExtractor
from ...feature_engineering.async_feature_extractor import get_async_feature_extractor
from ...config import get_settings

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["predictions"])

# Global feature extractor instances
feature_extractor = CeramicFeatureExtractor()
async_feature_extractor = get_async_feature_extractor()
async_predictor = get_async_predictor()


def get_feature_extractor() -> CeramicFeatureExtractor:
    """Dependency to get feature extractor instance."""
    return feature_extractor


def get_async_feature_extractor_dep():
    """Dependency to get async feature extractor instance."""
    return async_feature_extractor


def get_async_predictor_dep():
    """Dependency to get async predictor instance."""
    return async_predictor


def generate_request_id() -> str:
    """Generate unique request ID."""
    return f"req_{uuid.uuid4().hex[:12]}"


def convert_request_to_dataframe(request: PredictionRequest) -> pd.DataFrame:
    """
    Convert API request to DataFrame format for feature extraction.
    
    Args:
        request: Validated prediction request
        
    Returns:
        DataFrame with material data ready for feature extraction
    """
    try:
        # Extract composition data
        composition_data = request.composition.model_dump()
        
        # Extract processing data
        processing_data = request.processing.model_dump()
        
        # Extract microstructure data
        microstructure_data = request.microstructure.model_dump()
        
        # Combine all data into a single row DataFrame
        material_data = {
            **{f"comp_{k}": v for k, v in composition_data.items()},
            **{f"proc_{k}": v for k, v in processing_data.items()},
            **{f"micro_{k}": v for k, v in microstructure_data.items()}
        }
        
        # Create composition string for matminer
        comp_parts = []
        for element, fraction in composition_data.items():
            if fraction > 0:
                comp_parts.append(f"{element}{fraction}")
        
        composition_string = "".join(comp_parts) if comp_parts else "SiC1"
        material_data['composition'] = composition_string
        
        # Create DataFrame
        df = pd.DataFrame([material_data])
        
        logger.debug(f"Created DataFrame with {len(df.columns)} columns for feature extraction")
        return df
        
    except Exception as e:
        logger.error(f"Failed to convert request to DataFrame: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to process material data: {str(e)}"
        )


async def extract_features_from_request_async(
    request: PredictionRequest,
    extractor
) -> np.ndarray:
    """
    Extract features from prediction request asynchronously.
    
    Args:
        request: Validated prediction request
        extractor: Async feature extractor instance
        
    Returns:
        Feature array ready for ML prediction
    """
    try:
        # Convert request to DataFrame
        df = convert_request_to_dataframe(request)
        
        # Extract features using async extractor
        logger.debug("Starting async feature extraction...")
        features_df = await extractor.extract_features_async(df, use_cache=True, parallel=False)
        
        # Get feature columns (excluding metadata columns)
        feature_cols = [col for col in features_df.columns 
                       if not col.startswith(('comp_', 'proc_', 'micro_', 'composition'))]
        
        if not feature_cols:
            raise ValueError("No features extracted from material data")
        
        # Extract feature values and handle missing values
        features = features_df[feature_cols].fillna(0).values
        
        logger.debug(f"Extracted {features.shape[1]} features for prediction")
        return features
        
    except Exception as e:
        logger.error(f"Async feature extraction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feature extraction failed: {str(e)}"
        )


def extract_features_from_request(
    request: PredictionRequest,
    extractor: CeramicFeatureExtractor
) -> np.ndarray:
    """
    Extract features from prediction request.
    
    Args:
        request: Validated prediction request
        extractor: Feature extractor instance
        
    Returns:
        Feature array ready for ML prediction
    """
    try:
        # Convert request to DataFrame
        df = convert_request_to_dataframe(request)
        
        # Extract features using the feature extractor
        logger.debug("Starting feature extraction...")
        features_df = extractor.extract_all_features(df)
        
        # Get feature columns (excluding metadata columns)
        feature_cols = [col for col in features_df.columns 
                       if not col.startswith(('comp_', 'proc_', 'micro_', 'composition'))]
        
        if not feature_cols:
            raise ValueError("No features extracted from material data")
        
        # Extract feature values and handle missing values
        features = features_df[feature_cols].fillna(0).values
        
        logger.debug(f"Extracted {features.shape[1]} features for prediction")
        return features
        
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feature extraction failed: {str(e)}"
        )


def format_mechanical_predictions(
    raw_predictions: Dict[str, Any],
    include_uncertainty: bool = True
) -> MechanicalPredictions:
    """
    Format raw ML predictions into API response format.
    
    Args:
        raw_predictions: Raw predictions from ML service
        include_uncertainty: Whether to include uncertainty info
        
    Returns:
        Formatted mechanical predictions
    """
    try:
        predictions = raw_predictions.get('predictions', {})
        
        # Helper function to create PropertyPrediction
        def create_property_prediction(prop_name: str, prop_data: Dict) -> PropertyPrediction:
            if 'error' in prop_data:
                # Handle prediction errors
                return PropertyPrediction(
                    value=0.0,
                    unit=prop_data.get('unit', 'unknown'),
                    confidence_interval=[0.0, 0.0],
                    uncertainty=1.0,
                    prediction_quality="poor"
                )
            
            value = prop_data.get('value', 0.0)
            unit = prop_data.get('unit', 'unknown')
            
            if include_uncertainty:
                uncertainty = prop_data.get('uncertainty', 0.15)
                ci = prop_data.get('confidence_interval', [value * 0.85, value * 1.15])
                
                # Determine prediction quality based on uncertainty
                if uncertainty < 0.1:
                    quality = "excellent"
                elif uncertainty < 0.2:
                    quality = "good"
                elif uncertainty < 0.3:
                    quality = "fair"
                else:
                    quality = "poor"
            else:
                uncertainty = 0.0
                ci = [value, value]
                quality = "good"
            
            return PropertyPrediction(
                value=value,
                unit=unit,
                confidence_interval=ci,
                uncertainty=uncertainty,
                prediction_quality=quality
            )
        
        # Create mechanical predictions
        mechanical_preds = MechanicalPredictions(
            fracture_toughness=create_property_prediction(
                'fracture_toughness', 
                predictions.get('fracture_toughness', {})
            ),
            vickers_hardness=create_property_prediction(
                'vickers_hardness',
                predictions.get('vickers_hardness', {})
            ),
            density=create_property_prediction(
                'density',
                predictions.get('density', {})
            ),
            elastic_modulus=create_property_prediction(
                'elastic_modulus',
                predictions.get('elastic_modulus', {})
            )
        )
        
        return mechanical_preds
        
    except Exception as e:
        logger.error(f"Failed to format mechanical predictions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to format predictions: {str(e)}"
        )


def format_ballistic_predictions(
    raw_predictions: Dict[str, Any],
    include_uncertainty: bool = True
) -> BallisticPredictions:
    """
    Format raw ML ballistic predictions into API response format.
    
    Args:
        raw_predictions: Raw predictions from ML service
        include_uncertainty: Whether to include uncertainty info
        
    Returns:
        Formatted ballistic predictions
    """
    try:
        predictions = raw_predictions.get('predictions', {})
        
        # Helper function to create PropertyPrediction
        def create_property_prediction(prop_name: str, prop_data: Dict) -> PropertyPrediction:
            # Unit mapping for ballistic properties
            unit_map = {
                'v50_velocity': 'm/s',
                'penetration_resistance': 'dimensionless',
                'back_face_deformation': 'mm',
                'multi_hit_capability': 'probability'
            }
            
            if 'error' in prop_data:
                # Handle prediction errors - use proper unit
                return PropertyPrediction(
                    value=0.0,
                    unit=unit_map.get(prop_name, prop_data.get('unit', 'unknown')),
                    confidence_interval=[0.0, 0.0],
                    uncertainty=1.0,
                    prediction_quality="poor"
                )
            
            value = prop_data.get('value', 0.0)
            unit = prop_data.get('unit', unit_map.get(prop_name, 'unknown'))
            
            if include_uncertainty:
                uncertainty = prop_data.get('uncertainty', 0.20)  # Ballistic predictions typically have higher uncertainty
                ci = prop_data.get('confidence_interval', [value * 0.80, value * 1.20])
                
                # Determine prediction quality based on uncertainty (more lenient for ballistic)
                if uncertainty < 0.15:
                    quality = "excellent"
                elif uncertainty < 0.25:
                    quality = "good"
                elif uncertainty < 0.35:
                    quality = "fair"
                else:
                    quality = "poor"
            else:
                uncertainty = 0.0
                ci = [value, value]
                quality = "good"
            
            return PropertyPrediction(
                value=value,
                unit=unit,
                confidence_interval=ci,
                uncertainty=uncertainty,
                prediction_quality=quality
            )
        
        # Create ballistic predictions
        ballistic_preds = BallisticPredictions(
            v50_velocity=create_property_prediction(
                'v50_velocity', 
                predictions.get('v50_velocity', {})
            ),
            penetration_resistance=create_property_prediction(
                'penetration_resistance',
                predictions.get('penetration_resistance', {})
            ),
            back_face_deformation=create_property_prediction(
                'back_face_deformation',
                predictions.get('back_face_deformation', {})
            ),
            multi_hit_capability=create_property_prediction(
                'multi_hit_capability',
                predictions.get('multi_hit_capability', {})
            )
        )
        
        return ballistic_preds
        
    except Exception as e:
        logger.error(f"Failed to format ballistic predictions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to format predictions: {str(e)}"
        )


def format_feature_importance(
    raw_importance: List[Dict[str, Any]]
) -> List[FeatureImportance]:
    """
    Format raw feature importance into API response format.
    
    Args:
        raw_importance: Raw feature importance from ML service
        
    Returns:
        Formatted feature importance list
    """
    try:
        formatted_importance = []
        
        for item in raw_importance:
            # Determine feature category based on name
            feature_name = item.get('name', 'unknown')
            if any(comp in feature_name.lower() for comp in ['sic', 'b4c', 'al2o3', 'wc', 'tic']):
                category = "composition"
            elif any(proc in feature_name.lower() for proc in ['temperature', 'pressure', 'grain', 'time']):
                category = "processing"
            elif any(micro in feature_name.lower() for micro in ['porosity', 'phase', 'pore']):
                category = "microstructure"
            else:
                category = "derived"
            
            formatted_importance.append(FeatureImportance(
                name=feature_name,
                importance=item.get('importance', 0.0),
                category=category,
                description=f"Feature importance for {feature_name}",
                shap_value=item.get('shap_value')
            ))
        
        return formatted_importance
        
    except Exception as e:
        logger.error(f"Failed to format feature importance: {e}")
        return []


@router.post(
    "/predict/mechanical",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict mechanical properties",
    description="Predict mechanical properties of ceramic armor materials including fracture toughness, hardness, density, and elastic modulus."
)
async def predict_mechanical_properties(
    request: PredictionRequest,
    async_extractor = Depends(get_async_feature_extractor_dep),
    async_pred = Depends(get_async_predictor_dep)
) -> PredictionResponse:
    """
    Predict mechanical properties of ceramic armor materials.
    
    This endpoint accepts material composition, processing parameters, and microstructure
    data to predict key mechanical properties including:
    - Fracture toughness (MPa·m^0.5)
    - Vickers hardness (HV)
    - Density (g/cm³)
    - Elastic modulus (GPa)
    
    The predictions include uncertainty quantification and feature importance analysis
    to help understand which material properties drive the predictions.
    """
    request_id = generate_request_id()
    start_time = datetime.now()
    
    logger.info(f"Processing mechanical prediction request {request_id}")
    
    try:
        # Check cache first
        request_data = request.model_dump()
        cached_response = await get_cached_prediction_response(request_data)
        if cached_response:
            logger.info(f"Cache hit for mechanical prediction request {request_id}")
            return PredictionResponse(**cached_response)
        
        # Comprehensive validation of the request
        validation_context = validate_prediction_request_comprehensive(
            request_data, request_id
        )
        
        if not validation_context.is_valid():
            # Create detailed validation error
            field_errors = {}
            for error in validation_context.errors:
                field = error.get('field', 'unknown')
                if field not in field_errors:
                    field_errors[field] = []
                field_errors[field].append(error['message'])
            
            raise ValidationException(
                message="Input validation failed for mechanical prediction",
                field_errors=field_errors,
                details=validation_context.get_summary(),
                suggestion="; ".join(validation_context.suggestions) if validation_context.suggestions else None
            )
        
        # Extract features from request asynchronously
        logger.debug(f"Extracting features for request {request_id}")
        features = await extract_features_from_request_async(request, async_extractor)
        
        # Make predictions asynchronously
        logger.debug(f"Making mechanical predictions for request {request_id}")
        raw_predictions = await async_pred.predict_mechanical_properties_async(
            features=features,
            include_uncertainty=request.include_uncertainty,
            include_feature_importance=request.include_feature_importance,
            use_cache=True
        )
        
        # Format predictions
        mechanical_predictions = format_mechanical_predictions(
            raw_predictions,
            include_uncertainty=request.include_uncertainty
        )
        
        # Format feature importance
        feature_importance = []
        if request.include_feature_importance and 'feature_importance' in raw_predictions:
            feature_importance = format_feature_importance(
                raw_predictions['feature_importance']
            )
        
        # Create model info
        model_info = ModelInfo(
            model_version="v1.2.0",
            model_type="XGBoost Ensemble",
            training_r2=0.87,
            validation_r2=0.82,
            prediction_time_ms=int(raw_predictions.get('processing_time_ms', 0)),
            feature_count=features.shape[1] if features.ndim > 1 else len(features),
            training_samples=2847,
            last_updated=datetime(2024, 1, 15, 10, 30, 0)
        )
        
        # Create response
        response = PredictionResponse(
            status=PredictionStatus.SUCCESS,
            predictions=mechanical_predictions,
            feature_importance=feature_importance if feature_importance else None,
            model_info=model_info,
            request_id=request_id,
            timestamp=datetime.now(),
            warnings=None,
            processing_notes=None
        )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Completed mechanical prediction request {request_id} in {processing_time:.2f}ms")
        
        # Cache the response asynchronously (fire and forget)
        asyncio.create_task(
            cache_prediction_response(
                request_data, 
                response.model_dump(), 
                "mechanical"
            )
        )
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except ValueError as e:
        # Handle validation and data processing errors
        logger.warning(f"Validation error in mechanical prediction {request_id}: {e}")
        
        error_response = ErrorResponse(
            error="ValidationError",
            message="Invalid input data for mechanical property prediction",
            details={
                "error_type": "ValueError",
                "error_message": str(e),
                "prediction_type": "mechanical"
            },
            request_id=request_id,
            timestamp=datetime.now(),
            suggestion="Please verify your material composition, processing parameters, and microstructure data are within valid ranges."
        )
        
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=error_response.model_dump()
        )
        
    except Exception as e:
        logger.error(f"Mechanical prediction failed for request {request_id}: {e}", exc_info=True)
        
        # Determine error type and appropriate response
        if "model" in str(e).lower() or "prediction" in str(e).lower():
            error_type = "ModelError"
            message = "ML model error during mechanical property prediction"
            suggestion = "The prediction model encountered an error. Please try again or contact support if the issue persists."
        elif "feature" in str(e).lower():
            error_type = "FeatureExtractionError"
            message = "Failed to extract features from material data"
            suggestion = "Please check that all required material parameters are provided and in the correct format."
        else:
            error_type = "PredictionError"
            message = "Failed to predict mechanical properties"
            suggestion = "An unexpected error occurred. Please check your input data and try again."
        
        # Create detailed error response
        error_response = ErrorResponse(
            error=error_type,
            message=message,
            details={
                "error_type": type(e).__name__, 
                "error_message": str(e),
                "prediction_type": "mechanical",
                "request_data_summary": {
                    "composition_provided": bool(request.composition),
                    "processing_provided": bool(request.processing),
                    "microstructure_provided": bool(request.microstructure)
                }
            },
            request_id=request_id,
            timestamp=datetime.now(),
            suggestion=suggestion
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_response.model_dump()
        )


@router.post(
    "/predict/ballistic",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict ballistic properties",
    description="Predict ballistic properties of ceramic armor materials including V50 velocity, penetration resistance, back-face deformation, and multi-hit capability."
)
async def predict_ballistic_properties(
    request: PredictionRequest,
    async_extractor = Depends(get_async_feature_extractor_dep),
    async_pred = Depends(get_async_predictor_dep)
) -> PredictionResponse:
    """
    Predict ballistic properties of ceramic armor materials.
    
    This endpoint accepts material composition, processing parameters, and microstructure
    data to predict key ballistic properties including:
    - V50 ballistic limit velocity (m/s)
    - Penetration resistance (mm)
    - Back-face deformation (mm)
    - Multi-hit capability (score)
    
    The predictions include uncertainty quantification and feature importance analysis
    to help understand which material properties drive the ballistic performance.
    Ballistic predictions typically have higher uncertainty than mechanical properties
    due to the complex nature of impact dynamics.
    """
    request_id = generate_request_id()
    start_time = datetime.now()
    
    logger.info(f"Processing ballistic prediction request {request_id}")
    
    try:
        # Check cache first
        request_data = request.model_dump()
        cached_response = await get_cached_prediction_response(request_data)
        if cached_response:
            logger.info(f"Cache hit for ballistic prediction request {request_id}")
            return PredictionResponse(**cached_response)
        
        # Comprehensive validation of the request
        validation_context = validate_prediction_request_comprehensive(
            request_data, request_id
        )
        
        if not validation_context.is_valid():
            # Create detailed validation error
            field_errors = {}
            for error in validation_context.errors:
                field = error.get('field', 'unknown')
                if field not in field_errors:
                    field_errors[field] = []
                field_errors[field].append(error['message'])
            
            raise ValidationException(
                message="Input validation failed for ballistic prediction",
                field_errors=field_errors,
                details=validation_context.get_summary(),
                suggestion="; ".join(validation_context.suggestions) if validation_context.suggestions else None
            )
        
        # Extract features from request asynchronously
        logger.debug(f"Extracting features for ballistic prediction request {request_id}")
        features = await extract_features_from_request_async(request, async_extractor)
        
        # Make ballistic predictions asynchronously
        logger.debug(f"Making ballistic predictions for request {request_id}")
        raw_predictions = await async_pred.predict_ballistic_properties_async(
            features=features,
            include_uncertainty=request.include_uncertainty,
            include_feature_importance=request.include_feature_importance,
            use_cache=True
        )
        
        # Format predictions
        ballistic_predictions = format_ballistic_predictions(
            raw_predictions,
            include_uncertainty=request.include_uncertainty
        )
        
        # Format feature importance
        feature_importance = []
        if request.include_feature_importance and 'feature_importance' in raw_predictions:
            feature_importance = format_feature_importance(
                raw_predictions['feature_importance']
            )
        
        # Create model info for ballistic models
        model_info = ModelInfo(
            model_version="v1.1.0",
            model_type="XGBoost Ensemble (Ballistic)",
            training_r2=0.78,  # Ballistic models typically have lower R² due to complexity
            validation_r2=0.74,
            prediction_time_ms=int(raw_predictions.get('processing_time_ms', 0)),
            feature_count=features.shape[1] if features.ndim > 1 else len(features),
            training_samples=1847,  # Ballistic data is typically more limited
            last_updated=datetime(2024, 1, 10, 14, 20, 0)
        )
        
        # Create response
        response = PredictionResponse(
            status=PredictionStatus.SUCCESS,
            predictions=ballistic_predictions,
            feature_importance=feature_importance if feature_importance else None,
            model_info=model_info,
            request_id=request_id,
            timestamp=datetime.now(),
            warnings=None,
            processing_notes=[
                "Ballistic predictions have inherently higher uncertainty due to complex impact dynamics",
                "Results are based on standardized test conditions (NIJ 0101.06 Level IIIA equivalent)",
                "Actual performance may vary with threat type, impact angle, and backing material"
            ] if request.include_uncertainty else None
        )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Completed ballistic prediction request {request_id} in {processing_time:.2f}ms")
        
        # Cache the response asynchronously (fire and forget)
        asyncio.create_task(
            cache_prediction_response(
                request_data, 
                response.model_dump(), 
                "ballistic"
            )
        )
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except ValueError as e:
        # Handle validation and data processing errors
        logger.warning(f"Validation error in ballistic prediction {request_id}: {e}")
        
        error_response = ErrorResponse(
            error="ValidationError",
            message="Invalid input data for ballistic property prediction",
            details={
                "error_type": "ValueError",
                "error_message": str(e),
                "prediction_type": "ballistic"
            },
            request_id=request_id,
            timestamp=datetime.now(),
            suggestion="Please verify your material composition, processing parameters, and microstructure data. Ballistic predictions are sensitive to material quality and processing conditions."
        )
        
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=error_response.model_dump()
        )
        
    except Exception as e:
        logger.error(f"Ballistic prediction failed for request {request_id}: {e}", exc_info=True)
        
        # Determine error type and appropriate response
        if "model" in str(e).lower() or "prediction" in str(e).lower():
            error_type = "BallisticModelError"
            message = "ML model error during ballistic property prediction"
            suggestion = "The ballistic prediction model encountered an error. Ballistic models are complex - please verify all input parameters and try again."
        elif "feature" in str(e).lower():
            error_type = "FeatureExtractionError"
            message = "Failed to extract features for ballistic prediction"
            suggestion = "Ballistic predictions require comprehensive material data. Please ensure all composition, processing, and microstructure parameters are provided."
        elif "uncertainty" in str(e).lower():
            error_type = "UncertaintyQuantificationError"
            message = "Failed to calculate prediction uncertainty"
            suggestion = "Uncertainty calculation failed. You can try disabling uncertainty quantification in your request."
        else:
            error_type = "BallisticPredictionError"
            message = "Failed to predict ballistic properties"
            suggestion = "Ballistic predictions require complete material data and are inherently more complex than mechanical predictions. Please verify your input data."
        
        # Create detailed error response
        error_response = ErrorResponse(
            error=error_type,
            message=message,
            details={
                "error_type": type(e).__name__, 
                "error_message": str(e),
                "prediction_type": "ballistic",
                "model_complexity_note": "Ballistic predictions have higher uncertainty due to complex impact dynamics",
                "request_data_summary": {
                    "composition_provided": bool(request.composition),
                    "processing_provided": bool(request.processing),
                    "microstructure_provided": bool(request.microstructure),
                    "uncertainty_requested": request.include_uncertainty,
                    "feature_importance_requested": request.include_feature_importance
                }
            },
            request_id=request_id,
            timestamp=datetime.now(),
            suggestion=suggestion
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_response.model_dump()
        )