"""
Core Prediction Service for Ceramic Armor ML API.

This service integrates the ML model loader, feature engineering, and prediction
capabilities to provide a unified interface for the FastAPI endpoints.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np

from ...ml.model_loader import get_model_loader
from ...ml.predictor import get_predictor
from ...feature_engineering.ceramic_features import get_feature_engineer
from ..models.request_models import PredictionRequest
from ..models.response_models import (
    PredictionResponse,
    MechanicalPredictions,
    BallisticPredictions,
    PropertyPrediction,
    FeatureImportance,
    ModelInfo,
    PredictionStatus
)
from ..models.exceptions import (
    PredictionException,
    ModelException,
    DataProcessingException
)

logger = logging.getLogger(__name__)


class CeramicArmorPredictionService:
    """
    Core prediction service that orchestrates feature engineering,
    model loading, and prediction generation for ceramic armor materials.
    """
    
    def __init__(self):
        """Initialize the prediction service."""
        self.feature_engineer = get_feature_engineer()
        self.model_loader = get_model_loader()
        self.predictor = get_predictor()
        
        # Performance tracking
        self.prediction_count = 0
        self.total_processing_time = 0.0
        
        logger.info("Initialized CeramicArmorPredictionService")
    
    async def predict_mechanical_properties(
        self, 
        request: PredictionRequest
    ) -> MechanicalPredictions:
        """
        Predict mechanical properties for ceramic armor materials.
        
        Args:
            request: PredictionRequest with material data
            
        Returns:
            MechanicalPredictions with property values and uncertainties
            
        Raises:
            PredictionException: If prediction fails
            DataProcessingException: If feature extraction fails
        """
        start_time = datetime.now()
        
        try:
            logger.info("Starting mechanical properties prediction")
            
            # Extract features
            features, feature_names = await self._extract_features_async(request)
            
            # Make predictions using the predictor
            prediction_result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.predictor.predict_mechanical_properties,
                features,
                request.include_uncertainty,
                request.include_feature_importance
            )
            
            # Convert to response model
            mechanical_predictions = self._convert_to_mechanical_predictions(
                prediction_result, request.include_uncertainty, request.include_feature_importance
            )
            
            # Update performance metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_metrics(processing_time)
            
            logger.info(f"Mechanical prediction completed in {processing_time:.3f}s")
            return mechanical_predictions
            
        except Exception as e:
            logger.error(f"Mechanical prediction failed: {e}")
            if isinstance(e, (PredictionException, DataProcessingException)):
                raise
            raise PredictionException(f"Mechanical prediction failed: {str(e)}")
    
    async def predict_ballistic_properties(
        self, 
        request: PredictionRequest
    ) -> BallisticPredictions:
        """
        Predict ballistic properties for ceramic armor materials.
        
        Args:
            request: PredictionRequest with material data
            
        Returns:
            BallisticPredictions with property values and uncertainties
            
        Raises:
            PredictionException: If prediction fails
            DataProcessingException: If feature extraction fails
        """
        start_time = datetime.now()
        
        try:
            logger.info("Starting ballistic properties prediction")
            
            # Extract features
            features, feature_names = await self._extract_features_async(request)
            
            # Make predictions using the predictor
            prediction_result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.predictor.predict_ballistic_properties,
                features,
                request.include_uncertainty,
                request.include_feature_importance
            )
            
            # Convert to response model
            ballistic_predictions = self._convert_to_ballistic_predictions(
                prediction_result, request.include_uncertainty, request.include_feature_importance
            )
            
            # Update performance metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_metrics(processing_time)
            
            logger.info(f"Ballistic prediction completed in {processing_time:.3f}s")
            return ballistic_predictions
            
        except Exception as e:
            logger.error(f"Ballistic prediction failed: {e}")
            if isinstance(e, (PredictionException, DataProcessingException)):
                raise
            raise PredictionException(f"Ballistic prediction failed: {str(e)}")
    
    async def predict_all_properties(
        self, 
        request: PredictionRequest
    ) -> PredictionResponse:
        """
        Predict both mechanical and ballistic properties.
        
        Args:
            request: PredictionRequest with material data
            
        Returns:
            PredictionResponse with all predictions and metadata
            
        Raises:
            PredictionException: If prediction fails
        """
        start_time = datetime.now()
        
        try:
            logger.info("Starting comprehensive property prediction")
            
            # Extract features once for both predictions
            features, feature_names = await self._extract_features_async(request)
            
            # Run both predictions concurrently
            mechanical_task = asyncio.create_task(
                self._predict_mechanical_async(features, request)
            )
            ballistic_task = asyncio.create_task(
                self._predict_ballistic_async(features, request)
            )
            
            mechanical_result, ballistic_result = await asyncio.gather(
                mechanical_task, ballistic_task
            )
            
            # Convert results
            mechanical_predictions = self._convert_to_mechanical_predictions(
                mechanical_result, request.include_uncertainty, request.include_feature_importance
            )
            ballistic_predictions = self._convert_to_ballistic_predictions(
                ballistic_result, request.include_uncertainty, request.include_feature_importance
            )
            
            # Combine feature importance if requested
            feature_importance = []
            if request.include_feature_importance:
                feature_importance = self._combine_feature_importance(
                    mechanical_result.get('feature_importance', []),
                    ballistic_result.get('feature_importance', [])
                )
            
            # Create response
            processing_time = (datetime.now() - start_time).total_seconds()
            response = PredictionResponse(
                mechanical=mechanical_predictions,
                ballistic=ballistic_predictions,
                feature_importance=feature_importance,
                processing_time_ms=processing_time * 1000,
                timestamp=datetime.now(),
                status=PredictionStatus.SUCCESS,
                model_info=await self._get_model_info()
            )
            
            # Update performance metrics
            self._update_performance_metrics(processing_time)
            
            logger.info(f"Comprehensive prediction completed in {processing_time:.3f}s")
            return response
            
        except Exception as e:
            logger.error(f"Comprehensive prediction failed: {e}")
            if isinstance(e, (PredictionException, DataProcessingException)):
                raise
            raise PredictionException(f"Comprehensive prediction failed: {str(e)}")
    
    async def get_service_status(self) -> Dict[str, Any]:
        """
        Get prediction service status and performance metrics.
        
        Returns:
            Dictionary with service status information
        """
        try:
            # Get model status
            model_status = await asyncio.get_event_loop().run_in_executor(
                None, self.predictor.get_model_status
            )
            
            # Calculate average processing time
            avg_processing_time = (
                self.total_processing_time / self.prediction_count 
                if self.prediction_count > 0 else 0
            )
            
            return {
                'status': 'healthy',
                'prediction_count': self.prediction_count,
                'average_processing_time_ms': avg_processing_time * 1000,
                'model_status': model_status,
                'feature_engineer_status': 'healthy',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get service status: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _extract_features_async(self, request: PredictionRequest) -> Tuple[np.ndarray, List[str]]:
        """Extract features asynchronously."""
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, self.feature_engineer.extract_features, request
            )
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise DataProcessingException(f"Feature extraction failed: {str(e)}")
    
    async def _predict_mechanical_async(
        self, 
        features: np.ndarray, 
        request: PredictionRequest
    ) -> Dict[str, Any]:
        """Run mechanical prediction asynchronously."""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            self.predictor.predict_mechanical_properties,
            features,
            request.include_uncertainty,
            request.include_feature_importance
        )
    
    async def _predict_ballistic_async(
        self, 
        features: np.ndarray, 
        request: PredictionRequest
    ) -> Dict[str, Any]:
        """Run ballistic prediction asynchronously."""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            self.predictor.predict_ballistic_properties,
            features,
            request.include_uncertainty,
            request.include_feature_importance
        )
    
    def _convert_to_mechanical_predictions(
        self, 
        result: Dict[str, Any], 
        include_uncertainty: bool,
        include_feature_importance: bool
    ) -> MechanicalPredictions:
        """Convert prediction result to MechanicalPredictions model."""
        predictions = result.get('predictions', {})
        
        # Convert individual predictions
        fracture_toughness = self._convert_property_prediction(
            predictions.get('fracture_toughness', {}), include_uncertainty
        )
        vickers_hardness = self._convert_property_prediction(
            predictions.get('vickers_hardness', {}), include_uncertainty
        )
        density = self._convert_property_prediction(
            predictions.get('density', {}), include_uncertainty
        )
        elastic_modulus = self._convert_property_prediction(
            predictions.get('elastic_modulus', {}), include_uncertainty
        )
        
        # Feature importance
        feature_importance = []
        if include_feature_importance:
            feature_importance = [
                FeatureImportance(name=fi['name'], importance=fi['importance'])
                for fi in result.get('feature_importance', [])
            ]
        
        return MechanicalPredictions(
            fracture_toughness=fracture_toughness,
            vickers_hardness=vickers_hardness,
            density=density,
            elastic_modulus=elastic_modulus,
            feature_importance=feature_importance,
            processing_time_ms=result.get('processing_time_ms', 0)
        )
    
    def _convert_to_ballistic_predictions(
        self, 
        result: Dict[str, Any], 
        include_uncertainty: bool,
        include_feature_importance: bool
    ) -> BallisticPredictions:
        """Convert prediction result to BallisticPredictions model."""
        predictions = result.get('predictions', {})
        
        # Convert individual predictions
        v50_velocity = self._convert_property_prediction(
            predictions.get('v50_velocity', {}), include_uncertainty
        )
        penetration_resistance = self._convert_property_prediction(
            predictions.get('penetration_resistance', {}), include_uncertainty
        )
        back_face_deformation = self._convert_property_prediction(
            predictions.get('back_face_deformation', {}), include_uncertainty
        )
        multi_hit_capability = self._convert_property_prediction(
            predictions.get('multi_hit_capability', {}), include_uncertainty
        )
        
        # Feature importance
        feature_importance = []
        if include_feature_importance:
            feature_importance = [
                FeatureImportance(name=fi['name'], importance=fi['importance'])
                for fi in result.get('feature_importance', [])
            ]
        
        return BallisticPredictions(
            v50_velocity=v50_velocity,
            penetration_resistance=penetration_resistance,
            back_face_deformation=back_face_deformation,
            multi_hit_capability=multi_hit_capability,
            feature_importance=feature_importance,
            processing_time_ms=result.get('processing_time_ms', 0)
        )
    
    def _convert_property_prediction(
        self, 
        prediction_data: Dict[str, Any], 
        include_uncertainty: bool
    ) -> PropertyPrediction:
        """Convert individual property prediction to PropertyPrediction model."""
        value = prediction_data.get('value', 0.0)
        unit = prediction_data.get('unit', 'unknown')
        
        confidence_interval = None
        uncertainty = None
        
        if include_uncertainty:
            uncertainty = prediction_data.get('uncertainty', 0.1)
            confidence_interval = prediction_data.get('confidence_interval', [value * 0.9, value * 1.1])
        
        return PropertyPrediction(
            value=value,
            unit=unit,
            uncertainty=uncertainty,
            confidence_interval=confidence_interval
        )
    
    def _combine_feature_importance(
        self, 
        mechanical_fi: List[Dict[str, Any]], 
        ballistic_fi: List[Dict[str, Any]]
    ) -> List[FeatureImportance]:
        """Combine feature importance from mechanical and ballistic predictions."""
        # Create a dictionary to aggregate importance scores
        combined_importance = {}
        
        # Add mechanical feature importance
        for fi in mechanical_fi:
            name = fi['name']
            importance = fi['importance']
            combined_importance[name] = combined_importance.get(name, 0) + importance * 0.5
        
        # Add ballistic feature importance
        for fi in ballistic_fi:
            name = fi['name']
            importance = fi['importance']
            combined_importance[name] = combined_importance.get(name, 0) + importance * 0.5
        
        # Convert to list and sort by importance
        feature_importance = [
            FeatureImportance(name=name, importance=importance)
            for name, importance in combined_importance.items()
        ]
        
        # Sort by importance (descending) and return top 15
        feature_importance.sort(key=lambda x: x.importance, reverse=True)
        return feature_importance[:15]
    
    async def _get_model_info(self) -> ModelInfo:
        """Get model information for the response."""
        try:
            model_status = await asyncio.get_event_loop().run_in_executor(
                None, self.predictor.get_model_status
            )
            
            return ModelInfo(
                loaded_models=len(model_status.get('available_models', [])),
                model_status=model_status.get('status', 'unknown'),
                cache_info=model_status.get('cache_info', {}),
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.warning(f"Failed to get model info: {e}")
            return ModelInfo(
                loaded_models=0,
                model_status='error',
                cache_info={},
                last_updated=datetime.now()
            )
    
    def _update_performance_metrics(self, processing_time: float):
        """Update internal performance metrics."""
        self.prediction_count += 1
        self.total_processing_time += processing_time


# Global prediction service instance
_prediction_service: Optional[CeramicArmorPredictionService] = None


def get_prediction_service() -> CeramicArmorPredictionService:
    """Get global prediction service instance."""
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = CeramicArmorPredictionService()
    return _prediction_service