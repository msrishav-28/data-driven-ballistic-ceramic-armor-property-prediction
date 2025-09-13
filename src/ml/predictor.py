"""
Ceramic Armor ML Predictor Service.

This module provides the main prediction service that uses the model loader
to make predictions for mechanical and ballistic properties of ceramic armor materials.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from sklearn.base import BaseEstimator

from .model_loader import get_model_loader, ModelMetadata
from ..config import get_settings

logger = logging.getLogger(__name__)


class CeramicArmorPredictor:
    """
    Main predictor service for ceramic armor property predictions.
    
    Handles both mechanical and ballistic property predictions using
    cached ML models with uncertainty quantification.
    """
    
    def __init__(self):
        """Initialize the predictor service."""
        self.model_loader = get_model_loader()
        self.settings = get_settings()
        
        # Model mappings for different property types
        self.mechanical_models = {
            'fracture_toughness': 'mechanical_fracture_toughness',
            'vickers_hardness': 'mechanical_vickers_hardness', 
            'density': 'mechanical_density',
            'elastic_modulus': 'mechanical_elastic_modulus'
        }
        
        self.ballistic_models = {
            'v50_velocity': 'ballistic_v50_velocity',
            'penetration_resistance': 'ballistic_penetration_resistance',
            'back_face_deformation': 'ballistic_back_face_deformation',
            'multi_hit_capability': 'ballistic_multi_hit_capability'
        }
        
        logger.info("Initialized CeramicArmorPredictor")
    
    def predict_mechanical_properties(
        self, 
        features: np.ndarray,
        include_uncertainty: bool = True,
        include_feature_importance: bool = True
    ) -> Dict[str, Any]:
        """
        Predict mechanical properties of ceramic armor materials.
        
        Args:
            features: Engineered features array
            include_uncertainty: Whether to include uncertainty quantification
            include_feature_importance: Whether to include feature importance
            
        Returns:
            Dictionary with predictions, uncertainties, and metadata
        """
        start_time = datetime.now()
        predictions = {}
        
        try:
            for property_name, model_name in self.mechanical_models.items():
                try:
                    # Load model
                    model, metadata = self.model_loader.load_model(model_name)
                    
                    # Make prediction
                    prediction = self._make_single_prediction(
                        model, features, property_name, include_uncertainty
                    )
                    
                    predictions[property_name] = prediction
                    
                except Exception as e:
                    logger.error(f"Failed to predict {property_name}: {e}")
                    predictions[property_name] = {
                        'value': None,
                        'error': str(e)
                    }
            
            # Add feature importance if requested
            feature_importance = []
            if include_feature_importance:
                feature_importance = self._calculate_feature_importance(
                    self.mechanical_models, features
                )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                'predictions': predictions,
                'feature_importance': feature_importance,
                'processing_time_ms': round(processing_time, 2),
                'timestamp': datetime.now().isoformat(),
                'model_type': 'mechanical'
            }
            
        except Exception as e:
            logger.error(f"Mechanical prediction failed: {e}")
            raise
    
    def predict_ballistic_properties(
        self,
        features: np.ndarray,
        include_uncertainty: bool = True,
        include_feature_importance: bool = True
    ) -> Dict[str, Any]:
        """
        Predict ballistic properties of ceramic armor materials.
        
        Args:
            features: Engineered features array
            include_uncertainty: Whether to include uncertainty quantification
            include_feature_importance: Whether to include feature importance
            
        Returns:
            Dictionary with predictions, uncertainties, and metadata
        """
        start_time = datetime.now()
        predictions = {}
        
        try:
            for property_name, model_name in self.ballistic_models.items():
                try:
                    # Load model
                    model, metadata = self.model_loader.load_model(model_name)
                    
                    # Make prediction
                    prediction = self._make_single_prediction(
                        model, features, property_name, include_uncertainty
                    )
                    
                    predictions[property_name] = prediction
                    
                except Exception as e:
                    logger.error(f"Failed to predict {property_name}: {e}")
                    predictions[property_name] = {
                        'value': None,
                        'error': str(e)
                    }
            
            # Add feature importance if requested
            feature_importance = []
            if include_feature_importance:
                feature_importance = self._calculate_feature_importance(
                    self.ballistic_models, features
                )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                'predictions': predictions,
                'feature_importance': feature_importance,
                'processing_time_ms': round(processing_time, 2),
                'timestamp': datetime.now().isoformat(),
                'model_type': 'ballistic'
            }
            
        except Exception as e:
            logger.error(f"Ballistic prediction failed: {e}")
            raise
    
    def predict_batch(
        self,
        features_batch: np.ndarray,
        prediction_type: str = 'both'
    ) -> List[Dict[str, Any]]:
        """
        Make batch predictions for multiple materials.
        
        Args:
            features_batch: Array of feature vectors (n_samples, n_features)
            prediction_type: 'mechanical', 'ballistic', or 'both'
            
        Returns:
            List of prediction results for each sample
        """
        results = []
        
        for i, features in enumerate(features_batch):
            sample_result = {
                'sample_id': i,
                'predictions': {}
            }
            
            try:
                if prediction_type in ['mechanical', 'both']:
                    mechanical_pred = self.predict_mechanical_properties(
                        features.reshape(1, -1),
                        include_uncertainty=False,
                        include_feature_importance=False
                    )
                    sample_result['predictions']['mechanical'] = mechanical_pred['predictions']
                
                if prediction_type in ['ballistic', 'both']:
                    ballistic_pred = self.predict_ballistic_properties(
                        features.reshape(1, -1),
                        include_uncertainty=False,
                        include_feature_importance=False
                    )
                    sample_result['predictions']['ballistic'] = ballistic_pred['predictions']
                
            except Exception as e:
                logger.error(f"Batch prediction failed for sample {i}: {e}")
                sample_result['error'] = str(e)
            
            results.append(sample_result)
        
        return results
    
    def get_model_status(self) -> Dict[str, Any]:
        """
        Get status of all loaded models.
        
        Returns:
            Dictionary with model status information
        """
        try:
            cache_info = self.model_loader.cache.get_cache_info()
            available_models = self.model_loader.list_available_models()
            
            # Check which required models are available
            required_models = list(self.mechanical_models.values()) + list(self.ballistic_models.values())
            missing_models = [model for model in required_models if model not in available_models]
            
            return {
                'status': 'healthy' if not missing_models else 'degraded',
                'cache_info': cache_info,
                'available_models': available_models,
                'required_models': required_models,
                'missing_models': missing_models,
                'mechanical_models': self.mechanical_models,
                'ballistic_models': self.ballistic_models,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get model status: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _make_single_prediction(
        self,
        model: BaseEstimator,
        features: np.ndarray,
        property_name: str,
        include_uncertainty: bool
    ) -> Dict[str, Any]:
        """Make a single property prediction with optional uncertainty."""
        try:
            # Ensure features are 2D
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Make prediction
            prediction = model.predict(features)[0]
            
            result = {
                'value': float(prediction),
                'unit': self._get_property_unit(property_name)
            }
            
            # Add uncertainty quantification if requested and supported
            if include_uncertainty:
                uncertainty_info = self._calculate_uncertainty(model, features, prediction)
                result.update(uncertainty_info)
            
            return result
            
        except Exception as e:
            logger.error(f"Single prediction failed for {property_name}: {e}")
            raise
    
    def _calculate_uncertainty(
        self,
        model: BaseEstimator,
        features: np.ndarray,
        prediction: float
    ) -> Dict[str, Any]:
        """Calculate prediction uncertainty and confidence intervals."""
        try:
            uncertainty_info = {
                'uncertainty': 0.1,  # Default 10% uncertainty
                'confidence_interval': [prediction * 0.9, prediction * 1.1]
            }
            
            # Try to get uncertainty from model if it supports it
            if hasattr(model, 'predict_proba'):
                # For classification models
                probabilities = model.predict_proba(features)[0]
                uncertainty_info['uncertainty'] = 1.0 - np.max(probabilities)
                
            elif hasattr(model, 'predict') and hasattr(model, 'estimators_'):
                # For ensemble models, use prediction variance
                try:
                    predictions = np.array([
                        estimator.predict(features)[0] 
                        for estimator in model.estimators_
                    ])
                    std = np.std(predictions)
                    uncertainty_info['uncertainty'] = std / np.abs(prediction) if prediction != 0 else std
                    uncertainty_info['confidence_interval'] = [
                        prediction - 1.96 * std,
                        prediction + 1.96 * std
                    ]
                except Exception:
                    pass
            
            return uncertainty_info
            
        except Exception as e:
            logger.warning(f"Failed to calculate uncertainty: {e}")
            return {
                'uncertainty': 0.15,
                'confidence_interval': [prediction * 0.85, prediction * 1.15]
            }
    
    def _calculate_feature_importance(
        self,
        model_dict: Dict[str, str],
        features: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Calculate aggregated feature importance across models."""
        try:
            importance_scores = {}
            
            for property_name, model_name in model_dict.items():
                try:
                    model, metadata = self.model_loader.load_model(model_name)
                    
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        for i, importance in enumerate(importances):
                            feature_name = f"feature_{i}"
                            if feature_name not in importance_scores:
                                importance_scores[feature_name] = []
                            importance_scores[feature_name].append(importance)
                            
                except Exception as e:
                    logger.warning(f"Failed to get feature importance for {property_name}: {e}")
            
            # Aggregate importance scores
            aggregated_importance = []
            for feature_name, scores in importance_scores.items():
                avg_importance = np.mean(scores)
                aggregated_importance.append({
                    'name': feature_name,
                    'importance': float(avg_importance)
                })
            
            # Sort by importance and return top features
            aggregated_importance.sort(key=lambda x: x['importance'], reverse=True)
            return aggregated_importance[:10]  # Top 10 features
            
        except Exception as e:
            logger.error(f"Failed to calculate feature importance: {e}")
            return []
    
    def _get_property_unit(self, property_name: str) -> str:
        """Get the unit for a given property."""
        units = {
            # Mechanical properties
            'fracture_toughness': 'MPa·m^0.5',
            'vickers_hardness': 'HV',
            'density': 'g/cm³',
            'elastic_modulus': 'GPa',
            # Ballistic properties
            'v50_velocity': 'm/s',
            'penetration_resistance': 'dimensionless',  # Updated to match response model
            'back_face_deformation': 'mm',
            'multi_hit_capability': 'probability'  # Updated to match response model
        }
        return units.get(property_name, 'unknown')


# Global predictor instance
_predictor: Optional[CeramicArmorPredictor] = None


def get_predictor() -> CeramicArmorPredictor:
    """Get global predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = CeramicArmorPredictor()
    return _predictor