"""
ML Model Startup and Initialization Utilities.

This module provides utilities for initializing ML models at application startup,
including health checks and graceful error handling.
"""

import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

from .model_loader import get_model_loader, initialize_models
from .predictor import get_predictor
from ..config import get_settings

logger = logging.getLogger(__name__)


class ModelStartupManager:
    """
    Manages ML model initialization and startup procedures.
    
    Handles model loading, health checks, and provides startup status
    for the FastAPI application.
    """
    
    def __init__(self):
        """Initialize the startup manager."""
        self.settings = get_settings()
        self.startup_status: Dict[str, Any] = {
            'initialized': False,
            'startup_time': None,
            'models_loaded': 0,
            'errors': [],
            'last_health_check': None
        }
        
    async def startup_models(self) -> Dict[str, Any]:
        """
        Initialize all ML models at application startup.
        
        Returns:
            Dictionary with startup results and status
        """
        start_time = datetime.now()
        logger.info("Starting ML model initialization...")
        
        try:
            # Initialize model loader and load models
            initialization_result = await asyncio.get_event_loop().run_in_executor(
                None, initialize_models
            )
            
            # Initialize predictor
            predictor = get_predictor()
            
            # Initialize feature engineer
            from ..feature_engineering import get_feature_engineer
            feature_engineer = get_feature_engineer()
            
            # Initialize prediction service
            from ..api.services import get_prediction_service
            prediction_service = get_prediction_service()
            
            # Perform health check
            health_check = await self.health_check()
            
            # Update startup status
            self.startup_status.update({
                'initialized': True,
                'startup_time': (datetime.now() - start_time).total_seconds(),
                'models_loaded': initialization_result.get('loaded_models', 0),
                'initialization_result': initialization_result,
                'health_check': health_check,
                'services_initialized': {
                    'model_loader': True,
                    'predictor': True,
                    'feature_engineer': True,
                    'prediction_service': True
                },
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"ML models and services initialized successfully in {self.startup_status['startup_time']:.2f}s")
            return self.startup_status
            
        except Exception as e:
            error_msg = f"Failed to initialize ML models: {e}"
            logger.error(error_msg)
            
            self.startup_status.update({
                'initialized': False,
                'startup_time': (datetime.now() - start_time).total_seconds(),
                'errors': [error_msg],
                'timestamp': datetime.now().isoformat()
            })
            
            return self.startup_status
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check of ML system.
        
        Returns:
            Dictionary with health check results
        """
        try:
            # Get model loader health check
            model_loader = get_model_loader()
            loader_health = await asyncio.get_event_loop().run_in_executor(
                None, model_loader.health_check
            )
            
            # Get predictor status
            predictor = get_predictor()
            predictor_status = await asyncio.get_event_loop().run_in_executor(
                None, predictor.get_model_status
            )
            
            # Combine health information
            health_status = {
                'overall_status': 'healthy',
                'model_loader': loader_health,
                'predictor': predictor_status,
                'timestamp': datetime.now().isoformat()
            }
            
            # Determine overall status
            if (loader_health.get('status') != 'healthy' or 
                predictor_status.get('status') != 'healthy'):
                health_status['overall_status'] = 'degraded'
            
            if (loader_health.get('status') == 'unhealthy' or 
                predictor_status.get('status') == 'error'):
                health_status['overall_status'] = 'unhealthy'
            
            self.startup_status['last_health_check'] = health_status
            return health_status
            
        except Exception as e:
            error_msg = f"Health check failed: {e}"
            logger.error(error_msg)
            
            health_status = {
                'overall_status': 'unhealthy',
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            }
            
            self.startup_status['last_health_check'] = health_status
            return health_status
    
    def get_startup_status(self) -> Dict[str, Any]:
        """Get current startup status."""
        return self.startup_status.copy()
    
    async def shutdown_models(self) -> Dict[str, Any]:
        """
        Gracefully shutdown ML models and clean up resources.
        
        Returns:
            Dictionary with shutdown results
        """
        logger.info("Shutting down ML models...")
        
        try:
            # Clear model cache
            model_loader = get_model_loader()
            model_loader.cache.clear()
            
            shutdown_result = {
                'status': 'success',
                'message': 'ML models shut down successfully',
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info("ML models shut down successfully")
            return shutdown_result
            
        except Exception as e:
            error_msg = f"Failed to shutdown ML models: {e}"
            logger.error(error_msg)
            
            return {
                'status': 'error',
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            }


# Global startup manager instance
_startup_manager: Optional[ModelStartupManager] = None


def get_startup_manager() -> ModelStartupManager:
    """Get global startup manager instance."""
    global _startup_manager
    if _startup_manager is None:
        _startup_manager = ModelStartupManager()
    return _startup_manager


# Convenience functions for FastAPI lifespan events
async def startup_event() -> Dict[str, Any]:
    """FastAPI startup event handler for ML models."""
    manager = get_startup_manager()
    return await manager.startup_models()


async def shutdown_event() -> Dict[str, Any]:
    """FastAPI shutdown event handler for ML models."""
    manager = get_startup_manager()
    return await manager.shutdown_models()


async def health_check_event() -> Dict[str, Any]:
    """Health check endpoint handler."""
    manager = get_startup_manager()
    return await manager.health_check()