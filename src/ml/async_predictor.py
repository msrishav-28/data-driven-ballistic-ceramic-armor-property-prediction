"""
Async optimized ML predictor service for ceramic armor predictions.

This module provides async prediction capabilities with caching,
memory optimization, and parallel processing for improved performance.
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import time

from .predictor import CeramicArmorPredictor
from .model_loader import get_model_loader
from ..config import get_settings

logger = logging.getLogger(__name__)


class AsyncCeramicArmorPredictor:
    """
    Async optimized predictor service for ceramic armor properties.
    
    Provides async prediction capabilities with caching, parallel processing,
    and memory optimization for production environments.
    """
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize async predictor service.
        
        Args:
            max_workers: Maximum number of worker threads for CPU-bound tasks
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Base predictor
        self.base_predictor = CeramicArmorPredictor()
        
        # Prediction cache (simple in-memory cache)
        self._prediction_cache: Dict[str, Tuple[Dict[str, Any], datetime]] = {}
        self._cache_ttl_seconds = 3600  # 1 hour
        
        # Performance metrics
        self._prediction_times: List[float] = []
        self._cache_hits = 0
        self._cache_misses = 0
        
        logger.info(f"Initialized AsyncCeramicArmorPredictor with {max_workers} workers")
    
    async def predict_mechanical_properties_async(
        self,
        features: np.ndarray,
        include_uncertainty: bool = True,
        include_feature_importance: bool = True,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Predict mechanical properties asynchronously.
        
        Args:
            features: Engineered features array
            include_uncertainty: Whether to include uncertainty quantification
            include_feature_importance: Whether to include feature importance
            use_cache: Whether to use prediction caching
            
        Returns:
            Dictionary with predictions, uncertainties, and metadata
        """
        start_time = time.time()
        
        # Generate cache key if caching is enabled
        cache_key = None
        if use_cache:
            cache_key = self._generate_prediction_cache_key(
                features, "mechanical", include_uncertainty, include_feature_importance
            )
            
            # Check cache
            cached_result = self._get_cached_prediction(cache_key)
            if cached_result is not None:
                self._cache_hits += 1
                logger.debug("Cache hit for mechanical prediction")
                return cached_result
            
            self._cache_misses += 1
        
        # Run prediction in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self._predict_mechanical_sync,
            features,
            include_uncertainty,
            include_feature_importance
        )
        
        # Cache result if caching is enabled
        if use_cache and cache_key:
            self._cache_prediction(cache_key, result)
        
        # Record performance metrics
        prediction_time = time.time() - start_time
        self._prediction_times.append(prediction_time)
        
        # Keep only last 100 measurements
        if len(self._prediction_times) > 100:
            self._prediction_times = self._prediction_times[-100:]
        
        return result
    
    async def predict_ballistic_properties_async(
        self,
        features: np.ndarray,
        include_uncertainty: bool = True,
        include_feature_importance: bool = True,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Predict ballistic properties asynchronously.
        
        Args:
            features: Engineered features array
            include_uncertainty: Whether to include uncertainty quantification
            include_feature_importance: Whether to include feature importance
            use_cache: Whether to use prediction caching
            
        Returns:
            Dictionary with predictions, uncertainties, and metadata
        """
        start_time = time.time()
        
        # Generate cache key if caching is enabled
        cache_key = None
        if use_cache:
            cache_key = self._generate_prediction_cache_key(
                features, "ballistic", include_uncertainty, include_feature_importance
            )
            
            # Check cache
            cached_result = self._get_cached_prediction(cache_key)
            if cached_result is not None:
                self._cache_hits += 1
                logger.debug("Cache hit for ballistic prediction")
                return cached_result
            
            self._cache_misses += 1
        
        # Run prediction in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self._predict_ballistic_sync,
            features,
            include_uncertainty,
            include_feature_importance
        )
        
        # Cache result if caching is enabled
        if use_cache and cache_key:
            self._cache_prediction(cache_key, result)
        
        # Record performance metrics
        prediction_time = time.time() - start_time
        self._prediction_times.append(prediction_time)
        
        # Keep only last 100 measurements
        if len(self._prediction_times) > 100:
            self._prediction_times = self._prediction_times[-100:]
        
        return result
    
    async def predict_batch_async(
        self,
        features_batch: np.ndarray,
        prediction_type: str = 'both',
        batch_size: int = 10,
        use_parallel: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Make batch predictions asynchronously with parallel processing.
        
        Args:
            features_batch: Array of feature vectors (n_samples, n_features)
            prediction_type: 'mechanical', 'ballistic', or 'both'
            batch_size: Size of batches for parallel processing
            use_parallel: Whether to use parallel processing
            
        Returns:
            List of prediction results for each sample
        """
        start_time = time.time()
        
        if use_parallel and len(features_batch) > batch_size:
            # Process in parallel batches
            results = await self._predict_batch_parallel(
                features_batch, prediction_type, batch_size
            )
        else:
            # Process sequentially
            results = await self._predict_batch_sequential(
                features_batch, prediction_type
            )
        
        processing_time = time.time() - start_time
        logger.info(f"Batch prediction completed in {processing_time:.2f}s for {len(features_batch)} samples")
        
        return results
    
    async def _predict_batch_parallel(
        self,
        features_batch: np.ndarray,
        prediction_type: str,
        batch_size: int
    ) -> List[Dict[str, Any]]:
        """Process batch predictions in parallel."""
        # Split into chunks
        chunks = [
            features_batch[i:i + batch_size]
            for i in range(0, len(features_batch), batch_size)
        ]
        
        # Process chunks in parallel
        tasks = [
            self._predict_batch_sequential(chunk, prediction_type)
            for chunk in chunks
        ]
        
        # Wait for all tasks to complete
        chunk_results = await asyncio.gather(*tasks)
        
        # Flatten results
        results = []
        for chunk_result in chunk_results:
            results.extend(chunk_result)
        
        return results
    
    async def _predict_batch_sequential(
        self,
        features_batch: np.ndarray,
        prediction_type: str
    ) -> List[Dict[str, Any]]:
        """Process batch predictions sequentially."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._predict_batch_sync,
            features_batch,
            prediction_type
        )
    
    def _predict_mechanical_sync(
        self,
        features: np.ndarray,
        include_uncertainty: bool,
        include_feature_importance: bool
    ) -> Dict[str, Any]:
        """Synchronous mechanical prediction wrapper."""
        return self.base_predictor.predict_mechanical_properties(
            features=features,
            include_uncertainty=include_uncertainty,
            include_feature_importance=include_feature_importance
        )
    
    def _predict_ballistic_sync(
        self,
        features: np.ndarray,
        include_uncertainty: bool,
        include_feature_importance: bool
    ) -> Dict[str, Any]:
        """Synchronous ballistic prediction wrapper."""
        return self.base_predictor.predict_ballistic_properties(
            features=features,
            include_uncertainty=include_uncertainty,
            include_feature_importance=include_feature_importance
        )
    
    def _predict_batch_sync(
        self,
        features_batch: np.ndarray,
        prediction_type: str
    ) -> List[Dict[str, Any]]:
        """Synchronous batch prediction wrapper."""
        return self.base_predictor.predict_batch(
            features_batch=features_batch,
            prediction_type=prediction_type
        )
    
    def _generate_prediction_cache_key(
        self,
        features: np.ndarray,
        prediction_type: str,
        include_uncertainty: bool,
        include_feature_importance: bool
    ) -> str:
        """
        Generate cache key for prediction.
        
        Args:
            features: Feature array
            prediction_type: Type of prediction
            include_uncertainty: Whether uncertainty is included
            include_feature_importance: Whether feature importance is included
            
        Returns:
            Cache key string
        """
        import hashlib
        
        # Create a hash of the features and parameters
        features_hash = hashlib.md5(features.tobytes()).hexdigest()[:8]
        params_str = f"{prediction_type}_{include_uncertainty}_{include_feature_importance}"
        
        return f"{features_hash}_{params_str}"
    
    def _get_cached_prediction(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached prediction result.
        
        Args:
            cache_key: Cache key to look up
            
        Returns:
            Cached result or None if not found/expired
        """
        if cache_key in self._prediction_cache:
            result, timestamp = self._prediction_cache[cache_key]
            
            # Check if expired
            if (datetime.now() - timestamp).total_seconds() < self._cache_ttl_seconds:
                return result.copy()
            else:
                # Remove expired entry
                del self._prediction_cache[cache_key]
        
        return None
    
    def _cache_prediction(self, cache_key: str, result: Dict[str, Any]) -> None:
        """
        Cache prediction result.
        
        Args:
            cache_key: Cache key
            result: Prediction result to cache
        """
        # Limit cache size (simple LRU-like behavior)
        max_cache_size = 100
        if len(self._prediction_cache) >= max_cache_size:
            # Remove oldest entry
            oldest_key = min(
                self._prediction_cache.keys(),
                key=lambda k: self._prediction_cache[k][1]
            )
            del self._prediction_cache[oldest_key]
        
        self._prediction_cache[cache_key] = (result.copy(), datetime.now())
    
    async def get_model_status_async(self) -> Dict[str, Any]:
        """
        Get model status asynchronously.
        
        Returns:
            Dictionary with model status information
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.base_predictor.get_model_status
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self._prediction_times:
            return {
                'avg_prediction_time_ms': 0,
                'min_prediction_time_ms': 0,
                'max_prediction_time_ms': 0,
                'total_predictions': 0,
                'cache_hit_rate': 0
            }
        
        avg_time = np.mean(self._prediction_times) * 1000
        min_time = np.min(self._prediction_times) * 1000
        max_time = np.max(self._prediction_times) * 1000
        
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'avg_prediction_time_ms': round(avg_time, 2),
            'min_prediction_time_ms': round(min_time, 2),
            'max_prediction_time_ms': round(max_time, 2),
            'total_predictions': len(self._prediction_times),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'cache_hit_rate': round(hit_rate, 2),
            'cache_size': len(self._prediction_cache),
            'worker_threads': self.max_workers
        }
    
    async def cleanup_cache(self) -> int:
        """
        Clean up expired cache entries.
        
        Returns:
            Number of entries removed
        """
        current_time = datetime.now()
        expired_keys = [
            key for key, (_, timestamp) in self._prediction_cache.items()
            if (current_time - timestamp).total_seconds() >= self._cache_ttl_seconds
        ]
        
        for key in expired_keys:
            del self._prediction_cache[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired prediction cache entries")
        
        return len(expired_keys)
    
    async def warmup_models(self) -> Dict[str, Any]:
        """
        Warm up models by loading them into cache.
        
        Returns:
            Warmup results
        """
        start_time = time.time()
        
        try:
            # Get model status to trigger loading
            model_status = await self.get_model_status_async()
            
            # Create dummy features for warmup predictions
            dummy_features = np.random.random((1, 50))  # Assume 50 features
            
            # Make warmup predictions to load models into memory
            await self.predict_mechanical_properties_async(
                dummy_features,
                include_uncertainty=False,
                include_feature_importance=False,
                use_cache=False
            )
            
            await self.predict_ballistic_properties_async(
                dummy_features,
                include_uncertainty=False,
                include_feature_importance=False,
                use_cache=False
            )
            
            warmup_time = time.time() - start_time
            
            return {
                'status': 'success',
                'warmup_time_seconds': round(warmup_time, 2),
                'models_loaded': len(model_status.get('available_models', [])),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Model warmup failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'warmup_time_seconds': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.executor.shutdown(wait=True)


# Global async predictor instance
_async_predictor: Optional[AsyncCeramicArmorPredictor] = None


def get_async_predictor() -> AsyncCeramicArmorPredictor:
    """Get global async predictor instance."""
    global _async_predictor
    if _async_predictor is None:
        settings = get_settings()
        max_workers = getattr(settings, 'prediction_workers', 4)
        _async_predictor = AsyncCeramicArmorPredictor(max_workers=max_workers)
    return _async_predictor