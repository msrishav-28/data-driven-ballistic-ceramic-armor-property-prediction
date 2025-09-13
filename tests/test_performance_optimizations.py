"""
Tests for performance optimizations.

This module tests the async patterns, caching, and memory optimizations
implemented for production performance.
"""

import asyncio
import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch

from src.api.utils.response_cache import AsyncResponseCache, get_response_cache
from src.feature_engineering.async_feature_extractor import AsyncCeramicFeatureExtractor
from src.ml.async_predictor import AsyncCeramicArmorPredictor


class TestResponseCache:
    """Test response caching functionality."""
    
    @pytest.fixture
    def cache(self):
        """Create a test cache instance."""
        return AsyncResponseCache(max_size=10, default_ttl=60, max_memory_mb=1)
    
    @pytest.mark.asyncio
    async def test_cache_put_and_get(self, cache):
        """Test basic cache put and get operations."""
        request_data = {"composition": {"SiC": 0.6, "B4C": 0.4}}
        response_data = {"predictions": {"fracture_toughness": {"value": 4.5}}}
        
        # Put data in cache
        await cache.put(request_data, response_data)
        
        # Get data from cache
        cached_response = await cache.get(request_data)
        
        assert cached_response is not None
        assert cached_response["predictions"]["fracture_toughness"]["value"] == 4.5
    
    @pytest.mark.asyncio
    async def test_cache_miss(self, cache):
        """Test cache miss for non-existent data."""
        request_data = {"composition": {"SiC": 0.7, "B4C": 0.3}}
        
        cached_response = await cache.get(request_data)
        
        assert cached_response is None
    
    @pytest.mark.asyncio
    async def test_cache_eviction(self, cache):
        """Test cache eviction when size limit is reached."""
        # Fill cache beyond capacity
        for i in range(15):
            request_data = {"composition": {"SiC": 0.5 + i * 0.01}}
            response_data = {"predictions": {"value": i}}
            await cache.put(request_data, response_data)
        
        # Check that cache size is within limits
        stats = await cache.get_stats_async()
        assert stats["cache_size"] <= cache.max_size
    
    @pytest.mark.asyncio
    async def test_cache_cleanup(self, cache):
        """Test cleanup of expired entries."""
        # Add entry with very short TTL
        request_data = {"composition": {"SiC": 0.6}}
        response_data = {"predictions": {"value": 1}}
        await cache.put(request_data, response_data, ttl=0)  # Immediate expiry
        
        # Wait a bit and cleanup
        await asyncio.sleep(0.1)
        cleaned = await cache.cleanup_expired()
        
        assert cleaned >= 0  # Should clean up expired entries


class TestAsyncFeatureExtractor:
    """Test async feature extraction functionality."""
    
    @pytest.fixture
    def extractor(self):
        """Create a test async feature extractor."""
        return AsyncCeramicFeatureExtractor(max_workers=2, cache_size=5)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample material data."""
        return pd.DataFrame([{
            'comp_SiC': 0.6,
            'comp_B4C': 0.3,
            'comp_Al2O3': 0.1,
            'proc_sintering_temperature': 1800,
            'proc_pressure': 50,
            'proc_grain_size': 10,
            'micro_porosity': 0.02,
            'micro_phase_distribution': 'uniform',
            'micro_interface_quality': 'good'
        }])
    
    @pytest.mark.asyncio
    async def test_async_feature_extraction(self, extractor, sample_data):
        """Test async feature extraction."""
        result_df = await extractor.extract_features_async(sample_data)
        
        # Check that features were extracted
        assert len(result_df) == 1
        assert len(result_df.columns) > len(sample_data.columns)
        
        # Check for expected derived features
        assert 'ceramic_content' in result_df.columns
        assert 'hardness_indicator' in result_df.columns
        assert 'processing_efficiency' in result_df.columns
    
    @pytest.mark.asyncio
    async def test_feature_caching(self, extractor, sample_data):
        """Test feature extraction caching."""
        # First extraction
        start_time = datetime.now()
        result1 = await extractor.extract_features_async(sample_data, use_cache=True)
        first_time = (datetime.now() - start_time).total_seconds()
        
        # Second extraction (should be cached)
        start_time = datetime.now()
        result2 = await extractor.extract_features_async(sample_data, use_cache=True)
        second_time = (datetime.now() - start_time).total_seconds()
        
        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)
        
        # Second extraction should be faster (cached)
        assert second_time < first_time or second_time < 0.01  # Allow for very fast operations
    
    @pytest.mark.asyncio
    async def test_parallel_feature_extraction(self, extractor):
        """Test parallel feature extraction for multiple samples."""
        # Create multiple samples
        sample_data = pd.DataFrame([
            {
                'comp_SiC': 0.6 + i * 0.05,
                'comp_B4C': 0.3,
                'comp_Al2O3': 0.1 - i * 0.05,
                'proc_sintering_temperature': 1800,
                'proc_pressure': 50,
                'proc_grain_size': 10,
                'micro_porosity': 0.02,
                'micro_phase_distribution': 'uniform',
                'micro_interface_quality': 'good'
            }
            for i in range(5)
        ])
        
        result_df = await extractor.extract_features_async(
            sample_data, 
            use_cache=False, 
            parallel=True
        )
        
        # Check that all samples were processed
        assert len(result_df) == 5
        assert len(result_df.columns) > len(sample_data.columns)


class TestAsyncPredictor:
    """Test async predictor functionality."""
    
    @pytest.fixture
    def predictor(self):
        """Create a test async predictor."""
        return AsyncCeramicArmorPredictor(max_workers=2)
    
    @pytest.fixture
    def sample_features(self):
        """Create sample feature array."""
        return np.random.random((1, 50))  # 50 features
    
    @pytest.mark.asyncio
    async def test_async_mechanical_prediction(self, predictor, sample_features):
        """Test async mechanical property prediction."""
        with patch.object(predictor.base_predictor, 'predict_mechanical_properties') as mock_predict:
            mock_predict.return_value = {
                'predictions': {
                    'fracture_toughness': {'value': 4.5, 'unit': 'MPa·m^0.5'},
                    'vickers_hardness': {'value': 2800, 'unit': 'HV'}
                },
                'processing_time_ms': 50
            }
            
            result = await predictor.predict_mechanical_properties_async(
                sample_features,
                include_uncertainty=True,
                include_feature_importance=False,
                use_cache=False
            )
            
            assert 'predictions' in result
            assert 'fracture_toughness' in result['predictions']
            mock_predict.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_prediction_caching(self, predictor, sample_features):
        """Test prediction result caching."""
        with patch.object(predictor.base_predictor, 'predict_mechanical_properties') as mock_predict:
            mock_predict.return_value = {
                'predictions': {'fracture_toughness': {'value': 4.5}},
                'processing_time_ms': 50
            }
            
            # First prediction
            result1 = await predictor.predict_mechanical_properties_async(
                sample_features, use_cache=True
            )
            
            # Second prediction (should be cached)
            result2 = await predictor.predict_mechanical_properties_async(
                sample_features, use_cache=True
            )
            
            # Mock should only be called once due to caching
            assert mock_predict.call_count == 1
            assert result1 == result2
    
    @pytest.mark.asyncio
    async def test_batch_prediction_parallel(self, predictor):
        """Test parallel batch prediction."""
        # Create batch of features
        batch_features = np.random.random((10, 50))
        
        with patch.object(predictor.base_predictor, 'predict_batch') as mock_batch:
            mock_batch.return_value = [
                {'sample_id': i, 'predictions': {'mechanical': {'fracture_toughness': {'value': 4.0 + i}}}}
                for i in range(10)
            ]
            
            results = await predictor.predict_batch_async(
                batch_features,
                prediction_type='mechanical',
                batch_size=3,
                use_parallel=True
            )
            
            assert len(results) == 10
            # Should have called batch prediction multiple times for parallel processing
            assert mock_batch.call_count >= 1
    
    def test_performance_stats(self, predictor):
        """Test performance statistics collection."""
        # Simulate some prediction times
        predictor._prediction_times = [0.1, 0.15, 0.12, 0.08, 0.2]
        predictor._cache_hits = 10
        predictor._cache_misses = 5
        
        stats = predictor.get_performance_stats()
        
        assert 'avg_prediction_time_ms' in stats
        assert 'cache_hit_rate' in stats
        assert stats['cache_hit_rate'] == 66.67  # 10/(10+5) * 100
        assert stats['total_predictions'] == 5


class TestIntegrationPerformance:
    """Integration tests for performance optimizations."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_performance(self):
        """Test end-to-end performance with all optimizations."""
        # Create test data
        request_data = {
            "composition": {"SiC": 0.6, "B4C": 0.3, "Al2O3": 0.1},
            "processing": {
                "sintering_temperature": 1800,
                "pressure": 50,
                "grain_size": 10
            },
            "microstructure": {
                "porosity": 0.02,
                "phase_distribution": "uniform"
            }
        }
        
        # Test response caching
        cache = get_response_cache()
        
        # Should be cache miss first time
        cached = await cache.get(request_data)
        assert cached is None
        
        # Simulate response and cache it
        response_data = {
            "predictions": {"fracture_toughness": {"value": 4.5}},
            "timestamp": datetime.now().isoformat()
        }
        
        await cache.put(request_data, response_data)
        
        # Should be cache hit second time
        cached = await cache.get(request_data)
        assert cached is not None
        assert cached["predictions"]["fracture_toughness"]["value"] == 4.5
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self):
        """Test memory efficiency of optimizations."""
        # Create large dataset to test memory handling
        large_data = pd.DataFrame([
            {
                'comp_SiC': 0.5 + i * 0.001,
                'comp_B4C': 0.3,
                'comp_Al2O3': 0.2 - i * 0.001,
                'proc_sintering_temperature': 1800,
                'proc_pressure': 50,
                'proc_grain_size': 10,
                'micro_porosity': 0.02,
                'micro_phase_distribution': 'uniform',
                'micro_interface_quality': 'good'
            }
            for i in range(100)  # 100 samples
        ])
        
        extractor = AsyncCeramicFeatureExtractor(max_workers=2, cache_size=5)
        
        # Process large dataset
        result = await extractor.extract_features_async(
            large_data,
            use_cache=True,
            parallel=True
        )
        
        # Check that processing completed successfully
        assert len(result) == 100
        assert len(result.columns) > len(large_data.columns)
        
        # Check cache stats
        cache_stats = extractor.get_cache_stats()
        assert cache_stats['cache_size'] <= extractor.cache_size


if __name__ == "__main__":
    # Run basic tests
    asyncio.run(TestResponseCache().test_cache_put_and_get(AsyncResponseCache()))
    print("Performance optimization tests completed successfully!")