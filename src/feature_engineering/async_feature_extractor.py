"""
Async optimized feature extraction for ceramic materials.

This module provides async feature extraction with memory optimization
and caching for improved performance in production environments.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from datetime import datetime, timedelta

from .simple_feature_extractor import CeramicFeatureExtractor

logger = logging.getLogger(__name__)


class AsyncCeramicFeatureExtractor:
    """
    Async optimized feature extractor for ceramic materials.
    
    Provides async feature extraction with caching, memory optimization,
    and parallel processing for improved performance.
    """
    
    def __init__(self, max_workers: int = 4, cache_size: int = 128):
        """
        Initialize async feature extractor.
        
        Args:
            max_workers: Maximum number of worker threads
            cache_size: Size of feature cache
        """
        self.max_workers = max_workers
        self.cache_size = cache_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Base feature extractor
        self.base_extractor = CeramicFeatureExtractor()
        
        # Feature cache
        self._feature_cache: Dict[str, Tuple[pd.DataFrame, datetime]] = {}
        self._cache_ttl = timedelta(hours=1)  # Cache features for 1 hour
        
        # Pre-computed feature mappings
        self._feature_mappings = self._initialize_feature_mappings()
        
        logger.info(f"Initialized AsyncCeramicFeatureExtractor with {max_workers} workers")
    
    def _initialize_feature_mappings(self) -> Dict[str, Any]:
        """Initialize pre-computed feature mappings for optimization."""
        return {
            'composition_densities': {
                'SiC': 3.2,
                'B4C': 2.5,
                'Al2O3': 4.0,
                'WC': 15.6,
                'TiC': 4.9,
                'ZrO2': 6.1,
                'TiB2': 4.5
            },
            'hardness_coefficients': {
                'SiC': 0.8,
                'B4C': 0.9,
                'Al2O3': 0.6,
                'WC': 0.95,
                'TiC': 0.85,
                'ZrO2': 0.4,
                'TiB2': 0.88
            },
            'thermal_coefficients': {
                'SiC': 0.9,
                'B4C': 0.7,
                'Al2O3': 0.5,
                'WC': 0.8,
                'TiC': 0.85,
                'ZrO2': 0.3,
                'TiB2': 0.75
            }
        }
    
    def _generate_cache_key(self, df: pd.DataFrame) -> str:
        """
        Generate cache key for DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cache key string
        """
        # Create a hash of the DataFrame content
        df_dict = df.to_dict('records')
        json_str = json.dumps(df_dict, sort_keys=True, default=str)
        return hashlib.md5(json_str.encode()).hexdigest()[:16]
    
    def _is_cache_valid(self, timestamp: datetime) -> bool:
        """Check if cache entry is still valid."""
        return datetime.now() - timestamp < self._cache_ttl
    
    async def extract_features_async(
        self,
        df: pd.DataFrame,
        use_cache: bool = True,
        parallel: bool = True
    ) -> pd.DataFrame:
        """
        Extract features asynchronously with optimization.
        
        Args:
            df: Input DataFrame with material data
            use_cache: Whether to use feature caching
            parallel: Whether to use parallel processing
            
        Returns:
            DataFrame with extracted features
        """
        start_time = datetime.now()
        
        # Check cache first
        if use_cache:
            cache_key = self._generate_cache_key(df)
            if cache_key in self._feature_cache:
                cached_df, timestamp = self._feature_cache[cache_key]
                if self._is_cache_valid(timestamp):
                    logger.debug(f"Cache hit for feature extraction: {cache_key}")
                    return cached_df.copy()
                else:
                    # Remove expired cache entry
                    del self._feature_cache[cache_key]
        
        # Extract features
        if parallel and len(df) > 1:
            result_df = await self._extract_features_parallel(df)
        else:
            result_df = await self._extract_features_sequential(df)
        
        # Cache result
        if use_cache:
            # Limit cache size
            if len(self._feature_cache) >= self.cache_size:
                # Remove oldest entry
                oldest_key = min(
                    self._feature_cache.keys(),
                    key=lambda k: self._feature_cache[k][1]
                )
                del self._feature_cache[oldest_key]
            
            self._feature_cache[cache_key] = (result_df.copy(), datetime.now())
        
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.debug(f"Feature extraction completed in {processing_time:.3f}s")
        
        return result_df
    
    async def _extract_features_sequential(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features sequentially (async wrapper)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._extract_features_optimized,
            df
        )
    
    async def _extract_features_parallel(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features in parallel for large datasets."""
        # Split DataFrame into chunks for parallel processing
        chunk_size = max(1, len(df) // self.max_workers)
        chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
        
        # Process chunks in parallel
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(
                self.executor,
                self._extract_features_optimized,
                chunk
            )
            for chunk in chunks
        ]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Combine results
        return pd.concat(results, ignore_index=True)
    
    def _extract_features_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimized feature extraction with memory efficiency.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with extracted features
        """
        # Create a copy to avoid modifying original
        result_df = df.copy()
        
        # Pre-allocate feature columns for better memory usage
        feature_columns = []
        
        # Optimized composition features
        result_df, comp_features = self._add_composition_features_optimized(result_df)
        feature_columns.extend(comp_features)
        
        # Optimized processing features
        result_df, proc_features = self._add_processing_features_optimized(result_df)
        feature_columns.extend(proc_features)
        
        # Optimized microstructure features
        result_df, micro_features = self._add_microstructure_features_optimized(result_df)
        feature_columns.extend(micro_features)
        
        # Optimized derived features
        result_df, derived_features = self._add_derived_features_optimized(result_df)
        feature_columns.extend(derived_features)
        
        # Store feature column names
        result_df._feature_columns = feature_columns
        
        return result_df
    
    def _add_composition_features_optimized(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Add composition features with optimization."""
        feature_names = []
        
        # Extract composition columns efficiently
        comp_cols = [col for col in df.columns if col.startswith('comp_')]
        
        # Vectorized operations for better performance
        if comp_cols:
            # Total composition
            df['total_composition'] = df[comp_cols].sum(axis=1)
            feature_names.append('total_composition')
            
            # Ceramic content (vectorized)
            ceramic_cols = ['comp_SiC', 'comp_B4C', 'comp_Al2O3']
            available_ceramic = [col for col in ceramic_cols if col in df.columns]
            if available_ceramic:
                df['ceramic_content'] = df[available_ceramic].sum(axis=1)
                feature_names.append('ceramic_content')
            
            # Carbide content (vectorized)
            carbide_cols = ['comp_SiC', 'comp_B4C', 'comp_WC', 'comp_TiC']
            available_carbide = [col for col in carbide_cols if col in df.columns]
            if available_carbide:
                df['carbide_content'] = df[available_carbide].sum(axis=1)
                feature_names.append('carbide_content')
            
            # Density estimation (vectorized using pre-computed coefficients)
            density_sum = 0
            for comp, density in self._feature_mappings['composition_densities'].items():
                comp_col = f'comp_{comp}'
                if comp_col in df.columns:
                    density_sum += df[comp_col] * density
            
            if isinstance(density_sum, pd.Series):
                df['estimated_density'] = density_sum
                feature_names.append('estimated_density')
        
        # Add original composition columns to feature names
        feature_names.extend(comp_cols)
        
        return df, feature_names
    
    def _add_processing_features_optimized(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Add processing features with optimization."""
        feature_names = []
        
        # Extract processing columns
        proc_cols = [col for col in df.columns if col.startswith('proc_')]
        feature_names.extend(proc_cols)
        
        # Vectorized processing features
        if 'proc_sintering_temperature' in df.columns:
            # Normalized temperature (vectorized)
            df['normalized_temperature'] = df['proc_sintering_temperature'] / 2500.0
            feature_names.append('normalized_temperature')
            
            # Temperature categories (vectorized)
            df['high_temp'] = (df['proc_sintering_temperature'] > 2000).astype(np.int8)
            df['low_temp'] = (df['proc_sintering_temperature'] < 1500).astype(np.int8)
            feature_names.extend(['high_temp', 'low_temp'])
        
        if 'proc_pressure' in df.columns and 'proc_sintering_temperature' in df.columns:
            # Pressure-temperature interaction (vectorized)
            df['pressure_temp_product'] = df['proc_pressure'] * df['proc_sintering_temperature']
            feature_names.append('pressure_temp_product')
        
        if 'proc_grain_size' in df.columns:
            # Grain size categories (vectorized)
            df['fine_grain'] = (df['proc_grain_size'] < 1.0).astype(np.int8)
            df['coarse_grain'] = (df['proc_grain_size'] > 10.0).astype(np.int8)
            feature_names.extend(['fine_grain', 'coarse_grain'])
        
        return df, feature_names
    
    def _add_microstructure_features_optimized(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Add microstructure features with optimization."""
        feature_names = []
        
        # Extract microstructure columns
        micro_cols = [col for col in df.columns if col.startswith('micro_')]
        feature_names.extend(micro_cols)
        
        # Vectorized microstructure features
        if 'micro_porosity' in df.columns:
            # Porosity categories (vectorized)
            df['low_porosity'] = (df['micro_porosity'] < 0.02).astype(np.int8)
            df['high_porosity'] = (df['micro_porosity'] > 0.1).astype(np.int8)
            feature_names.extend(['low_porosity', 'high_porosity'])
        
        # Phase distribution encoding (optimized)
        if 'micro_phase_distribution' in df.columns:
            # Use categorical encoding for better memory efficiency
            phase_dummies = pd.get_dummies(
                df['micro_phase_distribution'],
                prefix='phase',
                dtype=np.int8
            )
            df = pd.concat([df, phase_dummies], axis=1)
            feature_names.extend(phase_dummies.columns.tolist())
        
        # Interface quality encoding (optimized)
        if 'micro_interface_quality' in df.columns:
            # Use mapping for better performance
            quality_map = {'poor': 1, 'fair': 2, 'good': 3, 'excellent': 4}
            df['interface_quality_numeric'] = df['micro_interface_quality'].map(
                quality_map
            ).fillna(3).astype(np.int8)
            feature_names.append('interface_quality_numeric')
        
        return df, feature_names
    
    def _add_derived_features_optimized(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Add derived features with optimization."""
        feature_names = []
        
        # Hardness indicator (vectorized using pre-computed coefficients)
        hardness_sum = 0
        for comp, coeff in self._feature_mappings['hardness_coefficients'].items():
            comp_col = f'comp_{comp}'
            if comp_col in df.columns:
                hardness_sum += df[comp_col] * coeff
        
        if isinstance(hardness_sum, pd.Series):
            df['hardness_indicator'] = hardness_sum
            feature_names.append('hardness_indicator')
            
            # Toughness indicator (inverse relationship)
            df['toughness_indicator'] = 1.0 - df['hardness_indicator']
            feature_names.append('toughness_indicator')
        
        # Processing efficiency (vectorized)
        if 'proc_sintering_temperature' in df.columns and 'proc_pressure' in df.columns:
            df['processing_efficiency'] = (
                (df['proc_sintering_temperature'] / 2000.0) * 0.7 +
                (df['proc_pressure'] / 100.0) * 0.3
            )
            feature_names.append('processing_efficiency')
        
        # Microstructure quality (vectorized)
        if 'micro_porosity' in df.columns and 'interface_quality_numeric' in df.columns:
            df['microstructure_quality'] = (
                (1.0 - df['micro_porosity']) * 0.6 +
                (df['interface_quality_numeric'] / 4.0) * 0.4
            )
            feature_names.append('microstructure_quality')
        
        # Thermal stability indicator (vectorized)
        thermal_sum = 0
        for comp, coeff in self._feature_mappings['thermal_coefficients'].items():
            comp_col = f'comp_{comp}'
            if comp_col in df.columns:
                thermal_sum += df[comp_col] * coeff
        
        if isinstance(thermal_sum, pd.Series):
            df['thermal_stability'] = thermal_sum
            feature_names.append('thermal_stability')
        
        return df, feature_names
    
    @lru_cache(maxsize=32)
    def get_feature_names(self) -> List[str]:
        """
        Get list of all possible feature names.
        
        Returns:
            List of feature names
        """
        # Create a sample DataFrame to extract feature names
        sample_data = {
            'comp_SiC': [0.6],
            'comp_B4C': [0.3],
            'comp_Al2O3': [0.1],
            'proc_sintering_temperature': [1800],
            'proc_pressure': [50],
            'proc_grain_size': [10],
            'micro_porosity': [0.02],
            'micro_phase_distribution': ['uniform'],
            'micro_interface_quality': ['good']
        }
        
        sample_df = pd.DataFrame(sample_data)
        result_df = self._extract_features_optimized(sample_df)
        
        return getattr(result_df, '_feature_columns', [])
    
    async def cleanup_cache(self) -> int:
        """
        Clean up expired cache entries.
        
        Returns:
            Number of entries removed
        """
        expired_keys = [
            key for key, (_, timestamp) in self._feature_cache.items()
            if not self._is_cache_valid(timestamp)
        ]
        
        for key in expired_keys:
            del self._feature_cache[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired feature cache entries")
        
        return len(expired_keys)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get feature cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'cache_size': len(self._feature_cache),
            'max_cache_size': self.cache_size,
            'cache_keys': list(self._feature_cache.keys()),
            'worker_threads': self.max_workers
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.executor.shutdown(wait=True)


# Global async feature extractor instance
_async_feature_extractor: Optional[AsyncCeramicFeatureExtractor] = None


def get_async_feature_extractor() -> AsyncCeramicFeatureExtractor:
    """Get global async feature extractor instance."""
    global _async_feature_extractor
    if _async_feature_extractor is None:
        _async_feature_extractor = AsyncCeramicFeatureExtractor()
    return _async_feature_extractor