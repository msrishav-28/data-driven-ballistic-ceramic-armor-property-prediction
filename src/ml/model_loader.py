"""
ML Model Loading and Caching System for Ceramic Armor Predictions.

This module provides utilities for loading, caching, and managing ML models
used for predicting mechanical and ballistic properties of ceramic armor materials.
"""

import os
import json
import pickle
import joblib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, asdict
from threading import Lock
import hashlib
import psutil
from sklearn.base import BaseEstimator

from ..config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for a loaded ML model."""
    
    model_name: str
    model_type: str  # 'mechanical' or 'ballistic'
    version: str
    file_path: str
    file_size: int
    file_hash: str
    loaded_at: datetime
    last_used: datetime
    use_count: int
    training_r2: float
    validation_r2: float
    feature_count: int
    target_properties: List[str]
    model_algorithm: str
    memory_usage_mb: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for JSON serialization."""
        data = asdict(self)
        data['loaded_at'] = self.loaded_at.isoformat()
        data['last_used'] = self.last_used.isoformat()
        return data


class ModelCache:
    """Thread-safe model cache with LRU eviction and memory management."""
    
    def __init__(self, max_size: int = 10, max_memory_mb: int = 2048):
        """
        Initialize model cache.
        
        Args:
            max_size: Maximum number of models to cache
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self._cache: Dict[str, Tuple[BaseEstimator, ModelMetadata]] = {}
        self._lock = Lock()
        
        # Performance optimization: pre-allocate memory tracking
        self._memory_usage = 0.0
        self._access_order = []  # Track access order for LRU
        
    def get(self, model_key: str) -> Optional[Tuple[BaseEstimator, ModelMetadata]]:
        """Get model from cache and update usage statistics."""
        with self._lock:
            if model_key in self._cache:
                model, metadata = self._cache[model_key]
                metadata.last_used = datetime.now()
                metadata.use_count += 1
                
                # Update access order for LRU
                if model_key in self._access_order:
                    self._access_order.remove(model_key)
                self._access_order.append(model_key)
                
                logger.debug(f"Cache hit for model: {model_key}")
                return model, metadata
            logger.debug(f"Cache miss for model: {model_key}")
            return None
    
    def put(self, model_key: str, model: BaseEstimator, metadata: ModelMetadata) -> None:
        """Add model to cache with LRU eviction if necessary."""
        with self._lock:
            # Check if we need to evict models
            self._evict_if_necessary(metadata.memory_usage_mb)
            
            # Add new model
            self._cache[model_key] = (model, metadata)
            self._memory_usage += metadata.memory_usage_mb
            self._access_order.append(model_key)
            
            logger.info(f"Cached model: {model_key} ({metadata.memory_usage_mb:.1f} MB)")
    
    def remove(self, model_key: str) -> bool:
        """Remove model from cache."""
        with self._lock:
            if model_key in self._cache:
                _, metadata = self._cache[model_key]
                self._memory_usage -= metadata.memory_usage_mb
                del self._cache[model_key]
                
                if model_key in self._access_order:
                    self._access_order.remove(model_key)
                
                logger.info(f"Removed model from cache: {model_key}")
                return True
            return False
    
    def clear(self) -> None:
        """Clear all models from cache."""
        with self._lock:
            self._cache.clear()
            self._memory_usage = 0.0
            self._access_order.clear()
            logger.info("Cleared model cache")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics and information."""
        with self._lock:
            total_memory = sum(metadata.memory_usage_mb for _, metadata in self._cache.values())
            return {
                'cache_size': len(self._cache),
                'max_size': self.max_size,
                'total_memory_mb': round(total_memory, 2),
                'max_memory_mb': self.max_memory_mb,
                'memory_usage_percent': round((total_memory / self.max_memory_mb) * 100, 2),
                'models': [metadata.to_dict() for _, metadata in self._cache.values()]
            }
    
    def _evict_if_necessary(self, new_model_memory: float = 0) -> None:
        """Evict least recently used models if cache is full or memory limit exceeded."""
        # Evict by memory usage (optimized with pre-tracked memory)
        while (self._memory_usage + new_model_memory > self.max_memory_mb and 
               self._access_order):
            lru_key = self._access_order[0]  # First item is least recently used
            _, metadata = self._cache[lru_key]
            evicted_memory = metadata.memory_usage_mb
            
            del self._cache[lru_key]
            self._access_order.remove(lru_key)
            self._memory_usage -= evicted_memory
            
            logger.info(f"Evicted model due to memory limit: {lru_key} ({evicted_memory:.1f} MB)")
        
        # Evict by cache size (optimized with access order tracking)
        while len(self._cache) >= self.max_size and self._access_order:
            lru_key = self._access_order[0]  # First item is least recently used
            _, metadata = self._cache[lru_key]
            
            del self._cache[lru_key]
            self._access_order.remove(lru_key)
            self._memory_usage -= metadata.memory_usage_mb
            
            logger.info(f"Evicted model due to size limit: {lru_key}")


class ModelLoader:
    """
    ML Model loader with caching, versioning, and health checks.
    
    Supports loading scikit-learn, XGBoost, LightGBM, and CatBoost models
    for ceramic armor property prediction.
    """
    
    def __init__(self, model_path: Optional[str] = None, cache_size: int = 10):
        """
        Initialize model loader.
        
        Args:
            model_path: Path to model directory
            cache_size: Maximum number of models to cache
        """
        settings = get_settings()
        self.model_path = Path(model_path or settings.model_path)
        self.cache = ModelCache(max_size=cache_size)
        self._supported_extensions = {'.pkl', '.joblib', '.json'}
        self._model_registry: Dict[str, ModelMetadata] = {}
        
        # Ensure model directory exists
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized ModelLoader with path: {self.model_path}")
    
    def load_model(self, model_name: str, force_reload: bool = False) -> Tuple[BaseEstimator, ModelMetadata]:
        """
        Load a model by name with caching support.
        
        Args:
            model_name: Name of the model to load
            force_reload: Force reload from disk even if cached
            
        Returns:
            Tuple of (model, metadata)
            
        Raises:
            FileNotFoundError: If model file not found
            ValueError: If model format not supported
        """
        model_key = f"{model_name}"
        
        # Check cache first (unless force reload)
        if not force_reload:
            cached = self.cache.get(model_key)
            if cached is not None:
                return cached
        
        # Load from disk
        model_file = self._find_model_file(model_name)
        if not model_file:
            raise FileNotFoundError(f"Model file not found for: {model_name}")
        
        logger.info(f"Loading model from disk: {model_file}")
        
        # Load the model
        model = self._load_model_file(model_file)
        
        # Create metadata
        metadata = self._create_model_metadata(model_name, model_file, model)
        
        # Cache the model
        self.cache.put(model_key, model, metadata)
        
        # Update registry
        self._model_registry[model_name] = metadata
        
        return model, metadata
    
    def load_all_models(self) -> Dict[str, Tuple[BaseEstimator, ModelMetadata]]:
        """
        Load all available models in the model directory.
        
        Returns:
            Dictionary mapping model names to (model, metadata) tuples
        """
        models = {}
        model_files = self._discover_model_files()
        
        for model_name, model_file in model_files.items():
            try:
                model, metadata = self.load_model(model_name)
                models[model_name] = (model, metadata)
                logger.info(f"Successfully loaded model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
        
        logger.info(f"Loaded {len(models)} models successfully")
        return models
    
    def get_model_info(self, model_name: str) -> Optional[ModelMetadata]:
        """Get metadata for a specific model."""
        if model_name in self._model_registry:
            return self._model_registry[model_name]
        
        # Try to load model to get info
        try:
            _, metadata = self.load_model(model_name)
            return metadata
        except Exception as e:
            logger.error(f"Failed to get info for model {model_name}: {e}")
            return None
    
    def list_available_models(self) -> List[str]:
        """List all available model names."""
        model_files = self._discover_model_files()
        return list(model_files.keys())
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on model loading system.
        
        Returns:
            Dictionary with health check results
        """
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'model_path_exists': self.model_path.exists(),
            'model_path_readable': os.access(self.model_path, os.R_OK),
            'available_models': [],
            'loaded_models': [],
            'cache_info': self.cache.get_cache_info(),
            'system_memory': self._get_system_memory_info(),
            'errors': []
        }
        
        try:
            # Check available models
            available_models = self.list_available_models()
            health_status['available_models'] = available_models
            
            # Check loaded models
            cache_info = self.cache.get_cache_info()
            health_status['loaded_models'] = [model['model_name'] for model in cache_info['models']]
            
            # Test loading a sample model if available
            if available_models:
                sample_model = available_models[0]
                try:
                    self.load_model(sample_model)
                    health_status['sample_model_load'] = 'success'
                except Exception as e:
                    health_status['sample_model_load'] = 'failed'
                    health_status['errors'].append(f"Sample model load failed: {e}")
                    health_status['status'] = 'degraded'
            
        except Exception as e:
            health_status['status'] = 'unhealthy'
            health_status['errors'].append(f"Health check failed: {e}")
            logger.error(f"Model loader health check failed: {e}")
        
        return health_status
    
    def _find_model_file(self, model_name: str) -> Optional[Path]:
        """Find model file by name, checking multiple extensions."""
        for ext in self._supported_extensions:
            model_file = self.model_path / f"{model_name}{ext}"
            if model_file.exists():
                return model_file
        return None
    
    def _discover_model_files(self) -> Dict[str, Path]:
        """Discover all model files in the model directory."""
        model_files = {}
        
        if not self.model_path.exists():
            return model_files
        
        for file_path in self.model_path.iterdir():
            if file_path.is_file() and file_path.suffix in self._supported_extensions:
                model_name = file_path.stem
                model_files[model_name] = file_path
        
        return model_files
    
    def _load_model_file(self, model_file: Path) -> BaseEstimator:
        """Load model from file based on extension."""
        try:
            if model_file.suffix == '.pkl':
                with open(model_file, 'rb') as f:
                    return pickle.load(f)
            elif model_file.suffix == '.joblib':
                return joblib.load(model_file)
            elif model_file.suffix == '.json':
                # For models that support JSON serialization (like LightGBM)
                import lightgbm as lgb
                return lgb.Booster(model_file=str(model_file))
            else:
                raise ValueError(f"Unsupported model file extension: {model_file.suffix}")
        except Exception as e:
            logger.error(f"Failed to load model file {model_file}: {e}")
            raise
    
    def _create_model_metadata(self, model_name: str, model_file: Path, model: BaseEstimator) -> ModelMetadata:
        """Create metadata for a loaded model."""
        file_stats = model_file.stat()
        file_hash = self._calculate_file_hash(model_file)
        
        # Extract model information
        model_info = self._extract_model_info(model)
        
        # Estimate memory usage
        memory_usage = self._estimate_model_memory(model)
        
        return ModelMetadata(
            model_name=model_name,
            model_type=self._infer_model_type(model_name),
            version=self._extract_model_version(model_file),
            file_path=str(model_file),
            file_size=file_stats.st_size,
            file_hash=file_hash,
            loaded_at=datetime.now(),
            last_used=datetime.now(),
            use_count=0,
            training_r2=model_info.get('training_r2', 0.0),
            validation_r2=model_info.get('validation_r2', 0.0),
            feature_count=model_info.get('feature_count', 0),
            target_properties=model_info.get('target_properties', []),
            model_algorithm=model_info.get('algorithm', 'unknown'),
            memory_usage_mb=memory_usage
        )
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of model file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _extract_model_info(self, model: BaseEstimator) -> Dict[str, Any]:
        """Extract information from the model object."""
        info = {
            'algorithm': type(model).__name__,
            'feature_count': 0,
            'target_properties': [],
            'training_r2': 0.0,
            'validation_r2': 0.0
        }
        
        # Try to extract feature count
        if hasattr(model, 'n_features_in_'):
            info['feature_count'] = model.n_features_in_
        elif hasattr(model, 'feature_importances_'):
            info['feature_count'] = len(model.feature_importances_)
        
        # Try to extract model-specific information
        if hasattr(model, '_ceramic_armor_metadata'):
            # Custom metadata we might have stored during training
            metadata = model._ceramic_armor_metadata
            info.update(metadata)
        
        return info
    
    def _infer_model_type(self, model_name: str) -> str:
        """Infer model type from model name."""
        name_lower = model_name.lower()
        if 'ballistic' in name_lower or 'v50' in name_lower or 'penetration' in name_lower:
            return 'ballistic'
        elif 'mechanical' in name_lower or 'fracture' in name_lower or 'hardness' in name_lower:
            return 'mechanical'
        else:
            return 'unknown'
    
    def _extract_model_version(self, model_file: Path) -> str:
        """Extract version from model filename or metadata."""
        # Look for version in filename (e.g., model_v1.2.0.pkl)
        name = model_file.stem
        if '_v' in name:
            version_part = name.split('_v')[-1]
            return version_part
        
        # Default version
        return "1.0.0"
    
    def _estimate_model_memory(self, model: BaseEstimator) -> float:
        """Estimate memory usage of model in MB."""
        try:
            import sys
            return sys.getsizeof(model) / (1024 * 1024)
        except Exception:
            # Fallback estimation based on model type
            return 10.0  # Default 10MB estimate
    
    def _get_system_memory_info(self) -> Dict[str, float]:
        """Get system memory information."""
        memory = psutil.virtual_memory()
        return {
            'total_gb': round(memory.total / (1024**3), 2),
            'available_gb': round(memory.available / (1024**3), 2),
            'used_percent': memory.percent
        }


# Global model loader instance
_model_loader: Optional[ModelLoader] = None


def get_model_loader() -> ModelLoader:
    """Get global model loader instance."""
    global _model_loader
    if _model_loader is None:
        settings = get_settings()
        _model_loader = ModelLoader(cache_size=settings.model_cache_size)
    return _model_loader


def initialize_models() -> Dict[str, Any]:
    """
    Initialize and load all models at application startup.
    
    Returns:
        Dictionary with initialization results
    """
    logger.info("Initializing ML models...")
    
    loader = get_model_loader()
    
    try:
        # Load all available models
        models = loader.load_all_models()
        
        result = {
            'status': 'success',
            'loaded_models': len(models),
            'model_names': list(models.keys()),
            'cache_info': loader.cache.get_cache_info(),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Successfully initialized {len(models)} models")
        return result
        
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'loaded_models': 0,
            'timestamp': datetime.now().isoformat()
        }