"""
Response caching system for prediction requests.

This module provides caching for identical prediction requests to improve
performance and reduce computational load on the ML models.
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Union
from threading import Lock
import pickle
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    
    key: str
    data: Dict[str, Any]
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl_seconds: int
    size_bytes: int
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl_seconds)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['last_accessed'] = self.last_accessed.isoformat()
        return data


class AsyncResponseCache:
    """
    Thread-safe async response cache with TTL and LRU eviction.
    
    Provides caching for prediction responses to improve performance
    for identical requests.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: int = 3600,  # 1 hour
        max_memory_mb: int = 100
    ):
        """
        Initialize response cache.
        
        Args:
            max_size: Maximum number of cache entries
            default_ttl: Default TTL in seconds
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.max_memory_mb = max_memory_mb
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self._sync_lock = Lock()  # For sync operations
        
        # Cache statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
        logger.info(f"Initialized AsyncResponseCache with max_size={max_size}, ttl={default_ttl}s")
    
    def _generate_cache_key(self, request_data: Dict[str, Any]) -> str:
        """
        Generate cache key from request data.
        
        Args:
            request_data: Request data to hash
            
        Returns:
            Cache key string
        """
        # Create a normalized representation of the request
        normalized_data = self._normalize_request_data(request_data)
        
        # Convert to JSON string with sorted keys for consistent hashing
        json_str = json.dumps(normalized_data, sort_keys=True, separators=(',', ':'))
        
        # Generate SHA-256 hash
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()[:16]
    
    def _normalize_request_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize request data for consistent caching.
        
        Args:
            data: Raw request data
            
        Returns:
            Normalized data for hashing
        """
        normalized = {}
        
        # Round floating point values to avoid precision issues
        for key, value in data.items():
            if isinstance(value, float):
                normalized[key] = round(value, 6)
            elif isinstance(value, dict):
                normalized[key] = self._normalize_request_data(value)
            elif isinstance(value, list):
                normalized[key] = [
                    round(v, 6) if isinstance(v, float) else v
                    for v in value
                ]
            else:
                normalized[key] = value
        
        return normalized
    
    def _estimate_size(self, data: Dict[str, Any]) -> int:
        """
        Estimate memory size of data in bytes.
        
        Args:
            data: Data to estimate size for
            
        Returns:
            Estimated size in bytes
        """
        try:
            return len(pickle.dumps(data))
        except Exception:
            # Fallback estimation
            return len(json.dumps(data, default=str)) * 2
    
    async def get(self, request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get cached response for request data.
        
        Args:
            request_data: Request data to look up
            
        Returns:
            Cached response data or None if not found/expired
        """
        cache_key = self._generate_cache_key(request_data)
        
        async with self._lock:
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                
                # Check if expired
                if entry.is_expired():
                    del self._cache[cache_key]
                    self._misses += 1
                    logger.debug(f"Cache entry expired: {cache_key}")
                    return None
                
                # Update access statistics
                entry.last_accessed = datetime.now()
                entry.access_count += 1
                self._hits += 1
                
                logger.debug(f"Cache hit: {cache_key}")
                return entry.data.copy()
            
            self._misses += 1
            logger.debug(f"Cache miss: {cache_key}")
            return None
    
    async def put(
        self,
        request_data: Dict[str, Any],
        response_data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> None:
        """
        Store response in cache.
        
        Args:
            request_data: Request data for key generation
            response_data: Response data to cache
            ttl: Time to live in seconds (uses default if None)
        """
        cache_key = self._generate_cache_key(request_data)
        ttl = ttl or self.default_ttl
        
        # Estimate size
        size_bytes = self._estimate_size(response_data)
        
        async with self._lock:
            # Check if we need to evict entries
            await self._evict_if_necessary(size_bytes)
            
            # Create cache entry
            entry = CacheEntry(
                key=cache_key,
                data=response_data.copy(),
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=0,
                ttl_seconds=ttl,
                size_bytes=size_bytes
            )
            
            self._cache[cache_key] = entry
            logger.debug(f"Cached response: {cache_key} ({size_bytes} bytes, TTL: {ttl}s)")
    
    async def _evict_if_necessary(self, new_entry_size: int) -> None:
        """
        Evict entries if necessary to make room for new entry.
        
        Args:
            new_entry_size: Size of new entry to be added
        """
        # Calculate current memory usage
        current_memory = sum(entry.size_bytes for entry in self._cache.values())
        max_memory_bytes = self.max_memory_mb * 1024 * 1024
        
        # Evict expired entries first
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            del self._cache[key]
            logger.debug(f"Evicted expired entry: {key}")
        
        # Evict by memory limit
        while (current_memory + new_entry_size > max_memory_bytes and self._cache):
            # Find LRU entry
            lru_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k].last_accessed
            )
            
            evicted_entry = self._cache[lru_key]
            current_memory -= evicted_entry.size_bytes
            del self._cache[lru_key]
            self._evictions += 1
            
            logger.debug(f"Evicted LRU entry for memory: {lru_key}")
        
        # Evict by size limit
        while len(self._cache) >= self.max_size:
            # Find LRU entry
            lru_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k].last_accessed
            )
            
            del self._cache[lru_key]
            self._evictions += 1
            
            logger.debug(f"Evicted LRU entry for size: {lru_key}")
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
            logger.info("Cleared response cache")
    
    async def cleanup_expired(self) -> int:
        """
        Remove expired entries from cache.
        
        Returns:
            Number of entries removed
        """
        async with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                del self._cache[key]
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._sync_lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
            
            current_memory = sum(entry.size_bytes for entry in self._cache.values())
            
            return {
                'cache_size': len(self._cache),
                'max_size': self.max_size,
                'hits': self._hits,
                'misses': self._misses,
                'evictions': self._evictions,
                'hit_rate_percent': round(hit_rate, 2),
                'memory_usage_mb': round(current_memory / (1024 * 1024), 2),
                'max_memory_mb': self.max_memory_mb,
                'memory_usage_percent': round((current_memory / (self.max_memory_mb * 1024 * 1024)) * 100, 2),
                'entries': [entry.to_dict() for entry in self._cache.values()]
            }
    
    async def get_stats_async(self) -> Dict[str, Any]:
        """
        Get cache statistics (async version).
        
        Returns:
            Dictionary with cache statistics
        """
        async with self._lock:
            return self.get_stats()


# Global cache instance
_response_cache: Optional[AsyncResponseCache] = None


def get_response_cache() -> AsyncResponseCache:
    """Get global response cache instance."""
    global _response_cache
    if _response_cache is None:
        from ...config import get_settings
        settings = get_settings()
        
        # Configure cache based on environment
        if settings.environment == "production":
            max_size = 2000
            ttl = 7200  # 2 hours
            max_memory = 200  # MB
        else:
            max_size = 500
            ttl = 1800  # 30 minutes
            max_memory = 50  # MB
        
        _response_cache = AsyncResponseCache(
            max_size=max_size,
            default_ttl=ttl,
            max_memory_mb=max_memory
        )
    
    return _response_cache


async def cache_prediction_response(
    request_data: Dict[str, Any],
    response_data: Dict[str, Any],
    prediction_type: str = "general",
    ttl: Optional[int] = None
) -> None:
    """
    Cache a prediction response.
    
    Args:
        request_data: Request data for key generation
        response_data: Response data to cache
        prediction_type: Type of prediction for TTL adjustment
        ttl: Custom TTL in seconds
    """
    cache = get_response_cache()
    
    # Adjust TTL based on prediction type
    if ttl is None:
        if prediction_type == "mechanical":
            ttl = 7200  # 2 hours - mechanical predictions are more stable
        elif prediction_type == "ballistic":
            ttl = 3600  # 1 hour - ballistic predictions have higher uncertainty
        else:
            ttl = cache.default_ttl
    
    await cache.put(request_data, response_data, ttl)


async def get_cached_prediction_response(
    request_data: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Get cached prediction response.
    
    Args:
        request_data: Request data to look up
        
    Returns:
        Cached response or None if not found
    """
    cache = get_response_cache()
    return await cache.get(request_data)


async def cleanup_cache_periodically():
    """
    Periodic cache cleanup task.
    Should be run as a background task.
    """
    cache = get_response_cache()
    
    while True:
        try:
            await asyncio.sleep(300)  # Run every 5 minutes
            cleaned = await cache.cleanup_expired()
            
            if cleaned > 0:
                logger.info(f"Cache cleanup: removed {cleaned} expired entries")
                
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")
            await asyncio.sleep(60)  # Wait 1 minute before retry