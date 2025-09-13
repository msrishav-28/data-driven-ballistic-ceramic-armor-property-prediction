"""
Rate limiting middleware for API protection.
Implements sliding window algorithm with configurable limits per endpoint type.
"""

import time
import logging
from collections import defaultdict, deque
from typing import Callable, Dict, Tuple

from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from src.config import get_settings


logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware to implement rate limiting using sliding window algorithm.
    
    Features:
    - Per-IP rate limiting with sliding window
    - Different limits for different endpoint types
    - Configurable burst allowance
    - Proper headers for client information
    - Whitelist support for trusted IPs
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.settings = get_settings()
        self.requests = defaultdict(deque)  # IP -> deque of request timestamps
        
        # Define different rate limits for different endpoint types
        self.endpoint_limits = {
            "prediction": (50, 3600),    # 50 requests per hour for predictions
            "batch": (10, 3600),         # 10 requests per hour for batch processing
            "upload": (20, 3600),        # 20 requests per hour for uploads
            "general": (100, 3600),      # 100 requests per hour for general endpoints
            "health": (1000, 3600),      # 1000 requests per hour for health checks
        }
        
        # Trusted IPs that bypass rate limiting (e.g., monitoring systems)
        self.trusted_ips = set(getattr(self.settings, 'trusted_ips', []))
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting using sliding window algorithm."""
        # Get client IP
        client_ip = self._get_client_ip(request)
        current_time = time.time()
        
        # Skip rate limiting for trusted IPs
        if client_ip in self.trusted_ips:
            logger.debug(f"Skipping rate limit for trusted IP: {client_ip}")
            return await call_next(request)
        
        # Determine endpoint type and get appropriate limits
        endpoint_type = self._get_endpoint_type(request.url.path)
        limit, window = self._get_rate_limit(endpoint_type)
        
        # Create unique key for this IP and endpoint type
        rate_key = f"{client_ip}:{endpoint_type}"
        
        # Clean old requests outside the sliding window
        self._clean_old_requests(rate_key, current_time, window)
        
        # Check rate limit
        current_requests = len(self.requests[rate_key])
        if current_requests >= limit:
            # Calculate when the oldest request will expire
            oldest_request = self.requests[rate_key][0] if self.requests[rate_key] else current_time
            retry_after = int(oldest_request + window - current_time)
            
            logger.warning(
                f"Rate limit exceeded for {client_ip} on {endpoint_type} endpoint. "
                f"Current: {current_requests}, Limit: {limit}"
            )
            
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests for {endpoint_type} endpoint. "
                              f"Limit: {limit} per {window} seconds",
                    "retry_after": max(1, retry_after),
                    "endpoint_type": endpoint_type,
                    "current_requests": current_requests,
                    "limit": limit
                },
                headers={
                    "Retry-After": str(max(1, retry_after)),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(oldest_request + window))
                }
            )
        
        # Add current request to the sliding window
        self.requests[rate_key].append(current_time)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to successful responses
        remaining = max(0, limit - len(self.requests[rate_key]))
        reset_time = int(current_time + window)
        
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_time)
        response.headers["X-RateLimit-Window"] = str(window)
        response.headers["X-RateLimit-Type"] = endpoint_type
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        # Check for forwarded headers (for reverse proxies)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct client IP
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _get_endpoint_type(self, path: str) -> str:
        """Determine the endpoint type based on the request path."""
        if path in ["/health", "/", "/docs", "/redoc", "/openapi.json"]:
            return "health"
        elif "/predict/batch" in path:
            return "batch"
        elif "/predict/" in path:
            return "prediction"
        elif "/upload" in path:
            return "upload"
        else:
            return "general"
    
    def _get_rate_limit(self, endpoint_type: str) -> Tuple[int, int]:
        """Get rate limit and window for the given endpoint type."""
        return self.endpoint_limits.get(endpoint_type, self.endpoint_limits["general"])
    
    def _clean_old_requests(self, rate_key: str, current_time: float, window: int):
        """Remove requests outside the sliding window."""
        window_start = current_time - window
        
        while (self.requests[rate_key] and 
               self.requests[rate_key][0] < window_start):
            self.requests[rate_key].popleft()
    
    def get_rate_limit_status(self, client_ip: str, endpoint_type: str = "general") -> Dict:
        """Get current rate limit status for debugging/monitoring."""
        rate_key = f"{client_ip}:{endpoint_type}"
        limit, window = self._get_rate_limit(endpoint_type)
        current_requests = len(self.requests[rate_key])
        
        return {
            "client_ip": client_ip,
            "endpoint_type": endpoint_type,
            "current_requests": current_requests,
            "limit": limit,
            "window": window,
            "remaining": max(0, limit - current_requests),
            "is_trusted": client_ip in self.trusted_ips
        }