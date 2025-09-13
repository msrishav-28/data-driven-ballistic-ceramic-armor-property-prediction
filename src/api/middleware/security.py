"""
Security middleware for input sanitization and security headers.
Implements comprehensive security measures for API protection.
"""

import re
import html
import logging
from typing import Callable, Dict, Any, List
from urllib.parse import unquote

from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from src.config import get_settings


logger = logging.getLogger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Middleware for comprehensive security measures including:
    - Input sanitization and validation
    - Security headers
    - XSS protection
    - SQL injection prevention
    - Path traversal protection
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.settings = get_settings()
        
        # Dangerous patterns to detect and block
        self.sql_injection_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
            r"(--|#|/\*|\*/)",
            r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
            r"(\bOR\s+\w+\s*=\s*\w+)",
            r"(\'\s*(OR|AND)\s*\'\w*\'\s*=\s*\'\w*\')",
        ]
        
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>.*?</iframe>",
            r"<object[^>]*>.*?</object>",
            r"<embed[^>]*>.*?</embed>",
        ]
        
        self.path_traversal_patterns = [
            r"\.\./",
            r"\.\.\\",
            r"%2e%2e%2f",
            r"%2e%2e\\",
            r"\.\.%2f",
            r"\.\.%5c",
        ]
        
        # Compile patterns for better performance
        self.compiled_sql_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.sql_injection_patterns]
        self.compiled_xss_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.xss_patterns]
        self.compiled_path_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.path_traversal_patterns]
        
        # Maximum allowed lengths for different input types
        self.max_lengths = {
            "query_param": 1000,
            "path_param": 200,
            "header_value": 2000,
            "json_string": 10000,
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with security checks and add security headers to response."""
        try:
            # Perform security checks on the request
            await self._validate_request_security(request)
            
            # Process the request
            response = await call_next(request)
            
            # Add security headers to the response
            self._add_security_headers(response)
            
            return response
            
        except HTTPException:
            # Re-raise HTTP exceptions (like security violations)
            raise
        except Exception as e:
            logger.error(f"Security middleware error: {e}")
            # Don't expose internal errors, return generic error
            raise HTTPException(
                status_code=500,
                detail="Internal security error"
            )
    
    async def _validate_request_security(self, request: Request):
        """Validate request for security threats."""
        # Check URL path for path traversal
        self._check_path_traversal(request.url.path)
        
        # Check query parameters
        for key, value in request.query_params.items():
            self._validate_input(value, "query_param", f"query parameter '{key}'")
        
        # Check headers (excluding standard headers)
        for key, value in request.headers.items():
            if key.lower() not in ['host', 'user-agent', 'accept', 'accept-encoding', 
                                  'accept-language', 'connection', 'content-type', 
                                  'content-length', 'authorization']:
                self._validate_input(value, "header_value", f"header '{key}'")
        
        # For POST/PUT requests, check body content if it's JSON
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")
            if "application/json" in content_type:
                try:
                    # Read body for validation (this consumes the stream)
                    body = await request.body()
                    if body:
                        body_str = body.decode('utf-8')
                        self._validate_json_content(body_str)
                        
                        # Re-create request with body for downstream processing
                        # Note: FastAPI will handle this automatically
                        
                except Exception as e:
                    logger.warning(f"Could not validate request body: {e}")
                    # Don't block request if we can't read body
                    pass
    
    def _check_path_traversal(self, path: str):
        """Check for path traversal attempts."""
        decoded_path = unquote(path)
        
        for pattern in self.compiled_path_patterns:
            if pattern.search(decoded_path):
                logger.warning(f"Path traversal attempt detected: {path}")
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Invalid request",
                        "message": "Path contains invalid characters"
                    }
                )
    
    def _validate_input(self, value: str, input_type: str, context: str):
        """Validate input for various security threats."""
        if not isinstance(value, str):
            return  # Skip non-string values
        
        # Check length limits
        max_length = self.max_lengths.get(input_type, 1000)
        if len(value) > max_length:
            logger.warning(f"Input too long in {context}: {len(value)} > {max_length}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Input validation failed",
                    "message": f"Input too long in {context}"
                }
            )
        
        # Check for SQL injection patterns
        for pattern in self.compiled_sql_patterns:
            if pattern.search(value):
                logger.warning(f"SQL injection attempt detected in {context}: {value[:100]}")
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Invalid input",
                        "message": "Input contains invalid characters"
                    }
                )
        
        # Check for XSS patterns
        for pattern in self.compiled_xss_patterns:
            if pattern.search(value):
                logger.warning(f"XSS attempt detected in {context}: {value[:100]}")
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Invalid input", 
                        "message": "Input contains invalid characters"
                    }
                )
    
    def _validate_json_content(self, content: str):
        """Validate JSON content for security threats."""
        # Basic validation for JSON content
        self._validate_input(content, "json_string", "request body")
        
        # Additional checks for common JSON-based attacks
        if len(content) > self.max_lengths["json_string"]:
            logger.warning(f"JSON content too large: {len(content)}")
            raise HTTPException(
                status_code=413,
                detail={
                    "error": "Payload too large",
                    "message": "Request body exceeds maximum allowed size"
                }
            )
    
    def _add_security_headers(self, response: Response):
        """Add comprehensive security headers to the response."""
        # Content Security Policy
        csp_policy = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
            "img-src 'self' data: https:; "
            "font-src 'self' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
            "connect-src 'self' https://api.materialsproject.org; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        )
        
        security_headers = {
            # Prevent XSS attacks
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            
            # Content Security Policy
            "Content-Security-Policy": csp_policy,
            
            # HTTPS enforcement (in production)
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains" if self.settings.environment == "production" else "",
            
            # Referrer policy
            "Referrer-Policy": "strict-origin-when-cross-origin",
            
            # Permissions policy
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
            
            # Cache control for sensitive endpoints
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            
            # Server identification
            "Server": "Ceramic-Armor-ML-API",
            
            # API version
            "X-API-Version": self.settings.app_version,
        }
        
        # Add headers to response
        for header, value in security_headers.items():
            if value:  # Only add non-empty headers
                response.headers[header] = value
    
    def sanitize_input(self, value: str) -> str:
        """Sanitize input by escaping dangerous characters."""
        if not isinstance(value, str):
            return value
        
        # HTML escape
        sanitized = html.escape(value)
        
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        
        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        return sanitized
    
    def get_security_report(self) -> Dict[str, Any]:
        """Get security configuration report for monitoring."""
        return {
            "sql_injection_patterns": len(self.sql_injection_patterns),
            "xss_patterns": len(self.xss_patterns),
            "path_traversal_patterns": len(self.path_traversal_patterns),
            "max_lengths": self.max_lengths,
            "environment": self.settings.environment,
            "security_headers_enabled": True,
        }


class CORSSecurityMiddleware(BaseHTTPMiddleware):
    """
    Enhanced CORS middleware with additional security features.
    Extends FastAPI's built-in CORS with more granular control.
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.settings = get_settings()
        
        # Parse allowed origins from settings
        self.allowed_origins = set(self.settings.cors_origins)
        
        # Add environment-specific origins
        if self.settings.environment == "development":
            self.allowed_origins.update([
                "http://localhost:3000",
                "http://localhost:8000", 
                "http://127.0.0.1:3000",
                "http://127.0.0.1:8000"
            ])
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process CORS with enhanced security checks."""
        origin = request.headers.get("origin")
        
        # Handle preflight requests
        if request.method == "OPTIONS":
            return self._handle_preflight(request, origin)
        
        # Process normal request
        response = await call_next(request)
        
        # Add CORS headers to response
        self._add_cors_headers(response, origin)
        
        return response
    
    def _handle_preflight(self, request: Request, origin: str) -> Response:
        """Handle CORS preflight requests."""
        if not self._is_origin_allowed(origin):
            logger.warning(f"CORS preflight rejected for origin: {origin}")
            return Response(status_code=403)
        
        response = Response()
        self._add_cors_headers(response, origin)
        
        # Add preflight-specific headers
        response.headers["Access-Control-Allow-Methods"] = ", ".join(self.settings.cors_allow_methods)
        response.headers["Access-Control-Allow-Headers"] = ", ".join(self.settings.cors_allow_headers)
        response.headers["Access-Control-Max-Age"] = "86400"  # 24 hours
        
        return response
    
    def _add_cors_headers(self, response: Response, origin: str):
        """Add CORS headers to response."""
        if self._is_origin_allowed(origin):
            response.headers["Access-Control-Allow-Origin"] = origin
            if self.settings.cors_allow_credentials:
                response.headers["Access-Control-Allow-Credentials"] = "true"
        
        # Add vary header for caching
        response.headers["Vary"] = "Origin"
    
    def _is_origin_allowed(self, origin: str) -> bool:
        """Check if origin is allowed."""
        if not origin:
            return False
        
        # Check exact matches
        if origin in self.allowed_origins:
            return True
        
        # Check wildcard patterns (if any)
        if "*" in self.allowed_origins:
            return True
        
        return False