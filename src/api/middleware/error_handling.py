"""
Comprehensive error handling middleware with detailed logging and context.
"""

import json
import logging
import traceback
from typing import Dict, Any, Optional, Union

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.middleware.logging import structured_logger
from src.config import get_settings


class ErrorContext:
    """Context manager for error information."""
    
    def __init__(self, request: Request):
        self.request = request
        self.settings = get_settings()
    
    def get_request_context(self) -> Dict[str, Any]:
        """Extract comprehensive request context for error logging."""
        # Safely get request_id
        request_id = 'unknown'
        if hasattr(self.request, 'state') and hasattr(self.request.state, 'request_id'):
            request_id = self.request.state.request_id
        
        context = {
            "method": self.request.method,
            "url": str(self.request.url),
            "path": self.request.url.path,
            "query_params": dict(self.request.query_params),
            "client_ip": self._get_client_ip(),
            "user_agent": self.request.headers.get("User-Agent", "Unknown"),
            "content_type": self.request.headers.get("Content-Type"),
            "content_length": self.request.headers.get("Content-Length", "0"),
            "request_id": request_id
        }
        
        # Include headers in debug mode only
        if self.settings.debug:
            context["headers"] = dict(self.request.headers)
        
        return context
    
    def _get_client_ip(self) -> str:
        """Extract client IP from request."""
        forwarded_for = self.request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = self.request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        if self.request.client:
            return self.request.client.host
        
        return "unknown"


class ErrorHandler:
    """Centralized error handler with comprehensive logging."""
    
    def __init__(self):
        self.logger = structured_logger
        self.settings = get_settings()
    
    def handle_http_exception(self, request: Request, exc: HTTPException) -> JSONResponse:
        """Handle HTTP exceptions with logging."""
        context = ErrorContext(request).get_request_context()
        
        # Log HTTP exceptions (but not 404s to avoid spam)
        if exc.status_code != 404:
            self.logger.log_structured(
                "WARNING" if exc.status_code < 500 else "ERROR",
                "http_exception",
                request_id=context["request_id"],
                status_code=exc.status_code,
                detail=exc.detail,
                **context
            )
        
        # Create error response
        error_response = {
            "error": "HTTP Exception",
            "message": exc.detail,
            "status_code": exc.status_code,
            "path": context["path"],
            "request_id": context["request_id"]
        }
        
        # Add debug info in development
        if self.settings.debug:
            error_response["debug_info"] = {
                "method": context["method"],
                "query_params": context["query_params"]
            }
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response
        )
    
    def handle_validation_error(self, request: Request, exc: Exception) -> JSONResponse:
        """Handle Pydantic validation errors."""
        context = ErrorContext(request).get_request_context()
        
        # Extract validation error details
        validation_errors = []
        if hasattr(exc, 'errors'):
            validation_errors = exc.errors()
        
        self.logger.log_structured(
            "WARNING",
            "validation_error",
            request_id=context["request_id"],
            validation_errors=validation_errors,
            **context
        )
        
        error_response = {
            "error": "Validation Error",
            "message": "Invalid input data",
            "status_code": 422,
            "path": context["path"],
            "request_id": context["request_id"],
            "validation_errors": validation_errors
        }
        
        return JSONResponse(
            status_code=422,
            content=error_response
        )
    
    def handle_ml_prediction_error(self, request: Request, exc: Exception, 
                                 prediction_context: Dict[str, Any] = None) -> JSONResponse:
        """Handle ML prediction specific errors."""
        context = ErrorContext(request).get_request_context()
        
        # Add ML-specific context
        ml_context = {
            "error_type": "ml_prediction_error",
            "prediction_context": prediction_context or {},
            "model_error": str(exc),
            "stack_trace": traceback.format_exc()
        }
        
        self.logger.log_structured(
            "ERROR",
            "ml_prediction_error",
            request_id=context["request_id"],
            **context,
            **ml_context
        )
        
        error_response = {
            "error": "Prediction Error",
            "message": "Failed to generate prediction. Please check your input data and try again.",
            "status_code": 500,
            "path": context["path"],
            "request_id": context["request_id"]
        }
        
        # Add technical details in debug mode
        if self.settings.debug:
            error_response["debug_info"] = {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "prediction_context": prediction_context
            }
        
        return JSONResponse(
            status_code=500,
            content=error_response
        )
    
    def handle_external_api_error(self, request: Request, exc: Exception,
                                api_name: str, operation: str) -> JSONResponse:
        """Handle external API errors (Materials Project, NIST, etc.)."""
        context = ErrorContext(request).get_request_context()
        
        api_context = {
            "error_type": "external_api_error",
            "api_name": api_name,
            "operation": operation,
            "api_error": str(exc),
            "stack_trace": traceback.format_exc()
        }
        
        self.logger.log_structured(
            "ERROR",
            "external_api_error",
            request_id=context["request_id"],
            **context,
            **api_context
        )
        
        error_response = {
            "error": "External API Error",
            "message": f"Failed to connect to {api_name}. Please try again later.",
            "status_code": 503,
            "path": context["path"],
            "request_id": context["request_id"]
        }
        
        if self.settings.debug:
            error_response["debug_info"] = {
                "api_name": api_name,
                "operation": operation,
                "error_message": str(exc)
            }
        
        return JSONResponse(
            status_code=503,
            content=error_response
        )
    
    def handle_general_exception(self, request: Request, exc: Exception) -> JSONResponse:
        """Handle unexpected exceptions with full context logging."""
        context = ErrorContext(request).get_request_context()
        
        # Log full exception details
        exception_context = {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "stack_trace": traceback.format_exc(),
            "exception_args": exc.args if exc.args else []
        }
        
        self.logger.log_structured(
            "ERROR",
            "unhandled_exception",
            **context,
            **exception_context
        )
        
        error_response = {
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please try again later.",
            "status_code": 500,
            "path": context["path"],
            "request_id": context["request_id"]
        }
        
        # Add debug information in development
        if self.settings.debug:
            error_response["debug_info"] = {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "stack_trace_lines": traceback.format_exc().split('\n')[-10:]  # Last 10 lines
            }
        
        return JSONResponse(
            status_code=500,
            content=error_response
        )


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive error handling and logging."""
    
    def __init__(self, app):
        super().__init__(app)
        self.error_handler = ErrorHandler()
        self.logger = structured_logger
    
    async def dispatch(self, request: Request, call_next):
        """Process request with comprehensive error handling."""
        # Add request ID to state for tracking
        if not hasattr(request.state, 'request_id'):
            import uuid
            request.state.request_id = f"req_{uuid.uuid4().hex[:12]}"
        
        try:
            response = await call_next(request)
            
            # Log successful requests (non-static files)
            if not request.url.path.startswith('/static'):
                self.logger.log_structured(
                    "INFO",
                    "request_completed",
                    request_id=request.state.request_id,
                    method=request.method,
                    path=request.url.path,
                    status_code=response.status_code,
                    response_time_ms=getattr(response, 'processing_time_ms', 0)
                )
            
            return response
            
        except HTTPException as exc:
            # Let FastAPI handle HTTP exceptions normally
            # They will be caught by the exception handlers in main.py
            raise
            
        except Exception as exc:
            # Handle unexpected exceptions
            return self.error_handler.handle_general_exception(request, exc)


# Custom exception classes for specific error types
class MLPredictionError(Exception):
    """Exception raised during ML prediction operations."""
    
    def __init__(self, message: str, model_name: str = None, 
                 prediction_type: str = None, context: Dict[str, Any] = None):
        super().__init__(message)
        self.model_name = model_name
        self.prediction_type = prediction_type
        self.context = context or {}


class ExternalAPIError(Exception):
    """Exception raised during external API operations."""
    
    def __init__(self, message: str, api_name: str, operation: str = None,
                 status_code: int = None, response_data: Any = None):
        super().__init__(message)
        self.api_name = api_name
        self.operation = operation
        self.status_code = status_code
        self.response_data = response_data


class DataValidationError(Exception):
    """Exception raised during data validation operations."""
    
    def __init__(self, message: str, field_errors: Dict[str, str] = None,
                 validation_context: Dict[str, Any] = None):
        super().__init__(message)
        self.field_errors = field_errors or {}
        self.validation_context = validation_context or {}


# Global error handler instance
error_handler = ErrorHandler()