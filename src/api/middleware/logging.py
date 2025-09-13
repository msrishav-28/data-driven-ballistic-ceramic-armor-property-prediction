"""
Comprehensive logging middleware for request/response tracking and monitoring.
"""

import json
import logging
import time
import traceback
import uuid
from typing import Callable, Dict, Any, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.config import get_settings


class StructuredLogger:
    """Structured logger for consistent log formatting."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.settings = get_settings()
    
    def log_structured(self, level: str, event: str, **kwargs) -> None:
        """Log structured data with consistent format."""
        log_data = {
            "timestamp": time.time(),
            "event": event,
            "environment": self.settings.environment,
            **kwargs
        }
        
        # Convert to JSON string for structured logging
        log_message = json.dumps(log_data, default=str, separators=(',', ':'))
        
        # Log at appropriate level
        log_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.log(log_level, log_message)
    
    def log_request(self, request_id: str, method: str, path: str, 
                   client_ip: str, user_agent: str = None, **kwargs) -> None:
        """Log HTTP request details."""
        self.log_structured(
            "INFO", "http_request",
            request_id=request_id,
            method=method,
            path=path,
            client_ip=client_ip,
            user_agent=user_agent,
            **kwargs
        )
    
    def log_response(self, request_id: str, method: str, path: str,
                    status_code: int, process_time: float, **kwargs) -> None:
        """Log HTTP response details."""
        self.log_structured(
            "INFO", "http_response",
            request_id=request_id,
            method=method,
            path=path,
            status_code=status_code,
            process_time_ms=round(process_time * 1000, 2),
            **kwargs
        )
    
    def log_error(self, request_id: str, error: Exception, 
                 context: Dict[str, Any] = None, **kwargs) -> None:
        """Log error with detailed context and stack trace."""
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "stack_trace": traceback.format_exc(),
            "context": context or {}
        }
        
        self.log_structured(
            "ERROR", "application_error",
            request_id=request_id,
            **error_data,
            **kwargs
        )
    
    def log_prediction(self, request_id: str, prediction_type: str,
                      model_name: str, process_time: float, 
                      feature_count: int = None, **kwargs) -> None:
        """Log ML prediction details."""
        self.log_structured(
            "INFO", "ml_prediction",
            request_id=request_id,
            prediction_type=prediction_type,
            model_name=model_name,
            process_time_ms=round(process_time * 1000, 2),
            feature_count=feature_count,
            **kwargs
        )
    
    def log_system_event(self, event_type: str, **kwargs) -> None:
        """Log system events like startup, shutdown, etc."""
        self.log_structured(
            "INFO", "system_event",
            event_type=event_type,
            **kwargs
        )


# Global structured logger instance
structured_logger = StructuredLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Enhanced middleware for comprehensive HTTP request/response logging."""
    
    def __init__(self, app):
        super().__init__(app)
        self.logger = structured_logger
        self.settings = get_settings()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with comprehensive logging."""
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Add request ID to request state for use in other parts of the app
        request.state.request_id = request_id
        
        start_time = time.time()
        
        # Extract request details
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("User-Agent", "Unknown")
        content_length = request.headers.get("Content-Length", "0")
        
        # Log request details
        self.logger.log_request(
            request_id=request_id,
            method=request.method,
            path=str(request.url.path),
            client_ip=client_ip,
            user_agent=user_agent,
            query_params=dict(request.query_params),
            content_length=int(content_length) if content_length.isdigit() else 0,
            headers=dict(request.headers) if self.settings.debug else {}
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log successful response
            self.logger.log_response(
                request_id=request_id,
                method=request.method,
                path=str(request.url.path),
                status_code=response.status_code,
                process_time=process_time,
                response_size=len(response.body) if hasattr(response, 'body') else 0
            )
            
            # Add custom headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{process_time:.4f}"
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            
            # Log error with context
            context = {
                "method": request.method,
                "path": str(request.url.path),
                "client_ip": client_ip,
                "process_time": process_time,
                "query_params": dict(request.query_params)
            }
            
            self.logger.log_error(
                request_id=request_id,
                error=e,
                context=context
            )
            
            # Re-raise the exception to be handled by FastAPI
            raise
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request headers."""
        # Check for forwarded headers (for reverse proxies like Render)
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


class ModelPredictionLogger:
    """Specialized logger for ML model predictions and performance."""
    
    def __init__(self):
        self.logger = structured_logger
    
    def log_prediction_start(self, request_id: str, prediction_type: str, 
                           input_data: Dict[str, Any]) -> None:
        """Log the start of a prediction request."""
        self.logger.log_structured(
            "INFO", "prediction_start",
            request_id=request_id,
            prediction_type=prediction_type,
            input_features=len(input_data) if isinstance(input_data, dict) else 0
        )
    
    def log_prediction_success(self, request_id: str, prediction_type: str,
                             model_name: str, process_time: float,
                             predictions: Dict[str, Any], 
                             feature_importance: Optional[Dict] = None) -> None:
        """Log successful prediction with results."""
        log_data = {
            "request_id": request_id,
            "prediction_type": prediction_type,
            "model_name": model_name,
            "process_time": process_time,
            "prediction_count": len(predictions) if predictions else 0
        }
        
        # Add feature importance if available
        if feature_importance:
            log_data["top_features"] = list(feature_importance.keys())[:5]
        
        self.logger.log_prediction(**log_data)
    
    def log_prediction_error(self, request_id: str, prediction_type: str,
                           error: Exception, input_data: Dict[str, Any]) -> None:
        """Log prediction errors with context."""
        context = {
            "prediction_type": prediction_type,
            "input_data_keys": list(input_data.keys()) if isinstance(input_data, dict) else []
        }
        
        self.logger.log_error(
            request_id=request_id,
            error=error,
            context=context
        )
    
    def log_model_performance(self, model_name: str, metrics: Dict[str, float]) -> None:
        """Log model performance metrics."""
        self.logger.log_structured(
            "INFO", "model_performance",
            model_name=model_name,
            **metrics
        )
    
    def log_batch_processing(self, request_id: str, batch_size: int, 
                           process_time: float, success_count: int, 
                           error_count: int) -> None:
        """Log batch processing results."""
        self.logger.log_structured(
            "INFO", "batch_processing",
            request_id=request_id,
            batch_size=batch_size,
            process_time=process_time,
            success_count=success_count,
            error_count=error_count,
            success_rate=success_count / batch_size if batch_size > 0 else 0
        )


# Global model prediction logger instance
model_logger = ModelPredictionLogger()