"""
Error response formatting utilities for the Ceramic Armor ML API.

This module provides comprehensive error formatting with detailed context,
user-friendly messages, and actionable suggestions for error resolution.
"""

import logging
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from fastapi import HTTPException, status
from pydantic import ValidationError

from ..models.exceptions import CeramicArmorMLException
from ..models.response_models import ErrorResponse

logger = logging.getLogger(__name__)


class ErrorFormatter:
    """Comprehensive error formatter with context-aware messaging."""
    
    # Error code to HTTP status mapping
    ERROR_STATUS_MAP = {
        "VALIDATION_ERROR": status.HTTP_422_UNPROCESSABLE_ENTITY,
        "MODEL_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "PREDICTION_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "DATA_PROCESSING_ERROR": status.HTTP_422_UNPROCESSABLE_ENTITY,
        "FILE_PROCESSING_ERROR": status.HTTP_422_UNPROCESSABLE_ENTITY,
        "FILE_UPLOAD_ERROR": status.HTTP_422_UNPROCESSABLE_ENTITY,
        "BATCH_PROCESSING_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "RATE_LIMIT_ERROR": status.HTTP_429_TOO_MANY_REQUESTS,
        "CONFIGURATION_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "EXTERNAL_API_ERROR": status.HTTP_503_SERVICE_UNAVAILABLE,
        "AUTHENTICATION_ERROR": status.HTTP_401_UNAUTHORIZED,
        "AUTHORIZATION_ERROR": status.HTTP_403_FORBIDDEN,
        "NOT_FOUND_ERROR": status.HTTP_404_NOT_FOUND,
        "TIMEOUT_ERROR": status.HTTP_408_REQUEST_TIMEOUT,
        "MEMORY_ERROR": status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        "DISK_SPACE_ERROR": status.HTTP_507_INSUFFICIENT_STORAGE
    }
    
    # User-friendly error messages
    ERROR_MESSAGES = {
        "VALIDATION_ERROR": "The provided data contains validation errors",
        "MODEL_ERROR": "The machine learning model encountered an error",
        "PREDICTION_ERROR": "Failed to generate predictions",
        "DATA_PROCESSING_ERROR": "Error processing the input data",
        "FILE_PROCESSING_ERROR": "Error processing the uploaded file",
        "FILE_UPLOAD_ERROR": "File upload failed",
        "BATCH_PROCESSING_ERROR": "Batch processing encountered an error",
        "RATE_LIMIT_ERROR": "Too many requests - rate limit exceeded",
        "CONFIGURATION_ERROR": "System configuration error",
        "EXTERNAL_API_ERROR": "External service unavailable",
        "AUTHENTICATION_ERROR": "Authentication required",
        "AUTHORIZATION_ERROR": "Access denied",
        "NOT_FOUND_ERROR": "Requested resource not found",
        "TIMEOUT_ERROR": "Request timed out",
        "MEMORY_ERROR": "Request too large to process",
        "DISK_SPACE_ERROR": "Insufficient storage space"
    }
    
    # Contextual suggestions for error resolution
    ERROR_SUGGESTIONS = {
        "VALIDATION_ERROR": "Please check your input data against the API documentation and ensure all required fields are provided with valid values.",
        "MODEL_ERROR": "This appears to be a temporary issue with our prediction models. Please try again in a few moments.",
        "PREDICTION_ERROR": "Please verify your material data is complete and within valid ranges. If the issue persists, contact support.",
        "DATA_PROCESSING_ERROR": "Please check that your data format matches the expected schema and all required fields are present.",
        "FILE_PROCESSING_ERROR": "Please ensure your file is in a supported format (CSV, Excel, JSON) and contains valid data.",
        "FILE_UPLOAD_ERROR": "Please check your file size (max 50MB) and format. Try uploading again or use a different file.",
        "BATCH_PROCESSING_ERROR": "There was an issue processing your batch request. Please check your data and try again.",
        "RATE_LIMIT_ERROR": "Please wait before making additional requests. Consider reducing request frequency.",
        "CONFIGURATION_ERROR": "This is a system issue. Please contact support if the problem persists.",
        "EXTERNAL_API_ERROR": "External services are temporarily unavailable. Please try again later.",
        "AUTHENTICATION_ERROR": "Please provide valid authentication credentials.",
        "AUTHORIZATION_ERROR": "You don't have permission to access this resource.",
        "NOT_FOUND_ERROR": "The requested resource could not be found. Please check the URL and try again.",
        "TIMEOUT_ERROR": "The request took too long to process. Please try again or reduce the request size.",
        "MEMORY_ERROR": "The request is too large. Please reduce the data size or split into smaller requests.",
        "DISK_SPACE_ERROR": "Server storage is full. Please try again later or contact support."
    }
    
    @classmethod
    def format_validation_error(
        cls,
        validation_error: ValidationError,
        request_id: str,
        context: Dict[str, Any] = None
    ) -> ErrorResponse:
        """
        Format Pydantic validation errors with detailed field information.
        
        Args:
            validation_error: Pydantic ValidationError
            request_id: Request identifier
            context: Additional context information
            
        Returns:
            Formatted ErrorResponse
        """
        field_errors = {}
        details = {"validation_errors": []}
        
        for error in validation_error.errors():
            field_path = ".".join(str(loc) for loc in error["loc"])
            error_msg = error["msg"]
            error_type = error["type"]
            
            # Group errors by field
            if field_path not in field_errors:
                field_errors[field_path] = []
            field_errors[field_path].append(error_msg)
            
            # Add detailed error information
            error_detail = {
                "field": field_path,
                "message": error_msg,
                "type": error_type,
                "input_value": error.get("input")
            }
            
            if "ctx" in error:
                error_detail["context"] = error["ctx"]
            
            details["validation_errors"].append(error_detail)
        
        # Generate contextual suggestions
        suggestion = cls._generate_validation_suggestions(field_errors)
        
        # Add context if provided
        if context:
            details.update(context)
        
        return ErrorResponse(
            error="VALIDATION_ERROR",
            message=cls.ERROR_MESSAGES["VALIDATION_ERROR"],
            details=details,
            field_errors=field_errors,
            request_id=request_id,
            timestamp=datetime.now(),
            suggestion=suggestion
        )
    
    @classmethod
    def format_custom_exception(
        cls,
        exception: CeramicArmorMLException,
        request_id: str,
        include_stack_trace: bool = False
    ) -> ErrorResponse:
        """
        Format custom application exceptions.
        
        Args:
            exception: Custom exception instance
            request_id: Request identifier
            include_stack_trace: Whether to include stack trace in details
            
        Returns:
            Formatted ErrorResponse
        """
        details = exception.details.copy() if exception.details else {}
        
        # Add exception-specific information
        details.update({
            "exception_type": type(exception).__name__,
            "error_code": exception.error_code
        })
        
        # Add stack trace in debug mode
        if include_stack_trace:
            details["stack_trace"] = traceback.format_exc().split('\n')
        
        # Add specific details for different exception types
        if hasattr(exception, 'field_errors') and exception.field_errors:
            field_errors = exception.field_errors
        else:
            field_errors = None
        
        # Get appropriate message and suggestion
        message = cls.ERROR_MESSAGES.get(exception.error_code, exception.message)
        suggestion = exception.suggestion or cls.ERROR_SUGGESTIONS.get(exception.error_code)
        
        return ErrorResponse(
            error=exception.error_code,
            message=message,
            details=details,
            field_errors=field_errors,
            request_id=request_id,
            timestamp=datetime.now(),
            suggestion=suggestion
        )
    
    @classmethod
    def format_general_exception(
        cls,
        exception: Exception,
        request_id: str,
        operation_context: str = None,
        include_stack_trace: bool = False
    ) -> ErrorResponse:
        """
        Format general Python exceptions.
        
        Args:
            exception: Exception instance
            request_id: Request identifier
            operation_context: Context of the operation that failed
            include_stack_trace: Whether to include stack trace
            
        Returns:
            Formatted ErrorResponse
        """
        exception_type = type(exception).__name__
        exception_message = str(exception)
        
        # Determine error category based on exception type
        error_code = cls._categorize_exception(exception)
        
        details = {
            "exception_type": exception_type,
            "exception_message": exception_message
        }
        
        if operation_context:
            details["operation_context"] = operation_context
        
        if include_stack_trace:
            details["stack_trace"] = traceback.format_exc().split('\n')
        
        # Add specific handling for common exception types
        if isinstance(exception, FileNotFoundError):
            details["file_path"] = getattr(exception, 'filename', 'unknown')
        elif isinstance(exception, PermissionError):
            details["permission_issue"] = "File or directory access denied"
        elif isinstance(exception, MemoryError):
            details["memory_issue"] = "Insufficient memory to complete operation"
        elif isinstance(exception, TimeoutError):
            details["timeout_info"] = "Operation exceeded time limit"
        
        message = cls.ERROR_MESSAGES.get(error_code, f"An unexpected {exception_type} occurred")
        suggestion = cls.ERROR_SUGGESTIONS.get(error_code, "Please try again or contact support if the issue persists.")
        
        return ErrorResponse(
            error=error_code,
            message=message,
            details=details,
            request_id=request_id,
            timestamp=datetime.now(),
            suggestion=suggestion
        )
    
    @classmethod
    def format_http_exception(
        cls,
        http_exception: HTTPException,
        request_id: str
    ) -> ErrorResponse:
        """
        Format FastAPI HTTP exceptions.
        
        Args:
            http_exception: HTTPException instance
            request_id: Request identifier
            
        Returns:
            Formatted ErrorResponse
        """
        status_code = http_exception.status_code
        detail = http_exception.detail
        
        # Determine error code based on status
        if status_code == 400:
            error_code = "BAD_REQUEST"
        elif status_code == 401:
            error_code = "AUTHENTICATION_ERROR"
        elif status_code == 403:
            error_code = "AUTHORIZATION_ERROR"
        elif status_code == 404:
            error_code = "NOT_FOUND_ERROR"
        elif status_code == 405:
            error_code = "METHOD_NOT_ALLOWED"
        elif status_code == 408:
            error_code = "TIMEOUT_ERROR"
        elif status_code == 413:
            error_code = "REQUEST_TOO_LARGE"
        elif status_code == 422:
            error_code = "VALIDATION_ERROR"
        elif status_code == 429:
            error_code = "RATE_LIMIT_ERROR"
        elif status_code >= 500:
            error_code = "INTERNAL_SERVER_ERROR"
        else:
            error_code = "HTTP_ERROR"
        
        # Handle different detail types
        if isinstance(detail, dict):
            message = detail.get("message", cls.ERROR_MESSAGES.get(error_code, "HTTP error occurred"))
            details = detail
        else:
            message = str(detail) if detail else cls.ERROR_MESSAGES.get(error_code, "HTTP error occurred")
            details = {"http_status": status_code, "detail": detail}
        
        suggestion = cls.ERROR_SUGGESTIONS.get(error_code, "Please check your request and try again.")
        
        return ErrorResponse(
            error=error_code,
            message=message,
            details=details,
            request_id=request_id,
            timestamp=datetime.now(),
            suggestion=suggestion
        )
    
    @classmethod
    def create_http_exception(
        cls,
        error_response: ErrorResponse,
        status_code: int = None
    ) -> HTTPException:
        """
        Create HTTPException from ErrorResponse.
        
        Args:
            error_response: Formatted error response
            status_code: HTTP status code (auto-determined if None)
            
        Returns:
            HTTPException instance
        """
        if status_code is None:
            status_code = cls.ERROR_STATUS_MAP.get(error_response.error, status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        return HTTPException(
            status_code=status_code,
            detail=error_response.model_dump()
        )
    
    @classmethod
    def _categorize_exception(cls, exception: Exception) -> str:
        """Categorize exception into error code."""
        exception_type = type(exception).__name__
        exception_message = str(exception).lower()
        
        # File-related errors
        if isinstance(exception, (FileNotFoundError, IsADirectoryError, NotADirectoryError)):
            return "FILE_PROCESSING_ERROR"
        
        # Permission errors
        if isinstance(exception, PermissionError):
            return "AUTHORIZATION_ERROR"
        
        # Memory errors
        if isinstance(exception, MemoryError):
            return "MEMORY_ERROR"
        
        # Timeout errors
        if isinstance(exception, TimeoutError) or "timeout" in exception_message:
            return "TIMEOUT_ERROR"
        
        # Network/connection errors
        if "connection" in exception_message or "network" in exception_message:
            return "EXTERNAL_API_ERROR"
        
        # Validation-related errors
        if "validation" in exception_message or "invalid" in exception_message:
            return "VALIDATION_ERROR"
        
        # Model/prediction errors
        if any(keyword in exception_message for keyword in ["model", "prediction", "inference"]):
            return "MODEL_ERROR"
        
        # Data processing errors
        if any(keyword in exception_message for keyword in ["data", "processing", "parsing"]):
            return "DATA_PROCESSING_ERROR"
        
        # Default to general error
        return "INTERNAL_SERVER_ERROR"
    
    @classmethod
    def _generate_validation_suggestions(cls, field_errors: Dict[str, List[str]]) -> str:
        """Generate contextual suggestions based on validation errors."""
        suggestions = []
        
        # Composition-related suggestions
        composition_fields = [field for field in field_errors.keys() if "composition" in field.lower()]
        if composition_fields:
            suggestions.append("Ensure all composition fractions are between 0 and 1, and sum to ≤ 1.0")
        
        # Temperature-related suggestions
        temp_fields = [field for field in field_errors.keys() if "temperature" in field.lower()]
        if temp_fields:
            suggestions.append("Sintering temperature should be between 1200-2500°C")
        
        # Pressure-related suggestions
        pressure_fields = [field for field in field_errors.keys() if "pressure" in field.lower()]
        if pressure_fields:
            suggestions.append("Applied pressure should be between 1-200 MPa")
        
        # Grain size suggestions
        grain_fields = [field for field in field_errors.keys() if "grain" in field.lower()]
        if grain_fields:
            suggestions.append("Grain size should be between 0.1-100 micrometers")
        
        # Porosity suggestions
        porosity_fields = [field for field in field_errors.keys() if "porosity" in field.lower()]
        if porosity_fields:
            suggestions.append("Porosity should be between 0-30% (0.0-0.3)")
        
        # File-related suggestions
        file_fields = [field for field in field_errors.keys() if any(keyword in field.lower() for keyword in ["file", "upload", "batch"])]
        if file_fields:
            suggestions.append("Ensure your file is in CSV, Excel, or JSON format and under 50MB")
        
        if suggestions:
            return "; ".join(suggestions)
        else:
            return "Please check the API documentation for valid input ranges and formats"


# Convenience functions for common error scenarios
def create_validation_error(
    message: str,
    field_errors: Dict[str, List[str]] = None,
    request_id: str = "unknown",
    details: Dict[str, Any] = None
) -> HTTPException:
    """Create a validation error HTTPException."""
    error_response = ErrorResponse(
        error="VALIDATION_ERROR",
        message=message,
        details=details or {},
        field_errors=field_errors,
        request_id=request_id,
        timestamp=datetime.now(),
        suggestion=ErrorFormatter.ERROR_SUGGESTIONS["VALIDATION_ERROR"]
    )
    
    return ErrorFormatter.create_http_exception(error_response)


def create_model_error(
    message: str,
    model_name: str = None,
    request_id: str = "unknown",
    details: Dict[str, Any] = None
) -> HTTPException:
    """Create a model error HTTPException."""
    error_details = details or {}
    if model_name:
        error_details["model_name"] = model_name
    
    error_response = ErrorResponse(
        error="MODEL_ERROR",
        message=message,
        details=error_details,
        request_id=request_id,
        timestamp=datetime.now(),
        suggestion=ErrorFormatter.ERROR_SUGGESTIONS["MODEL_ERROR"]
    )
    
    return ErrorFormatter.create_http_exception(error_response)


def create_file_processing_error(
    message: str,
    filename: str = None,
    file_size: int = None,
    request_id: str = "unknown",
    details: Dict[str, Any] = None
) -> HTTPException:
    """Create a file processing error HTTPException."""
    error_details = details or {}
    if filename:
        error_details["filename"] = filename
    if file_size:
        error_details["file_size_bytes"] = file_size
    
    error_response = ErrorResponse(
        error="FILE_PROCESSING_ERROR",
        message=message,
        details=error_details,
        request_id=request_id,
        timestamp=datetime.now(),
        suggestion=ErrorFormatter.ERROR_SUGGESTIONS["FILE_PROCESSING_ERROR"]
    )
    
    return ErrorFormatter.create_http_exception(error_response)