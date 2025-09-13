"""
Logging service integration for ML predictions and API operations.
"""

import time
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from fastapi import Request

from src.api.middleware import (
    model_logger,
    structured_logger,
    MLPredictionError,
    ExternalAPIError
)


class PredictionLoggingService:
    """Service for logging ML prediction operations with comprehensive context."""
    
    def __init__(self):
        self.model_logger = model_logger
        self.structured_logger = structured_logger
    
    @asynccontextmanager
    async def log_prediction_operation(self, request: Request, prediction_type: str,
                                     input_data: Dict[str, Any]):
        """Context manager for logging complete prediction operations."""
        request_id = getattr(request.state, 'request_id', 'unknown')
        start_time = time.time()
        
        # Log prediction start
        self.model_logger.log_prediction_start(
            request_id=request_id,
            prediction_type=prediction_type,
            input_data=input_data
        )
        
        try:
            yield request_id
            
        except Exception as e:
            # Log prediction error
            self.model_logger.log_prediction_error(
                request_id=request_id,
                prediction_type=prediction_type,
                error=e,
                input_data=input_data
            )
            
            # Re-raise as MLPredictionError for proper handling
            if not isinstance(e, MLPredictionError):
                raise MLPredictionError(
                    f"Prediction failed: {str(e)}",
                    prediction_type=prediction_type,
                    context={"input_data_keys": list(input_data.keys())}
                ) from e
            raise
    
    def log_prediction_success(self, request_id: str, prediction_type: str,
                             model_name: str, process_time: float,
                             predictions: Dict[str, Any],
                             feature_importance: Optional[Dict] = None,
                             model_metrics: Optional[Dict] = None) -> None:
        """Log successful prediction with comprehensive details."""
        # Log the prediction success
        self.model_logger.log_prediction_success(
            request_id=request_id,
            prediction_type=prediction_type,
            model_name=model_name,
            process_time=process_time,
            predictions=predictions,
            feature_importance=feature_importance
        )
        
        # Log model performance metrics if available
        if model_metrics:
            self.model_logger.log_model_performance(
                model_name=model_name,
                metrics=model_metrics
            )
    
    def log_batch_operation(self, request_id: str, batch_size: int,
                          process_time: float, results: List[Dict[str, Any]]) -> None:
        """Log batch processing operation results."""
        success_count = sum(1 for result in results if result.get('success', False))
        error_count = batch_size - success_count
        
        self.model_logger.log_batch_processing(
            request_id=request_id,
            batch_size=batch_size,
            process_time=process_time,
            success_count=success_count,
            error_count=error_count
        )
    
    def log_feature_extraction(self, request_id: str, feature_count: int,
                             extraction_time: float, feature_names: List[str] = None) -> None:
        """Log feature extraction operation."""
        log_data = {
            "request_id": request_id,
            "feature_count": feature_count,
            "extraction_time_ms": round(extraction_time * 1000, 2)
        }
        
        if feature_names:
            log_data["sample_features"] = feature_names[:10]  # First 10 features
        
        self.structured_logger.log_structured(
            "INFO", "feature_extraction",
            **log_data
        )


class ExternalAPILoggingService:
    """Service for logging external API operations (Materials Project, NIST, etc.)."""
    
    def __init__(self):
        self.structured_logger = structured_logger
    
    @asynccontextmanager
    async def log_api_operation(self, request_id: str, api_name: str, 
                              operation: str, parameters: Dict[str, Any] = None):
        """Context manager for logging external API operations."""
        start_time = time.time()
        
        # Log API operation start
        self.structured_logger.log_structured(
            "INFO", "external_api_start",
            request_id=request_id,
            api_name=api_name,
            operation=operation,
            parameters=parameters or {}
        )
        
        try:
            yield
            
            # Log successful operation
            process_time = time.time() - start_time
            self.structured_logger.log_structured(
                "INFO", "external_api_success",
                request_id=request_id,
                api_name=api_name,
                operation=operation,
                response_time_ms=round(process_time * 1000, 2)
            )
            
        except Exception as e:
            # Log API error
            process_time = time.time() - start_time
            self.structured_logger.log_structured(
                "ERROR", "external_api_error",
                request_id=request_id,
                api_name=api_name,
                operation=operation,
                error_type=type(e).__name__,
                error_message=str(e),
                response_time_ms=round(process_time * 1000, 2)
            )
            
            # Re-raise as ExternalAPIError for proper handling
            if not isinstance(e, ExternalAPIError):
                raise ExternalAPIError(
                    f"External API operation failed: {str(e)}",
                    api_name=api_name,
                    operation=operation
                ) from e
            raise
    
    def log_api_rate_limit(self, api_name: str, retry_after: int) -> None:
        """Log API rate limiting events."""
        self.structured_logger.log_structured(
            "WARNING", "external_api_rate_limit",
            api_name=api_name,
            retry_after_seconds=retry_after
        )
    
    def log_api_quota_exceeded(self, api_name: str, quota_type: str) -> None:
        """Log API quota exceeded events."""
        self.structured_logger.log_structured(
            "ERROR", "external_api_quota_exceeded",
            api_name=api_name,
            quota_type=quota_type
        )


class FileOperationLoggingService:
    """Service for logging file upload and processing operations."""
    
    def __init__(self):
        self.structured_logger = structured_logger
    
    def log_file_upload(self, request_id: str, filename: str, file_size: int,
                       content_type: str, validation_result: Dict[str, Any]) -> None:
        """Log file upload operation."""
        self.structured_logger.log_structured(
            "INFO", "file_upload",
            request_id=request_id,
            filename=filename,
            file_size_bytes=file_size,
            file_size_mb=round(file_size / (1024 * 1024), 2),
            content_type=content_type,
            validation_passed=validation_result.get('valid', False),
            validation_errors=validation_result.get('errors', [])
        )
    
    def log_file_processing(self, request_id: str, filename: str,
                          rows_processed: int, process_time: float,
                          success_count: int, error_count: int) -> None:
        """Log file processing results."""
        self.structured_logger.log_structured(
            "INFO", "file_processing",
            request_id=request_id,
            filename=filename,
            rows_processed=rows_processed,
            process_time_seconds=round(process_time, 2),
            success_count=success_count,
            error_count=error_count,
            success_rate=success_count / max(rows_processed, 1)
        )


# Global service instances
prediction_logging = PredictionLoggingService()
api_logging = ExternalAPILoggingService()
file_logging = FileOperationLoggingService()