"""
Custom exceptions and error handling for the Ceramic Armor ML API.

This module defines custom exception classes and error handling utilities
for comprehensive error management throughout the API.
"""

from typing import Dict, List, Optional, Any
from pydantic import ValidationError
from fastapi import HTTPException, status
import traceback
import logging


class CeramicArmorMLException(Exception):
    """Base exception class for Ceramic Armor ML API."""
    
    def __init__(
        self, 
        message: str, 
        error_code: str = "GENERAL_ERROR",
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.suggestion = suggestion
        super().__init__(self.message)


class ValidationException(CeramicArmorMLException):
    """Exception for input validation errors."""
    
    def __init__(
        self, 
        message: str, 
        field_errors: Optional[Dict[str, List[str]]] = None,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details=details,
            suggestion=suggestion
        )
        self.field_errors = field_errors or {}


class ModelException(CeramicArmorMLException):
    """Exception for ML model-related errors."""
    
    def __init__(
        self, 
        message: str, 
        model_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code="MODEL_ERROR",
            details=details,
            suggestion=suggestion
        )
        self.model_name = model_name


class PredictionException(CeramicArmorMLException):
    """Exception for prediction-related errors."""
    
    def __init__(
        self, 
        message: str, 
        prediction_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code="PREDICTION_ERROR",
            details=details,
            suggestion=suggestion
        )
        self.prediction_type = prediction_type


class DataProcessingException(CeramicArmorMLException):
    """Exception for data processing errors."""
    
    def __init__(
        self, 
        message: str, 
        processing_stage: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code="DATA_PROCESSING_ERROR",
            details=details,
            suggestion=suggestion
        )
        self.processing_stage = processing_stage


class FileProcessingException(CeramicArmorMLException):
    """Exception for file upload and processing errors."""
    
    def __init__(
        self, 
        message: str, 
        file_name: Optional[str] = None,
        file_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code="FILE_PROCESSING_ERROR",
            details=details,
            suggestion=suggestion
        )
        self.file_name = file_name
        self.file_type = file_type


class RateLimitException(CeramicArmorMLException):
    """Exception for rate limiting errors."""
    
    def __init__(
        self, 
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_ERROR",
            details=details,
            suggestion=f"Please wait {retry_after} seconds before retrying" if retry_after else "Please reduce request frequency"
        )
        self.retry_after = retry_after


def create_validation_error_response(
    validation_error: ValidationError,
    request_id: str
) -> Dict[str, Any]:
    """
    Create a standardized error response from Pydantic ValidationError.
    
    Args:
        validation_error: Pydantic ValidationError instance
        request_id: Unique request identifier
        
    Returns:
        Dictionary containing formatted error response
    """
    field_errors = {}
    details = {}
    
    for error in validation_error.errors():
        field_path = ".".join(str(loc) for loc in error["loc"])
        error_msg = error["msg"]
        error_type = error["type"]
        
        if field_path not in field_errors:
            field_errors[field_path] = []
        field_errors[field_path].append(error_msg)
        
        # Add error type to details
        details[f"{field_path}_error_type"] = error_type
        if "ctx" in error:
            details[f"{field_path}_context"] = error["ctx"]
    
    # Generate helpful suggestions based on common validation errors
    suggestion = generate_validation_suggestion(field_errors)
    
    return {
        "error": "ValidationError",
        "message": "Input validation failed",
        "details": details,
        "field_errors": field_errors,
        "request_id": request_id,
        "suggestion": suggestion
    }


def generate_validation_suggestion(field_errors: Dict[str, List[str]]) -> str:
    """
    Generate helpful suggestions based on validation errors.
    
    Args:
        field_errors: Dictionary of field-specific errors
        
    Returns:
        Helpful suggestion string
    """
    suggestions = []
    
    # Check for common composition errors
    if any("composition" in field for field in field_errors.keys()):
        suggestions.append("Ensure all composition fractions are between 0 and 1, and sum to ≤ 1.0")
    
    # Check for temperature-related errors
    if any("temperature" in field.lower() for field in field_errors.keys()):
        suggestions.append("Sintering temperature should be between 1200-2500°C")
    
    # Check for pressure-related errors
    if any("pressure" in field.lower() for field in field_errors.keys()):
        suggestions.append("Applied pressure should be between 1-200 MPa")
    
    # Check for grain size errors
    if any("grain_size" in field for field in field_errors.keys()):
        suggestions.append("Grain size should be between 0.1-100 micrometers")
    
    # Check for porosity errors
    if any("porosity" in field for field in field_errors.keys()):
        suggestions.append("Porosity should be between 0-30% (0.0-0.3)")
    
    return "; ".join(suggestions) if suggestions else "Please check the API documentation for valid input ranges"


def create_http_exception(
    exception: CeramicArmorMLException,
    request_id: str,
    status_code: int = status.HTTP_400_BAD_REQUEST
) -> HTTPException:
    """
    Create FastAPI HTTPException from custom exception.
    
    Args:
        exception: Custom exception instance
        request_id: Unique request identifier
        status_code: HTTP status code
        
    Returns:
        FastAPI HTTPException
    """
    error_detail = {
        "error": exception.error_code,
        "message": exception.message,
        "details": exception.details,
        "request_id": request_id,
        "suggestion": exception.suggestion
    }
    
    # Add field errors for validation exceptions
    if isinstance(exception, ValidationException):
        error_detail["field_errors"] = exception.field_errors
    
    # Add model-specific information
    if isinstance(exception, ModelException) and exception.model_name:
        error_detail["model_name"] = exception.model_name
    
    # Add prediction type for prediction exceptions
    if isinstance(exception, PredictionException) and exception.prediction_type:
        error_detail["prediction_type"] = exception.prediction_type
    
    # Add file information for file processing exceptions
    if isinstance(exception, FileProcessingException):
        if exception.file_name:
            error_detail["file_name"] = exception.file_name
        if exception.file_type:
            error_detail["file_type"] = exception.file_type
    
    # Add retry information for rate limit exceptions
    if isinstance(exception, RateLimitException) and exception.retry_after:
        error_detail["retry_after"] = exception.retry_after
    
    return HTTPException(status_code=status_code, detail=error_detail)


def log_exception(
    exception: Exception,
    request_id: str,
    logger: logging.Logger,
    additional_context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log exception with context information.
    
    Args:
        exception: Exception instance
        request_id: Unique request identifier
        logger: Logger instance
        additional_context: Additional context information
    """
    context = {
        "request_id": request_id,
        "exception_type": type(exception).__name__,
        "exception_message": str(exception),
        **(additional_context or {})
    }
    
    # Add custom exception details if available
    if isinstance(exception, CeramicArmorMLException):
        context.update({
            "error_code": exception.error_code,
            "details": exception.details,
            "suggestion": exception.suggestion
        })
    
    # Log with stack trace for debugging
    logger.error(
        f"Exception occurred: {exception}",
        extra=context,
        exc_info=True
    )


def validate_material_composition(composition_data: Dict[str, float]) -> List[str]:
    """
    Validate material composition data beyond Pydantic validation.
    
    Args:
        composition_data: Dictionary of material compositions
        
    Returns:
        List of validation error messages
    """
    errors = []
    
    # Check for realistic composition combinations
    sic_content = composition_data.get('SiC', 0)
    b4c_content = composition_data.get('B4C', 0)
    al2o3_content = composition_data.get('Al2O3', 0)
    
    # Check for incompatible high concentrations
    if sic_content > 0.8 and b4c_content > 0.15:
        errors.append("High SiC content (>80%) with significant B4C (>15%) may be difficult to process")
    
    # Check for very low total ceramic content
    ceramic_total = sic_content + b4c_content + al2o3_content
    if ceramic_total < 0.7:
        errors.append("Total ceramic content should typically be >70% for armor applications")
    
    # Check for unrealistic pure compositions
    if len([v for v in composition_data.values() if v > 0.95]) > 1:
        errors.append("Multiple materials cannot each exceed 95% composition")
    
    return errors


def validate_processing_parameters(
    processing_data: Dict[str, Any],
    composition_data: Dict[str, float]
) -> List[str]:
    """
    Validate processing parameters in context of material composition.
    
    Args:
        processing_data: Dictionary of processing parameters
        composition_data: Dictionary of material compositions
        
    Returns:
        List of validation error messages
    """
    errors = []
    
    temperature = processing_data.get('sintering_temperature', 0)
    pressure = processing_data.get('pressure', 0)
    grain_size = processing_data.get('grain_size', 0)
    
    # Check temperature-composition compatibility
    sic_content = composition_data.get('SiC', 0)
    b4c_content = composition_data.get('B4C', 0)
    
    if sic_content > 0.5 and temperature < 1600:
        errors.append("SiC-rich compositions typically require temperatures >1600°C")
    
    if b4c_content > 0.5 and temperature > 2200:
        errors.append("B4C-rich compositions may decompose at temperatures >2200°C")
    
    # Check pressure-grain size relationships
    if pressure > 100 and grain_size > 20:
        errors.append("High pressure (>100 MPa) typically produces finer grain sizes (<20 μm)")
    
    if pressure < 20 and grain_size < 5:
        errors.append("Low pressure (<20 MPa) rarely produces very fine grains (<5 μm)")
    
    return errors


class FileUploadException(CeramicArmorMLException):
    """Exception for file upload and processing errors."""
    
    def __init__(
        self, 
        message: str, 
        file_name: Optional[str] = None,
        file_size: Optional[int] = None,
        file_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code="FILE_UPLOAD_ERROR",
            details=details,
            suggestion=suggestion
        )
        self.file_name = file_name
        self.file_size = file_size
        self.file_type = file_type


class BatchProcessingException(CeramicArmorMLException):
    """Exception for batch processing errors."""
    
    def __init__(
        self, 
        message: str, 
        batch_id: Optional[str] = None,
        processed_count: Optional[int] = None,
        failed_count: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code="BATCH_PROCESSING_ERROR",
            details=details,
            suggestion=suggestion
        )
        self.batch_id = batch_id
        self.processed_count = processed_count
        self.failed_count = failed_count


class ConfigurationException(CeramicArmorMLException):
    """Exception for configuration and setup errors."""
    
    def __init__(
        self, 
        message: str, 
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details=details,
            suggestion=suggestion
        )
        self.config_key = config_key
        self.config_value = config_value


def validate_file_upload(
    file_content: bytes,
    filename: str,
    max_size_mb: int = 50,
    allowed_extensions: List[str] = None
) -> List[str]:
    """
    Validate uploaded file before processing.
    
    Args:
        file_content: Raw file content
        filename: Original filename
        max_size_mb: Maximum file size in MB
        allowed_extensions: List of allowed file extensions
        
    Returns:
        List of validation error messages
    """
    errors = []
    
    if allowed_extensions is None:
        allowed_extensions = ['.csv', '.xlsx', '.xls', '.json']
    
    # Check file size
    file_size_mb = len(file_content) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        errors.append(f"File size ({file_size_mb:.1f}MB) exceeds maximum allowed size ({max_size_mb}MB)")
    
    # Check file extension
    file_extension = filename.lower().split('.')[-1] if '.' in filename else ''
    if f'.{file_extension}' not in allowed_extensions:
        errors.append(f"File type '.{file_extension}' not supported. Allowed types: {', '.join(allowed_extensions)}")
    
    # Check if file is empty
    if len(file_content) == 0:
        errors.append("File is empty")
    
    # Basic content validation for CSV files
    if file_extension == 'csv':
        try:
            content_str = file_content.decode('utf-8')
            if not content_str.strip():
                errors.append("CSV file appears to be empty")
            elif len(content_str.split('\n')) < 2:
                errors.append("CSV file must contain at least a header row and one data row")
        except UnicodeDecodeError:
            # Try other encodings
            try:
                file_content.decode('latin-1')
            except UnicodeDecodeError:
                errors.append("CSV file encoding not supported. Please use UTF-8 or Latin-1 encoding")
    
    return errors


def validate_batch_data_quality(df, max_rows: int = 10000) -> Dict[str, Any]:
    """
    Validate batch data quality and provide detailed feedback.
    
    Args:
        df: DataFrame to validate
        max_rows: Maximum allowed rows
        
    Returns:
        Dictionary with validation results and quality metrics
    """
    validation_result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'quality_metrics': {},
        'suggestions': []
    }
    
    # Check row count
    if len(df) == 0:
        validation_result['errors'].append("Dataset is empty")
        validation_result['valid'] = False
        return validation_result
    
    if len(df) > max_rows:
        validation_result['errors'].append(f"Dataset has {len(df)} rows, maximum allowed is {max_rows}")
        validation_result['valid'] = False
    
    # Check for required columns
    required_cols = {
        'composition': ['SiC', 'B4C', 'Al2O3'],
        'processing': ['sintering_temperature', 'pressure', 'grain_size'],
        'microstructure': ['porosity', 'phase_distribution']
    }
    
    missing_cols = []
    for category, cols in required_cols.items():
        for col in cols:
            if col not in df.columns:
                missing_cols.append(f"{col} ({category})")
    
    if missing_cols:
        validation_result['errors'].append(f"Missing required columns: {', '.join(missing_cols)}")
        validation_result['valid'] = False
    
    # Data quality checks
    if validation_result['valid']:
        # Check for missing values
        missing_data = df.isnull().sum()
        critical_missing = missing_data[missing_data > len(df) * 0.5]  # More than 50% missing
        if len(critical_missing) > 0:
            validation_result['warnings'].append(f"Columns with >50% missing data: {list(critical_missing.index)}")
        
        # Check composition validity
        comp_cols = [col for col in ['SiC', 'B4C', 'Al2O3', 'WC', 'TiC'] if col in df.columns]
        if comp_cols:
            comp_sums = df[comp_cols].sum(axis=1)
            invalid_comp = (comp_sums > 1.1) | (comp_sums < 0.01)
            invalid_count = invalid_comp.sum()
            if invalid_count > 0:
                validation_result['warnings'].append(f"{invalid_count} rows have invalid composition sums")
                if invalid_count > len(df) * 0.1:  # More than 10% invalid
                    validation_result['suggestions'].append("Consider reviewing composition data - many rows have invalid totals")
        
        # Check temperature ranges
        if 'sintering_temperature' in df.columns:
            temp_out_of_range = ((df['sintering_temperature'] < 1200) | (df['sintering_temperature'] > 2500)).sum()
            if temp_out_of_range > 0:
                validation_result['warnings'].append(f"{temp_out_of_range} rows have temperatures outside valid range (1200-2500°C)")
        
        # Check porosity ranges
        if 'porosity' in df.columns:
            porosity_out_of_range = ((df['porosity'] < 0) | (df['porosity'] > 0.3)).sum()
            if porosity_out_of_range > 0:
                validation_result['warnings'].append(f"{porosity_out_of_range} rows have porosity outside valid range (0-30%)")
        
        # Calculate quality metrics
        validation_result['quality_metrics'] = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_data_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'complete_rows': len(df.dropna()),
            'duplicate_rows': df.duplicated().sum(),
            'numeric_columns': len(df.select_dtypes(include=['number']).columns),
            'text_columns': len(df.select_dtypes(include=['object']).columns)
        }
        
        # Add suggestions based on quality metrics
        if validation_result['quality_metrics']['missing_data_percentage'] > 20:
            validation_result['suggestions'].append("High percentage of missing data - consider data cleaning")
        
        if validation_result['quality_metrics']['duplicate_rows'] > 0:
            validation_result['suggestions'].append(f"Found {validation_result['quality_metrics']['duplicate_rows']} duplicate rows - consider removing duplicates")
    
    return validation_result