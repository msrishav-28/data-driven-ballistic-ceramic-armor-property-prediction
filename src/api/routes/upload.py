"""
File upload and batch processing endpoints for the Ceramic Armor ML API.

This module provides endpoints for batch predictions from uploaded CSV files,
including file validation, parsing, and downloadable results generation.
"""

import logging
import uuid
import os
import tempfile
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import io

import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, Depends, status, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse

from ..models.request_models import BatchPredictionRequest, CompositionModel, ProcessingModel, MicrostructureModel
from ..models.response_models import (
    BatchPredictionResponse, PredictionStatus, ErrorResponse
)
from ..models.exceptions import (
    FileProcessingException, BatchProcessingException, ValidationException
)
from ..utils.validation import validate_batch_file_comprehensive
from ..utils.error_formatter import ErrorFormatter, create_file_processing_error
from ...ml.predictor import get_predictor
from ...feature_engineering.simple_feature_extractor import CeramicFeatureExtractor
from ...config import get_settings

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["batch-processing"])

# Global feature extractor instance
feature_extractor = CeramicFeatureExtractor()

# In-memory storage for batch results (in production, use Redis or database)
batch_results_storage: Dict[str, Dict[str, Any]] = {}

# Temporary directory for batch files
BATCH_FILES_DIR = Path(tempfile.gettempdir()) / "ceramic_armor_batch"
BATCH_FILES_DIR.mkdir(exist_ok=True)


def get_feature_extractor() -> CeramicFeatureExtractor:
    """Dependency to get feature extractor instance."""
    return feature_extractor


def generate_batch_id() -> str:
    """Generate unique batch processing ID."""
    return f"batch_{uuid.uuid4().hex[:12]}"


def validate_csv_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate CSV file structure and required columns.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Validation result with status and details
    """
    required_composition_cols = ['SiC', 'B4C', 'Al2O3']
    required_processing_cols = ['sintering_temperature', 'pressure', 'grain_size']
    required_microstructure_cols = ['porosity', 'phase_distribution']
    
    validation_result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'row_count': len(df),
        'column_count': len(df.columns)
    }
    
    # Check for required composition columns
    missing_comp_cols = [col for col in required_composition_cols if col not in df.columns]
    if missing_comp_cols:
        validation_result['errors'].append(f"Missing required composition columns: {missing_comp_cols}")
        validation_result['valid'] = False
    
    # Check for required processing columns
    missing_proc_cols = [col for col in required_processing_cols if col not in df.columns]
    if missing_proc_cols:
        validation_result['errors'].append(f"Missing required processing columns: {missing_proc_cols}")
        validation_result['valid'] = False
    
    # Check for required microstructure columns
    missing_micro_cols = [col for col in required_microstructure_cols if col not in df.columns]
    if missing_micro_cols:
        validation_result['errors'].append(f"Missing required microstructure columns: {missing_micro_cols}")
        validation_result['valid'] = False
    
    # Check for empty DataFrame
    if len(df) == 0:
        validation_result['errors'].append("CSV file is empty")
        validation_result['valid'] = False
    
    # Check for reasonable row count
    if len(df) > 10000:
        validation_result['warnings'].append(f"Large file with {len(df)} rows may take significant time to process")
    
    # Validate data types and ranges for key columns
    if validation_result['valid']:
        # Check composition columns sum to reasonable values
        comp_cols = [col for col in required_composition_cols if col in df.columns]
        if comp_cols:
            comp_sums = df[comp_cols].sum(axis=1)
            invalid_comp_rows = comp_sums[(comp_sums > 1.1) | (comp_sums < 0.01)]
            if len(invalid_comp_rows) > 0:
                validation_result['warnings'].append(f"{len(invalid_comp_rows)} rows have invalid composition sums")
        
        # Check temperature ranges
        if 'sintering_temperature' in df.columns:
            temp_col = df['sintering_temperature']
            invalid_temps = temp_col[(temp_col < 1200) | (temp_col > 2500)]
            if len(invalid_temps) > 0:
                validation_result['warnings'].append(f"{len(invalid_temps)} rows have temperatures outside valid range (1200-2500°C)")
        
        # Check porosity ranges
        if 'porosity' in df.columns:
            porosity_col = df['porosity']
            invalid_porosity = porosity_col[(porosity_col < 0) | (porosity_col > 0.3)]
            if len(invalid_porosity) > 0:
                validation_result['warnings'].append(f"{len(invalid_porosity)} rows have porosity outside valid range (0-0.3)")
    
    return validation_result


def parse_uploaded_file(file_content: bytes, filename: str) -> pd.DataFrame:
    """
    Parse uploaded file content into DataFrame.
    
    Args:
        file_content: Raw file content
        filename: Original filename for format detection
        
    Returns:
        Parsed DataFrame
        
    Raises:
        HTTPException: If file parsing fails
    """
    try:
        file_extension = Path(filename).suffix.lower()
        
        if file_extension == '.csv':
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(io.BytesIO(file_content), encoding=encoding)
                    logger.info(f"Successfully parsed CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Could not decode CSV file with any supported encoding")
                
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(io.BytesIO(file_content))
            
        elif file_extension == '.json':
            df = pd.read_json(io.BytesIO(file_content))
            
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Clean column names (remove spaces, convert to lowercase)
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
        
        # Handle common column name variations
        column_mapping = {
            'sic': 'SiC',
            'b4c': 'B4C', 
            'al2o3': 'Al2O3',
            'wc': 'WC',
            'tic': 'TiC',
            'temperature': 'sintering_temperature',
            'temp': 'sintering_temperature',
            'grain': 'grain_size',
            'pore': 'porosity'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        logger.info(f"Parsed file {filename} with {len(df)} rows and {len(df.columns)} columns")
        return df
        
    except Exception as e:
        logger.error(f"Failed to parse file {filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to parse file: {str(e)}"
        )


def convert_dataframe_row_to_request(row: pd.Series) -> Dict[str, Any]:
    """
    Convert DataFrame row to prediction request format.
    
    Args:
        row: DataFrame row with material data
        
    Returns:
        Dictionary in PredictionRequest format
    """
    try:
        # Extract composition data
        composition_data = {}
        for comp in ['SiC', 'B4C', 'Al2O3', 'WC', 'TiC', 'ZrO2', 'TiB2']:
            composition_data[comp] = float(row.get(comp, 0.0))
        
        # Extract processing data
        processing_data = {
            'sintering_temperature': float(row.get('sintering_temperature', 1800)),
            'pressure': float(row.get('pressure', 50)),
            'grain_size': float(row.get('grain_size', 10)),
            'holding_time': float(row.get('holding_time', 60)),
            'heating_rate': float(row.get('heating_rate', 10)),
            'atmosphere': str(row.get('atmosphere', 'argon'))
        }
        
        # Extract microstructure data
        microstructure_data = {
            'porosity': float(row.get('porosity', 0.02)),
            'phase_distribution': str(row.get('phase_distribution', 'uniform')),
            'interface_quality': str(row.get('interface_quality', 'good')),
            'pore_size': float(row.get('pore_size', 1.0)),
            'connectivity': float(row.get('connectivity', 0.1))
        }
        
        return {
            'composition': composition_data,
            'processing': processing_data,
            'microstructure': microstructure_data,
            'include_uncertainty': True,
            'include_feature_importance': False,  # Disabled for batch to reduce size
            'prediction_type': 'both'
        }
        
    except Exception as e:
        logger.error(f"Failed to convert row to request format: {e}")
        raise ValueError(f"Invalid row data: {str(e)}")


async def process_batch_predictions(
    batch_id: str,
    df: pd.DataFrame,
    batch_request: BatchPredictionRequest,
    extractor: CeramicFeatureExtractor
) -> None:
    """
    Process batch predictions in background.
    
    Args:
        batch_id: Unique batch identifier
        df: DataFrame with material data
        batch_request: Batch processing configuration
        extractor: Feature extractor instance
    """
    start_time = datetime.now()
    
    try:
        logger.info(f"Starting batch processing for {batch_id} with {len(df)} materials")
        
        # Initialize results storage
        batch_results_storage[batch_id] = {
            'status': 'processing',
            'total_rows': len(df),
            'processed_rows': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'results': [],
            'errors': [],
            'start_time': start_time,
            'processing_time': None,
            'file_path': None
        }
        
        predictor = get_predictor()
        results = []
        
        # Process each row
        for idx, row in df.iterrows():
            try:
                # Convert row to request format
                request_data = convert_dataframe_row_to_request(row)
                
                # Create DataFrame for feature extraction
                material_df = pd.DataFrame([{
                    **{f"comp_{k}": v for k, v in request_data['composition'].items()},
                    **{f"proc_{k}": v for k, v in request_data['processing'].items()},
                    **{f"micro_{k}": v for k, v in request_data['microstructure'].items()}
                }])
                
                # Add composition string for matminer
                comp_parts = []
                for element, fraction in request_data['composition'].items():
                    if fraction > 0:
                        comp_parts.append(f"{element}{fraction}")
                material_df['composition'] = "".join(comp_parts) if comp_parts else "SiC1"
                
                # Extract features
                features_df = extractor.extract_all_features(material_df)
                feature_cols = [col for col in features_df.columns 
                               if not col.startswith(('comp_', 'proc_', 'micro_', 'composition'))]
                features = features_df[feature_cols].fillna(0).values
                
                # Make predictions
                result_row = {'row_index': idx}
                result_row.update(request_data['composition'])
                result_row.update(request_data['processing'])
                result_row.update(request_data['microstructure'])
                
                if batch_request.prediction_type in ['mechanical', 'both']:
                    mech_predictions = predictor.predict_mechanical_properties(
                        features=features,
                        include_uncertainty=batch_request.include_uncertainty,
                        include_feature_importance=False
                    )
                    
                    # Add mechanical predictions to result
                    for prop, pred_data in mech_predictions.get('predictions', {}).items():
                        result_row[f"mech_{prop}_value"] = pred_data.get('value', 0)
                        result_row[f"mech_{prop}_unit"] = pred_data.get('unit', '')
                        if batch_request.include_uncertainty:
                            result_row[f"mech_{prop}_uncertainty"] = pred_data.get('uncertainty', 0)
                            ci = pred_data.get('confidence_interval', [0, 0])
                            result_row[f"mech_{prop}_ci_lower"] = ci[0]
                            result_row[f"mech_{prop}_ci_upper"] = ci[1]
                
                if batch_request.prediction_type in ['ballistic', 'both']:
                    ball_predictions = predictor.predict_ballistic_properties(
                        features=features,
                        include_uncertainty=batch_request.include_uncertainty,
                        include_feature_importance=False
                    )
                    
                    # Add ballistic predictions to result
                    for prop, pred_data in ball_predictions.get('predictions', {}).items():
                        result_row[f"ball_{prop}_value"] = pred_data.get('value', 0)
                        result_row[f"ball_{prop}_unit"] = pred_data.get('unit', '')
                        if batch_request.include_uncertainty:
                            result_row[f"ball_{prop}_uncertainty"] = pred_data.get('uncertainty', 0)
                            ci = pred_data.get('confidence_interval', [0, 0])
                            result_row[f"ball_{prop}_ci_lower"] = ci[0]
                            result_row[f"ball_{prop}_ci_upper"] = ci[1]
                
                results.append(result_row)
                batch_results_storage[batch_id]['successful_predictions'] += 1
                
            except Exception as e:
                logger.error(f"Failed to process row {idx}: {e}")
                error_row = {'row_index': idx, 'error': str(e)}
                results.append(error_row)
                batch_results_storage[batch_id]['failed_predictions'] += 1
                batch_results_storage[batch_id]['errors'].append(f"Row {idx}: {str(e)}")
            
            batch_results_storage[batch_id]['processed_rows'] += 1
            
            # Add small delay to prevent overwhelming the system
            if idx % 10 == 0:
                await asyncio.sleep(0.01)
        
        # Save results to file
        results_df = pd.DataFrame(results)
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"batch_results_{batch_id}_{timestamp}.{batch_request.output_format}"
        output_path = BATCH_FILES_DIR / output_filename
        
        # Save in requested format
        if batch_request.output_format == 'csv':
            results_df.to_csv(output_path, index=False)
        elif batch_request.output_format == 'xlsx':
            results_df.to_excel(output_path, index=False)
        elif batch_request.output_format == 'json':
            results_df.to_json(output_path, orient='records', indent=2)
        
        # Update batch status
        processing_time = (datetime.now() - start_time).total_seconds()
        batch_results_storage[batch_id].update({
            'status': 'completed',
            'results': results,
            'processing_time': processing_time,
            'file_path': str(output_path),
            'file_size_mb': output_path.stat().st_size / (1024 * 1024),
            'expires_at': datetime.now() + timedelta(hours=24)  # Files expire after 24 hours
        })
        
        logger.info(f"Completed batch processing for {batch_id} in {processing_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Batch processing failed for {batch_id}: {e}")
        batch_results_storage[batch_id].update({
            'status': 'failed',
            'error': str(e),
            'processing_time': (datetime.now() - start_time).total_seconds()
        })


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload file for batch predictions",
    description="Upload a CSV, Excel, or JSON file containing multiple materials for batch property predictions."
)
async def upload_batch_prediction_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="CSV, Excel, or JSON file with material data"),
    file_format: str = Form("csv", pattern="^(csv|xlsx|json)$", description="Input file format"),
    output_format: str = Form("csv", pattern="^(csv|xlsx|json)$", description="Output file format"),
    include_uncertainty: bool = Form(True, description="Include uncertainty quantification"),
    include_feature_importance: bool = Form(False, description="Include feature importance"),
    prediction_type: str = Form("both", pattern="^(mechanical|ballistic|both)$", description="Prediction type"),
    max_rows: int = Form(1000, ge=1, le=10000, description="Maximum rows to process"),
    extractor: CeramicFeatureExtractor = Depends(get_feature_extractor)
) -> BatchPredictionResponse:
    """
    Upload and process a file containing multiple materials for batch predictions.
    
    This endpoint accepts CSV, Excel, or JSON files containing material data and
    processes them in the background to generate predictions for all materials.
    
    Required columns in the input file:
    - Composition: SiC, B4C, Al2O3 (minimum required)
    - Processing: sintering_temperature, pressure, grain_size
    - Microstructure: porosity, phase_distribution
    
    Optional columns:
    - Additional composition: WC, TiC, ZrO2, TiB2
    - Additional processing: holding_time, heating_rate, atmosphere
    - Additional microstructure: interface_quality, pore_size, connectivity
    
    The processing is done asynchronously, and results can be downloaded using
    the provided download URL once processing is complete.
    """
    batch_id = generate_batch_id()
    start_time = datetime.now()
    
    logger.info(f"Starting batch upload processing for {batch_id}, file: {file.filename}")
    
    try:
        # Validate file upload
        from src.api.models.exceptions import validate_file_upload, validate_batch_data_quality
        
        # Read file content first
        file_content = await file.read()
        
        # Validate file before processing
        file_errors = validate_file_upload(
            file_content=file_content,
            filename=file.filename or "unknown",
            max_size_mb=50,
            allowed_extensions=['.csv', '.xlsx', '.xls', '.json']
        )
        
        if file_errors:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": "FileValidationError",
                    "message": "File validation failed",
                    "errors": file_errors,
                    "request_id": batch_id,
                    "suggestion": "Please ensure your file is in the correct format (CSV, Excel, or JSON) and under 50MB"
                }
            )
        
        # Parse file
        df = parse_uploaded_file(file_content, file.filename)
        
        # Comprehensive file and data validation
        validation_context = validate_batch_file_comprehensive(df, file.filename, batch_id)
        
        if not validation_context.is_valid():
            # Create detailed validation error
            field_errors = {}
            for error in validation_context.errors:
                field = error.get('field', 'file_validation')
                if field not in field_errors:
                    field_errors[field] = []
                field_errors[field].append(error['message'])
            
            raise FileProcessingException(
                message="File validation failed",
                file_name=file.filename,
                file_size=len(file_content),
                file_type=file.content_type,
                details=validation_context.get_summary(),
                suggestion="; ".join(validation_context.suggestions) if validation_context.suggestions else None
            )
        
        # Limit number of rows
        if len(df) > max_rows:
            df = df.head(max_rows)
            logger.warning(f"Truncated file to {max_rows} rows for batch {batch_id}")
        
        # Create batch request object
        batch_request = BatchPredictionRequest(
            file_format=file_format,
            output_format=output_format,
            include_uncertainty=include_uncertainty,
            include_feature_importance=include_feature_importance,
            prediction_type=prediction_type,
            max_rows=max_rows
        )
        
        # Start background processing
        background_tasks.add_task(
            process_batch_predictions,
            batch_id,
            df,
            batch_request,
            extractor
        )
        
        # Return immediate response
        response = BatchPredictionResponse(
            status=PredictionStatus.SUCCESS,
            total_processed=0,  # Will be updated during processing
            successful_predictions=0,
            failed_predictions=0,
            processing_time_seconds=0,
            download_url=None,  # Will be available after processing
            file_size_mb=None,
            expires_at=None,
            summary_statistics=None,
            request_id=batch_id,
            timestamp=datetime.now(),
            warnings=validation_result.get('warnings', [])
        )
        
        logger.info(f"Initiated batch processing for {batch_id} with {len(df)} materials")
        return response
        
    except HTTPException:
        raise
        
    except UnicodeDecodeError as e:
        logger.error(f"File encoding error for batch {batch_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "FileEncodingError",
                "message": "Unable to read file due to encoding issues",
                "details": {"encoding_error": str(e)},
                "request_id": batch_id,
                "suggestion": "Please save your file with UTF-8 encoding or try a different file format"
            }
        )
        
    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty data error for batch {batch_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "EmptyDataError",
                "message": "The uploaded file appears to be empty or contains no valid data",
                "details": {"pandas_error": str(e)},
                "request_id": batch_id,
                "suggestion": "Please ensure your file contains data rows with the required columns"
            }
        )
        
    except pd.errors.ParserError as e:
        logger.error(f"File parsing error for batch {batch_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "FileParsingError",
                "message": "Unable to parse the uploaded file",
                "details": {"parser_error": str(e)},
                "request_id": batch_id,
                "suggestion": "Please check that your file is properly formatted and not corrupted"
            }
        )
        
    except MemoryError as e:
        logger.error(f"Memory error processing batch {batch_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "error": "MemoryError",
                "message": "File is too large to process",
                "details": {"memory_error": str(e)},
                "request_id": batch_id,
                "suggestion": "Please reduce the file size or split it into smaller files"
            }
        )
        
    except Exception as e:
        logger.error(f"Batch upload failed for {batch_id}: {e}", exc_info=True)
        
        # Determine error type for better user feedback
        error_message = str(e)
        if "permission" in error_message.lower():
            error_type = "PermissionError"
            suggestion = "File access permission denied. Please try uploading the file again."
        elif "timeout" in error_message.lower():
            error_type = "TimeoutError"
            suggestion = "File upload timed out. Please try with a smaller file or check your connection."
        elif "disk" in error_message.lower() or "space" in error_message.lower():
            error_type = "DiskSpaceError"
            suggestion = "Insufficient disk space to process the file. Please try again later."
        else:
            error_type = "BatchUploadError"
            suggestion = "An unexpected error occurred during file upload. Please try again or contact support."
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": error_type,
                "message": "Failed to process batch upload",
                "details": {
                    "error_type": type(e).__name__,
                    "error_message": error_message,
                    "file_name": file.filename,
                    "file_size_mb": len(file_content) / (1024 * 1024) if 'file_content' in locals() else 0
                },
                "request_id": batch_id,
                "timestamp": datetime.now().isoformat(),
                "suggestion": suggestion
            }
        )


@router.get(
    "/predict/batch/{batch_id}/status",
    response_model=BatchPredictionResponse,
    summary="Get batch processing status",
    description="Check the status of a batch prediction job and get download information when ready."
)
async def get_batch_status(batch_id: str) -> BatchPredictionResponse:
    """
    Get the current status of a batch prediction job.
    
    Returns processing progress, completion status, and download information
    when the batch processing is complete.
    """
    if batch_id not in batch_results_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch job {batch_id} not found"
        )
    
    batch_data = batch_results_storage[batch_id]
    
    # Determine status
    if batch_data['status'] == 'processing':
        status_enum = PredictionStatus.PARTIAL
    elif batch_data['status'] == 'completed':
        status_enum = PredictionStatus.SUCCESS
    else:
        status_enum = PredictionStatus.FAILED
    
    # Generate download URL if completed
    download_url = None
    if batch_data['status'] == 'completed' and batch_data.get('file_path'):
        settings = get_settings()
        download_url = f"{settings.api_v1_prefix}/predict/batch/{batch_id}/download"
    
    # Calculate summary statistics if available
    summary_stats = None
    if batch_data['status'] == 'completed' and batch_data.get('results'):
        try:
            results_df = pd.DataFrame(batch_data['results'])
            # Calculate basic statistics for mechanical properties
            mech_cols = [col for col in results_df.columns if col.startswith('mech_') and col.endswith('_value')]
            if mech_cols:
                summary_stats = {}
                for col in mech_cols:
                    prop_name = col.replace('mech_', '').replace('_value', '')
                    if col in results_df.columns:
                        summary_stats[f"avg_{prop_name}"] = float(results_df[col].mean())
                        summary_stats[f"std_{prop_name}"] = float(results_df[col].std())
        except Exception as e:
            logger.warning(f"Failed to calculate summary statistics for {batch_id}: {e}")
    
    response = BatchPredictionResponse(
        status=status_enum,
        total_processed=batch_data.get('total_rows', 0),
        successful_predictions=batch_data.get('successful_predictions', 0),
        failed_predictions=batch_data.get('failed_predictions', 0),
        processing_time_seconds=batch_data.get('processing_time', 0),
        download_url=download_url,
        file_size_mb=batch_data.get('file_size_mb'),
        expires_at=batch_data.get('expires_at'),
        summary_statistics=summary_stats,
        request_id=batch_id,
        timestamp=datetime.now(),
        warnings=batch_data.get('errors', [])[:5]  # Limit to first 5 errors
    )
    
    return response


@router.get(
    "/predict/batch/{batch_id}/download",
    response_class=FileResponse,
    summary="Download batch prediction results",
    description="Download the results file for a completed batch prediction job."
)
async def download_batch_results(batch_id: str) -> FileResponse:
    """
    Download the results file for a completed batch prediction job.
    
    The file will be in the format specified during upload (CSV, Excel, or JSON).
    Download links expire after 24 hours.
    """
    if batch_id not in batch_results_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch job {batch_id} not found"
        )
    
    batch_data = batch_results_storage[batch_id]
    
    if batch_data['status'] != 'completed':
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Batch job {batch_id} is not completed yet"
        )
    
    file_path = batch_data.get('file_path')
    if not file_path or not Path(file_path).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Results file not found or has expired"
        )
    
    # Check if file has expired
    expires_at = batch_data.get('expires_at')
    if expires_at and datetime.now() > expires_at:
        # Clean up expired file
        try:
            Path(file_path).unlink()
            del batch_results_storage[batch_id]
        except Exception as e:
            logger.warning(f"Failed to clean up expired file {file_path}: {e}")
        
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Results file has expired"
        )
    
    filename = Path(file_path).name
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/octet-stream'
    )


@router.delete(
    "/predict/batch/{batch_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Cancel or clean up batch job",
    description="Cancel a running batch job or clean up completed job data."
)
async def delete_batch_job(batch_id: str) -> None:
    """
    Cancel a running batch prediction job or clean up completed job data.
    
    This will stop processing (if still running) and remove all associated data.
    """
    if batch_id not in batch_results_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch job {batch_id} not found"
        )
    
    batch_data = batch_results_storage[batch_id]
    
    # Clean up result file if it exists
    file_path = batch_data.get('file_path')
    if file_path and Path(file_path).exists():
        try:
            Path(file_path).unlink()
            logger.info(f"Cleaned up result file for batch {batch_id}")
        except Exception as e:
            logger.warning(f"Failed to clean up file {file_path}: {e}")
    
    # Remove from storage
    del batch_results_storage[batch_id]
    logger.info(f"Deleted batch job {batch_id}")