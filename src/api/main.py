"""
Main FastAPI application for Ceramic Armor ML API.
Handles app initialization, middleware setup, and route registration.
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from src.config import get_settings, get_cors_origins, is_production
from src.api.middleware import (
    LoggingMiddleware,
    MonitoringMiddleware,
    ErrorHandlingMiddleware,
    RateLimitMiddleware,
    SecurityMiddleware,
    CORSSecurityMiddleware,
    app_logger,
    error_handler
)
from src.api.documentation import setup_api_documentation


# Import comprehensive logging configuration
from src.api.config import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    logger = logging.getLogger(__name__)
    
    # Startup
    logger.info("Starting Ceramic Armor ML API...")
    settings = get_settings()
    
    # Log comprehensive configuration
    app_logger.log_configuration()
    
    # Log basic startup info
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"API version: {settings.app_version}")
    
    # Initialize ML models and services
    logger.info("Initializing ML models and services...")
    try:
        from src.ml.startup import get_startup_manager
        startup_manager = get_startup_manager()
        startup_result = await startup_manager.startup_models()
        
        if startup_result.get('initialized', False):
            logger.info(f"ML initialization successful: {startup_result.get('models_loaded', 0)} models loaded")
            app_logger.log_model_initialization(startup_result)
        else:
            logger.error(f"ML initialization failed: {startup_result.get('errors', [])}")
            app_logger.log_model_initialization({
                "status": "failed",
                "errors": startup_result.get('errors', [])
            })
    except Exception as e:
        logger.error(f"Failed to initialize ML services: {e}")
        app_logger.log_model_initialization({
            "status": "error",
            "error": str(e)
        })
    
    # Start background tasks for performance optimization and monitoring
    background_tasks = []
    
    # Cache cleanup task
    async def cache_cleanup_task():
        """Background task for periodic cache cleanup."""
        from src.api.utils.response_cache import cleanup_cache_periodically
        await cleanup_cache_periodically()
    
    # Model warmup task
    async def model_warmup_task():
        """Background task for model warmup."""
        try:
            from src.ml.async_predictor import get_async_predictor
            async_predictor = get_async_predictor()
            warmup_result = await async_predictor.warmup_models()
            logger.info(f"Model warmup completed: {warmup_result}")
        except Exception as e:
            logger.error(f"Model warmup failed: {e}")
    
    # Production monitoring task
    async def monitoring_task():
        """Background task for production monitoring and alerting."""
        try:
            from src.api.services.monitoring_service import monitoring_service
            await monitoring_service.start_monitoring()
            logger.info("Production monitoring started")
        except Exception as e:
            logger.error(f"Failed to start production monitoring: {e}")
    
    # Start background tasks
    try:
        cache_task = asyncio.create_task(cache_cleanup_task())
        warmup_task = asyncio.create_task(model_warmup_task())
        monitor_task = asyncio.create_task(monitoring_task())
        background_tasks.extend([cache_task, warmup_task, monitor_task])
        logger.info("Started background optimization and monitoring tasks")
    except Exception as e:
        logger.error(f"Failed to start background tasks: {e}")
    
    # Log successful startup
    startup_info = {
        "status": "success",
        "models_initialized": startup_result.get('initialized', False) if 'startup_result' in locals() else False,
        "background_tasks_started": len(background_tasks)
    }
    app_logger.log_startup(startup_info)
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Ceramic Armor ML API...")
    shutdown_info = {}
    
    # Stop monitoring service
    try:
        from src.api.services.monitoring_service import monitoring_service
        await monitoring_service.stop_monitoring()
        logger.info("Production monitoring stopped")
    except Exception as e:
        logger.error(f"Error stopping monitoring service: {e}")
    
    # Cancel background tasks
    for task in background_tasks:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    logger.info("Cancelled background tasks")
    
    try:
        from src.ml.startup import get_startup_manager
        startup_manager = get_startup_manager()
        shutdown_result = await startup_manager.shutdown_models()
        logger.info(f"ML shutdown result: {shutdown_result.get('status', 'unknown')}")
        shutdown_info["ml_shutdown"] = shutdown_result
    except Exception as e:
        logger.error(f"Error during ML shutdown: {e}")
        shutdown_info["ml_shutdown_error"] = str(e)
    
    # Log shutdown with performance summary
    app_logger.log_shutdown(shutdown_info)
    logger.info("Application shutdown complete")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()
    
    # Setup comprehensive logging first
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Create comprehensive API description
    api_description = """
# Ceramic Armor ML API

A comprehensive machine learning API for predicting mechanical and ballistic properties of ceramic armor materials.

## Overview

This API provides state-of-the-art machine learning predictions for ceramic armor materials including:
- **Mechanical Properties**: Fracture toughness, Vickers hardness, density, elastic modulus
- **Ballistic Properties**: V50 velocity, penetration resistance, back-face deformation, multi-hit capability

## Key Features

- 🎯 **High Accuracy**: R² > 0.85 for mechanical properties, R² > 0.80 for ballistic properties
- 🔬 **Materials Supported**: SiC, B₄C, Al₂O₃, WC, TiC, ZrO₂, TiB₂ and composites
- 📊 **Uncertainty Quantification**: Confidence intervals and prediction quality assessment
- 🧠 **Model Interpretability**: SHAP-based feature importance analysis
- 📁 **Batch Processing**: Upload CSV/Excel files for multiple material predictions
- ⚡ **Fast Response**: < 500ms for single predictions
- 🔒 **Production Ready**: Rate limiting, security middleware, comprehensive monitoring

## Supported Materials

| Material | Formula | Primary Use |
|----------|---------|-------------|
| Silicon Carbide | SiC | High hardness, thermal conductivity |
| Boron Carbide | B₄C | Lightweight, high hardness |
| Aluminum Oxide | Al₂O₃ | Cost-effective, good toughness |
| Tungsten Carbide | WC | Ultra-high hardness |
| Titanium Carbide | TiC | High temperature resistance |
| Zirconium Oxide | ZrO₂ | Transformation toughening |
| Titanium Diboride | TiB₂ | Electrical conductivity |

## Quick Start

1. **Single Prediction**: Use `/api/v1/predict/mechanical` or `/api/v1/predict/ballistic`
2. **Batch Processing**: Upload files via `/api/v1/predict/batch`
3. **System Status**: Check health at `/health` or detailed status at `/api/v1/status`

## Authentication

Currently, the API is open for research use. Rate limiting applies:
- 100 requests per hour per IP for prediction endpoints
- 10 requests per hour per IP for batch processing

## Data Requirements

### Composition (Required)
- At least one of: SiC, B₄C, Al₂O₃ (minimum 50% total)
- All fractions must sum to ≤ 1.0

### Processing Parameters (Required)
- **Temperature**: 1200-2500°C
- **Pressure**: 1-200 MPa  
- **Grain Size**: 0.1-100 μm

### Microstructure (Required)
- **Porosity**: 0-30%
- **Phase Distribution**: uniform, gradient, or layered

## Response Format

All predictions include:
- **Value**: Predicted property value
- **Unit**: Measurement unit
- **Confidence Interval**: 95% confidence bounds
- **Uncertainty**: Relative uncertainty (0-1)
- **Quality**: Prediction quality assessment

## Error Handling

The API provides detailed error responses with:
- Error type and human-readable message
- Field-specific validation errors
- Suggestions for resolution
- Request tracking IDs

## Support

For technical support, feature requests, or research collaboration:
- 📧 Email: support@ceramic-armor-ml.com
- 📚 Documentation: [Full API Guide](https://docs.ceramic-armor-ml.com)
- 🐛 Issues: [GitHub Repository](https://github.com/ceramic-armor-ml/api)

---

*Developed for advanced materials research and defense applications.*
"""
    
    # Create FastAPI app with comprehensive configuration
    app = FastAPI(
        title="Ceramic Armor ML API",
        version=settings.app_version,
        description=api_description,
        summary="Machine Learning API for Ceramic Armor Material Property Prediction",
        terms_of_service="https://ceramic-armor-ml.com/terms",
        contact={
            "name": "Ceramic Armor ML Research Team",
            "url": "https://ceramic-armor-ml.com/contact",
            "email": "support@ceramic-armor-ml.com",
        },
        license_info={
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT",
        },
        docs_url="/docs" if not is_production() else None,
        redoc_url="/redoc" if not is_production() else None,
        openapi_url="/openapi.json" if not is_production() else None,
        lifespan=lifespan,
        servers=[
            {
                "url": "https://ceramic-armor-ml-api.onrender.com",
                "description": "Production server"
            },
            {
                "url": "http://localhost:8000",
                "description": "Development server"
            }
        ] if not is_production() else [
            {
                "url": "https://ceramic-armor-ml-api.onrender.com",
                "description": "Production server"
            }
        ],
        openapi_tags=[
            {
                "name": "predictions",
                "description": "Material property prediction endpoints for mechanical and ballistic properties",
                "externalDocs": {
                    "description": "Prediction Guide",
                    "url": "https://docs.ceramic-armor-ml.com/predictions"
                }
            },
            {
                "name": "batch-processing", 
                "description": "Batch processing endpoints for multiple material predictions from uploaded files",
                "externalDocs": {
                    "description": "Batch Processing Guide",
                    "url": "https://docs.ceramic-armor-ml.com/batch"
                }
            },
            {
                "name": "health",
                "description": "System health, status monitoring, and model information endpoints",
                "externalDocs": {
                    "description": "Monitoring Guide", 
                    "url": "https://docs.ceramic-armor-ml.com/monitoring"
                }
            }
        ]
    )
    
    # Add security middleware
    if is_production():
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]  # Configure with actual domains in production
        )
    
    # Add CORS middleware (FastAPI built-in for basic functionality)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_cors_origins(),
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )
    
    # Add custom middleware (order matters - last added is executed first)
    app.add_middleware(ErrorHandlingMiddleware)    # Outermost - catches all errors
    app.add_middleware(SecurityMiddleware)         # Security headers and input sanitization
    app.add_middleware(LoggingMiddleware)          # Logs requests/responses
    app.add_middleware(MonitoringMiddleware)       # Monitors system performance
    app.add_middleware(RateLimitMiddleware)        # Rate limiting with sliding window
    # Note: CORSSecurityMiddleware can replace the built-in CORS if more control is needed
    
    # Mount static files
    try:
        app.mount("/static", StaticFiles(directory="src/static"), name="static")
        logger.info("Static files mounted at /static")
    except Exception as e:
        logger.warning(f"Could not mount static files: {e}")
    
    # Add comprehensive exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions with comprehensive logging."""
        return error_handler.handle_http_exception(request, exc)
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions with comprehensive logging and context."""
        return error_handler.handle_general_exception(request, exc)
    
    # Add specific exception handlers for custom exceptions
    from src.api.middleware import MLPredictionError, ExternalAPIError, DataValidationError
    from src.api.models.exceptions import (
        CeramicArmorMLException, ValidationException, ModelException, 
        PredictionException, DataProcessingException, FileProcessingException,
        FileUploadException, BatchProcessingException, ConfigurationException
    )
    from src.api.utils.error_formatter import ErrorFormatter
    from pydantic import ValidationError
    
    @app.exception_handler(ValidationError)
    async def validation_exception_handler(request: Request, exc: ValidationError):
        """Handle Pydantic validation errors with comprehensive formatting."""
        request_id = getattr(request.state, 'request_id', 'unknown')
        error_response = ErrorFormatter.format_validation_error(exc, request_id)
        return ErrorFormatter.create_http_exception(error_response)
    
    @app.exception_handler(ValidationException)
    async def custom_validation_exception_handler(request: Request, exc: ValidationException):
        """Handle custom validation errors."""
        request_id = getattr(request.state, 'request_id', 'unknown')
        error_response = ErrorFormatter.format_custom_exception(exc, request_id)
        return ErrorFormatter.create_http_exception(error_response)
    
    @app.exception_handler(ModelException)
    async def model_exception_handler(request: Request, exc: ModelException):
        """Handle ML model errors."""
        request_id = getattr(request.state, 'request_id', 'unknown')
        error_response = ErrorFormatter.format_custom_exception(exc, request_id)
        return ErrorFormatter.create_http_exception(error_response)
    
    @app.exception_handler(PredictionException)
    async def prediction_exception_handler(request: Request, exc: PredictionException):
        """Handle prediction errors."""
        request_id = getattr(request.state, 'request_id', 'unknown')
        error_response = ErrorFormatter.format_custom_exception(exc, request_id)
        return ErrorFormatter.create_http_exception(error_response)
    
    @app.exception_handler(FileProcessingException)
    async def file_processing_exception_handler(request: Request, exc: FileProcessingException):
        """Handle file processing errors."""
        request_id = getattr(request.state, 'request_id', 'unknown')
        error_response = ErrorFormatter.format_custom_exception(exc, request_id)
        return ErrorFormatter.create_http_exception(error_response)
    
    @app.exception_handler(BatchProcessingException)
    async def batch_processing_exception_handler(request: Request, exc: BatchProcessingException):
        """Handle batch processing errors."""
        request_id = getattr(request.state, 'request_id', 'unknown')
        error_response = ErrorFormatter.format_custom_exception(exc, request_id)
        return ErrorFormatter.create_http_exception(error_response)
    
    @app.exception_handler(MLPredictionError)
    async def ml_prediction_exception_handler(request: Request, exc: MLPredictionError):
        """Handle ML prediction errors."""
        return error_handler.handle_ml_prediction_error(
            request, exc, {
                "model_name": exc.model_name,
                "prediction_type": exc.prediction_type,
                "context": exc.context
            }
        )
    
    @app.exception_handler(ExternalAPIError)
    async def external_api_exception_handler(request: Request, exc: ExternalAPIError):
        """Handle external API errors."""
        return error_handler.handle_external_api_error(
            request, exc, exc.api_name, exc.operation or "unknown"
        )
    
    @app.exception_handler(CeramicArmorMLException)
    async def ceramic_armor_exception_handler(request: Request, exc: CeramicArmorMLException):
        """Handle all custom Ceramic Armor ML exceptions."""
        request_id = getattr(request.state, 'request_id', 'unknown')
        error_response = ErrorFormatter.format_custom_exception(exc, request_id, include_stack_trace=settings.debug)
        return ErrorFormatter.create_http_exception(error_response)
    
    # Health check endpoint is now handled by health routes
    
    # Root endpoint - serve the frontend
    @app.get("/")
    async def root():
        """Serve the main frontend interface."""
        from fastapi.responses import FileResponse
        import os
        
        static_path = os.path.join("src", "static", "index.html")
        if os.path.exists(static_path):
            return FileResponse(static_path)
        else:
            # Fallback to API info if frontend not available
            return {
                "message": f"Welcome to {settings.app_name}",
                "version": settings.app_version,
                "docs": "/docs" if not is_production() else "Documentation disabled in production",
                "health": "/health"
            }
    
    # API info endpoint
    @app.get("/api")
    async def api_info() -> Dict[str, str]:
        """API information endpoint."""
        return {
            "message": f"Welcome to {settings.app_name}",
            "version": settings.app_version,
            "docs": "/docs" if not is_production() else "Documentation disabled in production",
            "documentation": "/documentation",
            "health": "/health"
        }
    
    # Documentation page endpoint
    @app.get("/documentation")
    async def documentation_page():
        """Serve the comprehensive API documentation page."""
        from fastapi.responses import FileResponse
        import os
        
        docs_path = os.path.join("src", "static", "docs.html")
        if os.path.exists(docs_path):
            return FileResponse(docs_path)
        else:
            return {
                "message": "Documentation page not found",
                "interactive_docs": "/docs" if not is_production() else "Disabled in production",
                "redoc": "/redoc" if not is_production() else "Disabled in production"
            }
    
    # Register API routes
    from src.api.routes import predictions, upload, health
    app.include_router(predictions.router, prefix=settings.api_v1_prefix)
    app.include_router(upload.router, prefix=settings.api_v1_prefix)
    
    # Register health routes (both with and without API prefix)
    app.include_router(health.router)  # For /health endpoint
    app.include_router(health.router, prefix=settings.api_v1_prefix)  # For /api/v1/status and /api/v1/models/info
    
    # Setup comprehensive API documentation
    setup_api_documentation(app)
    
    logger.info("FastAPI application created successfully")
    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    """Run the application directly."""
    settings = get_settings()
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True
    )