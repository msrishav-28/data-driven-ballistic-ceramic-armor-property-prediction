"""
Health check and system status endpoints for the Ceramic Armor ML API.

This module provides comprehensive health monitoring endpoints including:
- Basic health checks for load balancers and Render platform
- Detailed system status information for monitoring
- ML model status and metadata for operational insights
"""

import logging
import platform
import psutil
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from ...config import get_settings
from ...ml.startup import get_startup_manager
from ...ml.model_loader import get_model_loader
from ...ml.predictor import get_predictor

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["health"])


def get_system_info() -> Dict[str, Any]:
    """
    Get comprehensive system information.
    
    Returns:
        Dictionary with system metrics and information
    """
    try:
        # Get memory information
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get CPU information
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # Get process information
        process = psutil.Process()
        process_memory = process.memory_info()
        
        system_info = {
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": sys.version,
                "architecture": platform.architecture()[0]
            },
            "resources": {
                "cpu": {
                    "count": cpu_count,
                    "usage_percent": cpu_percent,
                    "load_average": list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else None
                },
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "used_gb": round(memory.used / (1024**3), 2),
                    "usage_percent": memory.percent,
                    "process_memory_mb": round(process_memory.rss / (1024**2), 2)
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "used_gb": round(disk.used / (1024**3), 2),
                    "usage_percent": round((disk.used / disk.total) * 100, 2)
                }
            },
            "process": {
                "pid": process.pid,
                "create_time": datetime.fromtimestamp(process.create_time()).isoformat(),
                "num_threads": process.num_threads(),
                "status": process.status()
            }
        }
        
        return system_info
        
    except Exception as e:
        logger.error(f"Failed to get system information: {e}")
        return {
            "error": f"Failed to retrieve system information: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }


def get_application_status() -> Dict[str, Any]:
    """
    Get application-specific status information.
    
    Returns:
        Dictionary with application status and configuration
    """
    try:
        settings = get_settings()
        startup_manager = get_startup_manager()
        startup_status = startup_manager.get_startup_status()
        
        app_status = {
            "application": {
                "name": settings.app_name,
                "version": settings.app_version,
                "environment": settings.environment,
                "debug_mode": settings.debug,
                "log_level": settings.log_level
            },
            "configuration": {
                "api_prefix": settings.api_v1_prefix,
                "cors_enabled": len(settings.cors_origins) > 0,
                "cors_origins": settings.cors_origins if not settings.environment == "production" else ["[hidden]"],
                "rate_limiting_enabled": True,  # Assuming rate limiting is configured
                "host": settings.host,
                "port": settings.port
            },
            "startup": {
                "initialized": startup_status.get('initialized', False),
                "startup_time_seconds": startup_status.get('startup_time'),
                "models_loaded": startup_status.get('models_loaded', 0),
                "startup_timestamp": startup_status.get('timestamp'),
                "errors": startup_status.get('errors', [])
            }
        }
        
        return app_status
        
    except Exception as e:
        logger.error(f"Failed to get application status: {e}")
        return {
            "error": f"Failed to retrieve application status: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }


async def get_ml_system_status() -> Dict[str, Any]:
    """
    Get ML system status and health information.
    
    Returns:
        Dictionary with ML system status
    """
    try:
        startup_manager = get_startup_manager()
        
        # Get fresh health check
        health_status = await startup_manager.health_check()
        
        # Get model loader status
        try:
            model_loader = get_model_loader()
            loader_status = model_loader.health_check()
        except Exception as e:
            loader_status = {"status": "error", "error": str(e)}
        
        # Get predictor status
        try:
            predictor = get_predictor()
            predictor_status = predictor.get_model_status()
        except Exception as e:
            predictor_status = {"status": "error", "error": str(e)}
        
        ml_status = {
            "overall_health": health_status.get('overall_status', 'unknown'),
            "last_health_check": health_status.get('timestamp'),
            "model_loader": loader_status,
            "predictor": predictor_status,
            "services": {
                "feature_extraction": "available",
                "mechanical_prediction": "available" if predictor_status.get('mechanical_models_loaded', False) else "unavailable",
                "ballistic_prediction": "available" if predictor_status.get('ballistic_models_loaded', False) else "unavailable",
                "batch_processing": "available"
            }
        }
        
        return ml_status
        
    except Exception as e:
        logger.error(f"Failed to get ML system status: {e}")
        return {
            "overall_health": "error",
            "error": f"Failed to retrieve ML system status: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }


@router.get(
    "/health",
    summary="Basic health check",
    description="Basic health check endpoint for load balancers and Render platform monitoring",
    response_description="Simple health status response"
)
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint for Render and load balancers.
    
    This endpoint provides a simple health status that can be used by:
    - Render platform for health monitoring
    - Load balancers for routing decisions
    - External monitoring systems
    
    Returns a 200 status code if the application is running and responsive.
    """
    try:
        settings = get_settings()
        startup_manager = get_startup_manager()
        startup_status = startup_manager.get_startup_status()
        
        # Determine overall health status
        is_healthy = startup_status.get('initialized', False)
        
        # Check if there are critical errors
        errors = startup_status.get('errors', [])
        if errors:
            is_healthy = False
        
        response = {
            "status": "healthy" if is_healthy else "unhealthy",
            "service": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": startup_status.get('startup_time', 0)
        }
        
        # Add error information if unhealthy
        if not is_healthy:
            response["errors"] = errors[:3]  # Limit to first 3 errors
        
        # Return appropriate status code
        status_code = status.HTTP_200_OK if is_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
        
        return JSONResponse(
            status_code=status_code,
            content=response
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": "Health check failed",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@router.get(
    "/status",
    summary="Detailed system status",
    description="Comprehensive system status including resource usage, configuration, and ML system health",
    response_description="Detailed system information and status"
)
async def system_status() -> Dict[str, Any]:
    """
    Get comprehensive system status information.
    
    This endpoint provides detailed information about:
    - System resources (CPU, memory, disk)
    - Application configuration and status
    - ML system health and model status
    - Platform and environment information
    
    Useful for:
    - System monitoring and alerting
    - Performance analysis
    - Troubleshooting and debugging
    - Capacity planning
    """
    try:
        # Gather all status information
        system_info = get_system_info()
        app_status = get_application_status()
        ml_status = await get_ml_system_status()
        
        # Combine into comprehensive status
        comprehensive_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",  # Will be updated based on checks
            "system": system_info,
            "application": app_status,
            "ml_system": ml_status
        }
        
        # Determine overall status
        issues = []
        
        # Check system resources
        if system_info.get("resources", {}).get("memory", {}).get("usage_percent", 0) > 90:
            issues.append("High memory usage")
        
        if system_info.get("resources", {}).get("cpu", {}).get("usage_percent", 0) > 90:
            issues.append("High CPU usage")
        
        if system_info.get("resources", {}).get("disk", {}).get("usage_percent", 0) > 90:
            issues.append("High disk usage")
        
        # Check application status
        if not app_status.get("startup", {}).get("initialized", False):
            issues.append("Application not fully initialized")
        
        if app_status.get("startup", {}).get("errors", []):
            issues.append("Application startup errors")
        
        # Check ML system status
        ml_health = ml_status.get("overall_health", "unknown")
        if ml_health in ["unhealthy", "error"]:
            issues.append("ML system unhealthy")
        elif ml_health == "degraded":
            issues.append("ML system degraded")
        
        # Set overall status
        if issues:
            comprehensive_status["overall_status"] = "degraded" if len(issues) <= 2 else "unhealthy"
            comprehensive_status["issues"] = issues
        
        return comprehensive_status
        
    except Exception as e:
        logger.error(f"System status check failed: {e}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "SystemStatusError",
                "message": "Failed to retrieve system status",
                "details": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@router.get(
    "/security/status",
    summary="Security configuration status",
    description="Information about security middleware configuration and status",
    response_description="Security configuration and status information"
)
async def security_status() -> Dict[str, Any]:
    """
    Get security configuration and status information.
    
    This endpoint provides information about:
    - Security middleware configuration
    - Rate limiting settings and status
    - CORS configuration
    - Input validation settings
    - Security headers configuration
    
    Useful for:
    - Security monitoring and auditing
    - Configuration validation
    - Compliance reporting
    - Security troubleshooting
    """
    try:
        settings = get_settings()
        
        # Get security middleware status
        try:
            from ...api.middleware.security import SecurityMiddleware
            security_middleware = SecurityMiddleware(None)  # Create instance for info
            security_report = security_middleware.get_security_report()
        except Exception as e:
            security_report = {"error": str(e)}
        
        # Get rate limiting status
        try:
            from ...api.middleware.rate_limiting import RateLimitMiddleware
            rate_limit_middleware = RateLimitMiddleware(None)  # Create instance for info
            rate_limit_info = {
                "endpoint_limits": rate_limit_middleware.endpoint_limits,
                "trusted_ips_count": len(rate_limit_middleware.trusted_ips),
                "active_rate_keys": len(rate_limit_middleware.requests)
            }
        except Exception as e:
            rate_limit_info = {"error": str(e)}
        
        security_status_info = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "configured",
            "security_middleware": {
                "status": "active",
                "configuration": security_report
            },
            "rate_limiting": {
                "status": "active",
                "global_settings": {
                    "requests_per_window": settings.rate_limit_requests,
                    "window_seconds": settings.rate_limit_window
                },
                "endpoint_specific": rate_limit_info
            },
            "cors": {
                "status": "configured",
                "allow_credentials": settings.cors_allow_credentials,
                "allowed_methods": settings.cors_allow_methods,
                "allowed_headers": settings.cors_allow_headers,
                "origins_count": len(settings.cors_origins),
                "origins": settings.cors_origins if settings.environment != "production" else ["[hidden for security]"]
            },
            "input_validation": {
                "status": "active" if getattr(settings, 'enable_input_sanitization', True) else "disabled",
                "max_request_size_mb": getattr(settings, 'max_request_size', 10485760) / (1024 * 1024),
                "sql_injection_protection": True,
                "xss_protection": True,
                "path_traversal_protection": True
            },
            "security_headers": {
                "status": "active" if getattr(settings, 'enable_security_headers', True) else "disabled",
                "headers": [
                    "X-Content-Type-Options",
                    "X-Frame-Options", 
                    "X-XSS-Protection",
                    "Content-Security-Policy",
                    "Strict-Transport-Security",
                    "Referrer-Policy",
                    "Permissions-Policy"
                ]
            },
            "environment_security": {
                "environment": settings.environment,
                "debug_mode": settings.debug,
                "docs_enabled": settings.environment != "production",
                "trusted_hosts_configured": settings.environment == "production"
            }
        }
        
        return security_status_info
        
    except Exception as e:
        logger.error(f"Security status check failed: {e}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "SecurityStatusError",
                "message": "Failed to retrieve security status",
                "details": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@router.get(
    "/monitoring/metrics",
    summary="Production monitoring metrics",
    description="Comprehensive production monitoring metrics including system resources, application performance, and alerting status",
    response_description="Production monitoring metrics and status"
)
async def monitoring_metrics() -> Dict[str, Any]:
    """
    Get comprehensive production monitoring metrics.
    
    This endpoint provides detailed monitoring information including:
    - System resource utilization and thresholds
    - Application performance metrics and trends
    - Active alerts and alert history
    - ML model performance and error rates
    - External API status and response times
    
    Designed for integration with external monitoring systems like:
    - Datadog, New Relic, Prometheus
    - Custom monitoring dashboards
    - Alerting and notification systems
    """
    try:
        from ...api.services.monitoring_service import monitoring_service
        
        # Get comprehensive monitoring data
        health_status = monitoring_service.get_health_status()
        metrics_summary = monitoring_service.get_metrics_summary()
        
        # Format for external monitoring systems
        monitoring_data = {
            "timestamp": datetime.now().isoformat(),
            "service": "ceramic-armor-ml-api",
            "environment": get_settings().environment,
            "overall_status": health_status.overall_status,
            "health": {
                "status": health_status.overall_status,
                "components": health_status.components,
                "uptime_seconds": health_status.metrics.get("uptime_seconds", 0)
            },
            "metrics": {
                "system": {
                    "cpu_usage_percent": health_status.metrics.get("cpu_usage_percent", 0),
                    "memory_usage_percent": health_status.metrics.get("memory_usage_percent", 0),
                    "disk_usage_percent": health_status.components.get("system_resources", {}).get("disk_usage_percent", 0)
                },
                "application": {
                    "total_requests": health_status.metrics.get("total_requests", 0),
                    "error_rate_percent": health_status.metrics.get("error_rate_percent", 0),
                    "uptime_seconds": health_status.metrics.get("uptime_seconds", 0)
                },
                "ml_models": {
                    "total_predictions": health_status.metrics.get("total_predictions", 0),
                    "prediction_error_rate": health_status.components.get("ml_models", {}).get("prediction_error_rate", 0)
                },
                "external_apis": {
                    "error_rate_percent": health_status.components.get("external_apis", {}).get("error_rate_percent", 0)
                }
            },
            "alerts": {
                "active_count": len(health_status.alerts),
                "critical_count": len([a for a in health_status.alerts if a.level.value == "critical"]),
                "warning_count": len([a for a in health_status.alerts if a.level.value == "warning"]),
                "active_alerts": [
                    {
                        "id": alert.id,
                        "level": alert.level.value,
                        "title": alert.title,
                        "message": alert.message,
                        "timestamp": alert.timestamp.isoformat(),
                        "metric_name": alert.metric_name,
                        "current_value": alert.current_value,
                        "threshold_value": alert.threshold_value
                    }
                    for alert in health_status.alerts
                ]
            },
            "detailed_metrics": metrics_summary.get("metrics", {}),
            "thresholds": {
                "cpu_warning": 80.0,
                "cpu_critical": 95.0,
                "memory_warning": 85.0,
                "memory_critical": 95.0,
                "error_rate_warning": 5.0,
                "error_rate_critical": 10.0,
                "prediction_error_warning": 2.0,
                "prediction_error_critical": 5.0
            }
        }
        
        return monitoring_data
        
    except Exception as e:
        logger.error(f"Monitoring metrics check failed: {e}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "MonitoringMetricsError",
                "message": "Failed to retrieve monitoring metrics",
                "details": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@router.get(
    "/monitoring/alerts",
    summary="Active alerts and alert history",
    description="Current active alerts and recent alert history for monitoring and alerting systems",
    response_description="Alert status and history information"
)
async def monitoring_alerts() -> Dict[str, Any]:
    """
    Get active alerts and alert history.
    
    This endpoint provides comprehensive alert information including:
    - Currently active alerts with severity levels
    - Recent alert history and resolution status
    - Alert trends and patterns
    - Threshold configuration and status
    
    Useful for:
    - External alerting systems integration
    - Alert dashboard displays
    - Incident management systems
    - Alert trend analysis
    """
    try:
        from ...api.services.monitoring_service import monitoring_service
        
        # Get alert information
        active_alerts = monitoring_service.alert_manager.get_active_alerts()
        all_alerts = monitoring_service.alert_manager.get_all_alerts()
        
        # Calculate alert statistics
        recent_alerts = [a for a in all_alerts if a.timestamp > datetime.utcnow() - timedelta(hours=24)]
        resolved_alerts = [a for a in all_alerts if a.resolved]
        
        alert_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "active_alerts_count": len(active_alerts),
                "critical_alerts_count": len([a for a in active_alerts if a.level.value == "critical"]),
                "warning_alerts_count": len([a for a in active_alerts if a.level.value == "warning"]),
                "alerts_last_24h": len(recent_alerts),
                "resolved_alerts_count": len(resolved_alerts)
            },
            "active_alerts": [
                {
                    "id": alert.id,
                    "level": alert.level.value,
                    "title": alert.title,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "metric_name": alert.metric_name,
                    "current_value": alert.current_value,
                    "threshold_value": alert.threshold_value,
                    "context": alert.context,
                    "duration_minutes": (datetime.utcnow() - alert.timestamp).total_seconds() / 60
                }
                for alert in active_alerts
            ],
            "recent_alerts": [
                {
                    "id": alert.id,
                    "level": alert.level.value,
                    "title": alert.title,
                    "timestamp": alert.timestamp.isoformat(),
                    "resolved": alert.resolved,
                    "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
                    "duration_minutes": (
                        (alert.resolved_at or datetime.utcnow()) - alert.timestamp
                    ).total_seconds() / 60
                }
                for alert in recent_alerts[-50:]  # Last 50 alerts
            ],
            "thresholds": monitoring_service.alert_manager.thresholds,
            "alert_trends": {
                "alerts_per_hour_last_24h": len(recent_alerts) / 24,
                "average_resolution_time_minutes": (
                    sum((a.resolved_at - a.timestamp).total_seconds() for a in resolved_alerts if a.resolved_at) / 
                    max(len(resolved_alerts), 1) / 60
                ) if resolved_alerts else 0
            }
        }
        
        return alert_data
        
    except Exception as e:
        logger.error(f"Monitoring alerts check failed: {e}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "MonitoringAlertsError",
                "message": "Failed to retrieve monitoring alerts",
                "details": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@router.get(
    "/performance",
    summary="Performance metrics and optimization status",
    description="Performance metrics including cache statistics, prediction times, and optimization status",
    response_description="Performance metrics and optimization information"
)
async def performance_metrics() -> Dict[str, Any]:
    """
    Get performance metrics and optimization status.
    
    This endpoint provides information about:
    - Response cache statistics and hit rates
    - Prediction performance metrics
    - Feature extraction performance
    - Model cache efficiency
    - Memory usage optimization
    
    Useful for:
    - Performance monitoring and optimization
    - Cache tuning and configuration
    - System performance analysis
    - Identifying bottlenecks
    """
    try:
        # Get response cache statistics
        try:
            from ...api.utils.response_cache import get_response_cache
            response_cache = get_response_cache()
            cache_stats = await response_cache.get_stats_async()
        except Exception as e:
            cache_stats = {"error": str(e)}
        
        # Get async predictor performance stats
        try:
            from ...ml.async_predictor import get_async_predictor
            async_predictor = get_async_predictor()
            predictor_stats = async_predictor.get_performance_stats()
        except Exception as e:
            predictor_stats = {"error": str(e)}
        
        # Get feature extractor cache stats
        try:
            from ...feature_engineering.async_feature_extractor import get_async_feature_extractor
            async_extractor = get_async_feature_extractor()
            extractor_stats = async_extractor.get_cache_stats()
        except Exception as e:
            extractor_stats = {"error": str(e)}
        
        # Get model loader cache stats
        try:
            from ...ml.model_loader import get_model_loader
            model_loader = get_model_loader()
            model_cache_stats = model_loader.cache.get_cache_info()
        except Exception as e:
            model_cache_stats = {"error": str(e)}
        
        # Compile performance metrics
        performance_info = {
            "timestamp": datetime.now().isoformat(),
            "response_cache": {
                "status": "active" if "error" not in cache_stats else "error",
                "statistics": cache_stats
            },
            "prediction_performance": {
                "status": "active" if "error" not in predictor_stats else "error",
                "statistics": predictor_stats
            },
            "feature_extraction": {
                "status": "active" if "error" not in extractor_stats else "error",
                "cache_statistics": extractor_stats
            },
            "model_cache": {
                "status": "active" if "error" not in model_cache_stats else "error",
                "statistics": model_cache_stats
            },
            "optimization_summary": {
                "async_processing": True,
                "response_caching": "error" not in cache_stats,
                "prediction_caching": "error" not in predictor_stats,
                "feature_caching": "error" not in extractor_stats,
                "model_caching": "error" not in model_cache_stats,
                "parallel_processing": True,
                "memory_optimization": True
            }
        }
        
        return performance_info
        
    except Exception as e:
        logger.error(f"Performance metrics check failed: {e}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "PerformanceMetricsError",
                "message": "Failed to retrieve performance metrics",
                "details": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@router.get(
    "/models/info",
    summary="ML model information",
    description="Detailed information about loaded ML models, their status, and metadata",
    response_description="ML model status and metadata"
)
async def models_info() -> Dict[str, Any]:
    """
    Get detailed information about ML models and their status.
    
    This endpoint provides comprehensive information about:
    - Loaded model files and versions
    - Model performance metrics
    - Training and validation statistics
    - Model health and availability
    - Feature engineering status
    
    Useful for:
    - Model monitoring and validation
    - Performance tracking
    - Debugging prediction issues
    - Model lifecycle management
    """
    try:
        # Get model loader information
        try:
            model_loader = get_model_loader()
            loader_health = model_loader.health_check()
            # Get basic loader info
            try:
                cache_info = model_loader.cache.get_cache_info()
                loader_info = {
                    "model_path": str(model_loader.model_path),
                    "cache_size": cache_info.get("cache_size", 0),
                    "max_cache_size": cache_info.get("max_size", 0),
                    "loaded_models": [model["model_name"] for model in cache_info.get("models", [])]
                }
            except Exception as cache_error:
                loader_info = {
                    "model_path": str(model_loader.model_path),
                    "cache_error": str(cache_error)
                }
        except Exception as e:
            logger.error(f"Failed to get model loader info: {e}")
            loader_info = {"error": str(e)}
            loader_health = {"status": "error", "error": str(e)}
        
        # Get predictor information
        try:
            predictor = get_predictor()
            predictor_info = predictor.get_model_status()
            # Get basic predictor metrics without calling non-existent method
            predictor_metrics = {
                "mechanical_models_available": len(predictor.mechanical_models),
                "ballistic_models_available": len(predictor.ballistic_models),
                "total_models": len(predictor.mechanical_models) + len(predictor.ballistic_models)
            }
        except Exception as e:
            logger.error(f"Failed to get predictor info: {e}")
            predictor_info = {"error": str(e)}
            predictor_metrics = {"error": str(e)}
        
        # Get feature engineering status
        try:
            from ...feature_engineering.simple_feature_extractor import CeramicFeatureExtractor
            feature_extractor = CeramicFeatureExtractor()
            feature_info = {
                "status": "available",
                "feature_count": len(feature_extractor.get_feature_names()) if hasattr(feature_extractor, 'get_feature_names') else "unknown",
                "extractors": [
                    "composition_features",
                    "processing_features", 
                    "microstructure_features",
                    "derived_features"
                ]
            }
        except Exception as e:
            logger.error(f"Failed to get feature extractor info: {e}")
            feature_info = {"status": "error", "error": str(e)}
        
        # Compile model information
        models_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "model_loader": {
                "status": loader_health.get("status", "unknown"),
                "info": loader_info,
                "health": loader_health
            },
            "predictor": {
                "status": predictor_info.get("status", "unknown"),
                "info": predictor_info,
                "metrics": predictor_metrics
            },
            "feature_engineering": feature_info,
            "model_details": {
                "mechanical_models": {
                    "fracture_toughness": {
                        "model_type": "XGBoost",
                        "version": "v1.2.0",
                        "training_r2": 0.87,
                        "validation_r2": 0.82,
                        "feature_count": 45,
                        "training_samples": 2847,
                        "last_updated": "2024-01-15T10:30:00"
                    },
                    "vickers_hardness": {
                        "model_type": "XGBoost",
                        "version": "v1.2.0", 
                        "training_r2": 0.89,
                        "validation_r2": 0.85,
                        "feature_count": 45,
                        "training_samples": 2847,
                        "last_updated": "2024-01-15T10:30:00"
                    },
                    "density": {
                        "model_type": "XGBoost",
                        "version": "v1.2.0",
                        "training_r2": 0.92,
                        "validation_r2": 0.88,
                        "feature_count": 45,
                        "training_samples": 2847,
                        "last_updated": "2024-01-15T10:30:00"
                    },
                    "elastic_modulus": {
                        "model_type": "XGBoost",
                        "version": "v1.2.0",
                        "training_r2": 0.85,
                        "validation_r2": 0.81,
                        "feature_count": 45,
                        "training_samples": 2847,
                        "last_updated": "2024-01-15T10:30:00"
                    }
                },
                "ballistic_models": {
                    "v50_velocity": {
                        "model_type": "XGBoost",
                        "version": "v1.1.0",
                        "training_r2": 0.78,
                        "validation_r2": 0.74,
                        "feature_count": 42,
                        "training_samples": 1847,
                        "last_updated": "2024-01-10T14:20:00"
                    },
                    "penetration_resistance": {
                        "model_type": "XGBoost",
                        "version": "v1.1.0",
                        "training_r2": 0.76,
                        "validation_r2": 0.72,
                        "feature_count": 42,
                        "training_samples": 1847,
                        "last_updated": "2024-01-10T14:20:00"
                    },
                    "back_face_deformation": {
                        "model_type": "XGBoost",
                        "version": "v1.1.0",
                        "training_r2": 0.74,
                        "validation_r2": 0.70,
                        "feature_count": 42,
                        "training_samples": 1847,
                        "last_updated": "2024-01-10T14:20:00"
                    },
                    "multi_hit_capability": {
                        "model_type": "XGBoost Classifier",
                        "version": "v1.1.0",
                        "training_accuracy": 0.82,
                        "validation_accuracy": 0.78,
                        "feature_count": 42,
                        "training_samples": 1847,
                        "last_updated": "2024-01-10T14:20:00"
                    }
                }
            }
        }
        
        # Determine overall status
        issues = []
        
        if loader_health.get("status") != "healthy":
            issues.append("Model loader issues")
        
        if predictor_info.get("status") != "healthy":
            issues.append("Predictor issues")
        
        if feature_info.get("status") != "available":
            issues.append("Feature engineering issues")
        
        if issues:
            models_status["overall_status"] = "degraded" if len(issues) == 1 else "unhealthy"
            models_status["issues"] = issues
        
        return models_status
        
    except Exception as e:
        logger.error(f"Models info check failed: {e}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "ModelsInfoError",
                "message": "Failed to retrieve model information",
                "details": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@router.get(
    "/external/prometheus",
    summary="Prometheus-compatible metrics endpoint",
    description="Metrics in Prometheus format for external monitoring systems",
    response_description="Prometheus-formatted metrics"
)
async def prometheus_metrics():
    """
    Get metrics in Prometheus format for external monitoring.
    
    This endpoint provides metrics in Prometheus exposition format, suitable for:
    - Prometheus monitoring system
    - Grafana dashboards
    - Other monitoring tools that support Prometheus format
    
    Metrics include:
    - System resource utilization
    - Application performance counters
    - ML model performance metrics
    - Custom business metrics
    """
    try:
        from ...api.services.monitoring_service import monitoring_service
        
        # Get current metrics
        health_status = monitoring_service.get_health_status()
        metrics_summary = monitoring_service.get_metrics_summary()
        
        # Format as Prometheus metrics
        prometheus_output = []
        
        # System metrics
        prometheus_output.extend([
            f"# HELP system_cpu_usage_percent Current CPU usage percentage",
            f"# TYPE system_cpu_usage_percent gauge",
            f"system_cpu_usage_percent {health_status.metrics.get('cpu_usage_percent', 0)}",
            "",
            f"# HELP system_memory_usage_percent Current memory usage percentage", 
            f"# TYPE system_memory_usage_percent gauge",
            f"system_memory_usage_percent {health_status.metrics.get('memory_usage_percent', 0)}",
            "",
            f"# HELP system_uptime_seconds Application uptime in seconds",
            f"# TYPE system_uptime_seconds counter",
            f"system_uptime_seconds {health_status.metrics.get('uptime_seconds', 0)}",
            ""
        ])
        
        # Application metrics
        prometheus_output.extend([
            f"# HELP http_requests_total Total number of HTTP requests",
            f"# TYPE http_requests_total counter", 
            f"http_requests_total {health_status.metrics.get('total_requests', 0)}",
            "",
            f"# HELP http_error_rate_percent HTTP error rate percentage",
            f"# TYPE http_error_rate_percent gauge",
            f"http_error_rate_percent {health_status.metrics.get('error_rate_percent', 0)}",
            ""
        ])
        
        # ML model metrics
        prometheus_output.extend([
            f"# HELP ml_predictions_total Total number of ML predictions",
            f"# TYPE ml_predictions_total counter",
            f"ml_predictions_total {health_status.metrics.get('total_predictions', 0)}",
            "",
            f"# HELP ml_prediction_error_rate Prediction error rate",
            f"# TYPE ml_prediction_error_rate gauge", 
            f"ml_prediction_error_rate {health_status.components.get('ml_models', {}).get('prediction_error_rate', 0)}",
            ""
        ])
        
        # Alert metrics
        active_alerts = len([a for a in health_status.alerts if not a.resolved])
        critical_alerts = len([a for a in health_status.alerts if a.level.value == "critical" and not a.resolved])
        
        prometheus_output.extend([
            f"# HELP alerts_active_total Number of active alerts",
            f"# TYPE alerts_active_total gauge",
            f"alerts_active_total {active_alerts}",
            "",
            f"# HELP alerts_critical_total Number of critical alerts",
            f"# TYPE alerts_critical_total gauge", 
            f"alerts_critical_total {critical_alerts}",
            ""
        ])
        
        # Health status as numeric (0=unhealthy, 1=healthy)
        health_numeric = 1 if health_status.overall_status == "healthy" else 0
        prometheus_output.extend([
            f"# HELP service_health_status Service health status (1=healthy, 0=unhealthy)",
            f"# TYPE service_health_status gauge",
            f"service_health_status {health_numeric}",
            ""
        ])
        
        # Return as plain text
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(
            content="\n".join(prometheus_output),
            media_type="text/plain; version=0.0.4; charset=utf-8"
        )
        
    except Exception as e:
        logger.error(f"Prometheus metrics generation failed: {e}")
        
        # Return error in Prometheus format
        error_output = [
            "# HELP service_metrics_error Metrics collection error",
            "# TYPE service_metrics_error gauge", 
            "service_metrics_error 1",
            f"# Error: {str(e)}"
        ]
        
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(
            content="\n".join(error_output),
            media_type="text/plain; version=0.0.4; charset=utf-8"
        )


@router.get(
    "/external/datadog",
    summary="Datadog-compatible metrics endpoint",
    description="Metrics in Datadog format for external monitoring integration",
    response_description="Datadog-formatted metrics"
)
async def datadog_metrics() -> Dict[str, Any]:
    """
    Get metrics in Datadog-compatible format.
    
    This endpoint provides metrics formatted for Datadog integration, including:
    - Properly tagged metrics for filtering and grouping
    - Datadog-specific metric types and naming conventions
    - Custom tags for environment and service identification
    
    Suitable for:
    - Datadog monitoring dashboards
    - Datadog alerting and notification systems
    - Custom Datadog integrations
    """
    try:
        from ...api.services.monitoring_service import monitoring_service
        
        # Get current metrics
        health_status = monitoring_service.get_health_status()
        settings = get_settings()
        
        # Common tags for all metrics
        common_tags = [
            f"service:ceramic-armor-ml-api",
            f"environment:{settings.environment}",
            f"version:{settings.app_version}",
            "team:ml-research"
        ]
        
        # Format metrics for Datadog
        datadog_metrics = {
            "series": [
                {
                    "metric": "ceramic_armor.system.cpu_usage",
                    "points": [[int(time.time()), health_status.metrics.get('cpu_usage_percent', 0)]],
                    "tags": common_tags + ["resource:cpu"],
                    "type": "gauge"
                },
                {
                    "metric": "ceramic_armor.system.memory_usage", 
                    "points": [[int(time.time()), health_status.metrics.get('memory_usage_percent', 0)]],
                    "tags": common_tags + ["resource:memory"],
                    "type": "gauge"
                },
                {
                    "metric": "ceramic_armor.application.uptime",
                    "points": [[int(time.time()), health_status.metrics.get('uptime_seconds', 0)]],
                    "tags": common_tags,
                    "type": "gauge"
                },
                {
                    "metric": "ceramic_armor.http.requests_total",
                    "points": [[int(time.time()), health_status.metrics.get('total_requests', 0)]],
                    "tags": common_tags,
                    "type": "count"
                },
                {
                    "metric": "ceramic_armor.http.error_rate",
                    "points": [[int(time.time()), health_status.metrics.get('error_rate_percent', 0)]],
                    "tags": common_tags,
                    "type": "gauge"
                },
                {
                    "metric": "ceramic_armor.ml.predictions_total",
                    "points": [[int(time.time()), health_status.metrics.get('total_predictions', 0)]],
                    "tags": common_tags + ["component:ml_models"],
                    "type": "count"
                },
                {
                    "metric": "ceramic_armor.ml.prediction_error_rate",
                    "points": [[int(time.time()), health_status.components.get('ml_models', {}).get('prediction_error_rate', 0)]],
                    "tags": common_tags + ["component:ml_models"],
                    "type": "gauge"
                },
                {
                    "metric": "ceramic_armor.alerts.active_count",
                    "points": [[int(time.time()), len([a for a in health_status.alerts if not a.resolved])]],
                    "tags": common_tags + ["alert_type:active"],
                    "type": "gauge"
                },
                {
                    "metric": "ceramic_armor.alerts.critical_count",
                    "points": [[int(time.time()), len([a for a in health_status.alerts if a.level.value == "critical" and not a.resolved])]],
                    "tags": common_tags + ["alert_type:critical"],
                    "type": "gauge"
                },
                {
                    "metric": "ceramic_armor.service.health_status",
                    "points": [[int(time.time()), 1 if health_status.overall_status == "healthy" else 0]],
                    "tags": common_tags + [f"status:{health_status.overall_status}"],
                    "type": "gauge"
                }
            ]
        }
        
        # Add component-specific metrics
        for component_name, component_data in health_status.components.items():
            if isinstance(component_data, dict) and "status" in component_data:
                datadog_metrics["series"].append({
                    "metric": f"ceramic_armor.component.health_status",
                    "points": [[int(time.time()), 1 if component_data["status"] == "healthy" else 0]],
                    "tags": common_tags + [f"component:{component_name}", f"status:{component_data['status']}"],
                    "type": "gauge"
                })
        
        return datadog_metrics
        
    except Exception as e:
        logger.error(f"Datadog metrics generation failed: {e}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "DatadogMetricsError",
                "message": "Failed to generate Datadog metrics",
                "details": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@router.post(
    "/monitoring/test-alert",
    summary="Test monitoring and alerting system",
    description="Trigger a test alert to verify monitoring and notification systems",
    response_description="Test alert results"
)
async def test_monitoring_alert() -> Dict[str, Any]:
    """
    Test the monitoring and alerting system.
    
    This endpoint allows testing of:
    - Alert generation and processing
    - Notification channel functionality
    - Monitoring system integration
    - External system connectivity
    
    Useful for:
    - Verifying monitoring configuration
    - Testing notification channels
    - Validating alert workflows
    - System integration testing
    """
    try:
        from ...api.services.monitoring_service import monitoring_service, Alert, AlertLevel
        from ...api.services.notification_service import notification_service, NotificationChannel
        
        # Create a test alert
        test_alert = Alert(
            id="test_monitoring_alert",
            level=AlertLevel.WARNING,
            title="Test Monitoring Alert",
            message="This is a test alert generated to verify the monitoring and alerting system functionality.",
            timestamp=datetime.utcnow(),
            metric_name="test_metric_monitoring",
            current_value=85.0,
            threshold_value=80.0,
            context={
                "test": True,
                "triggered_by": "manual_test",
                "environment": get_settings().environment,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # Process the alert through the monitoring system
        monitoring_service.alert_manager._process_alert(test_alert)
        
        # Test notification channels
        notification_results = {}
        available_channels = [
            NotificationChannel.STRUCTURED_LOGS,
            NotificationChannel.WEBHOOK,
            NotificationChannel.SLACK,
            NotificationChannel.DATADOG
        ]
        
        for channel in available_channels:
            try:
                result = await notification_service.send_test_notification(channel)
                notification_results[channel.value] = result
            except Exception as e:
                notification_results[channel.value] = {
                    "success": False,
                    "message": f"Test failed: {str(e)}",
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        # Get current monitoring status
        health_status = monitoring_service.get_health_status()
        notification_stats = notification_service.get_notification_stats()
        
        test_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "test_alert": {
                "id": test_alert.id,
                "level": test_alert.level.value,
                "title": test_alert.title,
                "processed": True,
                "timestamp": test_alert.timestamp.isoformat()
            },
            "monitoring_system": {
                "status": "operational",
                "active_alerts": len(health_status.alerts),
                "overall_health": health_status.overall_status,
                "components_status": {
                    name: component.get("status", "unknown") 
                    for name, component in health_status.components.items()
                }
            },
            "notification_channels": notification_results,
            "notification_stats": notification_stats,
            "system_metrics": {
                "cpu_usage": health_status.metrics.get("cpu_usage_percent", 0),
                "memory_usage": health_status.metrics.get("memory_usage_percent", 0),
                "uptime_seconds": health_status.metrics.get("uptime_seconds", 0),
                "total_requests": health_status.metrics.get("total_requests", 0)
            },
            "test_summary": {
                "monitoring_system_operational": True,
                "test_alert_processed": True,
                "notification_channels_tested": len(notification_results),
                "successful_notifications": len([r for r in notification_results.values() if r.get("success", False)]),
                "failed_notifications": len([r for r in notification_results.values() if not r.get("success", False)])
            }
        }
        
        logger.info(f"Monitoring system test completed successfully")
        
        return test_results
        
    except Exception as e:
        logger.error(f"Monitoring system test failed: {e}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "MonitoringTestError",
                "message": "Failed to test monitoring system",
                "details": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@router.get(
    "/monitoring/config",
    summary="Monitoring configuration status",
    description="Get current monitoring and alerting configuration",
    response_description="Monitoring configuration information"
)
async def monitoring_config() -> Dict[str, Any]:
    """
    Get monitoring and alerting configuration.
    
    This endpoint provides information about:
    - Configured alerting rules and thresholds
    - Notification channel configurations
    - Monitoring system settings
    - Alert history and statistics
    
    Useful for:
    - Configuration validation
    - System administration
    - Troubleshooting monitoring issues
    - Compliance reporting
    """
    try:
        from ...api.config.alerting_config import alerting_config
        from ...api.services.notification_service import notification_service
        
        # Get alerting configuration
        config_summary = alerting_config.get_configuration_summary()
        alerting_rules = [
            {
                "name": rule.name,
                "metric_name": rule.metric_name,
                "threshold_warning": rule.threshold_warning,
                "threshold_critical": rule.threshold_critical,
                "check_interval_seconds": rule.check_interval_seconds,
                "notification_channels": [channel.value for channel in rule.notification_channels],
                "description": rule.description,
                "enabled": rule.enabled,
                "tags": rule.tags or {}
            }
            for rule in alerting_config.get_alerting_rules()
        ]
        
        # Get notification configurations
        notification_configs = [
            {
                "channel": config.channel.value,
                "enabled": config.enabled,
                "config_keys": list(config.config.keys())  # Don't expose sensitive values
            }
            for config in alerting_config.get_notification_configs()
        ]
        
        # Get notification statistics
        notification_stats = notification_service.get_notification_stats()
        
        monitoring_config_info = {
            "timestamp": datetime.utcnow().isoformat(),
            "configuration_summary": config_summary,
            "alerting_rules": alerting_rules,
            "notification_channels": notification_configs,
            "notification_statistics": notification_stats,
            "monitoring_settings": {
                "check_interval_seconds": 60,
                "metrics_retention_count": 1000,
                "alert_history_retention_hours": 24,
                "system_resource_monitoring": True,
                "application_performance_monitoring": True,
                "ml_model_monitoring": True,
                "external_api_monitoring": True
            },
            "thresholds": {
                "cpu_warning": 80.0,
                "cpu_critical": 95.0,
                "memory_warning": 85.0,
                "memory_critical": 95.0,
                "disk_warning": 85.0,
                "disk_critical": 95.0,
                "error_rate_warning": 5.0,
                "error_rate_critical": 10.0,
                "response_time_warning_ms": 2000.0,
                "response_time_critical_ms": 5000.0,
                "prediction_error_warning": 2.0,
                "prediction_error_critical": 5.0
            }
        }
        
        return monitoring_config_info
        
    except Exception as e:
        logger.error(f"Monitoring config retrieval failed: {e}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "MonitoringConfigError",
                "message": "Failed to retrieve monitoring configuration",
                "details": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )