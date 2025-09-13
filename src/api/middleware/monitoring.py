"""
System monitoring middleware for performance tracking and health monitoring.
"""

import asyncio
import logging
import psutil
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.middleware.logging import structured_logger
from src.config import get_settings


class SystemMonitor:
    """System performance and health monitor."""
    
    def __init__(self):
        self.logger = structured_logger
        self.settings = get_settings()
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.last_health_check = time.time()
        self.performance_metrics = {
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "memory_used_mb": 0.0,
            "disk_usage_percent": 0.0
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get current system information."""
        try:
            # CPU and memory info
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Update performance metrics
            self.performance_metrics.update({
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_mb": memory.used / (1024 * 1024),
                "disk_usage_percent": disk.percent
            })
            
            return {
                "uptime_seconds": time.time() - self.start_time,
                "request_count": self.request_count,
                "error_count": self.error_count,
                "error_rate": self.error_count / max(self.request_count, 1),
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_mb": memory.used / (1024 * 1024),
                "memory_available_mb": memory.available / (1024 * 1024),
                "disk_usage_percent": disk.percent,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.log_error("system_monitor", e, {"operation": "get_system_info"})
            return {"error": "Failed to get system info", "timestamp": datetime.utcnow().isoformat()}
    
    def log_system_health(self) -> None:
        """Log system health metrics."""
        current_time = time.time()
        
        # Only log health metrics every 5 minutes to avoid spam
        if current_time - self.last_health_check > 300:  # 5 minutes
            system_info = self.get_system_info()
            
            self.logger.log_system_event(
                "system_health_check",
                **system_info
            )
            
            # Log warnings for high resource usage
            if system_info.get("cpu_percent", 0) > 80:
                self.logger.log_structured(
                    "WARNING", "high_cpu_usage",
                    cpu_percent=system_info["cpu_percent"]
                )
            
            if system_info.get("memory_percent", 0) > 85:
                self.logger.log_structured(
                    "WARNING", "high_memory_usage",
                    memory_percent=system_info["memory_percent"]
                )
            
            self.last_health_check = current_time
    
    def increment_request_count(self) -> None:
        """Increment total request count."""
        self.request_count += 1
    
    def increment_error_count(self) -> None:
        """Increment total error count."""
        self.error_count += 1
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for the last period."""
        return {
            "uptime_hours": (time.time() - self.start_time) / 3600,
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate_percent": (self.error_count / max(self.request_count, 1)) * 100,
            "avg_cpu_percent": self.performance_metrics["cpu_percent"],
            "avg_memory_percent": self.performance_metrics["memory_percent"],
            "current_memory_mb": self.performance_metrics["memory_used_mb"]
        }


# Global system monitor instance
system_monitor = SystemMonitor()


class MonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for system monitoring and performance tracking."""
    
    def __init__(self, app):
        super().__init__(app)
        self.monitor = system_monitor
        self.logger = structured_logger
        
        # Initialize production monitoring service
        try:
            from ..services.monitoring_service import monitoring_service
            self.monitoring_service = monitoring_service
        except ImportError:
            self.monitoring_service = None
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with monitoring."""
        # Increment request count
        self.monitor.increment_request_count()
        
        # Record request in monitoring service
        if self.monitoring_service:
            self.monitoring_service.metrics_collector.increment_counter(
                "http_requests_total",
                labels={
                    "method": request.method,
                    "endpoint": str(request.url.path),
                    "status": "processing"
                }
            )
        
        # Log system health periodically
        self.monitor.log_system_health()
        
        start_time = time.time()
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Record response time in monitoring service
            if self.monitoring_service:
                self.monitoring_service.metrics_collector.record_timer(
                    "http_request_duration",
                    process_time,
                    labels={
                        "method": request.method,
                        "endpoint": str(request.url.path),
                        "status_code": str(response.status_code)
                    }
                )
                
                # Record successful request
                self.monitoring_service.metrics_collector.increment_counter(
                    "http_requests_total",
                    labels={
                        "method": request.method,
                        "endpoint": str(request.url.path),
                        "status": "success",
                        "status_code": str(response.status_code)
                    }
                )
                
                # Record prediction-specific metrics
                if "/predict" in request.url.path:
                    self.monitoring_service.metrics_collector.increment_counter(
                        "predictions_total",
                        labels={
                            "endpoint": str(request.url.path),
                            "status": "success"
                        }
                    )
            
            # Log performance metrics for prediction endpoints
            if "/predict" in request.url.path:
                # Log slow requests
                if process_time > 5.0:  # 5 seconds threshold
                    self.logger.log_structured(
                        "WARNING", "slow_request",
                        request_id=getattr(request.state, 'request_id', 'unknown'),
                        path=str(request.url.path),
                        process_time=process_time,
                        method=request.method
                    )
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            
            # Increment error count
            self.monitor.increment_error_count()
            
            # Record error in monitoring service
            if self.monitoring_service:
                self.monitoring_service.metrics_collector.increment_counter(
                    "http_errors_total",
                    labels={
                        "method": request.method,
                        "endpoint": str(request.url.path),
                        "error_type": type(e).__name__
                    }
                )
                
                # Record prediction errors
                if "/predict" in request.url.path:
                    self.monitoring_service.metrics_collector.increment_counter(
                        "prediction_errors_total",
                        labels={
                            "endpoint": str(request.url.path),
                            "error_type": type(e).__name__
                        }
                    )
            
            # Log system state during error
            system_info = self.monitor.get_system_info()
            self.logger.log_structured(
                "ERROR", "request_error_with_system_state",
                request_id=getattr(request.state, 'request_id', 'unknown'),
                path=str(request.url.path),
                error_type=type(e).__name__,
                system_cpu=system_info.get("cpu_percent"),
                system_memory=system_info.get("memory_percent"),
                process_time=process_time
            )
            
            raise


class ApplicationLogger:
    """Application-level logger for startup, shutdown, and configuration events."""
    
    def __init__(self):
        self.logger = structured_logger
        self.settings = get_settings()
    
    def log_startup(self, startup_info: Dict[str, Any]) -> None:
        """Log application startup information."""
        self.logger.log_system_event(
            "application_startup",
            app_name=self.settings.app_name,
            app_version=self.settings.app_version,
            environment=self.settings.environment,
            debug_mode=self.settings.debug,
            host=self.settings.host,
            port=self.settings.port,
            **startup_info
        )
    
    def log_shutdown(self, shutdown_info: Dict[str, Any] = None) -> None:
        """Log application shutdown information."""
        uptime = time.time() - system_monitor.start_time
        performance_summary = system_monitor.get_performance_summary()
        
        self.logger.log_system_event(
            "application_shutdown",
            uptime_seconds=uptime,
            **performance_summary,
            **(shutdown_info or {})
        )
    
    def log_configuration(self) -> None:
        """Log application configuration (without sensitive data)."""
        config_info = {
            "app_name": self.settings.app_name,
            "app_version": self.settings.app_version,
            "environment": self.settings.environment,
            "debug": self.settings.debug,
            "api_prefix": self.settings.api_v1_prefix,
            "cors_origins_count": len(self.settings.cors_origins),
            "rate_limit_requests": self.settings.rate_limit_requests,
            "rate_limit_window": self.settings.rate_limit_window,
            "log_level": self.settings.log_level,
            "max_file_size_mb": self.settings.max_file_size / (1024 * 1024),
            "model_cache_size": self.settings.model_cache_size,
            "has_materials_project_key": bool(self.settings.materials_project_api_key),
            "has_nist_key": bool(self.settings.nist_api_key)
        }
        
        self.logger.log_system_event(
            "configuration_loaded",
            **config_info
        )
    
    def log_model_initialization(self, model_info: Dict[str, Any]) -> None:
        """Log ML model initialization results."""
        self.logger.log_system_event(
            "model_initialization",
            **model_info
        )
    
    def log_external_api_status(self, api_name: str, status: str, 
                              response_time: Optional[float] = None,
                              error: Optional[str] = None) -> None:
        """Log external API connectivity status."""
        log_data = {
            "api_name": api_name,
            "status": status
        }
        
        if response_time is not None:
            log_data["response_time_ms"] = round(response_time * 1000, 2)
        
        if error:
            log_data["error"] = error
        
        self.logger.log_system_event(
            "external_api_check",
            **log_data
        )


# Global application logger instance
app_logger = ApplicationLogger()