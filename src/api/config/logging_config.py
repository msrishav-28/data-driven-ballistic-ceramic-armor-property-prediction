"""
Comprehensive logging configuration for the Ceramic Armor ML API.
"""

import logging
import logging.config
import sys
from typing import Dict, Any

from src.config import get_settings, is_production


def get_logging_config() -> Dict[str, Any]:
    """Get comprehensive logging configuration."""
    settings = get_settings()
    
    # Base configuration
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(name)s %(levelname)s %(filename)s %(lineno)d %(funcName)s %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": settings.log_level.upper(),
                "formatter": "detailed" if settings.debug else "standard",
                "stream": sys.stdout
            },
            "error_console": {
                "class": "logging.StreamHandler",
                "level": "ERROR",
                "formatter": "detailed",
                "stream": sys.stderr
            }
        },
        "loggers": {
            # Root logger
            "": {
                "level": settings.log_level.upper(),
                "handlers": ["console"],
                "propagate": False
            },
            # Application loggers
            "src": {
                "level": settings.log_level.upper(),
                "handlers": ["console"],
                "propagate": False
            },
            "src.api": {
                "level": settings.log_level.upper(),
                "handlers": ["console"],
                "propagate": False
            },
            "src.ml": {
                "level": settings.log_level.upper(),
                "handlers": ["console"],
                "propagate": False
            },
            # FastAPI and Uvicorn loggers
            "fastapi": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False
            },
            "uvicorn.access": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False
            },
            "uvicorn.error": {
                "level": "INFO",
                "handlers": ["console", "error_console"],
                "propagate": False
            },
            # External library loggers (reduce noise)
            "httpx": {
                "level": "WARNING",
                "handlers": ["console"],
                "propagate": False
            },
            "urllib3": {
                "level": "WARNING",
                "handlers": ["console"],
                "propagate": False
            },
            "requests": {
                "level": "WARNING",
                "handlers": ["console"],
                "propagate": False
            },
            "matplotlib": {
                "level": "WARNING",
                "handlers": ["console"],
                "propagate": False
            }
        }
    }
    
    # Add file logging in production
    if is_production():
        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "json",
            "filename": "/tmp/ceramic_armor_api.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        }
        
        config["handlers"]["error_file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "detailed",
            "filename": "/tmp/ceramic_armor_errors.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 3
        }
        
        # Add file handlers to loggers
        for logger_name in ["", "src", "src.api", "src.ml"]:
            config["loggers"][logger_name]["handlers"].extend(["file"])
        
        # Add error file handler to error loggers
        config["loggers"]["uvicorn.error"]["handlers"].append("error_file")
    
    return config


def setup_logging() -> None:
    """Setup comprehensive logging configuration."""
    config = get_logging_config()
    logging.config.dictConfig(config)
    
    # Log configuration setup
    logger = logging.getLogger(__name__)
    settings = get_settings()
    
    logger.info(f"Logging configured - Level: {settings.log_level}, Environment: {settings.environment}")
    
    if is_production():
        logger.info("Production logging enabled with file output")
    else:
        logger.info("Development logging enabled with console output only")


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance."""
    return logging.getLogger(name)


# Performance logging utilities
class PerformanceLogger:
    """Utility class for performance logging."""
    
    def __init__(self, logger_name: str = __name__):
        self.logger = logging.getLogger(logger_name)
    
    def log_slow_operation(self, operation: str, duration: float, 
                          threshold: float = 1.0, **kwargs) -> None:
        """Log operations that exceed performance thresholds."""
        if duration > threshold:
            self.logger.warning(
                f"Slow operation detected: {operation} took {duration:.3f}s "
                f"(threshold: {threshold}s)",
                extra={
                    "operation": operation,
                    "duration_seconds": duration,
                    "threshold_seconds": threshold,
                    **kwargs
                }
            )
    
    def log_memory_usage(self, operation: str, memory_mb: float,
                        threshold_mb: float = 100.0) -> None:
        """Log high memory usage operations."""
        if memory_mb > threshold_mb:
            self.logger.warning(
                f"High memory usage: {operation} used {memory_mb:.1f}MB "
                f"(threshold: {threshold_mb}MB)",
                extra={
                    "operation": operation,
                    "memory_mb": memory_mb,
                    "threshold_mb": threshold_mb
                }
            )


# Security logging utilities
class SecurityLogger:
    """Utility class for security-related logging."""
    
    def __init__(self, logger_name: str = "security"):
        self.logger = logging.getLogger(logger_name)
    
    def log_suspicious_activity(self, client_ip: str, activity: str, 
                              details: Dict[str, Any] = None) -> None:
        """Log suspicious activity for security monitoring."""
        self.logger.warning(
            f"Suspicious activity from {client_ip}: {activity}",
            extra={
                "client_ip": client_ip,
                "activity": activity,
                "details": details or {},
                "security_event": True
            }
        )
    
    def log_rate_limit_exceeded(self, client_ip: str, endpoint: str,
                              request_count: int, time_window: int) -> None:
        """Log rate limit violations."""
        self.logger.warning(
            f"Rate limit exceeded by {client_ip} on {endpoint}: "
            f"{request_count} requests in {time_window}s",
            extra={
                "client_ip": client_ip,
                "endpoint": endpoint,
                "request_count": request_count,
                "time_window_seconds": time_window,
                "security_event": True
            }
        )
    
    def log_invalid_input(self, client_ip: str, endpoint: str,
                         validation_errors: list) -> None:
        """Log invalid input attempts."""
        self.logger.info(
            f"Invalid input from {client_ip} on {endpoint}",
            extra={
                "client_ip": client_ip,
                "endpoint": endpoint,
                "validation_errors": validation_errors,
                "security_event": True
            }
        )


# Global logger instances
performance_logger = PerformanceLogger()
security_logger = SecurityLogger()