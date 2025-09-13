# Comprehensive Logging and Monitoring Middleware

This document describes the comprehensive logging and monitoring middleware implemented for the Ceramic Armor ML API.

## Overview

The logging middleware provides:
- **Structured JSON logging** for all API requests and responses
- **Error handling** with detailed context and stack traces
- **System monitoring** with performance metrics
- **ML prediction logging** with model performance tracking
- **External API logging** for Materials Project, NIST, etc.
- **Security logging** for suspicious activity detection

## Components

### 1. LoggingMiddleware

Handles HTTP request/response logging with structured JSON format.

**Features:**
- Unique request ID generation for request tracing
- Request details: method, path, client IP, user agent, headers
- Response details: status code, processing time, response size
- Performance tracking with millisecond precision

**Example Log Output:**
```json
{
  "timestamp": 1757770798.667405,
  "event": "http_request",
  "environment": "development",
  "request_id": "10d04989-7f80-4909-accc-086afb8fb552",
  "method": "GET",
  "path": "/api/v1/predict/mechanical",
  "client_ip": "192.168.1.100",
  "user_agent": "Mozilla/5.0...",
  "query_params": {},
  "content_length": 1024
}
```

### 2. MonitoringMiddleware

Tracks system performance and health metrics.

**Features:**
- Request counting and error rate tracking
- System resource monitoring (CPU, memory, disk)
- Slow request detection and alerting
- Periodic health check logging

**System Metrics:**
- CPU usage percentage
- Memory usage and availability
- Disk usage percentage
- Application uptime
- Request/error counts and rates

### 3. ErrorHandlingMiddleware

Comprehensive error handling with detailed logging.

**Features:**
- Catches all unhandled exceptions
- Logs full stack traces with context
- Provides consistent error response format
- Handles specific error types (ML, API, validation)

**Error Types:**
- `MLPredictionError` - ML model prediction failures
- `ExternalAPIError` - External API connectivity issues
- `DataValidationError` - Input validation failures
- General exceptions with full context

### 4. StructuredLogger

Core logging utility for consistent JSON log formatting.

**Methods:**
- `log_structured()` - General structured logging
- `log_request()` - HTTP request logging
- `log_response()` - HTTP response logging
- `log_error()` - Error logging with context
- `log_prediction()` - ML prediction logging
- `log_system_event()` - System event logging

### 5. ModelPredictionLogger

Specialized logging for ML operations.

**Features:**
- Prediction start/success/error logging
- Model performance metrics tracking
- Feature importance logging
- Batch processing results

**Usage Example:**
```python
from src.api.middleware import model_logger

model_logger.log_prediction_success(
    request_id="abc-123",
    prediction_type="mechanical",
    model_name="xgboost_fracture_toughness",
    process_time=0.045,
    predictions={"fracture_toughness": 4.6},
    feature_importance={"SiC_content": 0.35}
)
```

### 6. SystemMonitor

Real-time system performance monitoring.

**Features:**
- Resource usage tracking
- Performance threshold alerting
- Health check reporting
- Uptime and statistics tracking

## Configuration

### Environment Variables

```bash
# Logging configuration
LOG_LEVEL=INFO                    # DEBUG, INFO, WARNING, ERROR
ENVIRONMENT=production            # development, production

# Rate limiting (affects logging frequency)
RATE_LIMIT_REQUESTS=100          # Requests per window
RATE_LIMIT_WINDOW=3600           # Window in seconds
```

### Logging Levels

- **DEBUG**: Detailed debugging information
- **INFO**: General operational messages
- **WARNING**: Warning conditions (high resource usage, slow requests)
- **ERROR**: Error conditions requiring attention

### Production Configuration

In production, the middleware automatically:
- Enables file logging with rotation
- Reduces debug information in error responses
- Implements structured JSON logging for log aggregation
- Monitors system health with alerting thresholds

## Integration with Services

### PredictionLoggingService

Context manager for ML prediction operations:

```python
from src.api.services import prediction_logging

async with prediction_logging.log_prediction_operation(
    request, "mechanical", input_data
) as request_id:
    # Perform prediction
    result = await predictor.predict(input_data)
    
    # Log success
    prediction_logging.log_prediction_success(
        request_id, "mechanical", "xgboost_model", 
        process_time, result
    )
```

### ExternalAPILoggingService

Logging for external API calls:

```python
from src.api.services import api_logging

async with api_logging.log_api_operation(
    request_id, "materials_project", "get_structure"
) as _:
    # Make API call
    structure = await mp_client.get_structure(material_id)
```

## Log Analysis

### Request Tracing

Each request gets a unique ID that can be used to trace the complete request lifecycle:

1. Request received (`http_request` event)
2. Processing steps (`ml_prediction`, `external_api_call` events)
3. Response sent (`http_response` event)
4. Any errors (`application_error`, `ml_prediction_error` events)

### Performance Monitoring

Key metrics to monitor:
- Average response time by endpoint
- Error rates by endpoint and error type
- System resource usage trends
- ML model performance metrics

### Error Analysis

Structured error logs include:
- Full stack traces
- Request context (method, path, parameters)
- System state (CPU, memory usage)
- User context (IP, user agent)

## Security Features

### Rate Limiting Integration

The logging middleware works with rate limiting to:
- Log rate limit violations
- Track suspicious activity patterns
- Monitor for potential abuse

### Input Sanitization Logging

Logs validation failures and potential security issues:
- Invalid input attempts
- Injection attack patterns
- Unusual request patterns

## Performance Impact

The logging middleware is designed for minimal performance impact:
- Asynchronous logging operations
- Efficient JSON serialization
- Configurable log levels
- Optional debug information

**Typical Overhead:**
- Request logging: < 1ms per request
- Error logging: < 5ms per error
- System monitoring: < 10ms every 5 minutes

## Troubleshooting

### Common Issues

1. **High Memory Usage**: Check log rotation settings and reduce log level
2. **Slow Responses**: Monitor for excessive debug logging in production
3. **Missing Logs**: Verify log level configuration and file permissions

### Debug Mode

Enable debug mode for detailed logging:
```bash
DEBUG=true
LOG_LEVEL=DEBUG
```

This provides:
- Full request/response headers
- Detailed stack traces
- System state information
- ML model internals

## Integration Examples

### Custom Error Handling

```python
from src.api.middleware import MLPredictionError

try:
    result = model.predict(features)
except Exception as e:
    raise MLPredictionError(
        "Prediction failed",
        model_name="xgboost_model",
        prediction_type="mechanical",
        context={"feature_count": len(features)}
    ) from e
```

### Performance Monitoring

```python
from src.api.middleware import performance_logger

# Log slow operations
performance_logger.log_slow_operation(
    "feature_extraction", 
    duration=2.5, 
    threshold=1.0
)

# Log high memory usage
performance_logger.log_memory_usage(
    "batch_prediction", 
    memory_mb=150, 
    threshold_mb=100
)
```

This comprehensive logging system provides full observability into the Ceramic Armor ML API operations, enabling effective monitoring, debugging, and performance optimization.