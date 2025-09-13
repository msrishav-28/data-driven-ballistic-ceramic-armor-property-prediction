# Ceramic Armor ML API Documentation Guide

## Overview

This guide provides comprehensive documentation for the Ceramic Armor ML API, including endpoint descriptions, request/response formats, examples, and best practices.

## Table of Contents

1. [API Overview](#api-overview)
2. [Authentication & Rate Limiting](#authentication--rate-limiting)
3. [Prediction Endpoints](#prediction-endpoints)
4. [Batch Processing](#batch-processing)
5. [Health & Monitoring](#health--monitoring)
6. [Error Handling](#error-handling)
7. [Examples & Use Cases](#examples--use-cases)
8. [Best Practices](#best-practices)

## API Overview

### Base URL
- **Production**: `https://ceramic-armor-ml-api.onrender.com`
- **Development**: `http://localhost:8000`

### API Version
- Current version: `v1`
- All endpoints are prefixed with `/api/v1/` except health endpoints

### Supported Formats
- **Request**: JSON
- **Response**: JSON
- **File Upload**: CSV, Excel (.xlsx), JSON

## Authentication & Rate Limiting

### Authentication
Currently, the API is open for research use. Future versions will support API key authentication.

### Rate Limits
- **Prediction Endpoints**: 100 requests per hour per IP
- **Batch Processing**: 10 requests per hour per IP  
- **Health Endpoints**: 1000 requests per hour per IP

### Rate Limit Headers
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642694400
```

## Prediction Endpoints

### Mechanical Properties Prediction

**Endpoint**: `POST /api/v1/predict/mechanical`

Predicts mechanical properties including fracture toughness, Vickers hardness, density, and elastic modulus.

#### Request Format

```json
{
  "composition": {
    "SiC": 0.6,
    "B4C": 0.3,
    "Al2O3": 0.1,
    "WC": 0.0,
    "TiC": 0.0
  },
  "processing": {
    "sintering_temperature": 1800,
    "pressure": 50,
    "grain_size": 10,
    "holding_time": 120,
    "heating_rate": 15,
    "atmosphere": "argon"
  },
  "microstructure": {
    "porosity": 0.02,
    "phase_distribution": "uniform",
    "interface_quality": "good",
    "pore_size": 1.0,
    "connectivity": 0.1
  },
  "include_uncertainty": true,
  "include_feature_importance": true,
  "prediction_type": "mechanical"
}
```

#### Response Format

```json
{
  "status": "success",
  "predictions": {
    "fracture_toughness": {
      "value": 4.6,
      "unit": "MPa·m^0.5",
      "confidence_interval": [4.2, 5.0],
      "uncertainty": 0.15,
      "prediction_quality": "good"
    },
    "vickers_hardness": {
      "value": 2800,
      "unit": "HV",
      "confidence_interval": [2650, 2950],
      "uncertainty": 0.12,
      "prediction_quality": "excellent"
    },
    "density": {
      "value": 3.21,
      "unit": "g/cm³",
      "confidence_interval": [3.18, 3.24],
      "uncertainty": 0.08,
      "prediction_quality": "excellent"
    },
    "elastic_modulus": {
      "value": 410,
      "unit": "GPa",
      "confidence_interval": [395, 425],
      "uncertainty": 0.10,
      "prediction_quality": "good"
    }
  },
  "feature_importance": [
    {
      "name": "SiC_content",
      "importance": 0.35,
      "category": "composition",
      "description": "Silicon Carbide content in material composition",
      "shap_value": 0.12
    }
  ],
  "model_info": {
    "model_version": "v1.2.0",
    "model_type": "XGBoost Ensemble",
    "training_r2": 0.87,
    "validation_r2": 0.82,
    "prediction_time_ms": 45,
    "feature_count": 156
  },
  "request_id": "req_abc123def456",
  "timestamp": "2024-01-15T14:30:00Z"
}
```

### Ballistic Properties Prediction

**Endpoint**: `POST /api/v1/predict/ballistic`

Predicts ballistic properties including V50 velocity, penetration resistance, back-face deformation, and multi-hit capability.

#### Key Differences from Mechanical Prediction
- Higher uncertainty values (ballistic predictions are inherently more uncertain)
- Different property units and ranges
- Additional processing notes about test conditions

#### Example Response Properties

```json
{
  "predictions": {
    "v50_velocity": {
      "value": 850,
      "unit": "m/s",
      "confidence_interval": [820, 880],
      "uncertainty": 0.18,
      "prediction_quality": "good"
    },
    "penetration_resistance": {
      "value": 7.2,
      "unit": "dimensionless",
      "confidence_interval": [6.8, 7.6],
      "uncertainty": 0.22,
      "prediction_quality": "fair"
    },
    "back_face_deformation": {
      "value": 12.5,
      "unit": "mm",
      "confidence_interval": [11.2, 13.8],
      "uncertainty": 0.25,
      "prediction_quality": "fair"
    },
    "multi_hit_capability": {
      "value": 0.75,
      "unit": "probability",
      "confidence_interval": [0.65, 0.85],
      "uncertainty": 0.30,
      "prediction_quality": "poor"
    }
  }
}
```

## Batch Processing

### Upload Batch File

**Endpoint**: `POST /api/v1/predict/batch`

Upload CSV, Excel, or JSON files for batch processing of multiple materials.

#### Request Format (Multipart Form)

```
Content-Type: multipart/form-data

file: [CSV/Excel/JSON file]
file_format: "csv"
output_format: "csv"
include_uncertainty: true
include_feature_importance: false
prediction_type: "both"
max_rows: 1000
```

#### Required CSV Columns

**Composition (Required)**:
- `SiC`, `B4C`, `Al2O3` (at least one required)

**Processing (Required)**:
- `sintering_temperature` (1200-2500°C)
- `pressure` (1-200 MPa)
- `grain_size` (0.1-100 μm)

**Microstructure (Required)**:
- `porosity` (0-0.3)
- `phase_distribution` (uniform/gradient/layered)

**Optional Columns**:
- Additional composition: `WC`, `TiC`, `ZrO2`, `TiB2`
- Additional processing: `holding_time`, `heating_rate`, `atmosphere`
- Additional microstructure: `interface_quality`, `pore_size`, `connectivity`

#### Response Format

```json
{
  "status": "success",
  "total_processed": 0,
  "successful_predictions": 0,
  "failed_predictions": 0,
  "processing_time_seconds": 0,
  "download_url": null,
  "request_id": "batch_abc123def456",
  "timestamp": "2024-01-15T14:30:00Z",
  "warnings": []
}
```

### Check Batch Status

**Endpoint**: `GET /api/v1/predict/batch/{batch_id}/status`

Check the processing status of a batch job.

### Download Batch Results

**Endpoint**: `GET /api/v1/predict/batch/{batch_id}/download`

Download the results file once processing is complete. Files expire after 24 hours.

## Health & Monitoring

### Basic Health Check

**Endpoint**: `GET /health`

Simple health check for load balancers and monitoring systems.

```json
{
  "status": "healthy",
  "service": "Ceramic Armor ML API",
  "version": "1.0.0",
  "environment": "production",
  "timestamp": "2024-01-15T14:30:00Z",
  "uptime_seconds": 3600.5
}
```

### Detailed System Status

**Endpoint**: `GET /api/v1/status`

Comprehensive system information including resource usage and ML system health.

### Model Information

**Endpoint**: `GET /api/v1/models/info`

Detailed information about loaded ML models and their performance metrics.

### Security Status

**Endpoint**: `GET /api/v1/security/status`

Information about security configuration, rate limiting, and middleware status.

## Error Handling

### Error Response Format

All errors follow a consistent format:

```json
{
  "error": "ValidationError",
  "message": "Invalid material composition data",
  "details": {
    "total_composition": 1.15,
    "max_allowed": 1.0
  },
  "field_errors": {
    "composition": ["Total composition exceeds 100%"]
  },
  "request_id": "req_abc123def456",
  "timestamp": "2024-01-15T14:30:00Z",
  "suggestion": "Ensure all composition fractions sum to 1.0 or less"
}
```

### Common Error Types

#### Validation Errors (400/422)
- Invalid composition (sum > 1.0 or < 0.01)
- Out-of-range processing parameters
- Missing required fields
- Invalid enum values

#### Rate Limit Errors (429)
- Too many requests within time window
- Includes reset time and current usage

#### Prediction Errors (500)
- ML model failures
- Feature extraction errors
- System resource issues

#### File Upload Errors (413/422)
- File too large (>50MB)
- Invalid file format
- Missing required columns
- Corrupted file data

## Examples & Use Cases

### High-Performance SiC Armor

```json
{
  "composition": {
    "SiC": 0.85,
    "B4C": 0.10,
    "Al2O3": 0.05
  },
  "processing": {
    "sintering_temperature": 2100,
    "pressure": 80,
    "grain_size": 5.0,
    "holding_time": 180,
    "atmosphere": "argon"
  },
  "microstructure": {
    "porosity": 0.01,
    "phase_distribution": "uniform",
    "interface_quality": "excellent"
  }
}
```

**Expected Results**:
- High fracture toughness (>5 MPa·m^0.5)
- Excellent hardness (>3000 HV)
- High V50 velocity (>900 m/s)

### Lightweight B₄C Composite

```json
{
  "composition": {
    "SiC": 0.20,
    "B4C": 0.70,
    "Al2O3": 0.10
  },
  "processing": {
    "sintering_temperature": 1950,
    "pressure": 60,
    "grain_size": 8.0,
    "atmosphere": "nitrogen"
  },
  "microstructure": {
    "porosity": 0.03,
    "phase_distribution": "uniform"
  }
}
```

**Expected Results**:
- Lower density (~2.8 g/cm³)
- Good ballistic performance
- Moderate mechanical properties

### Cost-Effective Al₂O₃ Armor

```json
{
  "composition": {
    "SiC": 0.15,
    "B4C": 0.05,
    "Al2O3": 0.75,
    "WC": 0.05
  },
  "processing": {
    "sintering_temperature": 1650,
    "pressure": 40,
    "grain_size": 15.0,
    "atmosphere": "air"
  },
  "microstructure": {
    "porosity": 0.05,
    "phase_distribution": "gradient"
  }
}
```

**Expected Results**:
- Lower cost materials
- Moderate performance
- Higher uncertainty due to gradient microstructure

## Best Practices

### Request Optimization

1. **Batch Processing**: Use batch endpoints for multiple materials
2. **Feature Importance**: Disable for batch processing to reduce response size
3. **Uncertainty**: Enable for critical applications, disable for screening

### Data Quality

1. **Composition Validation**: Ensure fractions sum to ≤1.0
2. **Realistic Parameters**: Stay within validated ranges
3. **Complete Data**: Provide all required fields for best accuracy

### Error Handling

1. **Retry Logic**: Implement exponential backoff for transient errors
2. **Validation**: Validate data client-side before API calls
3. **Rate Limiting**: Monitor rate limit headers and implement queuing

### Performance

1. **Caching**: Cache identical requests client-side
2. **Parallel Processing**: Use batch endpoints for multiple materials
3. **Monitoring**: Track response times and error rates

### Security

1. **Input Sanitization**: Validate all inputs before sending
2. **HTTPS**: Always use HTTPS in production
3. **Rate Limiting**: Respect rate limits and implement proper backoff

## Interactive Documentation

### Swagger UI
Access interactive API documentation at `/docs` (development only):
- Try out endpoints with real data
- View detailed request/response schemas
- Copy example requests and responses

### ReDoc
Alternative documentation interface at `/redoc` (development only):
- Clean, readable format
- Comprehensive schema documentation
- Downloadable OpenAPI specification

### OpenAPI Specification
Download the complete OpenAPI specification at `/openapi.json` for:
- Code generation
- API testing tools
- Custom documentation generation

## Support & Resources

### Getting Help
- **Email**: support@ceramic-armor-ml.com
- **Documentation**: https://docs.ceramic-armor-ml.com
- **Issues**: https://github.com/ceramic-armor-ml/api/issues

### Research Collaboration
For academic research and collaboration opportunities:
- **Research Portal**: https://research.ceramic-armor-ml.com
- **Publications**: https://ceramic-armor-ml.com/publications
- **Datasets**: https://data.ceramic-armor-ml.com

### Updates & Announcements
- **Changelog**: https://ceramic-armor-ml.com/changelog
- **Status Page**: https://status.ceramic-armor-ml.com
- **Newsletter**: https://ceramic-armor-ml.com/newsletter