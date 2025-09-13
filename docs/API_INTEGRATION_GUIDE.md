# API Integration Guide

This guide provides comprehensive instructions for integrating with the Ceramic Armor ML API, including authentication, request/response formats, error handling, and best practices.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Authentication](#authentication)
3. [API Endpoints](#api-endpoints)
4. [Request/Response Formats](#requestresponse-formats)
5. [Error Handling](#error-handling)
6. [Rate Limiting](#rate-limiting)
7. [Batch Processing](#batch-processing)
8. [Client Libraries](#client-libraries)
9. [Best Practices](#best-practices)
10. [Examples](#examples)

## Getting Started

### Base URL

- **Production**: `https://ceramic-armor-ml-api.onrender.com`
- **Development**: `http://localhost:8000`

### API Version

Current API version: `v1`
All endpoints are prefixed with `/api/v1`

### Content Type

All requests should use `Content-Type: application/json` except for file uploads which use `multipart/form-data`.

## Authentication

Currently, the API does not require authentication for basic predictions. However, rate limiting is applied based on IP address.

### Future Authentication (Planned)

```http
Authorization: Bearer your-api-key-here
```

## API Endpoints

### 1. Health Check

Check if the API is running and healthy.

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### 2. System Status

Get detailed system information including resource usage and model status.

```http
GET /api/v1/status
```

**Response:**
```json
{
  "overall_status": "healthy",
  "system": {
    "resources": {
      "memory": {
        "usage_percent": 45.2,
        "available_gb": 6.8
      },
      "cpu": {
        "usage_percent": 23.1,
        "cores": 4
      }
    }
  },
  "ml_system": {
    "overall_health": "healthy",
    "models_loaded": 8,
    "cache_usage": "67%"
  }
}
```

### 3. Model Information

Get information about loaded ML models.

```http
GET /api/v1/models/info
```

**Response:**
```json
{
  "models": {
    "mechanical": {
      "fracture_toughness": {
        "version": "v1.2.0",
        "training_r2": 0.87,
        "features": 156,
        "last_updated": "2024-01-10T00:00:00Z"
      },
      "vickers_hardness": {
        "version": "v1.2.0",
        "training_r2": 0.89,
        "features": 156,
        "last_updated": "2024-01-10T00:00:00Z"
      }
    },
    "ballistic": {
      "v50_velocity": {
        "version": "v1.1.0",
        "training_r2": 0.82,
        "features": 145,
        "last_updated": "2024-01-08T00:00:00Z"
      }
    }
  }
}
```

### 4. Mechanical Properties Prediction

Predict mechanical properties of ceramic armor materials.

```http
POST /api/v1/predict/mechanical
```

**Request Body:**
```json
{
  "composition": {
    "SiC": 0.85,
    "B4C": 0.10,
    "Al2O3": 0.05,
    "WC": 0.0,
    "TiC": 0.0
  },
  "processing": {
    "sintering_temperature": 2100,
    "pressure": 80,
    "grain_size": 5.0,
    "holding_time": 180,
    "heating_rate": 10,
    "atmosphere": "argon"
  },
  "microstructure": {
    "porosity": 0.01,
    "phase_distribution": "uniform",
    "interface_quality": "excellent",
    "pore_size": 0.5,
    "connectivity": 0.05
  },
  "include_uncertainty": true,
  "include_feature_importance": true
}
```

**Response:**
```json
{
  "request_id": "req_123456789",
  "status": "success",
  "predictions": {
    "fracture_toughness": {
      "value": 4.6,
      "unit": "MPa·m^0.5",
      "confidence_interval": [4.2, 5.0],
      "uncertainty": 0.15
    },
    "vickers_hardness": {
      "value": 2800,
      "unit": "HV",
      "confidence_interval": [2650, 2950],
      "uncertainty": 0.12
    },
    "density": {
      "value": 3.21,
      "unit": "g/cm³",
      "confidence_interval": [3.18, 3.24],
      "uncertainty": 0.08
    },
    "elastic_modulus": {
      "value": 410,
      "unit": "GPa",
      "confidence_interval": [395, 425],
      "uncertainty": 0.10
    }
  },
  "feature_importance": [
    {
      "name": "SiC_content",
      "importance": 0.35,
      "description": "Silicon carbide content fraction"
    },
    {
      "name": "sintering_temperature",
      "importance": 0.28,
      "description": "Sintering temperature in Celsius"
    },
    {
      "name": "grain_size",
      "importance": 0.22,
      "description": "Average grain size in micrometers"
    }
  ],
  "model_info": {
    "model_version": "v1.2.0",
    "prediction_time_ms": 45,
    "feature_count": 156
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### 5. Ballistic Properties Prediction

Predict ballistic properties of ceramic armor materials.

```http
POST /api/v1/predict/ballistic
```

**Request Body:** (Same structure as mechanical prediction)

**Response:**
```json
{
  "request_id": "req_123456790",
  "status": "success",
  "predictions": {
    "v50_velocity": {
      "value": 850,
      "unit": "m/s",
      "confidence_interval": [820, 880],
      "uncertainty": 0.18
    },
    "penetration_resistance": {
      "value": 0.85,
      "unit": "normalized",
      "confidence_interval": [0.78, 0.92],
      "uncertainty": 0.20
    },
    "back_face_deformation": {
      "value": 12.5,
      "unit": "mm",
      "confidence_interval": [11.2, 13.8],
      "uncertainty": 0.16
    },
    "multi_hit_capability": {
      "value": 0.75,
      "unit": "probability",
      "confidence_interval": [0.68, 0.82],
      "uncertainty": 0.22
    }
  },
  "processing_notes": [
    "High SiC content provides excellent ballistic performance",
    "Optimal grain size for multi-hit capability"
  ],
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### 6. Batch Processing

Upload a file for batch processing of multiple materials.

```http
POST /api/v1/predict/batch
Content-Type: multipart/form-data
```

**Form Data:**
- `file`: CSV/Excel file with material data
- `file_format`: "csv" or "xlsx"
- `output_format`: "csv", "xlsx", or "json"
- `include_uncertainty`: true/false
- `prediction_type`: "mechanical", "ballistic", or "both"
- `max_rows`: Maximum number of rows to process

**Response:**
```json
{
  "request_id": "batch_123456789",
  "status": "processing",
  "message": "Batch processing started",
  "estimated_completion_time": "2024-01-15T10:35:00Z",
  "total_rows": 150,
  "status_url": "/api/v1/predict/batch/batch_123456789/status"
}
```

### 7. Batch Status

Check the status of batch processing.

```http
GET /api/v1/predict/batch/{batch_id}/status
```

**Response:**
```json
{
  "batch_id": "batch_123456789",
  "status": "success",
  "total_processed": 150,
  "successful_predictions": 148,
  "failed_predictions": 2,
  "processing_time_seconds": 45.2,
  "download_url": "/api/v1/predict/batch/batch_123456789/download",
  "expires_at": "2024-01-16T10:30:00Z"
}
```

### 8. Download Batch Results

Download the results of batch processing.

```http
GET /api/v1/predict/batch/{batch_id}/download
```

Returns the processed file with predictions added as additional columns.

## Request/Response Formats

### Composition Object

```json
{
  "SiC": 0.0,      // Silicon Carbide fraction (0.0-1.0)
  "B4C": 0.0,      // Boron Carbide fraction (0.0-1.0)
  "Al2O3": 0.0,    // Aluminum Oxide fraction (0.0-1.0)
  "WC": 0.0,       // Tungsten Carbide fraction (0.0-1.0)
  "TiC": 0.0       // Titanium Carbide fraction (0.0-1.0)
}
```

**Validation Rules:**
- All values must be between 0.0 and 1.0
- Sum of all fractions must not exceed 1.0
- At least one component must be > 0.0

### Processing Object

```json
{
  "sintering_temperature": 1800,  // Temperature in Celsius (1200-2500)
  "pressure": 50,                 // Pressure in MPa (1-200)
  "grain_size": 10.0,            // Average grain size in μm (0.1-100)
  "holding_time": 120,           // Sintering time in minutes (1-600)
  "heating_rate": 10,            // Heating rate in °C/min (1-50)
  "atmosphere": "argon"          // Sintering atmosphere
}
```

### Microstructure Object

```json
{
  "porosity": 0.02,              // Porosity fraction (0.0-0.3)
  "phase_distribution": "uniform", // "uniform", "gradient", "layered"
  "interface_quality": "good",    // "poor", "fair", "good", "excellent"
  "pore_size": 1.0,              // Average pore size in μm (0.1-10)
  "connectivity": 0.1            // Pore connectivity (0.0-1.0)
}
```

## Error Handling

### HTTP Status Codes

- `200 OK`: Successful request
- `400 Bad Request`: Invalid request format
- `422 Unprocessable Entity`: Validation errors
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service temporarily unavailable

### Error Response Format

```json
{
  "error": "ValidationError",
  "message": "Input validation failed",
  "details": {
    "field": "composition.SiC",
    "error": "Value must be between 0.0 and 1.0",
    "received_value": 1.5
  },
  "request_id": "req_123456789",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Common Validation Errors

1. **Composition Sum Error**
   ```json
   {
     "error": "ValidationError",
     "message": "Total composition exceeds 100%",
     "details": {
       "total_composition": 1.15,
       "max_allowed": 1.0
     }
   }
   ```

2. **Temperature Range Error**
   ```json
   {
     "error": "ValidationError",
     "message": "Sintering temperature out of range",
     "details": {
       "field": "processing.sintering_temperature",
       "value": 3000,
       "valid_range": [1200, 2500]
     }
   }
   ```

## Rate Limiting

### Default Limits

- **Requests per hour**: 100
- **Batch uploads per hour**: 10
- **File size limit**: 10MB

### Rate Limit Headers

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642248600
Retry-After: 3600
```

### Rate Limit Exceeded Response

```json
{
  "error": "RateLimitExceeded",
  "message": "Too many requests",
  "details": {
    "limit": 100,
    "window_seconds": 3600,
    "retry_after_seconds": 1800
  }
}
```

## Batch Processing

### Supported File Formats

- **CSV**: Comma-separated values
- **Excel**: .xlsx format

### Required Columns

Your input file must contain columns for:

**Composition:**
- `SiC`, `B4C`, `Al2O3`, `WC`, `TiC`

**Processing:**
- `sintering_temperature`, `pressure`, `grain_size`
- Optional: `holding_time`, `heating_rate`, `atmosphere`

**Microstructure:**
- `porosity`, `phase_distribution`
- Optional: `interface_quality`, `pore_size`, `connectivity`

### Example CSV Format

```csv
material_name,SiC,B4C,Al2O3,WC,TiC,sintering_temperature,pressure,grain_size,porosity,phase_distribution
High_SiC_Armor,0.85,0.10,0.05,0.0,0.0,2100,80,5.0,0.01,uniform
Lightweight_B4C,0.20,0.70,0.10,0.0,0.0,1950,60,8.0,0.03,uniform
Cost_Effective_Al2O3,0.15,0.05,0.75,0.05,0.0,1650,40,15.0,0.05,gradient
```

### Output Format

The processed file will include additional columns with predictions:

```csv
# Original columns + prediction columns
fracture_toughness,fracture_toughness_ci_lower,fracture_toughness_ci_upper,
vickers_hardness,vickers_hardness_ci_lower,vickers_hardness_ci_upper,
density,density_ci_lower,density_ci_upper,
elastic_modulus,elastic_modulus_ci_lower,elastic_modulus_ci_upper
```

## Client Libraries

### Python Client

```python
import requests
from typing import Dict, Any

class CeramicArmorAPIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def predict_mechanical(self, composition: Dict, processing: Dict, 
                          microstructure: Dict) -> Dict[str, Any]:
        response = self.session.post(
            f"{self.base_url}/api/v1/predict/mechanical",
            json={
                "composition": composition,
                "processing": processing,
                "microstructure": microstructure
            }
        )
        response.raise_for_status()
        return response.json()

# Usage
client = CeramicArmorAPIClient("https://ceramic-armor-ml-api.onrender.com")
result = client.predict_mechanical(
    composition={"SiC": 0.85, "B4C": 0.10, "Al2O3": 0.05},
    processing={"sintering_temperature": 2100, "pressure": 80, "grain_size": 5.0},
    microstructure={"porosity": 0.01, "phase_distribution": "uniform"}
)
```

### JavaScript Client

```javascript
class CeramicArmorAPIClient {
    constructor(baseUrl) {
        this.baseUrl = baseUrl.replace(/\/$/, '');
    }
    
    async predictMechanical(composition, processing, microstructure) {
        const response = await fetch(`${this.baseUrl}/api/v1/predict/mechanical`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                composition,
                processing,
                microstructure
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    }
}

// Usage
const client = new CeramicArmorAPIClient('https://ceramic-armor-ml-api.onrender.com');
const result = await client.predictMechanical(
    { SiC: 0.85, B4C: 0.10, Al2O3: 0.05 },
    { sintering_temperature: 2100, pressure: 80, grain_size: 5.0 },
    { porosity: 0.01, phase_distribution: "uniform" }
);
```

## Best Practices

### 1. Error Handling

Always implement proper error handling:

```python
try:
    result = client.predict_mechanical(composition, processing, microstructure)
except requests.HTTPError as e:
    if e.response.status_code == 422:
        # Handle validation errors
        error_details = e.response.json()
        print(f"Validation error: {error_details['message']}")
    elif e.response.status_code == 429:
        # Handle rate limiting
        retry_after = int(e.response.headers.get('Retry-After', 60))
        print(f"Rate limited. Retry after {retry_after} seconds")
    else:
        print(f"HTTP error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### 2. Rate Limit Handling

Implement exponential backoff for rate limiting:

```python
import time
import random

def make_request_with_backoff(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except requests.HTTPError as e:
            if e.response.status_code == 429:
                retry_after = int(e.response.headers.get('Retry-After', 60))
                wait_time = retry_after + random.uniform(0, 10)
                time.sleep(wait_time)
            else:
                raise
    raise Exception("Max retries exceeded")
```

### 3. Input Validation

Validate inputs before sending requests:

```python
def validate_composition(composition):
    total = sum(composition.values())
    if total > 1.01:  # Allow small tolerance
        raise ValueError(f"Total composition {total:.3f} exceeds 1.0")
    
    for component, fraction in composition.items():
        if not 0 <= fraction <= 1:
            raise ValueError(f"{component} fraction {fraction} out of range [0, 1]")

def validate_processing(processing):
    temp = processing.get('sintering_temperature')
    if temp and not 1200 <= temp <= 2500:
        raise ValueError(f"Temperature {temp}°C out of range [1200, 2500]")
```

### 4. Batch Processing

For large datasets, use batch processing:

```python
def process_large_dataset(materials_df, batch_size=100):
    results = []
    
    for i in range(0, len(materials_df), batch_size):
        batch = materials_df.iloc[i:i+batch_size]
        batch_file = f"batch_{i}.csv"
        batch.to_csv(batch_file, index=False)
        
        # Upload and process batch
        batch_response = client.upload_batch_file(batch_file)
        batch_id = batch_response['request_id']
        
        # Wait for completion
        while True:
            status = client.get_batch_status(batch_id)
            if status['status'] == 'success':
                results.append(client.download_batch_results(batch_id))
                break
            elif status['status'] == 'failed':
                print(f"Batch {i} failed")
                break
            time.sleep(5)
    
    return results
```

### 5. Caching Results

Cache predictions to avoid redundant API calls:

```python
import hashlib
import json
from functools import lru_cache

def create_cache_key(composition, processing, microstructure):
    data = {
        'composition': composition,
        'processing': processing,
        'microstructure': microstructure
    }
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()

@lru_cache(maxsize=1000)
def cached_predict_mechanical(cache_key, composition, processing, microstructure):
    return client.predict_mechanical(composition, processing, microstructure)
```

## Examples

### Complete Integration Example

```python
#!/usr/bin/env python3
"""
Complete example of integrating with the Ceramic Armor ML API.
"""

import requests
import json
import time
from typing import Dict, Any, List

class CeramicArmorMLIntegration:
    def __init__(self, base_url: str = "https://ceramic-armor-ml-api.onrender.com"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'CeramicArmorML-Integration/1.0'
        })
    
    def health_check(self) -> bool:
        """Check if API is healthy."""
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.status_code == 200
        except:
            return False
    
    def predict_material_properties(self, material_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict both mechanical and ballistic properties."""
        results = {}
        
        # Predict mechanical properties
        try:
            mech_response = self.session.post(
                f"{self.base_url}/api/v1/predict/mechanical",
                json=material_data
            )
            mech_response.raise_for_status()
            results['mechanical'] = mech_response.json()
        except Exception as e:
            results['mechanical'] = {'error': str(e)}
        
        # Predict ballistic properties
        try:
            ball_response = self.session.post(
                f"{self.base_url}/api/v1/predict/ballistic",
                json=material_data
            )
            ball_response.raise_for_status()
            results['ballistic'] = ball_response.json()
        except Exception as e:
            results['ballistic'] = {'error': str(e)}
        
        return results
    
    def optimize_material(self, target_properties: Dict[str, float], 
                         constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find materials that meet target properties."""
        candidates = []
        
        # Generate candidate compositions
        for sic in [0.6, 0.7, 0.8, 0.9]:
            for b4c in [0.05, 0.1, 0.15, 0.2]:
                remaining = 1.0 - sic - b4c
                if remaining >= 0.05:  # Minimum Al2O3
                    composition = {
                        "SiC": sic,
                        "B4C": b4c,
                        "Al2O3": remaining,
                        "WC": 0.0,
                        "TiC": 0.0
                    }
                    
                    material_data = {
                        "composition": composition,
                        "processing": constraints.get("processing", {
                            "sintering_temperature": 2000,
                            "pressure": 70,
                            "grain_size": 6.0
                        }),
                        "microstructure": constraints.get("microstructure", {
                            "porosity": 0.02,
                            "phase_distribution": "uniform"
                        })
                    }
                    
                    results = self.predict_material_properties(material_data)
                    
                    # Check if meets targets
                    if self._meets_targets(results, target_properties):
                        candidates.append({
                            "composition": composition,
                            "predictions": results,
                            "score": self._calculate_score(results, target_properties)
                        })
        
        # Sort by score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:10]  # Return top 10
    
    def _meets_targets(self, results: Dict, targets: Dict[str, float]) -> bool:
        """Check if predictions meet target properties."""
        if 'mechanical' not in results or 'error' in results['mechanical']:
            return False
        
        predictions = results['mechanical']['predictions']
        
        for prop, target_value in targets.items():
            if prop in predictions:
                predicted_value = predictions[prop]['value']
                if predicted_value < target_value:
                    return False
        
        return True
    
    def _calculate_score(self, results: Dict, targets: Dict[str, float]) -> float:
        """Calculate optimization score."""
        if 'mechanical' not in results or 'error' in results['mechanical']:
            return 0.0
        
        predictions = results['mechanical']['predictions']
        score = 0.0
        
        for prop, target_value in targets.items():
            if prop in predictions:
                predicted_value = predictions[prop]['value']
                score += min(predicted_value / target_value, 2.0)  # Cap at 2x target
        
        return score / len(targets)

# Usage example
if __name__ == "__main__":
    # Initialize integration
    api = CeramicArmorMLIntegration()
    
    # Check API health
    if not api.health_check():
        print("API is not available")
        exit(1)
    
    # Define target properties
    targets = {
        "fracture_toughness": 4.0,  # MPa·m^0.5
        "vickers_hardness": 2500,   # HV
        "density": 3.0              # g/cm³
    }
    
    # Define constraints
    constraints = {
        "processing": {
            "sintering_temperature": 2000,
            "pressure": 70,
            "grain_size": 6.0
        },
        "microstructure": {
            "porosity": 0.02,
            "phase_distribution": "uniform"
        }
    }
    
    # Find optimal materials
    print("Optimizing material composition...")
    candidates = api.optimize_material(targets, constraints)
    
    print(f"\nFound {len(candidates)} candidate materials:")
    for i, candidate in enumerate(candidates[:3]):
        print(f"\n{i+1}. Score: {candidate['score']:.3f}")
        print(f"   Composition: SiC={candidate['composition']['SiC']:.2f}, "
              f"B4C={candidate['composition']['B4C']:.2f}, "
              f"Al2O3={candidate['composition']['Al2O3']:.2f}")
        
        if 'mechanical' in candidate['predictions']:
            mech = candidate['predictions']['mechanical']['predictions']
            print(f"   Fracture Toughness: {mech['fracture_toughness']['value']:.2f} MPa·m^0.5")
            print(f"   Vickers Hardness: {mech['vickers_hardness']['value']:.0f} HV")
            print(f"   Density: {mech['density']['value']:.2f} g/cm³")
```

This comprehensive integration guide provides everything needed to successfully integrate with the Ceramic Armor ML API, from basic usage to advanced optimization scenarios.