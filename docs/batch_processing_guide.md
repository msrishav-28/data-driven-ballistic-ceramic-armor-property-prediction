# Batch Processing Guide

## Overview

The Ceramic Armor ML API supports batch processing of multiple materials through file uploads. This allows you to predict properties for hundreds or thousands of materials efficiently.

## Supported File Formats

- **CSV** (.csv) - Comma-separated values
- **Excel** (.xlsx, .xls) - Microsoft Excel files  
- **JSON** (.json) - JavaScript Object Notation

## Required Columns

Your input file must contain the following minimum columns:

### Composition (Required)
- `SiC` - Silicon Carbide fraction (0-1)
- `B4C` - Boron Carbide fraction (0-1) 
- `Al2O3` - Aluminum Oxide fraction (0-1)

### Processing Parameters (Required)
- `sintering_temperature` - Temperature in Celsius (1200-2500)
- `pressure` - Applied pressure in MPa (1-200)
- `grain_size` - Average grain size in micrometers (0.1-100)

### Microstructure (Required)
- `porosity` - Porosity fraction (0-0.3)
- `phase_distribution` - Distribution type: "uniform", "gradient", or "layered"

## Optional Columns

### Additional Composition
- `WC` - Tungsten Carbide fraction (0-1)
- `TiC` - Titanium Carbide fraction (0-1)
- `ZrO2` - Zirconium Oxide fraction (0-1)
- `TiB2` - Titanium Diboride fraction (0-1)

### Additional Processing
- `holding_time` - Sintering time in minutes (1-600)
- `heating_rate` - Heating rate in °C/min (1-50)
- `atmosphere` - Sintering atmosphere: "air", "argon", "nitrogen", "vacuum", "hydrogen"

### Additional Microstructure
- `interface_quality` - Interface quality: "poor", "fair", "good", "excellent"
- `pore_size` - Average pore size in micrometers (0.01-50)
- `connectivity` - Pore connectivity factor (0-1)

## API Endpoints

### 1. Upload Batch File

```http
POST /api/v1/predict/batch
Content-Type: multipart/form-data
```

**Parameters:**
- `file` - The CSV/Excel/JSON file to upload
- `file_format` - Input format: "csv", "xlsx", or "json"
- `output_format` - Output format: "csv", "xlsx", or "json"
- `include_uncertainty` - Include uncertainty quantification (true/false)
- `include_feature_importance` - Include feature importance (true/false)
- `prediction_type` - Type of predictions: "mechanical", "ballistic", or "both"
- `max_rows` - Maximum number of rows to process (1-10000)

**Response:**
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

### 2. Check Processing Status

```http
GET /api/v1/predict/batch/{batch_id}/status
```

**Response:**
```json
{
  "status": "success",
  "total_processed": 150,
  "successful_predictions": 147,
  "failed_predictions": 3,
  "processing_time_seconds": 12.5,
  "download_url": "/api/v1/predict/batch/batch_abc123def456/download",
  "file_size_mb": 2.3,
  "expires_at": "2024-01-16T14:30:00Z",
  "summary_statistics": {
    "avg_fracture_toughness": 4.2,
    "std_fracture_toughness": 0.8
  },
  "request_id": "batch_abc123def456",
  "timestamp": "2024-01-15T14:30:00Z"
}
```

### 3. Download Results

```http
GET /api/v1/predict/batch/{batch_id}/download
```

Returns the results file in the requested format.

### 4. Cancel/Delete Batch Job

```http
DELETE /api/v1/predict/batch/{batch_id}
```

Cancels processing or cleans up completed job data.

## Output Format

The results file contains all input columns plus prediction results:

### Mechanical Properties (if requested)
- `mech_fracture_toughness_value` - Predicted fracture toughness
- `mech_fracture_toughness_unit` - Unit (MPa·m^0.5)
- `mech_fracture_toughness_uncertainty` - Uncertainty (if enabled)
- `mech_fracture_toughness_ci_lower` - Lower confidence interval
- `mech_fracture_toughness_ci_upper` - Upper confidence interval
- Similar columns for: `vickers_hardness`, `density`, `elastic_modulus`

### Ballistic Properties (if requested)
- `ball_v50_velocity_value` - Predicted V50 velocity
- `ball_v50_velocity_unit` - Unit (m/s)
- `ball_v50_velocity_uncertainty` - Uncertainty (if enabled)
- `ball_v50_velocity_ci_lower` - Lower confidence interval
- `ball_v50_velocity_ci_upper` - Upper confidence interval
- Similar columns for: `penetration_resistance`, `back_face_deformation`, `multi_hit_capability`

## Example Usage

### Python with requests

```python
import requests
import time

# Upload file
with open('materials.csv', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/predict/batch',
        files={'file': f},
        data={
            'file_format': 'csv',
            'output_format': 'csv',
            'include_uncertainty': 'true',
            'prediction_type': 'both',
            'max_rows': 1000
        }
    )

batch_id = response.json()['request_id']

# Check status
while True:
    status_response = requests.get(f'http://localhost:8000/api/v1/predict/batch/{batch_id}/status')
    status_data = status_response.json()
    
    if status_data['status'] == 'success' and status_data['download_url']:
        break
    
    time.sleep(5)  # Wait 5 seconds

# Download results
download_response = requests.get(f'http://localhost:8000{status_data["download_url"]}')
with open('results.csv', 'wb') as f:
    f.write(download_response.content)
```

### cURL

```bash
# Upload file
curl -X POST "http://localhost:8000/api/v1/predict/batch" \
  -F "file=@materials.csv" \
  -F "file_format=csv" \
  -F "output_format=csv" \
  -F "include_uncertainty=true" \
  -F "prediction_type=both"

# Check status (replace batch_id with actual ID)
curl "http://localhost:8000/api/v1/predict/batch/batch_abc123def456/status"

# Download results
curl -O "http://localhost:8000/api/v1/predict/batch/batch_abc123def456/download"
```

## Limitations

- Maximum file size: 50MB
- Maximum rows per batch: 10,000
- Results expire after 24 hours
- Processing is done asynchronously in the background
- Feature importance is disabled by default for batch processing to reduce file size

## Error Handling

Common errors and solutions:

1. **Missing Required Columns**: Ensure your file has all required composition, processing, and microstructure columns
2. **Invalid Composition**: Total composition fractions must sum to ≤ 1.0
3. **Out of Range Values**: Check that temperatures, pressures, and other parameters are within valid ranges
4. **File Too Large**: Split large files into smaller batches
5. **Invalid File Format**: Ensure file extension matches the specified format

## Performance Tips

- Use CSV format for fastest processing
- Disable feature importance for large batches
- Process mechanical and ballistic properties separately if only one type is needed
- Use reasonable max_rows limits to avoid timeouts
- Monitor processing status rather than waiting synchronously