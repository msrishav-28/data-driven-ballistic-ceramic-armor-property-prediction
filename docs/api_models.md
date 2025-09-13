# API Models Documentation

This document provides comprehensive documentation for the Pydantic data models used in the Ceramic Armor ML API.

## Overview

The API models are organized into three main categories:
- **Request Models**: Define the structure for incoming API requests
- **Response Models**: Define the structure for API responses
- **Exception Models**: Handle validation errors and provide detailed error information

## Request Models

### CompositionModel

Defines the material composition with validation for ceramic armor materials.

**Fields:**
- `SiC` (float): Silicon Carbide fraction (0-1)
- `B4C` (float): Boron Carbide fraction (0-1) 
- `Al2O3` (float): Aluminum Oxide fraction (0-1)
- `WC` (float, optional): Tungsten Carbide fraction (0-1)
- `TiC` (float, optional): Titanium Carbide fraction (0-1)
- `ZrO2` (float, optional): Zirconium Oxide fraction (0-1)
- `TiB2` (float, optional): Titanium Diboride fraction (0-1)
- `other` (float, optional): Other materials fraction (0-1)

**Validation Rules:**
- Total composition cannot exceed 100% (1.0)
- Total composition must be at least 1% (0.01)
- Primary ceramic materials (SiC, B4C, Al2O3, WC, TiC) must comprise at least 50%

**Example:**
```json
{
  "SiC": 0.6,
  "B4C": 0.3,
  "Al2O3": 0.1,
  "WC": 0.0,
  "TiC": 0.0
}
```

### ProcessingModel

Defines processing parameters for ceramic manufacturing conditions.

**Fields:**
- `sintering_temperature` (float): Sintering temperature in Celsius (1200-2500°C)
- `pressure` (float): Applied pressure in MPa (1-200 MPa)
- `grain_size` (float): Average grain size in micrometers (0.1-100 μm)
- `holding_time` (float, optional): Sintering holding time in minutes (1-600 min)
- `heating_rate` (float, optional): Heating rate in °C/min (1-50 °C/min)
- `atmosphere` (str, optional): Sintering atmosphere (air|argon|nitrogen|vacuum|hydrogen)

**Validation Rules:**
- Temperature-grain size relationship validation
- Very high temperatures (>2000°C) typically produce grain sizes >1μm
- Low temperatures (<1400°C) rarely produce grain sizes >50μm

**Example:**
```json
{
  "sintering_temperature": 1800,
  "pressure": 50,
  "grain_size": 10,
  "holding_time": 120,
  "heating_rate": 15,
  "atmosphere": "argon"
}
```

### MicrostructureModel

Defines microstructure characteristics for ceramic materials.

**Fields:**
- `porosity` (float): Porosity fraction (0-0.3)
- `phase_distribution` (PhaseDistribution): Type of phase distribution (uniform|gradient|layered)
- `interface_quality` (InterfaceQuality, optional): Quality of interfaces (poor|fair|good|excellent)
- `pore_size` (float, optional): Average pore size in micrometers (0.01-50 μm)
- `connectivity` (float, optional): Pore connectivity factor (0-1)

**Validation Rules:**
- Pore size-porosity relationship validation
- High porosity (>15%) typically has pore sizes >0.1μm
- Low porosity (<2%) rarely has pore sizes >10μm

**Example:**
```json
{
  "porosity": 0.02,
  "phase_distribution": "uniform",
  "interface_quality": "good",
  "pore_size": 1.0,
  "connectivity": 0.1
}
```

### PredictionRequest

Complete prediction request combining all material parameters.

**Fields:**
- `composition` (CompositionModel): Material composition data
- `processing` (ProcessingModel): Processing parameters
- `microstructure` (MicrostructureModel): Microstructure characteristics
- `include_uncertainty` (bool): Include uncertainty quantification (default: true)
- `include_feature_importance` (bool): Include feature importance analysis (default: true)
- `prediction_type` (str, optional): Type of properties to predict (mechanical|ballistic|both)

## Response Models

### PropertyPrediction

Individual property prediction with uncertainty quantification.

**Fields:**
- `value` (float): Predicted property value
- `unit` (str): Unit of measurement
- `confidence_interval` (List[float]): 95% confidence interval [lower, upper]
- `uncertainty` (float): Relative uncertainty (0-1)
- `prediction_quality` (str): Quality assessment (excellent|good|fair|poor)

### MechanicalPredictions

Mechanical property predictions for ceramic armor materials.

**Fields:**
- `fracture_toughness` (PropertyPrediction): Fracture toughness prediction
- `vickers_hardness` (PropertyPrediction): Vickers hardness prediction
- `density` (PropertyPrediction): Material density prediction
- `elastic_modulus` (PropertyPrediction): Elastic modulus prediction
- `compressive_strength` (PropertyPrediction, optional): Compressive strength prediction
- `flexural_strength` (PropertyPrediction, optional): Flexural strength prediction
- `poisson_ratio` (PropertyPrediction, optional): Poisson's ratio prediction

### BallisticPredictions

Ballistic property predictions for ceramic armor materials.

**Fields:**
- `v50_velocity` (PropertyPrediction): V50 ballistic limit velocity
- `penetration_resistance` (PropertyPrediction): Penetration resistance index
- `back_face_deformation` (PropertyPrediction): Back face deformation under impact
- `multi_hit_capability` (PropertyPrediction): Multi-hit capability score
- `energy_absorption` (PropertyPrediction, optional): Energy absorption capacity
- `damage_tolerance` (PropertyPrediction, optional): Damage tolerance index

### PredictionResponse

Complete prediction response containing all results and metadata.

**Fields:**
- `status` (PredictionStatus): Prediction status (success|partial|failed|warning)
- `predictions` (Union[MechanicalPredictions, BallisticPredictions, Dict]): Property predictions
- `feature_importance` (List[FeatureImportance], optional): Feature importance analysis
- `model_info` (ModelInfo): Model information and performance metrics
- `request_id` (str): Unique request identifier
- `timestamp` (datetime): Prediction timestamp
- `warnings` (List[str], optional): Any warnings generated during prediction
- `processing_notes` (List[str], optional): Notes about data processing or assumptions

## Exception Handling

### Custom Exceptions

- `CeramicArmorMLException`: Base exception class
- `ValidationException`: Input validation errors
- `ModelException`: ML model-related errors
- `PredictionException`: Prediction-related errors
- `DataProcessingException`: Data processing errors
- `FileProcessingException`: File upload and processing errors
- `RateLimitException`: Rate limiting errors

### Validation Utilities

#### validate_material_composition(composition_data)

Validates material composition beyond basic Pydantic validation.

**Checks:**
- Realistic composition combinations
- Incompatible high concentrations
- Very low total ceramic content
- Unrealistic pure compositions

#### validate_processing_parameters(processing_data, composition_data)

Validates processing parameters in context of material composition.

**Checks:**
- Temperature-composition compatibility
- SiC-rich compositions require temperatures >1600°C
- B4C-rich compositions may decompose at temperatures >2200°C
- Pressure-grain size relationships

## Usage Examples

### Basic Prediction Request

```python
from src.api.models import CompositionModel, ProcessingModel, MicrostructureModel, PredictionRequest, PhaseDistribution

# Create composition
composition = CompositionModel(
    SiC=0.6,
    B4C=0.3,
    Al2O3=0.1
)

# Create processing parameters
processing = ProcessingModel(
    sintering_temperature=1800,
    pressure=50,
    grain_size=10,
    holding_time=120,
    atmosphere="argon"
)

# Create microstructure
microstructure = MicrostructureModel(
    porosity=0.02,
    phase_distribution=PhaseDistribution.UNIFORM,
    interface_quality="good"
)

# Create complete request
request = PredictionRequest(
    composition=composition,
    processing=processing,
    microstructure=microstructure,
    prediction_type="both"
)
```

### Error Handling

```python
from src.api.models import ValidationException, create_validation_error_response
from pydantic import ValidationError

try:
    # Invalid composition (total > 1.0)
    composition = CompositionModel(
        SiC=0.7,
        B4C=0.4,
        Al2O3=0.2
    )
except ValidationError as e:
    error_response = create_validation_error_response(e, "req_123")
    print(error_response)
```

## Validation Rules Summary

### Composition Validation
- All fractions must be between 0 and 1
- Total composition ≤ 100% (1.0)
- Primary ceramics ≥ 50% of total
- At least 1% total material required

### Processing Validation
- Temperature: 1200-2500°C
- Pressure: 1-200 MPa
- Grain size: 0.1-100 μm
- Temperature-grain size compatibility
- Material-specific temperature requirements

### Microstructure Validation
- Porosity: 0-30% (0.0-0.3)
- Pore size: 0.01-50 μm
- Pore size-porosity relationship
- Connectivity: 0-1

### Cross-Parameter Validation
- SiC-rich (>50%) + low temp (<1600°C) → Warning
- B4C-rich (>50%) + high temp (>2200°C) → Warning
- High pressure (>100 MPa) + large grains (>20 μm) → Warning
- High porosity (>15%) + small pores (<0.1 μm) → Warning

This comprehensive validation ensures that only realistic and physically meaningful material parameters are accepted by the API.