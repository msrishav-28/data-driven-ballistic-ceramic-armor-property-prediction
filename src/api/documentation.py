"""
API Documentation Configuration for Ceramic Armor ML API.

This module provides comprehensive documentation enhancements including
detailed examples, response schemas, and interactive documentation features.
"""

from typing import Dict, Any, List
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi


def get_custom_openapi_schema(app: FastAPI) -> Dict[str, Any]:
    """
    Generate custom OpenAPI schema with enhanced documentation.
    
    Args:
        app: FastAPI application instance
        
    Returns:
        Enhanced OpenAPI schema dictionary
    """
    if app.openapi_schema:
        return app.openapi_schema
    
    # Generate base OpenAPI schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        servers=app.servers
    )
    
    # Add comprehensive examples and documentation
    openapi_schema = add_comprehensive_examples(openapi_schema)
    openapi_schema = add_response_examples(openapi_schema)
    openapi_schema = add_security_documentation(openapi_schema)
    openapi_schema = add_error_examples(openapi_schema)
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


def add_comprehensive_examples(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Add comprehensive request examples to the OpenAPI schema."""
    
    # Mechanical prediction examples
    mechanical_examples = {
        "high_performance_sic": {
            "summary": "High Performance SiC Armor",
            "description": "Silicon carbide armor optimized for maximum hardness and toughness",
            "value": {
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
                "include_uncertainty": True,
                "include_feature_importance": True,
                "prediction_type": "mechanical"
            }
        },
        "lightweight_b4c": {
            "summary": "Lightweight B₄C Composite",
            "description": "Boron carbide composite optimized for weight reduction while maintaining protection",
            "value": {
                "composition": {
                    "SiC": 0.20,
                    "B4C": 0.70,
                    "Al2O3": 0.10,
                    "WC": 0.0,
                    "TiC": 0.0
                },
                "processing": {
                    "sintering_temperature": 1950,
                    "pressure": 60,
                    "grain_size": 8.0,
                    "holding_time": 120,
                    "heating_rate": 15,
                    "atmosphere": "nitrogen"
                },
                "microstructure": {
                    "porosity": 0.03,
                    "phase_distribution": "uniform",
                    "interface_quality": "good",
                    "pore_size": 1.2,
                    "connectivity": 0.15
                },
                "include_uncertainty": True,
                "include_feature_importance": True,
                "prediction_type": "mechanical"
            }
        },
        "cost_effective_alumina": {
            "summary": "Cost-Effective Al₂O₃ Armor",
            "description": "Aluminum oxide based armor for cost-sensitive applications",
            "value": {
                "composition": {
                    "SiC": 0.15,
                    "B4C": 0.05,
                    "Al2O3": 0.75,
                    "WC": 0.05,
                    "TiC": 0.0
                },
                "processing": {
                    "sintering_temperature": 1650,
                    "pressure": 40,
                    "grain_size": 15.0,
                    "holding_time": 90,
                    "heating_rate": 12,
                    "atmosphere": "air"
                },
                "microstructure": {
                    "porosity": 0.05,
                    "phase_distribution": "gradient",
                    "interface_quality": "fair",
                    "pore_size": 2.0,
                    "connectivity": 0.25
                },
                "include_uncertainty": True,
                "include_feature_importance": False,
                "prediction_type": "mechanical"
            }
        }
    }
    
    # Ballistic prediction examples
    ballistic_examples = {
        "multi_hit_armor": {
            "summary": "Multi-Hit Capable Armor",
            "description": "Ceramic armor designed for multiple impact resistance",
            "value": {
                "composition": {
                    "SiC": 0.60,
                    "B4C": 0.25,
                    "Al2O3": 0.10,
                    "WC": 0.05,
                    "TiC": 0.0
                },
                "processing": {
                    "sintering_temperature": 2000,
                    "pressure": 70,
                    "grain_size": 6.0,
                    "holding_time": 150,
                    "heating_rate": 8,
                    "atmosphere": "argon"
                },
                "microstructure": {
                    "porosity": 0.02,
                    "phase_distribution": "layered",
                    "interface_quality": "excellent",
                    "pore_size": 0.8,
                    "connectivity": 0.08
                },
                "include_uncertainty": True,
                "include_feature_importance": True,
                "prediction_type": "ballistic"
            }
        },
        "high_velocity_protection": {
            "summary": "High Velocity Protection",
            "description": "Armor optimized for high-velocity projectile protection",
            "value": {
                "composition": {
                    "SiC": 0.45,
                    "B4C": 0.35,
                    "Al2O3": 0.05,
                    "WC": 0.15,
                    "TiC": 0.0
                },
                "processing": {
                    "sintering_temperature": 2200,
                    "pressure": 90,
                    "grain_size": 3.0,
                    "holding_time": 200,
                    "heating_rate": 5,
                    "atmosphere": "vacuum"
                },
                "microstructure": {
                    "porosity": 0.008,
                    "phase_distribution": "uniform",
                    "interface_quality": "excellent",
                    "pore_size": 0.3,
                    "connectivity": 0.02
                },
                "include_uncertainty": True,
                "include_feature_importance": True,
                "prediction_type": "ballistic"
            }
        }
    }
    
    # Add examples to schema
    if "components" not in schema:
        schema["components"] = {}
    if "examples" not in schema["components"]:
        schema["components"]["examples"] = {}
    
    schema["components"]["examples"].update({
        "MechanicalPredictionExamples": mechanical_examples,
        "BallisticPredictionExamples": ballistic_examples
    })
    
    return schema


def add_response_examples(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Add comprehensive response examples to the OpenAPI schema."""
    
    # Mechanical prediction response example
    mechanical_response_example = {
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
            },
            {
                "name": "sintering_temperature",
                "importance": 0.28,
                "category": "processing",
                "description": "Sintering temperature during processing",
                "shap_value": 0.08
            },
            {
                "name": "grain_size",
                "importance": 0.22,
                "category": "processing",
                "description": "Average grain size in microstructure",
                "shap_value": -0.05
            }
        ],
        "model_info": {
            "model_version": "v1.2.0",
            "model_type": "XGBoost Ensemble",
            "training_r2": 0.87,
            "validation_r2": 0.82,
            "prediction_time_ms": 45,
            "feature_count": 156,
            "training_samples": 2847,
            "last_updated": "2024-01-15T10:30:00Z"
        },
        "request_id": "req_abc123def456",
        "timestamp": "2024-01-15T14:30:00Z",
        "warnings": None,
        "processing_notes": None
    }
    
    # Ballistic prediction response example
    ballistic_response_example = {
        "status": "success",
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
        },
        "feature_importance": [
            {
                "name": "B4C_content",
                "importance": 0.42,
                "category": "composition",
                "description": "Boron Carbide content affecting ballistic performance",
                "shap_value": 0.18
            },
            {
                "name": "density",
                "importance": 0.31,
                "category": "derived",
                "description": "Material density from composition and processing",
                "shap_value": 0.14
            }
        ],
        "model_info": {
            "model_version": "v1.1.0",
            "model_type": "XGBoost Ensemble (Ballistic)",
            "training_r2": 0.78,
            "validation_r2": 0.74,
            "prediction_time_ms": 52,
            "feature_count": 142,
            "training_samples": 1847,
            "last_updated": "2024-01-10T14:20:00Z"
        },
        "request_id": "req_xyz789ghi012",
        "timestamp": "2024-01-15T14:35:00Z",
        "warnings": None,
        "processing_notes": [
            "Ballistic predictions have inherently higher uncertainty due to complex impact dynamics",
            "Results are based on standardized test conditions (NIJ 0101.06 Level IIIA equivalent)",
            "Actual performance may vary with threat type, impact angle, and backing material"
        ]
    }
    
    # Batch processing response example
    batch_response_example = {
        "status": "success",
        "total_processed": 150,
        "successful_predictions": 147,
        "failed_predictions": 3,
        "processing_time_seconds": 12.5,
        "download_url": "https://ceramic-armor-ml-api.onrender.com/api/v1/predict/batch/batch_abc123def456/download",
        "file_size_mb": 2.3,
        "expires_at": "2024-01-16T14:30:00Z",
        "summary_statistics": {
            "avg_fracture_toughness": 4.2,
            "std_fracture_toughness": 0.8,
            "avg_vickers_hardness": 2650,
            "std_vickers_hardness": 320,
            "avg_v50_velocity": 825,
            "std_v50_velocity": 95
        },
        "request_id": "batch_abc123def456",
        "timestamp": "2024-01-15T14:30:00Z",
        "warnings": ["3 materials had incomplete composition data and used default values"]
    }
    
    # Add response examples to schema
    if "components" not in schema:
        schema["components"] = {}
    if "examples" not in schema["components"]:
        schema["components"]["examples"] = {}
    
    schema["components"]["examples"].update({
        "MechanicalPredictionResponse": {
            "summary": "Successful Mechanical Prediction",
            "description": "Example response for mechanical property prediction",
            "value": mechanical_response_example
        },
        "BallisticPredictionResponse": {
            "summary": "Successful Ballistic Prediction", 
            "description": "Example response for ballistic property prediction",
            "value": ballistic_response_example
        },
        "BatchProcessingResponse": {
            "summary": "Completed Batch Processing",
            "description": "Example response for completed batch processing job",
            "value": batch_response_example
        }
    })
    
    return schema


def add_security_documentation(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Add security and rate limiting documentation."""
    
    # Add security schemes
    if "components" not in schema:
        schema["components"] = {}
    if "securitySchemes" not in schema["components"]:
        schema["components"]["securitySchemes"] = {}
    
    schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key for authentication (currently optional for research use)"
        },
        "RateLimiting": {
            "type": "http",
            "scheme": "bearer",
            "description": "Rate limiting: 100 requests/hour for predictions, 10 requests/hour for batch processing"
        }
    }
    
    # Add rate limiting information to info
    if "info" not in schema:
        schema["info"] = {}
    
    schema["info"]["x-rate-limits"] = {
        "prediction_endpoints": {
            "limit": 100,
            "window": "1 hour",
            "scope": "per IP address"
        },
        "batch_processing": {
            "limit": 10,
            "window": "1 hour", 
            "scope": "per IP address"
        },
        "health_endpoints": {
            "limit": 1000,
            "window": "1 hour",
            "scope": "per IP address"
        }
    }
    
    return schema


def add_error_examples(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Add comprehensive error response examples."""
    
    error_examples = {
        "ValidationError": {
            "summary": "Validation Error",
            "description": "Request data validation failed",
            "value": {
                "error": "ValidationError",
                "message": "Invalid material composition data",
                "details": {
                    "total_composition": 1.15,
                    "max_allowed": 1.0
                },
                "field_errors": {
                    "composition": ["Total composition exceeds 100%"],
                    "processing.sintering_temperature": ["Temperature must be between 1200-2500°C"]
                },
                "request_id": "req_abc123def456",
                "timestamp": "2024-01-15T14:30:00Z",
                "suggestion": "Ensure all composition fractions sum to 1.0 or less and check processing parameter ranges"
            }
        },
        "PredictionError": {
            "summary": "Prediction Error",
            "description": "ML model prediction failed",
            "value": {
                "error": "PredictionError",
                "message": "Failed to predict mechanical properties",
                "details": {
                    "error_type": "ModelError",
                    "error_message": "Feature extraction failed for unusual composition",
                    "model_name": "mechanical_ensemble"
                },
                "request_id": "req_xyz789ghi012",
                "timestamp": "2024-01-15T14:30:00Z",
                "suggestion": "Please check your material composition. Ensure primary ceramic materials (SiC, B4C, Al2O3) comprise at least 50% of the composition."
            }
        },
        "RateLimitError": {
            "summary": "Rate Limit Exceeded",
            "description": "API rate limit exceeded",
            "value": {
                "error": "RateLimitExceeded",
                "message": "Rate limit exceeded for prediction endpoints",
                "details": {
                    "limit": 100,
                    "window": "1 hour",
                    "reset_time": "2024-01-15T15:30:00Z",
                    "requests_made": 101
                },
                "request_id": "req_rate_limit_001",
                "timestamp": "2024-01-15T14:30:00Z",
                "suggestion": "Please wait until the rate limit window resets or contact support for higher limits"
            }
        },
        "FileUploadError": {
            "summary": "File Upload Error",
            "description": "Batch file upload validation failed",
            "value": {
                "error": "FileUploadError",
                "message": "Invalid file structure",
                "details": {
                    "file_size_mb": 52.3,
                    "max_size_mb": 50.0,
                    "missing_columns": ["SiC", "sintering_temperature"]
                },
                "request_id": "batch_upload_001",
                "timestamp": "2024-01-15T14:30:00Z",
                "suggestion": "Ensure file is under 50MB and contains required columns: SiC, B4C, Al2O3, sintering_temperature, pressure, grain_size, porosity, phase_distribution"
            }
        }
    }
    
    # Add error examples to schema
    if "components" not in schema:
        schema["components"] = {}
    if "examples" not in schema["components"]:
        schema["components"]["examples"] = {}
    
    schema["components"]["examples"].update(error_examples)
    
    return schema


def setup_api_documentation(app: FastAPI) -> None:
    """
    Set up comprehensive API documentation for the FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    
    def custom_openapi():
        return get_custom_openapi_schema(app)
    
    app.openapi = custom_openapi


# Custom CSS for enhanced documentation appearance
CUSTOM_DOCS_CSS = """
<style>
.swagger-ui .topbar { display: none; }
.swagger-ui .info { margin: 20px 0; }
.swagger-ui .info .title { color: #1f2937; font-size: 2.5rem; }
.swagger-ui .info .description { font-size: 1.1rem; line-height: 1.6; }
.swagger-ui .scheme-container { background: #f8fafc; padding: 20px; border-radius: 8px; margin: 20px 0; }
.swagger-ui .opblock.opblock-post { border-color: #059669; }
.swagger-ui .opblock.opblock-get { border-color: #0284c7; }
.swagger-ui .opblock.opblock-delete { border-color: #dc2626; }
.swagger-ui .opblock-summary { font-weight: 600; }
.swagger-ui .parameter__name { font-weight: 600; color: #374151; }
.swagger-ui .response-col_status { font-weight: 600; }
.swagger-ui .example { background: #f1f5f9; border-radius: 6px; }
.swagger-ui .model-box { background: #fafafa; border: 1px solid #e5e7eb; }
</style>
"""

# Custom JavaScript for enhanced interactivity
CUSTOM_DOCS_JS = """
<script>
// Add custom functionality for better UX
document.addEventListener('DOMContentLoaded', function() {
    // Add copy buttons to code examples
    const codeBlocks = document.querySelectorAll('pre code');
    codeBlocks.forEach(block => {
        const button = document.createElement('button');
        button.textContent = 'Copy';
        button.className = 'copy-btn';
        button.onclick = () => {
            navigator.clipboard.writeText(block.textContent);
            button.textContent = 'Copied!';
            setTimeout(() => button.textContent = 'Copy', 2000);
        };
        block.parentNode.appendChild(button);
    });
    
    // Add expand/collapse for large examples
    const examples = document.querySelectorAll('.example');
    examples.forEach(example => {
        if (example.textContent.length > 1000) {
            example.style.maxHeight = '200px';
            example.style.overflow = 'hidden';
            const expandBtn = document.createElement('button');
            expandBtn.textContent = 'Show more';
            expandBtn.onclick = () => {
                example.style.maxHeight = example.style.maxHeight === 'none' ? '200px' : 'none';
                expandBtn.textContent = example.style.maxHeight === 'none' ? 'Show less' : 'Show more';
            };
            example.parentNode.appendChild(expandBtn);
        }
    });
});
</script>
"""