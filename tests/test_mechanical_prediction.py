"""
Comprehensive unit tests for mechanical properties prediction endpoint.
"""

import pytest
import json
from fastapi.testclient import TestClient


class TestMechanicalPredictionEndpoint:
    """Test mechanical properties prediction endpoint functionality."""
    
    def test_endpoint_exists(self, client):
        """Test that the mechanical prediction endpoint exists."""
        response = client.post("/api/v1/predict/mechanical", json={})
        # Should not return 404 (endpoint exists)
        assert response.status_code != 404
    
    def test_valid_prediction_request(self, client, valid_prediction_request):
        """Test successful mechanical property prediction."""
        request_data = valid_prediction_request.model_dump()
        
        response = client.post("/api/v1/predict/mechanical", json=request_data)
        
        # Should return successful response
        assert response.status_code == 200
        
        data = response.json()
        
        # Verify response structure
        assert "status" in data
        assert "predictions" in data
        assert "model_info" in data
        assert "request_id" in data
        assert "timestamp" in data
        
        # Verify prediction structure
        predictions = data["predictions"]
        expected_properties = ["fracture_toughness", "vickers_hardness", "density", "elastic_modulus"]
        
        for prop in expected_properties:
            assert prop in predictions
            prop_data = predictions[prop]
            assert "value" in prop_data
            assert "unit" in prop_data
            assert "confidence_interval" in prop_data
            assert "uncertainty" in prop_data
            assert "prediction_quality" in prop_data
            
            # Verify data types
            assert isinstance(prop_data["value"], (int, float))
            assert isinstance(prop_data["unit"], str)
            assert isinstance(prop_data["confidence_interval"], list)
            assert len(prop_data["confidence_interval"]) == 2
            assert isinstance(prop_data["uncertainty"], (int, float))
            assert prop_data["prediction_quality"] in ["poor", "fair", "good", "excellent"]
    
    def test_prediction_without_uncertainty(self, client, valid_prediction_request):
        """Test prediction request without uncertainty quantification."""
        request_data = valid_prediction_request.model_dump()
        request_data["include_uncertainty"] = False
        
        response = client.post("/api/v1/predict/mechanical", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        predictions = data["predictions"]
        
        # Should still have uncertainty fields but with default values
        for prop in ["fracture_toughness", "vickers_hardness", "density", "elastic_modulus"]:
            prop_data = predictions[prop]
            assert "uncertainty" in prop_data
            assert "confidence_interval" in prop_data
    
    def test_prediction_without_feature_importance(self, client, valid_prediction_request):
        """Test prediction request without feature importance."""
        request_data = valid_prediction_request.model_dump()
        request_data["include_feature_importance"] = False
        
        response = client.post("/api/v1/predict/mechanical", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        # Feature importance should be None or empty
        assert data.get("feature_importance") is None or len(data.get("feature_importance", [])) == 0
    
    def test_model_info_structure(self, client, valid_prediction_request):
        """Test model information structure in response."""
        request_data = valid_prediction_request.model_dump()
        
        response = client.post("/api/v1/predict/mechanical", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        model_info = data["model_info"]
        
        required_fields = [
            "model_version", "model_type", "training_r2", "validation_r2",
            "prediction_time_ms", "feature_count", "training_samples", "last_updated"
        ]
        
        for field in required_fields:
            assert field in model_info
        
        # Verify data types
        assert isinstance(model_info["training_r2"], (int, float))
        assert isinstance(model_info["validation_r2"], (int, float))
        assert isinstance(model_info["prediction_time_ms"], int)
        assert isinstance(model_info["feature_count"], int)
        assert isinstance(model_info["training_samples"], int)


class TestMechanicalPredictionValidation:
    """Test input validation for mechanical prediction endpoint."""
    
    def test_invalid_composition_sum(self, client, valid_processing, valid_microstructure):
        """Test validation error for composition sum > 1."""
        invalid_data = {
            "composition": {
                "SiC": 0.8,
                "B4C": 0.5,  # Total > 1.0
                "Al2O3": 0.1,
                "WC": 0.0,
                "TiC": 0.0
            },
            "processing": valid_processing.model_dump(),
            "microstructure": valid_microstructure.model_dump()
        }
        
        response = client.post("/api/v1/predict/mechanical", json=invalid_data)
        assert response.status_code == 422
        
        error_data = response.json()
        assert "detail" in error_data
    
    def test_negative_composition_values(self, client, valid_processing, valid_microstructure):
        """Test validation error for negative composition values."""
        invalid_data = {
            "composition": {
                "SiC": -0.1,  # Negative value
                "B4C": 0.5,
                "Al2O3": 0.5,
                "WC": 0.0,
                "TiC": 0.0
            },
            "processing": valid_processing.model_dump(),
            "microstructure": valid_microstructure.model_dump()
        }
        
        response = client.post("/api/v1/predict/mechanical", json=invalid_data)
        assert response.status_code == 422
    
    def test_invalid_temperature_range(self, client, valid_composition, valid_microstructure):
        """Test validation error for temperature out of range."""
        invalid_data = {
            "composition": valid_composition.model_dump(),
            "processing": {
                "sintering_temperature": 1000,  # Too low
                "pressure": 50,
                "grain_size": 10,
                "holding_time": 120,
                "heating_rate": 15,
                "atmosphere": "argon"
            },
            "microstructure": valid_microstructure.model_dump()
        }
        
        response = client.post("/api/v1/predict/mechanical", json=invalid_data)
        assert response.status_code == 422
    
    def test_invalid_pressure_range(self, client, valid_composition, valid_microstructure):
        """Test validation error for pressure out of range."""
        invalid_data = {
            "composition": valid_composition.model_dump(),
            "processing": {
                "sintering_temperature": 1800,
                "pressure": 300,  # Too high
                "grain_size": 10,
                "holding_time": 120,
                "heating_rate": 15,
                "atmosphere": "argon"
            },
            "microstructure": valid_microstructure.model_dump()
        }
        
        response = client.post("/api/v1/predict/mechanical", json=invalid_data)
        assert response.status_code == 422
    
    def test_invalid_porosity_range(self, client, valid_composition, valid_processing):
        """Test validation error for porosity out of range."""
        invalid_data = {
            "composition": valid_composition.model_dump(),
            "processing": valid_processing.model_dump(),
            "microstructure": {
                "porosity": 0.5,  # Too high (50%)
                "phase_distribution": "uniform",
                "interface_quality": "good",
                "pore_size": 1.0,
                "connectivity": 0.1
            }
        }
        
        response = client.post("/api/v1/predict/mechanical", json=invalid_data)
        assert response.status_code == 422
    
    def test_invalid_phase_distribution(self, client, valid_composition, valid_processing):
        """Test validation error for invalid phase distribution."""
        invalid_data = {
            "composition": valid_composition.model_dump(),
            "processing": valid_processing.model_dump(),
            "microstructure": {
                "porosity": 0.02,
                "phase_distribution": "invalid_distribution",  # Invalid value
                "interface_quality": "good",
                "pore_size": 1.0,
                "connectivity": 0.1
            }
        }
        
        response = client.post("/api/v1/predict/mechanical", json=invalid_data)
        assert response.status_code == 422
    
    def test_missing_required_fields(self, client):
        """Test validation error for missing required fields."""
        incomplete_data = {
            "composition": {
                "SiC": 0.6,
                "B4C": 0.3,
                "Al2O3": 0.1
            }
            # Missing processing and microstructure
        }
        
        response = client.post("/api/v1/predict/mechanical", json=incomplete_data)
        assert response.status_code == 422
    
    def test_empty_request_body(self, client):
        """Test validation error for empty request body."""
        response = client.post("/api/v1/predict/mechanical", json={})
        assert response.status_code == 422
    
    def test_invalid_json_format(self, client):
        """Test error handling for invalid JSON."""
        response = client.post(
            "/api/v1/predict/mechanical",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422


class TestMechanicalPredictionEdgeCases:
    """Test edge cases for mechanical prediction endpoint."""
    
    def test_minimal_valid_composition(self, client, valid_processing, valid_microstructure):
        """Test prediction with minimal valid composition."""
        minimal_data = {
            "composition": {
                "SiC": 0.5,  # Minimum primary ceramic content
                "B4C": 0.0,
                "Al2O3": 0.5,
                "WC": 0.0,
                "TiC": 0.0
            },
            "processing": valid_processing.model_dump(),
            "microstructure": valid_microstructure.model_dump()
        }
        
        response = client.post("/api/v1/predict/mechanical", json=minimal_data)
        # Should succeed with minimal valid data
        assert response.status_code == 200
    
    def test_maximum_valid_values(self, client):
        """Test prediction with maximum valid parameter values."""
        max_data = {
            "composition": {
                "SiC": 1.0,  # Pure SiC
                "B4C": 0.0,
                "Al2O3": 0.0,
                "WC": 0.0,
                "TiC": 0.0
            },
            "processing": {
                "sintering_temperature": 2500,  # Maximum temperature
                "pressure": 200,  # Maximum pressure
                "grain_size": 100,  # Maximum grain size
                "holding_time": 600,  # Maximum holding time
                "heating_rate": 50,  # Maximum heating rate
                "atmosphere": "vacuum"
            },
            "microstructure": {
                "porosity": 0.3,  # Maximum porosity
                "phase_distribution": "layered",
                "interface_quality": "excellent",
                "pore_size": 10.0,  # Maximum pore size
                "connectivity": 0.5  # Maximum connectivity
            }
        }
        
        response = client.post("/api/v1/predict/mechanical", json=max_data)
        assert response.status_code == 200
    
    def test_zero_porosity(self, client, valid_composition, valid_processing):
        """Test prediction with zero porosity (theoretical dense material)."""
        zero_porosity_data = {
            "composition": valid_composition.model_dump(),
            "processing": valid_processing.model_dump(),
            "microstructure": {
                "porosity": 0.0,  # Zero porosity
                "phase_distribution": "uniform",
                "interface_quality": "excellent",
                "pore_size": 0.1,
                "connectivity": 0.01
            }
        }
        
        response = client.post("/api/v1/predict/mechanical", json=zero_porosity_data)
        assert response.status_code == 200


class TestMechanicalPredictionResponseFormat:
    """Test response format and data consistency."""
    
    def test_response_json_format(self, client, valid_prediction_request):
        """Test that response is valid JSON."""
        request_data = valid_prediction_request.model_dump()
        
        response = client.post("/api/v1/predict/mechanical", json=request_data)
        assert response.status_code == 200
        
        # Should be valid JSON
        data = response.json()
        assert isinstance(data, dict)
    
    def test_confidence_interval_consistency(self, client, valid_prediction_request):
        """Test that confidence intervals are consistent with predicted values."""
        request_data = valid_prediction_request.model_dump()
        
        response = client.post("/api/v1/predict/mechanical", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        predictions = data["predictions"]
        
        for prop_name, prop_data in predictions.items():
            value = prop_data["value"]
            ci = prop_data["confidence_interval"]
            
            # Confidence interval should contain the predicted value
            assert ci[0] <= value <= ci[1], f"Confidence interval inconsistent for {prop_name}"
            
            # Lower bound should be less than upper bound
            assert ci[0] < ci[1], f"Invalid confidence interval for {prop_name}"
    
    def test_uncertainty_values_range(self, client, valid_prediction_request):
        """Test that uncertainty values are in valid range [0, 1]."""
        request_data = valid_prediction_request.model_dump()
        
        response = client.post("/api/v1/predict/mechanical", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        predictions = data["predictions"]
        
        for prop_name, prop_data in predictions.items():
            uncertainty = prop_data["uncertainty"]
            assert 0 <= uncertainty <= 1, f"Invalid uncertainty value for {prop_name}: {uncertainty}"
    
    def test_request_id_uniqueness(self, client, valid_prediction_request):
        """Test that each request gets a unique request ID."""
        request_data = valid_prediction_request.model_dump()
        
        # Make multiple requests
        response1 = client.post("/api/v1/predict/mechanical", json=request_data)
        response2 = client.post("/api/v1/predict/mechanical", json=request_data)
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        data1 = response1.json()
        data2 = response2.json()
        
        # Request IDs should be different
        assert data1["request_id"] != data2["request_id"]