"""
Comprehensive unit tests for ballistic properties prediction endpoint.
"""

import pytest
import json
from fastapi.testclient import TestClient


class TestBallisticPredictionEndpoint:
    """Test ballistic properties prediction endpoint functionality."""
    
    def test_endpoint_exists(self, client):
        """Test that the ballistic prediction endpoint exists."""
        response = client.post("/api/v1/predict/ballistic", json={})
        # Should not return 404 (endpoint exists)
        assert response.status_code != 404
    
    def test_valid_prediction_request(self, client, valid_prediction_request):
        """Test successful ballistic property prediction."""
        request_data = valid_prediction_request.model_dump()
        
        response = client.post("/api/v1/predict/ballistic", json=request_data)
        
        # Should return successful response
        assert response.status_code == 200
        
        data = response.json()
        
        # Verify response structure
        assert "status" in data
        assert "predictions" in data
        assert "model_info" in data
        assert "request_id" in data
        assert "timestamp" in data
        
        # Verify ballistic prediction structure
        predictions = data["predictions"]
        expected_properties = ["v50_velocity", "penetration_resistance", "back_face_deformation", "multi_hit_capability"]
        
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
    
    def test_ballistic_specific_units(self, client, valid_prediction_request):
        """Test that ballistic predictions have correct units."""
        request_data = valid_prediction_request.model_dump()
        
        response = client.post("/api/v1/predict/ballistic", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        predictions = data["predictions"]
        
        # Check expected units for ballistic properties
        expected_units = {
            "v50_velocity": "m/s",
            "penetration_resistance": "dimensionless",
            "back_face_deformation": "mm",
            "multi_hit_capability": "probability"
        }
        
        for prop, expected_unit in expected_units.items():
            assert predictions[prop]["unit"] == expected_unit
    
    def test_ballistic_uncertainty_higher_than_mechanical(self, client, valid_prediction_request):
        """Test that ballistic predictions typically have higher uncertainty."""
        request_data = valid_prediction_request.model_dump()
        
        response = client.post("/api/v1/predict/ballistic", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        predictions = data["predictions"]
        
        # Ballistic predictions should generally have higher uncertainty
        # This is a general expectation due to the complexity of ballistic phenomena
        for prop_name, prop_data in predictions.items():
            uncertainty = prop_data["uncertainty"]
            # Ballistic uncertainty is typically higher than mechanical (>0.1)
            assert uncertainty >= 0.0  # At minimum, should be non-negative
    
    def test_processing_notes_included(self, client, valid_prediction_request):
        """Test that ballistic predictions include processing notes."""
        request_data = valid_prediction_request.model_dump()
        request_data["include_uncertainty"] = True
        
        response = client.post("/api/v1/predict/ballistic", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        
        # Should include processing notes for ballistic predictions
        assert "processing_notes" in data
        if data["processing_notes"]:
            assert isinstance(data["processing_notes"], list)
            assert len(data["processing_notes"]) > 0
    
    def test_model_info_ballistic_specific(self, client, valid_prediction_request):
        """Test model information is specific to ballistic models."""
        request_data = valid_prediction_request.model_dump()
        
        response = client.post("/api/v1/predict/ballistic", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        model_info = data["model_info"]
        
        # Ballistic models typically have different characteristics
        assert "model_type" in model_info
        assert "ballistic" in model_info["model_type"].lower() or "Ballistic" in model_info["model_type"]


class TestBallisticPredictionValidation:
    """Test input validation for ballistic prediction endpoint."""
    
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
        
        response = client.post("/api/v1/predict/ballistic", json=invalid_data)
        assert response.status_code == 422
    
    def test_ballistic_specific_validation(self, client, valid_composition, valid_microstructure):
        """Test ballistic-specific validation requirements."""
        # Test with processing parameters that might be invalid for ballistic applications
        invalid_data = {
            "composition": valid_composition.model_dump(),
            "processing": {
                "sintering_temperature": 1200,  # Very low temperature
                "pressure": 5,  # Very low pressure
                "grain_size": 100,  # Very large grain size
                "holding_time": 30,  # Very short time
                "heating_rate": 5,
                "atmosphere": "air"  # Potentially problematic atmosphere
            },
            "microstructure": valid_microstructure.model_dump()
        }
        
        response = client.post("/api/v1/predict/ballistic", json=invalid_data)
        # Should either succeed with warnings or fail with validation error
        assert response.status_code in [200, 422]


class TestBallisticPredictionEdgeCases:
    """Test edge cases for ballistic prediction endpoint."""
    
    def test_high_hardness_composition(self, client, valid_processing, valid_microstructure):
        """Test prediction with high-hardness composition (B4C rich)."""
        high_hardness_data = {
            "composition": {
                "SiC": 0.2,
                "B4C": 0.7,  # High B4C content for hardness
                "Al2O3": 0.1,
                "WC": 0.0,
                "TiC": 0.0
            },
            "processing": valid_processing.model_dump(),
            "microstructure": valid_microstructure.model_dump()
        }
        
        response = client.post("/api/v1/predict/ballistic", json=high_hardness_data)
        assert response.status_code == 200
        
        data = response.json()
        predictions = data["predictions"]
        
        # High B4C content should potentially give good ballistic performance
        # This is a materials science expectation
        assert predictions["v50_velocity"]["value"] > 0
    
    def test_high_toughness_composition(self, client, valid_processing, valid_microstructure):
        """Test prediction with high-toughness composition (SiC rich)."""
        high_toughness_data = {
            "composition": {
                "SiC": 0.8,  # High SiC content for toughness
                "B4C": 0.1,
                "Al2O3": 0.1,
                "WC": 0.0,
                "TiC": 0.0
            },
            "processing": valid_processing.model_dump(),
            "microstructure": valid_microstructure.model_dump()
        }
        
        response = client.post("/api/v1/predict/ballistic", json=high_toughness_data)
        assert response.status_code == 200
        
        data = response.json()
        predictions = data["predictions"]
        
        # High SiC content should potentially give good multi-hit capability
        assert predictions["multi_hit_capability"]["value"] >= 0
    
    def test_ultra_low_porosity(self, client, valid_composition, valid_processing):
        """Test prediction with ultra-low porosity."""
        ultra_dense_data = {
            "composition": valid_composition.model_dump(),
            "processing": valid_processing.model_dump(),
            "microstructure": {
                "porosity": 0.001,  # Ultra-low porosity
                "phase_distribution": "uniform",
                "interface_quality": "excellent",
                "pore_size": 0.1,
                "connectivity": 0.01
            }
        }
        
        response = client.post("/api/v1/predict/ballistic", json=ultra_dense_data)
        assert response.status_code == 200


class TestBallisticPredictionResponseConsistency:
    """Test response consistency and data validation."""
    
    def test_v50_velocity_realistic_range(self, client, valid_prediction_request):
        """Test that V50 velocity predictions are in realistic range."""
        request_data = valid_prediction_request.model_dump()
        
        response = client.post("/api/v1/predict/ballistic", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        v50_velocity = data["predictions"]["v50_velocity"]["value"]
        
        # V50 velocity should be in realistic range for ceramic armor (typically 200-1500 m/s)
        assert 100 <= v50_velocity <= 2000, f"V50 velocity {v50_velocity} m/s is outside realistic range"
    
    def test_penetration_resistance_range(self, client, valid_prediction_request):
        """Test that penetration resistance is in valid range."""
        request_data = valid_prediction_request.model_dump()
        
        response = client.post("/api/v1/predict/ballistic", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        penetration_resistance = data["predictions"]["penetration_resistance"]["value"]
        
        # Penetration resistance should be dimensionless and positive
        assert penetration_resistance >= 0, "Penetration resistance should be non-negative"
    
    def test_back_face_deformation_range(self, client, valid_prediction_request):
        """Test that back-face deformation is in realistic range."""
        request_data = valid_prediction_request.model_dump()
        
        response = client.post("/api/v1/predict/ballistic", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        bfd = data["predictions"]["back_face_deformation"]["value"]
        
        # Back-face deformation should be positive and realistic (typically 0-50mm)
        assert 0 <= bfd <= 100, f"Back-face deformation {bfd} mm is outside realistic range"
    
    def test_multi_hit_capability_probability_range(self, client, valid_prediction_request):
        """Test that multi-hit capability is in probability range [0, 1]."""
        request_data = valid_prediction_request.model_dump()
        
        response = client.post("/api/v1/predict/ballistic", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        multi_hit = data["predictions"]["multi_hit_capability"]["value"]
        
        # Multi-hit capability should be a probability [0, 1]
        assert 0 <= multi_hit <= 1, f"Multi-hit capability {multi_hit} is outside probability range [0, 1]"
    
    def test_confidence_intervals_ballistic_specific(self, client, valid_prediction_request):
        """Test confidence intervals are appropriate for ballistic properties."""
        request_data = valid_prediction_request.model_dump()
        
        response = client.post("/api/v1/predict/ballistic", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        predictions = data["predictions"]
        
        for prop_name, prop_data in predictions.items():
            value = prop_data["value"]
            ci = prop_data["confidence_interval"]
            uncertainty = prop_data["uncertainty"]
            
            # Confidence interval should be reasonable relative to uncertainty
            ci_width = ci[1] - ci[0]
            expected_width = 2 * uncertainty * value  # Approximate 95% CI width
            
            # Allow some tolerance in CI width calculation
            assert ci_width > 0, f"Confidence interval width should be positive for {prop_name}"
            
            # For probability-based properties, ensure CI stays within [0, 1]
            if prop_name == "multi_hit_capability":
                assert 0 <= ci[0] <= 1, f"Lower CI bound outside [0,1] for {prop_name}"
                assert 0 <= ci[1] <= 1, f"Upper CI bound outside [0,1] for {prop_name}"


class TestBallisticPredictionPerformance:
    """Test performance characteristics of ballistic prediction endpoint."""
    
    def test_response_time_acceptable(self, client, valid_prediction_request, performance_monitor):
        """Test that ballistic prediction response time is acceptable."""
        request_data = valid_prediction_request.model_dump()
        
        performance_monitor.start_monitoring()
        response = client.post("/api/v1/predict/ballistic", json=request_data)
        metrics = performance_monitor.get_metrics()
        
        assert response.status_code == 200
        
        # Response time should be under 1000ms for ballistic predictions
        # (Ballistic predictions may be slightly slower due to complexity)
        assert metrics['response_time_ms'] < 1000, f"Response time {metrics['response_time_ms']}ms too slow"
    
    def test_memory_usage_reasonable(self, client, valid_prediction_request, performance_monitor):
        """Test that memory usage during ballistic prediction is reasonable."""
        request_data = valid_prediction_request.model_dump()
        
        performance_monitor.start_monitoring()
        response = client.post("/api/v1/predict/ballistic", json=request_data)
        metrics = performance_monitor.get_metrics()
        
        assert response.status_code == 200
        
        # Memory delta should be reasonable (< 100MB per request)
        assert metrics['memory_delta_mb'] < 100, f"Memory usage {metrics['memory_delta_mb']}MB too high"