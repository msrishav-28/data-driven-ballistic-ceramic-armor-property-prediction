"""
Integration tests for complete prediction workflows.

These tests verify end-to-end functionality including:
- Complete prediction workflows from request to response
- Cross-endpoint consistency
- Feature extraction and ML pipeline integration
- Error handling across the full stack
"""

import pytest
import json
import time
from typing import Dict, Any, List
from fastapi.testclient import TestClient


class TestCompletePredictionWorkflow:
    """Test complete prediction workflows from request to response."""
    
    def test_mechanical_prediction_complete_workflow(self, client, valid_prediction_request):
        """Test complete mechanical prediction workflow."""
        request_data = valid_prediction_request.model_dump()
        
        # Step 1: Make prediction request
        response = client.post("/api/v1/predict/mechanical", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        
        # Step 2: Verify complete response structure
        assert "status" in data
        assert "predictions" in data
        assert "model_info" in data
        assert "request_id" in data
        assert "timestamp" in data
        
        # Step 3: Verify all mechanical properties are predicted
        predictions = data["predictions"]
        mechanical_properties = ["fracture_toughness", "vickers_hardness", "density", "elastic_modulus"]
        
        for prop in mechanical_properties:
            assert prop in predictions
            prop_data = predictions[prop]
            
            # Verify complete property data
            assert "value" in prop_data
            assert "unit" in prop_data
            assert "confidence_interval" in prop_data
            assert "uncertainty" in prop_data
            assert "prediction_quality" in prop_data
            
            # Verify data consistency
            assert isinstance(prop_data["value"], (int, float))
            assert prop_data["value"] > 0  # Physical properties should be positive
            assert len(prop_data["confidence_interval"]) == 2
            assert prop_data["confidence_interval"][0] < prop_data["confidence_interval"][1]
        
        # Step 4: Verify model information completeness
        model_info = data["model_info"]
        required_model_fields = [
            "model_version", "model_type", "training_r2", "validation_r2",
            "prediction_time_ms", "feature_count", "training_samples"
        ]
        
        for field in required_model_fields:
            assert field in model_info
        
        # Step 5: Verify request tracking
        assert data["request_id"].startswith("req_")
        assert len(data["request_id"]) > 10  # Should be sufficiently unique
    
    def test_ballistic_prediction_complete_workflow(self, client, valid_prediction_request):
        """Test complete ballistic prediction workflow."""
        request_data = valid_prediction_request.model_dump()
        
        # Step 1: Make ballistic prediction request
        response = client.post("/api/v1/predict/ballistic", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        
        # Step 2: Verify complete response structure
        assert "status" in data
        assert "predictions" in data
        assert "model_info" in data
        assert "request_id" in data
        assert "timestamp" in data
        
        # Step 3: Verify all ballistic properties are predicted
        predictions = data["predictions"]
        ballistic_properties = ["v50_velocity", "penetration_resistance", "back_face_deformation", "multi_hit_capability"]
        
        for prop in ballistic_properties:
            assert prop in predictions
            prop_data = predictions[prop]
            
            # Verify complete property data
            assert "value" in prop_data
            assert "unit" in prop_data
            assert "confidence_interval" in prop_data
            assert "uncertainty" in prop_data
            assert "prediction_quality" in prop_data
            
            # Verify data consistency
            assert isinstance(prop_data["value"], (int, float))
            assert prop_data["value"] >= 0  # Ballistic properties should be non-negative
        
        # Step 4: Verify ballistic-specific processing notes
        if data.get("processing_notes"):
            assert isinstance(data["processing_notes"], list)
            # Should contain ballistic-specific information
            notes_text = " ".join(data["processing_notes"])
            assert any(keyword in notes_text.lower() for keyword in ["ballistic", "impact", "dynamics"])
    
    def test_feature_importance_workflow(self, client, valid_prediction_request):
        """Test feature importance analysis workflow."""
        request_data = valid_prediction_request.model_dump()
        request_data["include_feature_importance"] = True
        
        # Test both mechanical and ballistic predictions
        for endpoint in ["/api/v1/predict/mechanical", "/api/v1/predict/ballistic"]:
            response = client.post(endpoint, json=request_data)
            assert response.status_code == 200
            
            data = response.json()
            
            # Should include feature importance when requested
            if data.get("feature_importance"):
                feature_importance = data["feature_importance"]
                assert isinstance(feature_importance, list)
                assert len(feature_importance) > 0
                
                # Verify feature importance structure
                for feature in feature_importance:
                    assert "name" in feature
                    assert "importance" in feature
                    assert "category" in feature
                    
                    # Importance should be between 0 and 1
                    assert 0 <= feature["importance"] <= 1
                    
                    # Category should be valid
                    assert feature["category"] in ["composition", "processing", "microstructure", "derived"]
    
    def test_uncertainty_quantification_workflow(self, client, valid_prediction_request):
        """Test uncertainty quantification workflow."""
        request_data = valid_prediction_request.model_dump()
        
        # Test with uncertainty enabled
        request_data["include_uncertainty"] = True
        response = client.post("/api/v1/predict/mechanical", json=request_data)
        assert response.status_code == 200
        
        data_with_uncertainty = response.json()
        predictions_with_uncertainty = data_with_uncertainty["predictions"]
        
        # Test without uncertainty
        request_data["include_uncertainty"] = False
        response = client.post("/api/v1/predict/mechanical", json=request_data)
        assert response.status_code == 200
        
        data_without_uncertainty = response.json()
        predictions_without_uncertainty = data_without_uncertainty["predictions"]
        
        # Compare uncertainty information
        for prop in predictions_with_uncertainty:
            with_unc = predictions_with_uncertainty[prop]
            without_unc = predictions_without_uncertainty[prop]
            
            # Both should have uncertainty fields, but values may differ
            assert "uncertainty" in with_unc
            assert "uncertainty" in without_unc
            assert "confidence_interval" in with_unc
            assert "confidence_interval" in without_unc


class TestCrossEndpointConsistency:
    """Test consistency between different endpoints and prediction types."""
    
    def test_mechanical_vs_ballistic_consistency(self, client, valid_prediction_request):
        """Test consistency between mechanical and ballistic predictions."""
        request_data = valid_prediction_request.model_dump()
        
        # Get mechanical predictions
        mech_response = client.post("/api/v1/predict/mechanical", json=request_data)
        assert mech_response.status_code == 200
        mech_data = mech_response.json()
        
        # Get ballistic predictions
        ball_response = client.post("/api/v1/predict/ballistic", json=request_data)
        assert ball_response.status_code == 200
        ball_data = ball_response.json()
        
        # Both should have same request structure validation
        assert mech_data["status"] == ball_data["status"]
        
        # Both should have valid request IDs (different)
        assert mech_data["request_id"] != ball_data["request_id"]
        
        # Both should have model info with appropriate fields
        assert "model_info" in mech_data
        assert "model_info" in ball_data
        
        # Model types should be different but both valid
        mech_model_type = mech_data["model_info"]["model_type"]
        ball_model_type = ball_data["model_info"]["model_type"]
        
        assert mech_model_type != ball_model_type or "ballistic" in ball_model_type.lower()
    
    def test_same_material_different_requests(self, client, valid_prediction_request):
        """Test that same material gives consistent results across requests."""
        request_data = valid_prediction_request.model_dump()
        
        # Make multiple requests with same data
        responses = []
        for _ in range(3):
            response = client.post("/api/v1/predict/mechanical", json=request_data)
            assert response.status_code == 200
            responses.append(response.json())
        
        # Predictions should be consistent (same values)
        base_predictions = responses[0]["predictions"]
        
        for i in range(1, len(responses)):
            current_predictions = responses[i]["predictions"]
            
            for prop in base_predictions:
                base_value = base_predictions[prop]["value"]
                current_value = current_predictions[prop]["value"]
                
                # Values should be identical for same input
                assert abs(base_value - current_value) < 1e-6, f"Inconsistent predictions for {prop}"
    
    def test_health_status_consistency(self, client):
        """Test consistency between health and status endpoints."""
        # Get basic health
        health_response = client.get("/health")
        assert health_response.status_code == 200
        health_data = health_response.json()
        
        # Get detailed status
        status_response = client.get("/api/v1/status")
        assert status_response.status_code == 200
        status_data = status_response.json()
        
        # Both should report healthy status
        assert health_data["status"] == "healthy"
        assert status_data["status"] == "healthy"
        
        # Version should be consistent
        if "version" in health_data and "version" in status_data:
            assert health_data["version"] == status_data["version"]


class TestErrorHandlingIntegration:
    """Test error handling across the full application stack."""
    
    def test_validation_error_propagation(self, client):
        """Test that validation errors are properly propagated through the stack."""
        invalid_data = {
            "composition": {
                "SiC": 1.5,  # Invalid: > 1.0
                "B4C": 0.3,
                "Al2O3": 0.2
            },
            "processing": {
                "sintering_temperature": 1800,
                "pressure": 50,
                "grain_size": 10
            },
            "microstructure": {
                "porosity": 0.02,
                "phase_distribution": "uniform"
            }
        }
        
        response = client.post("/api/v1/predict/mechanical", json=invalid_data)
        assert response.status_code == 422
        
        error_data = response.json()
        assert "detail" in error_data
        
        # Error should be informative
        error_detail = str(error_data["detail"])
        assert len(error_detail) > 0
    
    def test_ml_prediction_error_handling(self, client, valid_prediction_request):
        """Test handling of ML prediction errors."""
        # This test would require mocking ML prediction failures
        # For now, test that the endpoint handles requests gracefully
        request_data = valid_prediction_request.model_dump()
        
        response = client.post("/api/v1/predict/mechanical", json=request_data)
        
        # Should either succeed or fail gracefully with proper error response
        if response.status_code != 200:
            assert response.status_code in [422, 500, 503]
            error_data = response.json()
            assert "detail" in error_data or "error" in error_data
    
    def test_malformed_request_handling(self, client):
        """Test handling of malformed requests."""
        # Test completely invalid JSON
        response = client.post(
            "/api/v1/predict/mechanical",
            data="not json at all",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
        
        # Test missing content type
        response = client.post("/api/v1/predict/mechanical", data="{}")
        assert response.status_code in [422, 415]  # Unprocessable Entity or Unsupported Media Type
    
    def test_large_request_handling(self, client, valid_prediction_request):
        """Test handling of unusually large requests."""
        request_data = valid_prediction_request.model_dump()
        
        # Add large amount of extra data
        request_data["extra_large_field"] = "x" * 10000  # 10KB of extra data
        
        response = client.post("/api/v1/predict/mechanical", json=request_data)
        
        # Should either succeed (ignoring extra field) or fail gracefully
        assert response.status_code in [200, 422]


class TestPerformanceIntegration:
    """Test performance characteristics of integrated workflows."""
    
    def test_concurrent_requests_handling(self, client, valid_prediction_request):
        """Test handling of concurrent requests."""
        import threading
        import queue
        
        request_data = valid_prediction_request.model_dump()
        results = queue.Queue()
        
        def make_request():
            try:
                response = client.post("/api/v1/predict/mechanical", json=request_data)
                results.put(("success", response.status_code, response.json()))
            except Exception as e:
                results.put(("error", str(e), None))
        
        # Start multiple concurrent requests
        threads = []
        num_concurrent = 5
        
        for _ in range(num_concurrent):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all requests to complete
        for thread in threads:
            thread.join(timeout=10)  # 10 second timeout
        
        # Collect results
        successful_requests = 0
        while not results.empty():
            result_type, status_or_error, data = results.get()
            if result_type == "success" and status_or_error == 200:
                successful_requests += 1
        
        # At least some requests should succeed
        assert successful_requests > 0, "No concurrent requests succeeded"
        
        # Ideally, all should succeed
        assert successful_requests >= num_concurrent * 0.8, "Too many concurrent requests failed"
    
    def test_memory_usage_stability(self, client, valid_prediction_request, performance_monitor):
        """Test memory usage stability across multiple requests."""
        request_data = valid_prediction_request.model_dump()
        
        # Make multiple requests and monitor memory
        memory_measurements = []
        
        for i in range(10):
            performance_monitor.start_monitoring()
            response = client.post("/api/v1/predict/mechanical", json=request_data)
            metrics = performance_monitor.get_metrics()
            
            assert response.status_code == 200
            memory_measurements.append(metrics['memory_usage_mb'])
        
        # Memory usage should be relatively stable (not growing significantly)
        max_memory = max(memory_measurements)
        min_memory = min(memory_measurements)
        memory_growth = max_memory - min_memory
        
        # Memory growth should be reasonable (< 100MB across 10 requests)
        assert memory_growth < 100, f"Memory growth {memory_growth}MB too high across requests"
    
    def test_response_time_consistency(self, client, valid_prediction_request, performance_monitor):
        """Test response time consistency across multiple requests."""
        request_data = valid_prediction_request.model_dump()
        
        response_times = []
        
        for i in range(5):
            performance_monitor.start_monitoring()
            response = client.post("/api/v1/predict/mechanical", json=request_data)
            metrics = performance_monitor.get_metrics()
            
            assert response.status_code == 200
            response_times.append(metrics['response_time_ms'])
        
        # Response times should be consistent
        avg_time = sum(response_times) / len(response_times)
        max_time = max(response_times)
        
        # No single request should be more than 3x the average
        assert max_time < avg_time * 3, f"Response time inconsistency: max {max_time}ms vs avg {avg_time}ms"


class TestDataFlowIntegration:
    """Test data flow through the complete application stack."""
    
    def test_feature_extraction_integration(self, client, valid_prediction_request):
        """Test that feature extraction integrates properly with predictions."""
        request_data = valid_prediction_request.model_dump()
        
        response = client.post("/api/v1/predict/mechanical", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        model_info = data["model_info"]
        
        # Should report feature count from actual feature extraction
        assert "feature_count" in model_info
        assert model_info["feature_count"] > 0
        
        # Feature count should be reasonable (not too low or too high)
        feature_count = model_info["feature_count"]
        assert 10 <= feature_count <= 1000, f"Feature count {feature_count} seems unrealistic"
    
    def test_composition_processing_integration(self, client):
        """Test integration between composition and processing parameter validation."""
        # Test SiC-rich composition with appropriate processing
        sic_rich_data = {
            "composition": {
                "SiC": 0.8,
                "B4C": 0.1,
                "Al2O3": 0.1,
                "WC": 0.0,
                "TiC": 0.0
            },
            "processing": {
                "sintering_temperature": 2000,  # High temp for SiC
                "pressure": 80,
                "grain_size": 5,
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
            }
        }
        
        response = client.post("/api/v1/predict/mechanical", json=sic_rich_data)
        assert response.status_code == 200
        
        # Should produce reasonable predictions for SiC-rich material
        data = response.json()
        predictions = data["predictions"]
        
        # SiC-rich materials typically have high fracture toughness
        fracture_toughness = predictions["fracture_toughness"]["value"]
        assert fracture_toughness > 0, "Fracture toughness should be positive"
    
    def test_microstructure_property_correlation(self, client, valid_composition, valid_processing):
        """Test correlation between microstructure and predicted properties."""
        # Test low porosity material
        low_porosity_data = {
            "composition": valid_composition.model_dump(),
            "processing": valid_processing.model_dump(),
            "microstructure": {
                "porosity": 0.005,  # Very low porosity
                "phase_distribution": "uniform",
                "interface_quality": "excellent",
                "pore_size": 0.2,
                "connectivity": 0.01
            }
        }
        
        # Test high porosity material
        high_porosity_data = {
            "composition": valid_composition.model_dump(),
            "processing": valid_processing.model_dump(),
            "microstructure": {
                "porosity": 0.1,  # Higher porosity
                "phase_distribution": "uniform",
                "interface_quality": "good",
                "pore_size": 2.0,
                "connectivity": 0.2
            }
        }
        
        # Get predictions for both
        low_por_response = client.post("/api/v1/predict/mechanical", json=low_porosity_data)
        high_por_response = client.post("/api/v1/predict/mechanical", json=high_porosity_data)
        
        assert low_por_response.status_code == 200
        assert high_por_response.status_code == 200
        
        low_por_data = low_por_response.json()
        high_por_data = high_por_response.json()
        
        # Low porosity should generally give higher density
        low_density = low_por_data["predictions"]["density"]["value"]
        high_density = high_por_data["predictions"]["density"]["value"]
        
        # This is a materials science expectation
        assert low_density >= high_density, "Lower porosity should give higher or equal density"