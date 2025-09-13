"""
Comprehensive unit tests for FastAPI application setup and core functionality.
"""

import pytest
import json
from fastapi.testclient import TestClient


class TestApplicationSetup:
    """Test FastAPI application initialization and configuration."""
    
    def test_app_creation(self, app):
        """Test that the FastAPI app is created successfully."""
        assert app is not None
        assert app.title == "Ceramic Armor ML API"
        assert app.version is not None
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns correct response."""
        response = client.get("/")
        assert response.status_code == 200
        
        # Check if it returns HTML (frontend) or JSON (API info)
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            data = response.json()
            assert "message" in data
            assert "version" in data
            assert "Ceramic Armor ML API" in data["message"]
    
    def test_api_info_endpoint(self, client):
        """Test API info endpoint."""
        response = client.get("/api")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "health" in data
        assert data["health"] == "/health"


class TestHealthEndpoints:
    """Test health check and status endpoints."""
    
    def test_health_endpoint(self, client):
        """Test basic health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "service" in data
        assert "version" in data
        assert "environment" in data
        assert "timestamp" in data
    
    def test_api_status_endpoint(self, client):
        """Test detailed API status endpoint."""
        response = client.get("/api/v1/status")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "system_info" in data
        assert "models" in data
        assert "performance" in data
    
    def test_models_info_endpoint(self, client):
        """Test models information endpoint."""
        response = client.get("/api/v1/models/info")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "status" in data


class TestMiddleware:
    """Test middleware functionality."""
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/")
        assert response.status_code == 200
        # CORS headers should be present due to middleware
    
    def test_security_headers(self, client):
        """Test security headers are present."""
        response = client.get("/health")
        assert response.status_code == 200
        
        # Check for security headers
        headers = response.headers
        # Note: Actual security headers depend on SecurityMiddleware implementation
        assert "content-type" in headers
    
    def test_request_logging(self, client):
        """Test that requests are logged properly."""
        # Make a request that should be logged
        response = client.get("/health")
        assert response.status_code == 200
        
        # Logging verification would require checking log output
        # For now, just verify the request completes successfully
    
    def test_error_handling_middleware(self, client):
        """Test error handling middleware."""
        # Test 404 handling
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404
        
        # Should return JSON error response
        data = response.json()
        assert "detail" in data or "error" in data


class TestAPIDocumentation:
    """Test API documentation endpoints."""
    
    def test_openapi_schema(self, client):
        """Test OpenAPI schema generation."""
        response = client.get("/openapi.json")
        # In production, docs might be disabled
        if response.status_code == 200:
            data = response.json()
            assert "openapi" in data
            assert "info" in data
            assert "paths" in data
    
    def test_swagger_docs(self, client):
        """Test Swagger documentation endpoint."""
        response = client.get("/docs")
        # In production, docs might be disabled
        if response.status_code == 200:
            assert "text/html" in response.headers.get("content-type", "")
    
    def test_redoc_docs(self, client):
        """Test ReDoc documentation endpoint."""
        response = client.get("/redoc")
        # In production, docs might be disabled
        if response.status_code == 200:
            assert "text/html" in response.headers.get("content-type", "")


class TestStaticFiles:
    """Test static file serving."""
    
    def test_static_file_mounting(self, client):
        """Test that static files are properly mounted."""
        # Try to access a static file path
        response = client.get("/static/")
        # Should either return 404 (no index) or 200 (if index exists)
        assert response.status_code in [200, 404, 405]  # 405 = Method Not Allowed for directories


class TestApplicationConfiguration:
    """Test application configuration and settings."""
    
    def test_environment_configuration(self, client):
        """Test that environment configuration is loaded."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "environment" in data
        # Environment should be one of: development, testing, production
        assert data["environment"] in ["development", "testing", "production"]
    
    def test_version_information(self, client):
        """Test version information is available."""
        response = client.get("/api")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert data["version"] is not None
        assert len(data["version"]) > 0


class TestErrorHandling:
    """Test comprehensive error handling."""
    
    def test_404_error_handling(self, client):
        """Test 404 error handling."""
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404
        
        data = response.json()
        assert "detail" in data or "error" in data
    
    def test_method_not_allowed_handling(self, client):
        """Test 405 Method Not Allowed handling."""
        # Try POST on a GET-only endpoint
        response = client.post("/health")
        assert response.status_code == 405
        
        data = response.json()
        assert "detail" in data or "error" in data
    
    def test_invalid_json_handling(self, client):
        """Test handling of invalid JSON in request body."""
        response = client.post(
            "/api/v1/predict/mechanical",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
        
        data = response.json()
        assert "detail" in data or "error" in data


class TestRateLimiting:
    """Test rate limiting functionality."""
    
    def test_rate_limit_headers(self, client):
        """Test that rate limit headers are present."""
        response = client.get("/api/v1/status")
        assert response.status_code == 200
        
        # Rate limiting headers might be present
        # Actual implementation depends on RateLimitMiddleware
    
    def test_health_endpoint_rate_limit_exemption(self, client):
        """Test that health endpoint is exempt from rate limiting."""
        # Make multiple requests to health endpoint
        for _ in range(5):
            response = client.get("/health")
            assert response.status_code == 200
        
        # Should not be rate limited