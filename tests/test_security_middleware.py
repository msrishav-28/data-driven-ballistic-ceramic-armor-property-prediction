"""
Tests for security middleware functionality.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch

from src.api.middleware.security import SecurityMiddleware, CORSSecurityMiddleware
from src.config import Settings


@pytest.fixture
def app_with_security():
    """Create FastAPI app with security middleware for testing."""
    app = FastAPI()
    app.add_middleware(SecurityMiddleware)
    
    @app.get("/test")
    async def test_endpoint():
        return {"message": "test"}
    
    @app.post("/test")
    async def test_post_endpoint(data: dict):
        return {"received": data}
    
    return app


@pytest.fixture
def client(app_with_security):
    """Create test client."""
    return TestClient(app_with_security)


class TestSecurityMiddleware:
    """Test security middleware functionality."""
    
    def test_security_headers_added(self, client):
        """Test that security headers are added to responses."""
        response = client.get("/test")
        
        assert response.status_code == 200
        
        # Check for security headers
        assert "X-Content-Type-Options" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        
        assert "X-Frame-Options" in response.headers
        assert response.headers["X-Frame-Options"] == "DENY"
        
        assert "X-XSS-Protection" in response.headers
        assert response.headers["X-XSS-Protection"] == "1; mode=block"
        
        assert "Content-Security-Policy" in response.headers
        assert "Referrer-Policy" in response.headers
        assert "Cache-Control" in response.headers
    
    def test_sql_injection_detection(self, client):
        """Test SQL injection pattern detection."""
        # Test SQL injection in query parameters
        response = client.get("/test?param='; DROP TABLE users; --")
        assert response.status_code == 400
        assert "Invalid input" in response.json()["detail"]["error"]
        
        # Test SQL injection in different formats
        response = client.get("/test?param=1 OR 1=1")
        assert response.status_code == 400
        
        response = client.get("/test?param=UNION SELECT * FROM users")
        assert response.status_code == 400
    
    def test_xss_detection(self, client):
        """Test XSS pattern detection."""
        # Test script tag injection
        response = client.get("/test?param=<script>alert('xss')</script>")
        assert response.status_code == 400
        assert "Invalid input" in response.json()["detail"]["error"]
        
        # Test javascript: protocol
        response = client.get("/test?param=javascript:alert('xss')")
        assert response.status_code == 400
        
        # Test event handler injection
        response = client.get("/test?param=<img onerror='alert(1)' src='x'>")
        assert response.status_code == 400
    
    def test_path_traversal_detection(self, client):
        """Test path traversal detection."""
        # Test directory traversal
        response = client.get("/../../../etc/passwd")
        assert response.status_code == 400
        
        # Test URL encoded traversal
        response = client.get("/%2e%2e%2f%2e%2e%2fpasswd")
        assert response.status_code == 400
    
    def test_input_length_limits(self, client):
        """Test input length validation."""
        # Create very long query parameter
        long_param = "a" * 2000  # Exceeds default limit of 1000
        response = client.get(f"/test?param={long_param}")
        assert response.status_code == 400
        assert "Input too long" in response.json()["detail"]["message"]
    
    def test_valid_requests_pass(self, client):
        """Test that valid requests pass through successfully."""
        # Normal query parameters
        response = client.get("/test?param=normal_value")
        assert response.status_code == 200
        
        # Normal POST request
        response = client.post("/test", json={"key": "value"})
        assert response.status_code == 200
        
        # Alphanumeric with special chars (safe)
        response = client.get("/test?param=test-value_123")
        assert response.status_code == 200


class TestRateLimitMiddleware:
    """Test enhanced rate limiting functionality."""
    
    @patch('src.config.get_settings')
    def test_endpoint_specific_limits(self, mock_settings, client):
        """Test that different endpoints have different rate limits."""
        # Mock settings for testing
        mock_settings.return_value = Settings(
            rate_limit_requests=5,
            rate_limit_window=60,
            trusted_ips=[]
        )
        
        # This would require setting up the rate limit middleware
        # and testing different endpoint types
        pass
    
    def test_rate_limit_headers(self, client):
        """Test that rate limit headers are included in responses."""
        response = client.get("/test")
        
        # Check for rate limit headers
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers


class TestCORSSecurityMiddleware:
    """Test enhanced CORS middleware."""
    
    def test_cors_preflight_handling(self):
        """Test CORS preflight request handling."""
        app = FastAPI()
        app.add_middleware(CORSSecurityMiddleware)
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}
        
        client = TestClient(app)
        
        # Test preflight request
        response = client.options(
            "/test",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET"
            }
        )
        
        # Should handle preflight appropriately
        assert response.status_code in [200, 204]
    
    def test_origin_validation(self):
        """Test origin validation logic."""
        app = FastAPI()
        app.add_middleware(CORSSecurityMiddleware)
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}
        
        client = TestClient(app)
        
        # Test with allowed origin
        response = client.get(
            "/test",
            headers={"Origin": "http://localhost:3000"}
        )
        assert response.status_code == 200
        
        # Test with disallowed origin (in production)
        with patch('src.config.get_settings') as mock_settings:
            mock_settings.return_value.environment = "production"
            mock_settings.return_value.cors_origins = ["https://allowed-domain.com"]
            
            response = client.get(
                "/test",
                headers={"Origin": "https://malicious-domain.com"}
            )
            # Should still process request but not add CORS headers


if __name__ == "__main__":
    pytest.main([__file__])