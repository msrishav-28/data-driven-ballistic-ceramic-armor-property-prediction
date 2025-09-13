"""
Tests for comprehensive logging and monitoring middleware.
"""

import json
import logging
import pytest
from unittest.mock import Mock, patch
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.api.middleware import (
    LoggingMiddleware,
    MonitoringMiddleware,
    ErrorHandlingMiddleware,
    StructuredLogger,
    MLPredictionError,
    ExternalAPIError
)
from src.api.services.logging_service import (
    PredictionLoggingService,
    ExternalAPILoggingService
)


@pytest.fixture
def app():
    """Create test FastAPI app with logging middleware."""
    app = FastAPI()
    
    # Add middleware in correct order
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(MonitoringMiddleware)
    
    @app.get("/test")
    async def test_endpoint():
        return {"message": "test"}
    
    @app.get("/error")
    async def error_endpoint():
        raise Exception("Test error")
    
    @app.get("/ml-error")
    async def ml_error_endpoint():
        raise MLPredictionError("Test ML error", model_name="test_model")
    
    @app.get("/api-error")
    async def api_error_endpoint():
        raise ExternalAPIError("Test API error", api_name="test_api")
    
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


class TestLoggingMiddleware:
    """Test logging middleware functionality."""
    
    def test_request_logging(self, client, caplog):
        """Test that requests are logged properly."""
        with caplog.at_level(logging.INFO):
            response = client.get("/test")
        
        assert response.status_code == 200
        
        # Check that request and response were logged
        log_messages = [record.message for record in caplog.records]
        
        # Should have structured JSON logs
        json_logs = []
        for message in log_messages:
            try:
                log_data = json.loads(message)
                json_logs.append(log_data)
            except json.JSONDecodeError:
                continue
        
        # Find request and response logs
        request_logs = [log for log in json_logs if log.get('event') == 'http_request']
        response_logs = [log for log in json_logs if log.get('event') == 'http_response']
        
        assert len(request_logs) >= 1
        assert len(response_logs) >= 1
        
        # Verify request log structure
        request_log = request_logs[0]
        assert request_log['method'] == 'GET'
        assert request_log['path'] == '/test'
        assert 'request_id' in request_log
        assert 'client_ip' in request_log
        
        # Verify response log structure
        response_log = response_logs[0]
        assert response_log['status_code'] == 200
        assert response_log['method'] == 'GET'
        assert response_log['path'] == '/test'
        assert 'process_time_ms' in response_log
    
    def test_error_logging(self, client, caplog):
        """Test that errors are logged with proper context."""
        with caplog.at_level(logging.ERROR):
            response = client.get("/error")
        
        assert response.status_code == 500
        
        # Check error logs
        error_records = [record for record in caplog.records if record.levelno >= logging.ERROR]
        assert len(error_records) >= 1
        
        # Check for structured error log
        json_logs = []
        for record in error_records:
            try:
                log_data = json.loads(record.message)
                if log_data.get('event') == 'unhandled_exception':
                    json_logs.append(log_data)
            except json.JSONDecodeError:
                continue
        
        assert len(json_logs) >= 1
        error_log = json_logs[0]
        
        assert error_log['error_type'] == 'Exception'
        assert 'Test error' in error_log['error_message']
        assert 'stack_trace' in error_log
        assert 'request_id' in error_log
    
    def test_ml_error_handling(self, client, caplog):
        """Test ML prediction error handling."""
        with caplog.at_level(logging.ERROR):
            response = client.get("/ml-error")
        
        assert response.status_code == 500
        
        # Check response format
        response_data = response.json()
        assert response_data['error'] == 'Prediction Error'
        assert 'request_id' in response_data
    
    def test_external_api_error_handling(self, client, caplog):
        """Test external API error handling."""
        with caplog.at_level(logging.ERROR):
            response = client.get("/api-error")
        
        assert response.status_code == 503
        
        # Check response format
        response_data = response.json()
        assert response_data['error'] == 'External API Error'
        assert 'test_api' in response_data['message']
        assert 'request_id' in response_data


class TestStructuredLogger:
    """Test structured logger functionality."""
    
    def test_structured_logging(self, caplog):
        """Test structured log format."""
        logger = StructuredLogger("test")
        
        with caplog.at_level(logging.INFO):
            logger.log_structured("INFO", "test_event", key1="value1", key2=42)
        
        # Parse the JSON log
        log_record = caplog.records[0]
        log_data = json.loads(log_record.message)
        
        assert log_data['event'] == 'test_event'
        assert log_data['key1'] == 'value1'
        assert log_data['key2'] == 42
        assert 'timestamp' in log_data
        assert 'environment' in log_data
    
    def test_prediction_logging(self, caplog):
        """Test ML prediction logging."""
        logger = StructuredLogger("test")
        
        with caplog.at_level(logging.INFO):
            logger.log_prediction(
                request_id="test-123",
                prediction_type="mechanical",
                model_name="test_model",
                process_time=0.5,
                feature_count=50
            )
        
        log_record = caplog.records[0]
        log_data = json.loads(log_record.message)
        
        assert log_data['event'] == 'ml_prediction'
        assert log_data['request_id'] == 'test-123'
        assert log_data['prediction_type'] == 'mechanical'
        assert log_data['model_name'] == 'test_model'
        assert log_data['process_time_ms'] == 500.0
        assert log_data['feature_count'] == 50


class TestPredictionLoggingService:
    """Test prediction logging service."""
    
    @pytest.mark.asyncio
    async def test_prediction_operation_context(self, caplog):
        """Test prediction operation context manager."""
        service = PredictionLoggingService()
        
        # Mock request with state
        request = Mock()
        request.state = Mock()
        request.state.request_id = "test-456"
        
        input_data = {"composition": {"SiC": 0.6}, "processing": {"temp": 1800}}
        
        with caplog.at_level(logging.INFO):
            async with service.log_prediction_operation(request, "mechanical", input_data) as request_id:
                assert request_id == "test-456"
        
        # Check that prediction start was logged
        json_logs = []
        for record in caplog.records:
            try:
                log_data = json.loads(record.message)
                json_logs.append(log_data)
            except json.JSONDecodeError:
                continue
        
        start_logs = [log for log in json_logs if log.get('event') == 'prediction_start']
        assert len(start_logs) == 1
        
        start_log = start_logs[0]
        assert start_log['request_id'] == 'test-456'
        assert start_log['prediction_type'] == 'mechanical'
        assert start_log['input_features'] == 2
    
    @pytest.mark.asyncio
    async def test_prediction_error_handling(self, caplog):
        """Test prediction error handling in context manager."""
        service = PredictionLoggingService()
        
        request = Mock()
        request.state = Mock()
        request.state.request_id = "test-789"
        
        input_data = {"test": "data"}
        
        with caplog.at_level(logging.ERROR):
            with pytest.raises(MLPredictionError):
                async with service.log_prediction_operation(request, "ballistic", input_data):
                    raise ValueError("Test prediction error")
        
        # Check error logging
        error_records = [record for record in caplog.records if record.levelno >= logging.ERROR]
        assert len(error_records) >= 1


class TestMonitoringMiddleware:
    """Test monitoring middleware functionality."""
    
    def test_request_counting(self, client):
        """Test that requests are counted properly."""
        from src.api.middleware.monitoring import system_monitor
        
        initial_count = system_monitor.request_count
        
        # Make several requests
        for _ in range(3):
            client.get("/test")
        
        assert system_monitor.request_count >= initial_count + 3
    
    def test_error_counting(self, client):
        """Test that errors are counted properly."""
        from src.api.middleware.monitoring import system_monitor
        
        initial_error_count = system_monitor.error_count
        
        # Make error request
        client.get("/error")
        
        assert system_monitor.error_count >= initial_error_count + 1
    
    def test_system_info_collection(self):
        """Test system information collection."""
        from src.api.middleware.monitoring import system_monitor
        
        system_info = system_monitor.get_system_info()
        
        assert 'uptime_seconds' in system_info
        assert 'request_count' in system_info
        assert 'error_count' in system_info
        assert 'cpu_percent' in system_info
        assert 'memory_percent' in system_info
        assert 'timestamp' in system_info
        
        # Verify data types
        assert isinstance(system_info['uptime_seconds'], (int, float))
        assert isinstance(system_info['request_count'], int)
        assert isinstance(system_info['error_count'], int)


if __name__ == "__main__":
    pytest.main([__file__])