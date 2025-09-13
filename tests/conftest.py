"""
Pytest configuration and shared fixtures for comprehensive testing.

This module provides shared fixtures and configuration for all test modules,
including test client setup, mock data, and performance monitoring utilities.
"""

import pytest
import asyncio
import time
import psutil
import os
from typing import Dict, Any, Generator
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.models.request_models import (
    CompositionModel, ProcessingModel, MicrostructureModel, PredictionRequest
)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def app():
    """Create FastAPI application for testing."""
    return create_app()


@pytest.fixture(scope="function")
def client(app):
    """Create test client for each test function."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def mock_ml_predictor():
    """Mock ML predictor for testing without actual model loading."""
    mock_predictor = Mock()
    
    # Mock mechanical predictions
    mock_predictor.predict_mechanical_properties.return_value = {
        'predictions': {
            'fracture_toughness': {
                'value': 4.6,
                'unit': 'MPa·m^0.5',
                'confidence_interval': [4.2, 5.0],
                'uncertainty': 0.15
            },
            'vickers_hardness': {
                'value': 2800,
                'unit': 'HV',
                'confidence_interval': [2650, 2950],
                'uncertainty': 0.12
            },
            'density': {
                'value': 3.21,
                'unit': 'g/cm³',
                'confidence_interval': [3.18, 3.24],
                'uncertainty': 0.08
            },
            'elastic_modulus': {
                'value': 410,
                'unit': 'GPa',
                'confidence_interval': [395, 425],
                'uncertainty': 0.10
            }
        },
        'feature_importance': [
            {'name': 'SiC_content', 'importance': 0.35, 'shap_value': 0.12},
            {'name': 'sintering_temperature', 'importance': 0.28, 'shap_value': 0.09},
            {'name': 'grain_size', 'importance': 0.22, 'shap_value': 0.07}
        ],
        'processing_time_ms': 45
    }
    
    # Mock ballistic predictions
    mock_predictor.predict_ballistic_properties.return_value = {
        'predictions': {
            'v50_velocity': {
                'value': 850,
                'unit': 'm/s',
                'confidence_interval': [780, 920],
                'uncertainty': 0.18
            },
            'penetration_resistance': {
                'value': 0.85,
                'unit': 'dimensionless',
                'confidence_interval': [0.75, 0.95],
                'uncertainty': 0.12
            },
            'back_face_deformation': {
                'value': 12.5,
                'unit': 'mm',
                'confidence_interval': [10.2, 14.8],
                'uncertainty': 0.20
            },
            'multi_hit_capability': {
                'value': 0.72,
                'unit': 'probability',
                'confidence_interval': [0.65, 0.79],
                'uncertainty': 0.10
            }
        },
        'feature_importance': [
            {'name': 'B4C_content', 'importance': 0.42, 'shap_value': 0.15},
            {'name': 'density', 'importance': 0.31, 'shap_value': 0.11},
            {'name': 'porosity', 'importance': 0.27, 'shap_value': -0.08}
        ],
        'processing_time_ms': 52
    }
    
    return mock_predictor


@pytest.fixture
def valid_composition():
    """Valid material composition for testing."""
    return CompositionModel(
        SiC=0.6,
        B4C=0.3,
        Al2O3=0.1,
        WC=0.0,
        TiC=0.0
    )


@pytest.fixture
def valid_processing():
    """Valid processing parameters for testing."""
    return ProcessingModel(
        sintering_temperature=1800,
        pressure=50,
        grain_size=10,
        holding_time=120,
        heating_rate=15,
        atmosphere="argon"
    )


@pytest.fixture
def valid_microstructure():
    """Valid microstructure parameters for testing."""
    return MicrostructureModel(
        porosity=0.02,
        phase_distribution="uniform",
        interface_quality="good",
        pore_size=1.0,
        connectivity=0.1
    )


@pytest.fixture
def valid_prediction_request(valid_composition, valid_processing, valid_microstructure):
    """Valid complete prediction request for testing."""
    return PredictionRequest(
        composition=valid_composition,
        processing=valid_processing,
        microstructure=valid_microstructure,
        include_uncertainty=True,
        include_feature_importance=True
    )


@pytest.fixture
def invalid_composition_data():
    """Invalid composition data for validation testing."""
    return {
        "SiC": 0.8,
        "B4C": 0.5,  # Total > 1.0
        "Al2O3": 0.1
    }


@pytest.fixture
def invalid_processing_data():
    """Invalid processing data for validation testing."""
    return {
        "sintering_temperature": 1000,  # Too low
        "pressure": 300,  # Too high
        "grain_size": -5  # Negative
    }


@pytest.fixture
def performance_monitor():
    """Performance monitoring fixture for response time and memory usage tests."""
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.start_memory = None
            self.process = psutil.Process(os.getpid())
        
        def start_monitoring(self):
            """Start performance monitoring."""
            self.start_time = time.time()
            self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        def get_metrics(self) -> Dict[str, float]:
            """Get performance metrics."""
            if self.start_time is None:
                raise ValueError("Monitoring not started")
            
            end_time = time.time()
            end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            
            return {
                'response_time_ms': (end_time - self.start_time) * 1000,
                'memory_usage_mb': end_memory,
                'memory_delta_mb': end_memory - self.start_memory,
                'cpu_percent': self.process.cpu_percent()
            }
    
    return PerformanceMonitor()


@pytest.fixture
def batch_test_data():
    """Sample batch data for batch processing tests."""
    return [
        {
            "material_id": "SiC_001",
            "SiC": 0.7,
            "B4C": 0.2,
            "Al2O3": 0.1,
            "sintering_temperature": 1850,
            "pressure": 55,
            "grain_size": 8,
            "porosity": 0.015
        },
        {
            "material_id": "B4C_001", 
            "SiC": 0.3,
            "B4C": 0.6,
            "Al2O3": 0.1,
            "sintering_temperature": 1750,
            "pressure": 45,
            "grain_size": 12,
            "porosity": 0.025
        },
        {
            "material_id": "Al2O3_001",
            "SiC": 0.2,
            "B4C": 0.2,
            "Al2O3": 0.6,
            "sintering_temperature": 1600,
            "pressure": 40,
            "grain_size": 15,
            "porosity": 0.03
        }
    ]


@pytest.fixture(autouse=True)
def mock_external_apis():
    """Mock external API calls to prevent network requests during testing."""
    with patch('src.ml.predictor.get_predictor') as mock_get_predictor:
        # Use the mock predictor fixture
        mock_predictor = Mock()
        mock_get_predictor.return_value = mock_predictor
        
        # Configure mock responses
        mock_predictor.predict_mechanical_properties.return_value = {
            'predictions': {
                'fracture_toughness': {'value': 4.6, 'unit': 'MPa·m^0.5', 'uncertainty': 0.15},
                'vickers_hardness': {'value': 2800, 'unit': 'HV', 'uncertainty': 0.12},
                'density': {'value': 3.21, 'unit': 'g/cm³', 'uncertainty': 0.08},
                'elastic_modulus': {'value': 410, 'unit': 'GPa', 'uncertainty': 0.10}
            },
            'feature_importance': [],
            'processing_time_ms': 45
        }
        
        mock_predictor.predict_ballistic_properties.return_value = {
            'predictions': {
                'v50_velocity': {'value': 850, 'unit': 'm/s', 'uncertainty': 0.18},
                'penetration_resistance': {'value': 0.85, 'unit': 'dimensionless', 'uncertainty': 0.12},
                'back_face_deformation': {'value': 12.5, 'unit': 'mm', 'uncertainty': 0.20},
                'multi_hit_capability': {'value': 0.72, 'unit': 'probability', 'uncertainty': 0.10}
            },
            'feature_importance': [],
            'processing_time_ms': 52
        }
        
        yield mock_predictor


# Performance test thresholds
PERFORMANCE_THRESHOLDS = {
    'max_response_time_ms': 500,  # Maximum acceptable response time
    'max_memory_usage_mb': 512,   # Maximum memory usage per request
    'max_memory_delta_mb': 50,    # Maximum memory increase per request
    'max_cpu_percent': 80         # Maximum CPU usage during request
}


@pytest.fixture
def performance_thresholds():
    """Performance test thresholds."""
    return PERFORMANCE_THRESHOLDS


# Test data generators
def generate_test_materials(count: int = 10) -> list:
    """Generate test materials for load testing."""
    import random
    
    materials = []
    for i in range(count):
        # Generate random but valid compositions
        sic = random.uniform(0.3, 0.8)
        b4c = random.uniform(0.1, 0.7 - sic)
        al2o3 = 1.0 - sic - b4c
        
        materials.append({
            "composition": {
                "SiC": sic,
                "B4C": b4c,
                "Al2O3": al2o3,
                "WC": 0.0,
                "TiC": 0.0
            },
            "processing": {
                "sintering_temperature": random.randint(1400, 2200),
                "pressure": random.randint(20, 100),
                "grain_size": random.uniform(1, 50),
                "holding_time": random.randint(60, 300),
                "heating_rate": random.uniform(5, 25),
                "atmosphere": random.choice(["argon", "nitrogen", "vacuum"])
            },
            "microstructure": {
                "porosity": random.uniform(0.001, 0.1),
                "phase_distribution": random.choice(["uniform", "gradient", "layered"]),
                "interface_quality": random.choice(["poor", "fair", "good", "excellent"]),
                "pore_size": random.uniform(0.1, 5.0),
                "connectivity": random.uniform(0.01, 0.3)
            }
        })
    
    return materials


@pytest.fixture
def load_test_materials():
    """Generate materials for load testing."""
    return generate_test_materials(50)