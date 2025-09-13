"""
Tests for API Pydantic models.

This module contains unit tests for request and response models,
including validation logic and error handling.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from src.api.models import (
    CompositionModel,
    ProcessingModel,
    MicrostructureModel,
    PredictionRequest,
    PropertyPrediction,
    MechanicalPredictions,
    PredictionResponse,
    ValidationException,
    validate_material_composition,
    validate_processing_parameters,
    PhaseDistribution,
    InterfaceQuality,
    PredictionStatus
)


class TestCompositionModel:
    """Test cases for CompositionModel validation."""
    
    def test_valid_composition(self):
        """Test valid material composition."""
        composition = CompositionModel(
            SiC=0.6,
            B4C=0.3,
            Al2O3=0.1
        )
        assert composition.SiC == 0.6
        assert composition.B4C == 0.3
        assert composition.Al2O3 == 0.1
    
    def test_composition_sum_validation(self):
        """Test that composition sum validation works."""
        with pytest.raises(ValidationError) as exc_info:
            CompositionModel(
                SiC=0.7,
                B4C=0.4,
                Al2O3=0.2  # Total = 1.3 > 1.0
            )
        
        error = exc_info.value
        assert "Total composition" in str(error)
    
    def test_primary_materials_validation(self):
        """Test that primary materials validation works."""
        with pytest.raises(ValidationError) as exc_info:
            CompositionModel(
                SiC=0.1,
                B4C=0.1,
                Al2O3=0.1,
                other=0.7  # Primary materials < 50%
            )
        
        error = exc_info.value
        assert "Primary ceramic materials" in str(error)
    
    def test_negative_values_rejected(self):
        """Test that negative composition values are rejected."""
        with pytest.raises(ValidationError):
            CompositionModel(
                SiC=-0.1,
                B4C=0.5,
                Al2O3=0.5
            )


class TestProcessingModel:
    """Test cases for ProcessingModel validation."""
    
    def test_valid_processing_parameters(self):
        """Test valid processing parameters."""
        processing = ProcessingModel(
            sintering_temperature=1800,
            pressure=50,
            grain_size=10,
            holding_time=120
        )
        assert processing.sintering_temperature == 1800
        assert processing.pressure == 50
        assert processing.grain_size == 10
        assert processing.holding_time == 120
    
    def test_temperature_range_validation(self):
        """Test temperature range validation."""
        with pytest.raises(ValidationError):
            ProcessingModel(
                sintering_temperature=1000,  # Too low
                pressure=50,
                grain_size=10
            )
        
        with pytest.raises(ValidationError):
            ProcessingModel(
                sintering_temperature=3000,  # Too high
                pressure=50,
                grain_size=10
            )
    
    def test_pressure_range_validation(self):
        """Test pressure range validation."""
        with pytest.raises(ValidationError):
            ProcessingModel(
                sintering_temperature=1800,
                pressure=0,  # Too low
                grain_size=10
            )
        
        with pytest.raises(ValidationError):
            ProcessingModel(
                sintering_temperature=1800,
                pressure=300,  # Too high
                grain_size=10
            )


class TestMicrostructureModel:
    """Test cases for MicrostructureModel validation."""
    
    def test_valid_microstructure(self):
        """Test valid microstructure parameters."""
        microstructure = MicrostructureModel(
            porosity=0.02,
            phase_distribution=PhaseDistribution.UNIFORM,
            interface_quality=InterfaceQuality.GOOD
        )
        assert microstructure.porosity == 0.02
        assert microstructure.phase_distribution == PhaseDistribution.UNIFORM
        assert microstructure.interface_quality == InterfaceQuality.GOOD
    
    def test_porosity_range_validation(self):
        """Test porosity range validation."""
        with pytest.raises(ValidationError):
            MicrostructureModel(
                porosity=-0.1,  # Negative porosity
                phase_distribution=PhaseDistribution.UNIFORM
            )
        
        with pytest.raises(ValidationError):
            MicrostructureModel(
                porosity=0.5,  # Too high porosity
                phase_distribution=PhaseDistribution.UNIFORM
            )


class TestPredictionRequest:
    """Test cases for complete PredictionRequest validation."""
    
    def test_valid_prediction_request(self):
        """Test valid complete prediction request."""
        request = PredictionRequest(
            composition=CompositionModel(
                SiC=0.6,
                B4C=0.3,
                Al2O3=0.1
            ),
            processing=ProcessingModel(
                sintering_temperature=1800,
                pressure=50,
                grain_size=10
            ),
            microstructure=MicrostructureModel(
                porosity=0.02,
                phase_distribution=PhaseDistribution.UNIFORM
            )
        )
        
        assert request.composition.SiC == 0.6
        assert request.processing.sintering_temperature == 1800
        assert request.microstructure.porosity == 0.02
        assert request.include_uncertainty is True  # Default value
        assert request.include_feature_importance is True  # Default value


class TestPropertyPrediction:
    """Test cases for PropertyPrediction response model."""
    
    def test_valid_property_prediction(self):
        """Test valid property prediction."""
        prediction = PropertyPrediction(
            value=4.6,
            unit="MPa·m^0.5",
            confidence_interval=[4.2, 5.0],
            uncertainty=0.15,
            prediction_quality="good"
        )
        
        assert prediction.value == 4.6
        assert prediction.unit == "MPa·m^0.5"
        assert prediction.confidence_interval == [4.2, 5.0]
        assert prediction.uncertainty == 0.15
        assert prediction.prediction_quality == "good"
    
    def test_confidence_interval_validation(self):
        """Test confidence interval validation."""
        with pytest.raises(ValidationError):
            PropertyPrediction(
                value=4.6,
                unit="MPa·m^0.5",
                confidence_interval=[4.2],  # Only one value
                uncertainty=0.15,
                prediction_quality="good"
            )
        
        with pytest.raises(ValidationError):
            PropertyPrediction(
                value=4.6,
                unit="MPa·m^0.5",
                confidence_interval=[4.2, 5.0, 5.5],  # Too many values
                uncertainty=0.15,
                prediction_quality="good"
            )


class TestValidationUtilities:
    """Test cases for validation utility functions."""
    
    def test_validate_material_composition(self):
        """Test material composition validation utility."""
        # Valid composition
        errors = validate_material_composition({
            'SiC': 0.6,
            'B4C': 0.3,
            'Al2O3': 0.1
        })
        assert len(errors) == 0
        
        # Invalid composition - high SiC with high B4C
        errors = validate_material_composition({
            'SiC': 0.85,
            'B4C': 0.2
        })
        assert len(errors) > 0
        assert any("SiC content" in error for error in errors)
    
    def test_validate_processing_parameters(self):
        """Test processing parameters validation utility."""
        composition = {'SiC': 0.6, 'B4C': 0.3, 'Al2O3': 0.1}
        
        # Valid parameters
        errors = validate_processing_parameters({
            'sintering_temperature': 1800,
            'pressure': 50,
            'grain_size': 10
        }, composition)
        assert len(errors) == 0
        
        # Invalid - SiC-rich with low temperature
        errors = validate_processing_parameters({
            'sintering_temperature': 1400,
            'pressure': 50,
            'grain_size': 10
        }, {'SiC': 0.8, 'B4C': 0.1, 'Al2O3': 0.1})
        assert len(errors) > 0
        assert any("SiC-rich" in error for error in errors)


if __name__ == "__main__":
    pytest.main([__file__])