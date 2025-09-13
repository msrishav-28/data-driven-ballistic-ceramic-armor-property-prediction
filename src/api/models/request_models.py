"""
Pydantic request models for the Ceramic Armor ML API.

This module defines the data models for API requests, including material composition,
processing parameters, and microstructure data with comprehensive validation.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict, Optional, List, Any
from enum import Enum
import math


class PhaseDistribution(str, Enum):
    """Enumeration for phase distribution types."""
    UNIFORM = "uniform"
    GRADIENT = "gradient"
    LAYERED = "layered"


class InterfaceQuality(str, Enum):
    """Enumeration for interface quality levels."""
    POOR = "poor"
    FAIR = "fair"
    GOOD = "good"
    EXCELLENT = "excellent"


class CompositionModel(BaseModel):
    """
    Material composition model with validation for ceramic armor materials.
    
    All composition values should be fractions (0-1) and must sum to <= 1.0.
    """
    SiC: float = Field(
        0.0, 
        ge=0, 
        le=1, 
        description="Silicon Carbide fraction (0-1)"
    )
    B4C: float = Field(
        0.0, 
        ge=0, 
        le=1, 
        description="Boron Carbide fraction (0-1)"
    )
    Al2O3: float = Field(
        0.0, 
        ge=0, 
        le=1, 
        description="Aluminum Oxide fraction (0-1)"
    )
    WC: Optional[float] = Field(
        0.0, 
        ge=0, 
        le=1, 
        description="Tungsten Carbide fraction (0-1)"
    )
    TiC: Optional[float] = Field(
        0.0, 
        ge=0, 
        le=1, 
        description="Titanium Carbide fraction (0-1)"
    )
    ZrO2: Optional[float] = Field(
        0.0, 
        ge=0, 
        le=1, 
        description="Zirconium Oxide fraction (0-1)"
    )
    TiB2: Optional[float] = Field(
        0.0, 
        ge=0, 
        le=1, 
        description="Titanium Diboride fraction (0-1)"
    )
    other: Optional[float] = Field(
        0.0, 
        ge=0, 
        le=1, 
        description="Other materials fraction (0-1)"
    )

    @model_validator(mode='after')
    def validate_composition_sum(self) -> 'CompositionModel':
        """Validate that total composition does not exceed 100%."""
        values = self.model_dump()
        total = sum(v for v in values.values() if v is not None)
        if total > 1.01:  # Allow small tolerance for floating point precision
            raise ValueError(f"Total composition ({total:.3f}) cannot exceed 100%")
        if total < 0.01:  # Require at least some material
            raise ValueError("Total composition must be greater than 1%")
        return self

    @model_validator(mode='after')
    def validate_primary_materials(self) -> 'CompositionModel':
        """Ensure at least one primary ceramic material is present."""
        values = self.model_dump()
        primary_materials = ['SiC', 'B4C', 'Al2O3', 'WC', 'TiC']
        primary_total = sum(values.get(mat, 0) for mat in primary_materials)
        if primary_total < 0.5:
            raise ValueError("Primary ceramic materials (SiC, B4C, Al2O3, WC, TiC) must comprise at least 50% of composition")
        return self
    
    @model_validator(mode='after')
    def validate_realistic_combinations(self) -> 'CompositionModel':
        """Validate realistic material combinations."""
        values = self.model_dump()
        
        # Check for incompatible high concentrations
        sic_content = values.get('SiC', 0)
        b4c_content = values.get('B4C', 0)
        al2o3_content = values.get('Al2O3', 0)
        wc_content = values.get('WC', 0)
        
        # SiC and B4C compatibility
        if sic_content > 0.8 and b4c_content > 0.15:
            raise ValueError("High SiC content (>80%) with significant B4C (>15%) may be difficult to process")
        
        # WC compatibility with ceramics
        if wc_content > 0.3 and (sic_content + b4c_content) > 0.6:
            raise ValueError("High WC content (>30%) with high ceramic content may result in poor bonding")
        
        # Al2O3 as pure phase check
        if al2o3_content > 0.95 and (sic_content + b4c_content + wc_content) > 0.04:
            raise ValueError("Nearly pure Al2O3 (>95%) should not contain significant amounts of other ceramics")
        
        return self


class ProcessingModel(BaseModel):
    """
    Processing parameters model for ceramic manufacturing conditions.
    
    Includes sintering temperature, pressure, grain size, and timing parameters.
    """
    sintering_temperature: float = Field(
        ..., 
        ge=1200, 
        le=2500, 
        description="Sintering temperature in Celsius (1200-2500°C)"
    )
    pressure: float = Field(
        ..., 
        ge=1, 
        le=200, 
        description="Applied pressure in MPa (1-200 MPa)"
    )
    grain_size: float = Field(
        ..., 
        ge=0.1, 
        le=100, 
        description="Average grain size in micrometers (0.1-100 μm)"
    )
    holding_time: Optional[float] = Field(
        60, 
        ge=1, 
        le=600, 
        description="Sintering holding time in minutes (1-600 min)"
    )
    heating_rate: Optional[float] = Field(
        10, 
        ge=1, 
        le=50, 
        description="Heating rate in °C/min (1-50 °C/min)"
    )
    atmosphere: Optional[str] = Field(
        "argon", 
        pattern="^(air|argon|nitrogen|vacuum|hydrogen)$",
        description="Sintering atmosphere"
    )

    @model_validator(mode='after')
    def validate_temperature_grain_size_relationship(self) -> 'ProcessingModel':
        """Validate realistic temperature-grain size relationships."""
        temp = self.sintering_temperature
        grain_size = self.grain_size
        pressure = self.pressure
        
        # Higher temperatures generally produce larger grains
        if temp > 2000 and grain_size < 1:
            raise ValueError("Very high temperatures (>2000°C) typically produce grain sizes >1μm")
        if temp < 1400 and grain_size > 50:
            raise ValueError("Low temperatures (<1400°C) rarely produce grain sizes >50μm")
        
        # Pressure-temperature relationships
        if pressure > 150 and temp < 1500:
            raise ValueError("Very high pressure (>150 MPa) typically requires higher temperatures (>1500°C)")
        
        # Heating rate validation
        heating_rate = getattr(self, 'heating_rate', 10)
        if heating_rate > 30 and temp > 2200:
            raise ValueError("High heating rates (>30°C/min) with very high temperatures (>2200°C) may cause thermal shock")
        
        return self
    
    @model_validator(mode='after')
    def validate_atmosphere_compatibility(self) -> 'ProcessingModel':
        """Validate atmosphere compatibility with temperature and materials."""
        temp = self.sintering_temperature
        atmosphere = getattr(self, 'atmosphere', 'argon')
        
        # Air atmosphere limitations
        if atmosphere == 'air' and temp > 1600:
            raise ValueError("Air atmosphere not recommended above 1600°C due to oxidation risks")
        
        # Hydrogen atmosphere safety
        if atmosphere == 'hydrogen' and temp > 2000:
            raise ValueError("Hydrogen atmosphere requires special safety considerations above 2000°C")
        
        # Vacuum processing requirements
        if atmosphere == 'vacuum' and temp < 1400:
            raise ValueError("Vacuum processing typically requires temperatures >1400°C for effective sintering")
        
        return self


class MicrostructureModel(BaseModel):
    """
    Microstructure characteristics model for ceramic materials.
    
    Describes porosity, phase distribution, and interface properties.
    """
    porosity: float = Field(
        ..., 
        ge=0, 
        le=0.3, 
        description="Porosity fraction (0-0.3)"
    )
    phase_distribution: PhaseDistribution = Field(
        ..., 
        description="Type of phase distribution in the material"
    )
    interface_quality: Optional[InterfaceQuality] = Field(
        InterfaceQuality.GOOD, 
        description="Quality of grain boundaries and interfaces"
    )
    pore_size: Optional[float] = Field(
        1.0, 
        ge=0.01, 
        le=50, 
        description="Average pore size in micrometers (0.01-50 μm)"
    )
    connectivity: Optional[float] = Field(
        0.1, 
        ge=0, 
        le=1, 
        description="Pore connectivity factor (0-1)"
    )

    @model_validator(mode='after')
    def validate_pore_size_porosity_relationship(self) -> 'MicrostructureModel':
        """Validate realistic pore size-porosity relationships."""
        porosity = self.porosity
        pore_size = self.pore_size or 1.0  # Default value
        
        # High porosity should correlate with larger or more connected pores
        if porosity > 0.15 and pore_size < 0.1:
            raise ValueError("High porosity (>15%) typically has pore sizes >0.1μm")
        if porosity < 0.02 and pore_size > 10:
            raise ValueError("Low porosity (<2%) rarely has pore sizes >10μm")
        return self


class PredictionRequest(BaseModel):
    """
    Complete prediction request model combining all material parameters.
    
    This is the main request model for both mechanical and ballistic property predictions.
    """
    composition: CompositionModel = Field(
        ..., 
        description="Material composition data"
    )
    processing: ProcessingModel = Field(
        ..., 
        description="Processing parameters"
    )
    microstructure: MicrostructureModel = Field(
        ..., 
        description="Microstructure characteristics"
    )
    include_uncertainty: bool = Field(
        True, 
        description="Include uncertainty quantification in predictions"
    )
    include_feature_importance: bool = Field(
        True, 
        description="Include feature importance analysis"
    )
    prediction_type: Optional[str] = Field(
        "both", 
        pattern="^(mechanical|ballistic|both)$",
        description="Type of properties to predict"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "composition": {
                    "SiC": 0.6,
                    "B4C": 0.3,
                    "Al2O3": 0.1,
                    "WC": 0.0,
                    "TiC": 0.0
                },
                "processing": {
                    "sintering_temperature": 1800,
                    "pressure": 50,
                    "grain_size": 10,
                    "holding_time": 120,
                    "heating_rate": 15,
                    "atmosphere": "argon"
                },
                "microstructure": {
                    "porosity": 0.02,
                    "phase_distribution": "uniform",
                    "interface_quality": "good",
                    "pore_size": 1.0,
                    "connectivity": 0.1
                },
                "include_uncertainty": True,
                "include_feature_importance": True,
                "prediction_type": "both"
            }
        }


class BatchPredictionRequest(BaseModel):
    """
    Request model for batch predictions from uploaded files.
    
    Includes file processing options and output format preferences.
    """
    file_format: str = Field(
        "csv", 
        pattern="^(csv|xlsx|json)$",
        description="Input file format"
    )
    output_format: str = Field(
        "csv", 
        pattern="^(csv|xlsx|json)$",
        description="Desired output format"
    )
    include_uncertainty: bool = Field(
        True, 
        description="Include uncertainty quantification"
    )
    include_feature_importance: bool = Field(
        False, 
        description="Include feature importance (increases file size)"
    )
    prediction_type: str = Field(
        "both", 
        pattern="^(mechanical|ballistic|both)$",
        description="Type of properties to predict"
    )
    max_rows: Optional[int] = Field(
        1000, 
        ge=1, 
        le=10000,
        description="Maximum number of rows to process"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "file_format": "csv",
                "output_format": "csv",
                "include_uncertainty": True,
                "include_feature_importance": False,
                "prediction_type": "both",
                "max_rows": 1000
            }
        }
