"""
Comprehensive validation utilities for the Ceramic Armor ML API.

This module provides advanced validation functions for input data,
file uploads, and business logic validation with detailed error reporting.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from ..models.exceptions import (
    ValidationException, 
    FileProcessingException,
    DataProcessingException
)

logger = logging.getLogger(__name__)


class ValidationContext:
    """Context manager for validation operations with detailed reporting."""
    
    def __init__(self, operation_name: str, request_id: str = None):
        self.operation_name = operation_name
        self.request_id = request_id or "unknown"
        self.errors = []
        self.warnings = []
        self.suggestions = []
        self.start_time = datetime.now()
    
    def add_error(self, message: str, field: str = None, code: str = None):
        """Add a validation error."""
        error_entry = {
            "message": message,
            "field": field,
            "code": code,
            "timestamp": datetime.now().isoformat()
        }
        self.errors.append(error_entry)
        logger.warning(f"Validation error in {self.operation_name}: {message}")
    
    def add_warning(self, message: str, field: str = None):
        """Add a validation warning."""
        warning_entry = {
            "message": message,
            "field": field,
            "timestamp": datetime.now().isoformat()
        }
        self.warnings.append(warning_entry)
        logger.info(f"Validation warning in {self.operation_name}: {message}")
    
    def add_suggestion(self, message: str):
        """Add a validation suggestion."""
        self.suggestions.append(message)
    
    def is_valid(self) -> bool:
        """Check if validation passed (no errors)."""
        return len(self.errors) == 0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        duration = (datetime.now() - self.start_time).total_seconds()
        return {
            "operation": self.operation_name,
            "request_id": self.request_id,
            "valid": self.is_valid(),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "suggestion_count": len(self.suggestions),
            "validation_time_seconds": duration,
            "errors": self.errors,
            "warnings": self.warnings,
            "suggestions": self.suggestions
        }


class MaterialCompositionValidator:
    """Advanced validator for material composition data."""
    
    # Known material properties for validation
    MATERIAL_PROPERTIES = {
        'SiC': {
            'density_range': (3.1, 3.3),
            'melting_point': 2830,
            'typical_fraction_range': (0.0, 1.0),
            'common_applications': ['high_hardness', 'thermal_conductivity']
        },
        'B4C': {
            'density_range': (2.4, 2.6),
            'melting_point': 2450,
            'typical_fraction_range': (0.0, 1.0),
            'common_applications': ['lightweight', 'neutron_absorption']
        },
        'Al2O3': {
            'density_range': (3.9, 4.1),
            'melting_point': 2072,
            'typical_fraction_range': (0.0, 1.0),
            'common_applications': ['cost_effective', 'toughness']
        },
        'WC': {
            'density_range': (15.6, 15.8),
            'melting_point': 2870,
            'typical_fraction_range': (0.0, 0.5),
            'common_applications': ['ultra_hardness', 'wear_resistance']
        },
        'TiC': {
            'density_range': (4.9, 5.0),
            'melting_point': 3067,
            'typical_fraction_range': (0.0, 0.3),
            'common_applications': ['high_temperature', 'hardness']
        }
    }
    
    @classmethod
    def validate_composition_advanced(
        cls, 
        composition: Dict[str, float], 
        context: ValidationContext
    ) -> None:
        """
        Perform advanced composition validation with materials science rules.
        
        Args:
            composition: Dictionary of material fractions
            context: Validation context for error reporting
        """
        # Basic sum validation
        total = sum(composition.values())
        if total > 1.01:
            context.add_error(
                f"Total composition ({total:.3f}) exceeds 100%",
                field="composition_total",
                code="COMPOSITION_OVERFLOW"
            )
        elif total < 0.01:
            context.add_error(
                "Total composition must be greater than 1%",
                field="composition_total",
                code="COMPOSITION_UNDERFLOW"
            )
        
        # Individual material validation
        for material, fraction in composition.items():
            if material in cls.MATERIAL_PROPERTIES:
                props = cls.MATERIAL_PROPERTIES[material]
                min_frac, max_frac = props['typical_fraction_range']
                
                if fraction > max_frac:
                    context.add_error(
                        f"{material} fraction ({fraction:.3f}) exceeds typical maximum ({max_frac})",
                        field=f"composition.{material}",
                        code="MATERIAL_FRACTION_HIGH"
                    )
                
                # Warn about unusual but not invalid fractions
                if fraction > 0.9 and material != 'Al2O3':
                    context.add_warning(
                        f"Very high {material} content ({fraction:.1%}) - ensure this is intentional",
                        field=f"composition.{material}"
                    )
        
        # Materials compatibility checks
        cls._validate_material_compatibility(composition, context)
        
        # Density estimation and validation
        cls._validate_estimated_density(composition, context)
    
    @classmethod
    def _validate_material_compatibility(
        cls, 
        composition: Dict[str, float], 
        context: ValidationContext
    ) -> None:
        """Validate material compatibility based on processing considerations."""
        sic = composition.get('SiC', 0)
        b4c = composition.get('B4C', 0)
        al2o3 = composition.get('Al2O3', 0)
        wc = composition.get('WC', 0)
        tic = composition.get('TiC', 0)
        
        # SiC-B4C compatibility
        if sic > 0.7 and b4c > 0.2:
            context.add_warning(
                "High SiC with significant B4C may require careful processing to avoid phase separation"
            )
        
        # WC compatibility
        if wc > 0.3:
            ceramic_total = sic + b4c + al2o3
            if ceramic_total > 0.6:
                context.add_error(
                    "High WC content (>30%) with high ceramic content (>60%) may result in poor sintering",
                    code="INCOMPATIBLE_WC_CERAMIC"
                )
        
        # TiC processing considerations
        if tic > 0.2 and (sic + b4c) > 0.7:
            context.add_warning(
                "TiC with high SiC/B4C content may require inert atmosphere processing"
            )
        
        # Pure material warnings
        pure_materials = [mat for mat, frac in composition.items() if frac > 0.95]
        if len(pure_materials) > 0:
            context.add_suggestion(
                f"Nearly pure {pure_materials[0]} composition - consider if additives might improve properties"
            )
    
    @classmethod
    def _validate_estimated_density(
        cls, 
        composition: Dict[str, float], 
        context: ValidationContext
    ) -> None:
        """Estimate and validate theoretical density."""
        try:
            estimated_density = 0
            for material, fraction in composition.items():
                if material in cls.MATERIAL_PROPERTIES and fraction > 0:
                    # Use average density for estimation
                    density_range = cls.MATERIAL_PROPERTIES[material]['density_range']
                    avg_density = (density_range[0] + density_range[1]) / 2
                    estimated_density += fraction * avg_density
            
            if estimated_density > 0:
                # Validate reasonable density range for ceramics
                if estimated_density < 2.0:
                    context.add_warning(
                        f"Estimated density ({estimated_density:.2f} g/cm³) is unusually low for ceramic armor"
                    )
                elif estimated_density > 16.0:
                    context.add_warning(
                        f"Estimated density ({estimated_density:.2f} g/cm³) is very high - may affect ballistic performance"
                    )
                
                # Add density estimate as suggestion
                context.add_suggestion(
                    f"Estimated theoretical density: {estimated_density:.2f} g/cm³"
                )
        
        except Exception as e:
            logger.warning(f"Failed to estimate density: {e}")


class ProcessingParametersValidator:
    """Advanced validator for processing parameters."""
    
    # Processing parameter relationships and constraints
    PARAMETER_RELATIONSHIPS = {
        'temperature_pressure': {
            'high_temp_high_pressure': (2000, 100),  # °C, MPa
            'low_temp_low_pressure': (1400, 20)
        },
        'grain_size_factors': {
            'temperature_coefficient': 0.01,  # μm per °C above 1400
            'pressure_coefficient': -0.1     # μm per MPa above 20
        }
    }
    
    @classmethod
    def validate_processing_advanced(
        cls,
        processing: Dict[str, Any],
        composition: Dict[str, float],
        context: ValidationContext
    ) -> None:
        """
        Perform advanced processing parameter validation.
        
        Args:
            processing: Processing parameters dictionary
            composition: Material composition for context
            context: Validation context for error reporting
        """
        temp = processing.get('sintering_temperature', 0)
        pressure = processing.get('pressure', 0)
        grain_size = processing.get('grain_size', 0)
        holding_time = processing.get('holding_time', 60)
        heating_rate = processing.get('heating_rate', 10)
        atmosphere = processing.get('atmosphere', 'argon')
        
        # Temperature-composition compatibility
        cls._validate_temperature_composition(temp, composition, context)
        
        # Pressure-grain size relationships
        cls._validate_pressure_grain_relationships(pressure, grain_size, temp, context)
        
        # Time-temperature relationships
        cls._validate_time_temperature_relationships(holding_time, temp, heating_rate, context)
        
        # Atmosphere compatibility
        cls._validate_atmosphere_compatibility(atmosphere, temp, composition, context)
        
        # Energy efficiency suggestions
        cls._suggest_energy_optimization(temp, pressure, holding_time, context)
    
    @classmethod
    def _validate_temperature_composition(
        cls,
        temperature: float,
        composition: Dict[str, float],
        context: ValidationContext
    ) -> None:
        """Validate temperature compatibility with composition."""
        sic_content = composition.get('SiC', 0)
        b4c_content = composition.get('B4C', 0)
        al2o3_content = composition.get('Al2O3', 0)
        
        # SiC processing requirements
        if sic_content > 0.5:
            if temperature < 1600:
                context.add_error(
                    f"SiC-rich compositions ({sic_content:.1%}) typically require temperatures >1600°C (current: {temperature}°C)",
                    field="sintering_temperature",
                    code="TEMP_TOO_LOW_FOR_SIC"
                )
            elif temperature > 2300:
                context.add_warning(
                    f"Very high temperature ({temperature}°C) with SiC may cause grain growth"
                )
        
        # B4C decomposition risk
        if b4c_content > 0.3 and temperature > 2200:
            context.add_error(
                f"B4C content ({b4c_content:.1%}) may decompose at {temperature}°C (limit: 2200°C)",
                field="sintering_temperature",
                code="TEMP_TOO_HIGH_FOR_B4C"
            )
        
        # Al2O3 processing optimization
        if al2o3_content > 0.7:
            if temperature < 1500:
                context.add_warning(
                    f"Al2O3-rich compositions benefit from temperatures >1500°C for better densification"
                )
    
    @classmethod
    def _validate_pressure_grain_relationships(
        cls,
        pressure: float,
        grain_size: float,
        temperature: float,
        context: ValidationContext
    ) -> None:
        """Validate pressure-grain size relationships."""
        # High pressure typically produces fine grains
        if pressure > 100 and grain_size > 20:
            context.add_error(
                f"High pressure ({pressure} MPa) typically produces grain sizes <20μm (specified: {grain_size}μm)",
                field="grain_size",
                code="INCONSISTENT_PRESSURE_GRAIN"
            )
        
        # Low pressure limitations
        if pressure < 20 and grain_size < 5:
            context.add_error(
                f"Low pressure ({pressure} MPa) rarely produces very fine grains (<5μm)",
                field="grain_size",
                code="UNREALISTIC_LOW_PRESSURE_FINE_GRAIN"
            )
        
        # Temperature-grain size consistency
        if temperature > 2000 and grain_size < 2:
            context.add_warning(
                f"High temperature ({temperature}°C) with very fine grains ({grain_size}μm) may be difficult to achieve"
            )
    
    @classmethod
    def _validate_time_temperature_relationships(
        cls,
        holding_time: float,
        temperature: float,
        heating_rate: float,
        context: ValidationContext
    ) -> None:
        """Validate time-temperature processing relationships."""
        # High temperature short time vs low temperature long time
        if temperature > 2200 and holding_time > 300:  # 5 hours
            context.add_warning(
                f"Very high temperature ({temperature}°C) with long holding time ({holding_time} min) may cause excessive grain growth"
            )
        
        if temperature < 1500 and holding_time < 30:
            context.add_warning(
                f"Low temperature ({temperature}°C) with short time ({holding_time} min) may result in poor densification"
            )
        
        # Heating rate considerations
        if heating_rate > 25 and temperature > 2000:
            context.add_error(
                f"High heating rate ({heating_rate}°C/min) with high temperature ({temperature}°C) may cause thermal shock",
                field="heating_rate",
                code="THERMAL_SHOCK_RISK"
            )
    
    @classmethod
    def _validate_atmosphere_compatibility(
        cls,
        atmosphere: str,
        temperature: float,
        composition: Dict[str, float],
        context: ValidationContext
    ) -> None:
        """Validate atmosphere compatibility."""
        # Air atmosphere limitations
        if atmosphere == 'air':
            if temperature > 1600:
                context.add_error(
                    f"Air atmosphere not recommended above 1600°C (current: {temperature}°C) due to oxidation",
                    field="atmosphere",
                    code="AIR_OXIDATION_RISK"
                )
            
            # Check for oxidation-sensitive materials
            sensitive_materials = ['TiC', 'WC']
            for material in sensitive_materials:
                if composition.get(material, 0) > 0.1:
                    context.add_warning(
                        f"{material} content ({composition[material]:.1%}) may oxidize in air atmosphere"
                    )
        
        # Vacuum processing requirements
        if atmosphere == 'vacuum' and temperature < 1400:
            context.add_warning(
                "Vacuum processing typically requires temperatures >1400°C for effective sintering"
            )
    
    @classmethod
    def _suggest_energy_optimization(
        cls,
        temperature: float,
        pressure: float,
        holding_time: float,
        context: ValidationContext
    ) -> None:
        """Suggest energy optimization opportunities."""
        # High energy consumption warning
        energy_score = (temperature / 2000) * (pressure / 100) * (holding_time / 120)
        
        if energy_score > 2.0:
            context.add_suggestion(
                "Consider optimizing processing parameters to reduce energy consumption while maintaining quality"
            )
        
        # Specific optimization suggestions
        if temperature > 2100 and holding_time > 180:
            context.add_suggestion(
                "High temperature processing - consider reducing holding time or temperature slightly"
            )


class DataQualityValidator:
    """Validator for data quality and statistical properties."""
    
    @classmethod
    def validate_dataframe_quality(
        cls,
        df: pd.DataFrame,
        context: ValidationContext,
        required_columns: List[str] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive DataFrame quality validation.
        
        Args:
            df: DataFrame to validate
            context: Validation context
            required_columns: List of required column names
            
        Returns:
            Quality metrics dictionary
        """
        quality_metrics = {}
        
        # Basic structure validation
        if df.empty:
            context.add_error("Dataset is empty", code="EMPTY_DATASET")
            return quality_metrics
        
        # Required columns check
        if required_columns:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                context.add_error(
                    f"Missing required columns: {missing_cols}",
                    code="MISSING_REQUIRED_COLUMNS"
                )
        
        # Calculate quality metrics
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        missing_percentage = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
        
        quality_metrics.update({
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_cells': missing_cells,
            'missing_percentage': missing_percentage,
            'duplicate_rows': df.duplicated().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
        })
        
        # Missing data analysis
        if missing_percentage > 50:
            context.add_error(
                f"Excessive missing data: {missing_percentage:.1f}% of cells are empty",
                code="EXCESSIVE_MISSING_DATA"
            )
        elif missing_percentage > 20:
            context.add_warning(
                f"High missing data: {missing_percentage:.1f}% of cells are empty"
            )
        
        # Duplicate analysis
        duplicate_count = quality_metrics['duplicate_rows']
        if duplicate_count > 0:
            duplicate_percentage = (duplicate_count / len(df)) * 100
            if duplicate_percentage > 10:
                context.add_warning(
                    f"High duplicate rate: {duplicate_count} rows ({duplicate_percentage:.1f}%)"
                )
            else:
                context.add_suggestion(
                    f"Found {duplicate_count} duplicate rows - consider removing if unintentional"
                )
        
        # Column-specific analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        text_cols = df.select_dtypes(include=['object']).columns
        
        quality_metrics.update({
            'numeric_columns': len(numeric_cols),
            'text_columns': len(text_cols),
            'column_types': dict(df.dtypes.astype(str))
        })
        
        # Numeric data quality
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                # Check for outliers using IQR method
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                outlier_mask = (col_data < (Q1 - 1.5 * IQR)) | (col_data > (Q3 + 1.5 * IQR))
                outlier_count = outlier_mask.sum()
                
                if outlier_count > len(col_data) * 0.1:  # More than 10% outliers
                    context.add_warning(
                        f"Column '{col}' has {outlier_count} potential outliers ({outlier_count/len(col_data)*100:.1f}%)"
                    )
        
        return quality_metrics


def validate_prediction_request_comprehensive(
    request_data: Dict[str, Any],
    request_id: str = None
) -> ValidationContext:
    """
    Comprehensive validation for prediction requests.
    
    Args:
        request_data: Complete request data dictionary
        request_id: Request identifier for tracking
        
    Returns:
        ValidationContext with results
    """
    context = ValidationContext("prediction_request_validation", request_id)
    
    try:
        # Extract components
        composition = request_data.get('composition', {})
        processing = request_data.get('processing', {})
        microstructure = request_data.get('microstructure', {})
        
        # Validate composition
        if composition:
            MaterialCompositionValidator.validate_composition_advanced(composition, context)
        else:
            context.add_error("Missing composition data", field="composition", code="MISSING_COMPOSITION")
        
        # Validate processing parameters
        if processing:
            ProcessingParametersValidator.validate_processing_advanced(processing, composition, context)
        else:
            context.add_error("Missing processing data", field="processing", code="MISSING_PROCESSING")
        
        # Validate microstructure (basic validation)
        if microstructure:
            porosity = microstructure.get('porosity', 0)
            if porosity > 0.3:
                context.add_error(
                    f"Porosity ({porosity:.1%}) exceeds maximum (30%)",
                    field="microstructure.porosity",
                    code="POROSITY_TOO_HIGH"
                )
        else:
            context.add_error("Missing microstructure data", field="microstructure", code="MISSING_MICROSTRUCTURE")
        
        # Cross-validation between components
        if composition and processing:
            # Example: High WC content requires specific processing
            wc_content = composition.get('WC', 0)
            temperature = processing.get('sintering_temperature', 0)
            
            if wc_content > 0.2 and temperature < 1700:
                context.add_warning(
                    f"WC content ({wc_content:.1%}) may require higher sintering temperature (>1700°C)"
                )
        
        # Add overall assessment
        if context.is_valid():
            context.add_suggestion("All validation checks passed - request is ready for processing")
        else:
            context.add_suggestion("Please address the validation errors before submitting the request")
    
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        context.add_error(f"Validation process failed: {str(e)}", code="VALIDATION_SYSTEM_ERROR")
    
    return context


def validate_batch_file_comprehensive(
    df: pd.DataFrame,
    filename: str,
    request_id: str = None
) -> ValidationContext:
    """
    Comprehensive validation for batch upload files.
    
    Args:
        df: DataFrame from uploaded file
        filename: Original filename
        request_id: Request identifier for tracking
        
    Returns:
        ValidationContext with results
    """
    context = ValidationContext("batch_file_validation", request_id)
    
    try:
        # Required columns for batch processing
        required_columns = [
            'SiC', 'B4C', 'Al2O3',  # Minimum composition
            'sintering_temperature', 'pressure', 'grain_size',  # Processing
            'porosity', 'phase_distribution'  # Microstructure
        ]
        
        # Data quality validation
        quality_metrics = DataQualityValidator.validate_dataframe_quality(
            df, context, required_columns
        )
        
        # File-specific validation
        if len(df) > 10000:
            context.add_error(
                f"File contains {len(df)} rows, maximum allowed is 10,000",
                code="FILE_TOO_LARGE"
            )
        
        # Validate each row's composition (sample check)
        if not df.empty and context.is_valid():
            # Check first few rows for composition validity
            sample_size = min(10, len(df))
            composition_cols = ['SiC', 'B4C', 'Al2O3', 'WC', 'TiC']
            available_comp_cols = [col for col in composition_cols if col in df.columns]
            
            if available_comp_cols:
                sample_df = df.head(sample_size)
                for idx, row in sample_df.iterrows():
                    composition = {col: row.get(col, 0) for col in available_comp_cols}
                    total = sum(composition.values())
                    
                    if total > 1.1 or total < 0.01:
                        context.add_warning(
                            f"Row {idx + 1}: Invalid composition sum ({total:.3f})"
                        )
        
        # Add quality metrics to context
        context.quality_metrics = quality_metrics
        
        # Final recommendations
        if context.is_valid():
            context.add_suggestion(f"File '{filename}' is ready for batch processing")
            if quality_metrics.get('missing_percentage', 0) > 5:
                context.add_suggestion("Consider filling missing values for better prediction accuracy")
        
    except Exception as e:
        logger.error(f"Batch file validation failed: {e}")
        context.add_error(f"File validation failed: {str(e)}", code="FILE_VALIDATION_ERROR")
    
    return context