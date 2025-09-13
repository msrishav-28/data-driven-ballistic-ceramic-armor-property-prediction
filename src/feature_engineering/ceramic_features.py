"""
Ceramic Armor Feature Engineering Module.

This module provides comprehensive feature engineering for ceramic armor materials,
converting API request data into ML-ready feature vectors compatible with trained models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import logging
from dataclasses import dataclass

from ..api.models.request_models import (
    CompositionModel, 
    ProcessingModel, 
    MicrostructureModel,
    PredictionRequest,
    PhaseDistribution,
    InterfaceQuality
)

logger = logging.getLogger(__name__)


@dataclass
class FeatureMetadata:
    """Metadata for engineered features."""
    
    name: str
    description: str
    unit: str
    feature_type: str  # 'composition', 'processing', 'microstructure', 'derived'
    importance_rank: Optional[int] = None


class CeramicFeatureEngineer:
    """
    Comprehensive feature engineering for ceramic armor materials.
    
    Converts material composition, processing parameters, and microstructure
    characteristics into ML-ready feature vectors with domain-specific
    derived features based on materials science principles.
    """
    
    def __init__(self):
        """Initialize the feature engineer with material property databases."""
        self.feature_metadata = self._initialize_feature_metadata()
        self.material_properties = self._initialize_material_properties()
        
        # Feature scaling parameters (would be loaded from training data in production)
        self.feature_means = {}
        self.feature_stds = {}
        
        logger.info("Initialized CeramicFeatureEngineer")
    
    def extract_features(self, request: PredictionRequest) -> Tuple[np.ndarray, List[str]]:
        """
        Extract comprehensive feature vector from prediction request.
        
        Args:
            request: PredictionRequest containing material data
            
        Returns:
            Tuple of (feature_vector, feature_names)
        """
        try:
            features = {}
            
            # Extract composition features
            comp_features = self._extract_composition_features(request.composition)
            features.update(comp_features)
            
            # Extract processing features
            proc_features = self._extract_processing_features(request.processing)
            features.update(proc_features)
            
            # Extract microstructure features
            micro_features = self._extract_microstructure_features(request.microstructure)
            features.update(micro_features)
            
            # Extract derived features
            derived_features = self._extract_derived_features(
                request.composition, request.processing, request.microstructure
            )
            features.update(derived_features)
            
            # Convert to arrays
            feature_names = list(features.keys())
            feature_vector = np.array([features[name] for name in feature_names])
            
            # Apply scaling if available
            if self.feature_means and self.feature_stds:
                feature_vector = self._scale_features(feature_vector, feature_names)
            
            logger.debug(f"Extracted {len(feature_names)} features")
            return feature_vector, feature_names
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise
    
    def _extract_composition_features(self, composition: CompositionModel) -> Dict[str, float]:
        """Extract composition-based features."""
        features = {}
        
        # Direct composition features
        comp_dict = composition.model_dump()
        for material, fraction in comp_dict.items():
            if fraction is not None:
                features[f"comp_{material}"] = fraction
        
        # Composition-derived features
        features.update({
            # Total ceramic content
            "comp_total_ceramic": composition.SiC + composition.B4C + composition.Al2O3,
            
            # Carbide content
            "comp_total_carbide": composition.SiC + composition.B4C + (composition.WC or 0) + (composition.TiC or 0),
            
            # Oxide content
            "comp_total_oxide": composition.Al2O3 + (composition.ZrO2 or 0),
            
            # Hard phase content (SiC, B4C, WC)
            "comp_hard_phase": composition.SiC + composition.B4C + (composition.WC or 0),
            
            # Toughening phase content (Al2O3, ZrO2)
            "comp_toughening_phase": composition.Al2O3 + (composition.ZrO2 or 0),
            
            # Composition ratios
            "comp_sic_b4c_ratio": composition.SiC / (composition.B4C + 1e-6),
            "comp_carbide_oxide_ratio": (composition.SiC + composition.B4C) / (composition.Al2O3 + 1e-6),
        })
        
        # Material property weighted averages
        features.update(self._calculate_weighted_properties(comp_dict))
        
        return features
    
    def _extract_processing_features(self, processing: ProcessingModel) -> Dict[str, float]:
        """Extract processing parameter features."""
        features = {
            # Direct processing parameters
            "proc_temperature": processing.sintering_temperature,
            "proc_pressure": processing.pressure,
            "proc_grain_size": processing.grain_size,
            "proc_holding_time": processing.holding_time or 60,
            "proc_heating_rate": processing.heating_rate or 10,
            
            # Normalized processing parameters
            "proc_temperature_norm": (processing.sintering_temperature - 1200) / (2500 - 1200),
            "proc_pressure_norm": (processing.pressure - 1) / (200 - 1),
            "proc_grain_size_log": np.log10(processing.grain_size),
            
            # Processing-derived features
            "proc_thermal_energy": processing.sintering_temperature * (processing.holding_time or 60),
            "proc_pressure_temperature_product": processing.pressure * processing.sintering_temperature,
            "proc_densification_parameter": processing.pressure / (processing.grain_size ** 0.5),
        }
        
        # Atmosphere encoding
        atmosphere_map = {
            "air": [1, 0, 0, 0, 0],
            "argon": [0, 1, 0, 0, 0],
            "nitrogen": [0, 0, 1, 0, 0],
            "vacuum": [0, 0, 0, 1, 0],
            "hydrogen": [0, 0, 0, 0, 1]
        }
        atmosphere = processing.atmosphere or "argon"
        atmosphere_encoding = atmosphere_map.get(atmosphere, atmosphere_map["argon"])
        
        for i, atm_type in enumerate(["air", "argon", "nitrogen", "vacuum", "hydrogen"]):
            features[f"proc_atmosphere_{atm_type}"] = atmosphere_encoding[i]
        
        return features
    
    def _extract_microstructure_features(self, microstructure: MicrostructureModel) -> Dict[str, float]:
        """Extract microstructure-based features."""
        features = {
            # Direct microstructure parameters
            "micro_porosity": microstructure.porosity,
            "micro_pore_size": microstructure.pore_size or 1.0,
            "micro_connectivity": microstructure.connectivity or 0.1,
            
            # Derived microstructure features
            "micro_relative_density": 1.0 - microstructure.porosity,
            "micro_pore_volume_fraction": microstructure.porosity,
            "micro_pore_size_log": np.log10(microstructure.pore_size or 1.0),
            "micro_porosity_connectivity_product": microstructure.porosity * (microstructure.connectivity or 0.1),
        }
        
        # Phase distribution encoding
        phase_dist_map = {
            PhaseDistribution.UNIFORM: [1, 0, 0],
            PhaseDistribution.GRADIENT: [0, 1, 0],
            PhaseDistribution.LAYERED: [0, 0, 1]
        }
        phase_encoding = phase_dist_map[microstructure.phase_distribution]
        
        for i, dist_type in enumerate(["uniform", "gradient", "layered"]):
            features[f"micro_phase_dist_{dist_type}"] = phase_encoding[i]
        
        # Interface quality encoding
        interface_quality_map = {
            InterfaceQuality.POOR: 0.25,
            InterfaceQuality.FAIR: 0.5,
            InterfaceQuality.GOOD: 0.75,
            InterfaceQuality.EXCELLENT: 1.0
        }
        features["micro_interface_quality"] = interface_quality_map[microstructure.interface_quality or InterfaceQuality.GOOD]
        
        return features
    
    def _extract_derived_features(
        self, 
        composition: CompositionModel, 
        processing: ProcessingModel, 
        microstructure: MicrostructureModel
    ) -> Dict[str, float]:
        """Extract advanced derived features based on materials science principles."""
        features = {}
        
        # Pugh ratio approximation (G/K ratio for brittleness)
        weighted_modulus = self._calculate_weighted_elastic_modulus(composition)
        features["derived_pugh_ratio_approx"] = weighted_modulus / (weighted_modulus * 0.6)  # Approximation
        
        # Cauchy pressure approximation
        features["derived_cauchy_pressure_approx"] = weighted_modulus * 0.1  # Simplified
        
        # Thermal shock resistance parameter
        thermal_conductivity = self._calculate_weighted_thermal_conductivity(composition)
        thermal_expansion = self._calculate_weighted_thermal_expansion(composition)
        features["derived_thermal_shock_resistance"] = thermal_conductivity / (weighted_modulus * thermal_expansion)
        
        # Densification efficiency
        theoretical_density = self._calculate_theoretical_density(composition)
        actual_density = theoretical_density * (1 - microstructure.porosity)
        features["derived_densification_efficiency"] = actual_density / theoretical_density
        
        # Processing efficiency metrics
        features["derived_temperature_pressure_efficiency"] = (
            processing.sintering_temperature * processing.pressure / 
            (processing.grain_size * (processing.holding_time or 60))
        )
        
        # Microstructure quality index
        interface_quality_score = {
            InterfaceQuality.POOR: 0.25,
            InterfaceQuality.FAIR: 0.5,
            InterfaceQuality.GOOD: 0.75,
            InterfaceQuality.EXCELLENT: 1.0
        }[microstructure.interface_quality or InterfaceQuality.GOOD]
        
        features["derived_microstructure_quality"] = (
            (1 - microstructure.porosity) * interface_quality_score * 
            (1 - microstructure.connectivity or 0.1)
        )
        
        # Ballistic performance indicators
        features["derived_hardness_toughness_balance"] = (
            composition.SiC + composition.B4C  # Hard phases
        ) * (
            composition.Al2O3 + (composition.ZrO2 or 0)  # Toughening phases
        )
        
        # Multi-hit capability indicator
        features["derived_multi_hit_indicator"] = (
            features["derived_microstructure_quality"] * 
            features["derived_hardness_toughness_balance"] *
            (1 - microstructure.porosity)
        )
        
        return features
    
    def _calculate_weighted_properties(self, composition: Dict[str, float]) -> Dict[str, float]:
        """Calculate composition-weighted material properties."""
        properties = {}
        
        total_fraction = sum(v for v in composition.values() if v is not None and v > 0)
        if total_fraction == 0:
            return properties
        
        # Weighted density
        weighted_density = 0
        for material, fraction in composition.items():
            if fraction and fraction > 0:
                density = self.material_properties.get(material, {}).get('density', 3.2)
                weighted_density += fraction * density
        
        properties["comp_weighted_density"] = weighted_density
        
        # Weighted hardness
        weighted_hardness = 0
        for material, fraction in composition.items():
            if fraction and fraction > 0:
                hardness = self.material_properties.get(material, {}).get('hardness', 2000)
                weighted_hardness += fraction * hardness
        
        properties["comp_weighted_hardness"] = weighted_hardness
        
        return properties
    
    def _calculate_weighted_elastic_modulus(self, composition: CompositionModel) -> float:
        """Calculate composition-weighted elastic modulus."""
        comp_dict = composition.model_dump()
        weighted_modulus = 0
        
        for material, fraction in comp_dict.items():
            if fraction and fraction > 0:
                modulus = self.material_properties.get(material, {}).get('elastic_modulus', 400)
                weighted_modulus += fraction * modulus
        
        return weighted_modulus
    
    def _calculate_weighted_thermal_conductivity(self, composition: CompositionModel) -> float:
        """Calculate composition-weighted thermal conductivity."""
        comp_dict = composition.model_dump()
        weighted_conductivity = 0
        
        for material, fraction in comp_dict.items():
            if fraction and fraction > 0:
                conductivity = self.material_properties.get(material, {}).get('thermal_conductivity', 50)
                weighted_conductivity += fraction * conductivity
        
        return weighted_conductivity
    
    def _calculate_weighted_thermal_expansion(self, composition: CompositionModel) -> float:
        """Calculate composition-weighted thermal expansion coefficient."""
        comp_dict = composition.model_dump()
        weighted_expansion = 0
        
        for material, fraction in comp_dict.items():
            if fraction and fraction > 0:
                expansion = self.material_properties.get(material, {}).get('thermal_expansion', 4.5e-6)
                weighted_expansion += fraction * expansion
        
        return weighted_expansion
    
    def _calculate_theoretical_density(self, composition: CompositionModel) -> float:
        """Calculate theoretical density based on composition."""
        comp_dict = composition.model_dump()
        theoretical_density = 0
        
        for material, fraction in comp_dict.items():
            if fraction and fraction > 0:
                density = self.material_properties.get(material, {}).get('density', 3.2)
                theoretical_density += fraction * density
        
        return theoretical_density
    
    def _scale_features(self, features: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """Apply feature scaling using stored parameters."""
        scaled_features = features.copy()
        
        for i, name in enumerate(feature_names):
            if name in self.feature_means and name in self.feature_stds:
                mean = self.feature_means[name]
                std = self.feature_stds[name]
                if std > 0:
                    scaled_features[i] = (features[i] - mean) / std
        
        return scaled_features
    
    def _initialize_feature_metadata(self) -> Dict[str, FeatureMetadata]:
        """Initialize feature metadata for documentation and validation."""
        metadata = {}
        
        # This would be expanded with comprehensive metadata for all features
        # For now, including key examples
        metadata.update({
            "comp_SiC": FeatureMetadata(
                "comp_SiC", "Silicon Carbide fraction", "fraction", "composition"
            ),
            "proc_temperature": FeatureMetadata(
                "proc_temperature", "Sintering temperature", "°C", "processing"
            ),
            "micro_porosity": FeatureMetadata(
                "micro_porosity", "Material porosity", "fraction", "microstructure"
            ),
            "derived_pugh_ratio_approx": FeatureMetadata(
                "derived_pugh_ratio_approx", "Approximate Pugh ratio (G/K)", "dimensionless", "derived"
            )
        })
        
        return metadata
    
    def _initialize_material_properties(self) -> Dict[str, Dict[str, float]]:
        """Initialize material property database."""
        return {
            'SiC': {
                'density': 3.21,  # g/cm³
                'hardness': 2800,  # HV
                'elastic_modulus': 410,  # GPa
                'thermal_conductivity': 120,  # W/m·K
                'thermal_expansion': 4.0e-6  # /K
            },
            'B4C': {
                'density': 2.52,
                'hardness': 3000,
                'elastic_modulus': 460,
                'thermal_conductivity': 30,
                'thermal_expansion': 5.6e-6
            },
            'Al2O3': {
                'density': 3.95,
                'hardness': 2000,
                'elastic_modulus': 370,
                'thermal_conductivity': 25,
                'thermal_expansion': 8.1e-6
            },
            'WC': {
                'density': 15.6,
                'hardness': 2200,
                'elastic_modulus': 700,
                'thermal_conductivity': 85,
                'thermal_expansion': 5.2e-6
            },
            'TiC': {
                'density': 4.93,
                'hardness': 2400,
                'elastic_modulus': 450,
                'thermal_conductivity': 20,
                'thermal_expansion': 7.4e-6
            },
            'ZrO2': {
                'density': 6.05,
                'hardness': 1200,
                'elastic_modulus': 200,
                'thermal_conductivity': 2,
                'thermal_expansion': 10.5e-6
            },
            'TiB2': {
                'density': 4.52,
                'hardness': 2500,
                'elastic_modulus': 370,
                'thermal_conductivity': 65,
                'thermal_expansion': 8.1e-6
            }
        }
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names that would be generated."""
        # This would return the complete list of feature names
        # For now, returning a representative subset
        return [
            # Composition features
            "comp_SiC", "comp_B4C", "comp_Al2O3", "comp_WC", "comp_TiC",
            "comp_total_ceramic", "comp_total_carbide", "comp_hard_phase",
            "comp_weighted_density", "comp_weighted_hardness",
            
            # Processing features
            "proc_temperature", "proc_pressure", "proc_grain_size",
            "proc_temperature_norm", "proc_thermal_energy",
            
            # Microstructure features
            "micro_porosity", "micro_relative_density", "micro_interface_quality",
            
            # Derived features
            "derived_pugh_ratio_approx", "derived_thermal_shock_resistance",
            "derived_microstructure_quality", "derived_multi_hit_indicator"
        ]


# Global feature engineer instance
_feature_engineer: Optional[CeramicFeatureEngineer] = None


def get_feature_engineer() -> CeramicFeatureEngineer:
    """Get global feature engineer instance."""
    global _feature_engineer
    if _feature_engineer is None:
        _feature_engineer = CeramicFeatureEngineer()
    return _feature_engineer