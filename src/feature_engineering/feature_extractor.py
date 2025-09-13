"""
Comprehensive feature extraction for ceramic materials.
Generates 150+ features using matminer and custom descriptors.
"""

from matminer.featurizers.composition import (
    ElementProperty, Stoichiometry, ValenceOrbital,
    IonProperty, ElementFraction, TMetalFraction,
    BandCenter, AtomicOrbitals
)
from matminer.featurizers.structure import (
    SiteStatsFingerprint, DensityFeatures,
    StructuralHeterogeneity, ChemicalOrdering,
    MaximumPackingEfficiency, RadialDistributionFunction
)
from pymatgen.core import Composition, Structure
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class CeramicFeatureExtractor:
    """
    Comprehensive feature extraction for ceramic materials
    Generates 150+ features using matminer and custom descriptors
    """
    
    def __init__(self):
        """Initialize featurizers"""
        # Composition-based featurizers (90+ features)
        self.composition_featurizers = [
            ElementProperty.from_preset('magpie'),  # 132 features
            Stoichiometry(),  # 7 features
            ValenceOrbital(),  # 18 features
            IonProperty(),  # 12 features
            ElementFraction(),  # Element fractions
            TMetalFraction(),  # Transition metal fraction
            BandCenter(),  # Band center features
        ]
        
        # Structure-based featurizers (60+ features)
        self.structure_featurizers = [
            DensityFeatures(),  # 3 features
            StructuralHeterogeneity(),  # 6 features
            ChemicalOrdering(),  # 3 features
            MaximumPackingEfficiency(),  # 3 features
            SiteStatsFingerprint.from_preset('CoordinationNumber_ward-prb-2017'),  # 22 features
            RadialDistributionFunction(cutoff=10.0, bin_size=0.1),  # 100 features
        ]
        
        self.feature_columns = []
        
    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract comprehensive feature set from materials data
        
        Args:
            df: DataFrame with 'composition' and optionally 'structure' columns
            
        Returns:
            DataFrame with 150+ engineered features
        """
        logger.info("Starting comprehensive feature extraction...")
        
        # Extract composition features
        if 'composition' in df.columns:
            df = self._extract_composition_features(df)
        
        # Extract structure features if available
        if 'structure' in df.columns:
            df = self._extract_structure_features(df)
        
        # Add custom ceramic-specific features
        df = self._add_custom_ceramic_features(df)
        
        # Add interaction features
        df = self._add_interaction_features(df)
        
        logger.info(f"Extracted {len(self.feature_columns)} total features")
        
        return df

    def _extract_composition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract composition-based features"""
        logger.info("Extracting composition features...")
        
        # Ensure composition column is Composition objects
        if df['composition'].dtype == 'object' and isinstance(df['composition'].iloc[0], str):
            df['composition'] = df['composition'].apply(Composition)
        
        for featurizer in tqdm(self.composition_featurizers, desc="Composition featurizers"):
            try:
                df = featurizer.featurize_dataframe(df, 'composition', ignore_errors=True)
                # Track new feature columns
                new_cols = [col for col in df.columns if col not in self.feature_columns]
                self.feature_columns.extend(new_cols)
            except Exception as e:
                logger.warning(f"Error with featurizer {featurizer.__class__.__name__}: {str(e)}")
                continue
        
        return df

    def _extract_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract structure-based features"""
        logger.info("Extracting structure features...")
        
        # Filter rows with valid structures
        valid_structures = df['structure'].notna()
        
        if valid_structures.sum() == 0:
            logger.warning("No valid structures found for feature extraction")
            return df
        
        for featurizer in tqdm(self.structure_featurizers, desc="Structure featurizers"):
            try:
                # Only featurize rows with valid structures
                df_valid = df[valid_structures].copy()
                df_valid = featurizer.featurize_dataframe(df_valid, 'structure', ignore_errors=True)
                
                # Merge back with original dataframe
                new_cols = [col for col in df_valid.columns if col not in df.columns]
                for col in new_cols:
                    df[col] = np.nan
                    df.loc[valid_structures, col] = df_valid[col]
                
                self.feature_columns.extend(new_cols)
                
            except Exception as e:
                logger.warning(f"Error with featurizer {featurizer.__class__.__name__}: {str(e)}")
                continue
        
        return df

    def _add_custom_ceramic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add domain-specific features for ceramic armor materials"""
        logger.info("Adding custom ceramic features...")
        
        # Pugh's ratio (G/K) - brittleness indicator
        if 'shear_modulus_vrh' in df.columns and 'bulk_modulus_vrh' in df.columns:
            df['pugh_ratio'] = df['shear_modulus_vrh'] / df['bulk_modulus_vrh']
            df['inverse_pugh_ratio'] = df['bulk_modulus_vrh'] / df['shear_modulus_vrh']
            self.feature_columns.extend(['pugh_ratio', 'inverse_pugh_ratio'])
        
        # Cauchy pressure - bonding character
        if 'cauchy_pressure' not in df.columns:
            df['cauchy_pressure'] = np.nan  # Will be calculated if elastic tensor available
        
        # Ceramic hardness predictors
        if 'shear_modulus_vrh' in df.columns and 'bulk_modulus_vrh' in df.columns:
            # Chen's model for hardness
            df['chen_hardness'] = 2 * (df['shear_modulus_vrh'] ** 0.585) - 3
            
            # Tian's model for hardness
            k = df['bulk_modulus_vrh']
            g = df['shear_modulus_vrh']
            df['tian_hardness'] = 0.92 * (g / k) ** 1.137 * g ** 0.708
            
            self.feature_columns.extend(['chen_hardness', 'tian_hardness'])
        
        # Density-normalized properties
        if 'density' in df.columns:
            for prop in ['bulk_modulus_vrh', 'shear_modulus_vrh', 'youngs_modulus']:
                if prop in df.columns:
                    df[f'{prop}_per_density'] = df[prop] / df['density']
                    self.feature_columns.append(f'{prop}_per_density')
        
        # Average atomic volume
        if 'volume' in df.columns and 'nsites' in df.columns:
            df['avg_atomic_volume'] = df['volume'] / df['nsites']
            self.feature_columns.append('avg_atomic_volume')
        
        # Fracture toughness predictors
        if 'youngs_modulus' in df.columns and 'vickers_hardness' in df.columns:
            # Lawn-Evans-Marshall model
            df['lem_toughness'] = 0.016 * np.sqrt(df['youngs_modulus'] / df['vickers_hardness'])
            self.feature_columns.append('lem_toughness')
        
        # Ballistic performance indicators
        if 'density' in df.columns and 'youngs_modulus' in df.columns:
            # Acoustic impedance (related to shock wave propagation)
            df['acoustic_impedance'] = df['density'] * np.sqrt(df['youngs_modulus'] / df['density'])
            self.feature_columns.append('acoustic_impedance')
        
        # Material indices for armor applications
        if 'youngs_modulus' in df.columns and 'density' in df.columns:
            # Specific stiffness
            df['specific_stiffness'] = df['youngs_modulus'] / df['density']
            # Material index for minimum weight design
            df['weight_index'] = df['youngs_modulus'] ** 0.5 / df['density']
            self.feature_columns.extend(['specific_stiffness', 'weight_index'])
        
        return df

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between key properties"""
        logger.info("Adding interaction features...")
        
        # Key properties for interactions
        key_props = ['bulk_modulus_vrh', 'shear_modulus_vrh', 'density', 
                    'formation_energy', 'band_gap']
        
        # Create polynomial features for key properties
        for i, prop1 in enumerate(key_props):
            if prop1 not in df.columns:
                continue
            for prop2 in key_props[i+1:]:
                if prop2 not in df.columns:
                    continue
                
                # Multiplication interaction
                df[f'{prop1}_x_{prop2}'] = df[prop1] * df[prop2]
                # Ratio interaction
                df[f'{prop1}_div_{prop2}'] = df[prop1] / (df[prop2] + 1e-10)
                
                self.feature_columns.extend([f'{prop1}_x_{prop2}', f'{prop1}_div_{prop2}'])
        
        return df