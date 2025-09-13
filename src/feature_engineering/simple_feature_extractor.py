"""
Simplified feature extraction for ceramic materials.
Basic feature extraction without external dependencies for testing.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class CeramicFeatureExtractor:
    """
    Simplified feature extraction for ceramic materials
    Generates basic features from composition, processing, and microstructure data
    """
    
    def __init__(self):
        """Initialize feature extractor"""
        self.feature_columns = []
        
    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract basic feature set from materials data
        
        Args:
            df: DataFrame with material data
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting basic feature extraction...")
        
        # Add composition-based features
        df = self._add_composition_features(df)
        
        # Add processing-based features
        df = self._add_processing_features(df)
        
        # Add microstructure-based features
        df = self._add_microstructure_features(df)
        
        # Add derived features
        df = self._add_derived_features(df)
        
        logger.info(f"Extracted {len(self.feature_columns)} total features")
        
        return df

    def _add_composition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add composition-based features"""
        logger.debug("Adding composition features...")
        
        # Extract composition fractions
        comp_cols = [col for col in df.columns if col.startswith('comp_')]
        for col in comp_cols:
            if col in df.columns:
                self.feature_columns.append(col)
        
        # Add total composition check
        if comp_cols:
            df['total_composition'] = df[comp_cols].sum(axis=1)
            self.feature_columns.append('total_composition')
        
        # Add ceramic content (SiC + B4C + Al2O3)
        ceramic_cols = ['comp_SiC', 'comp_B4C', 'comp_Al2O3']
        available_ceramic_cols = [col for col in ceramic_cols if col in df.columns]
        if available_ceramic_cols:
            df['ceramic_content'] = df[available_ceramic_cols].sum(axis=1)
            self.feature_columns.append('ceramic_content')
        
        # Add carbide content (SiC + B4C + WC + TiC)
        carbide_cols = ['comp_SiC', 'comp_B4C', 'comp_WC', 'comp_TiC']
        available_carbide_cols = [col for col in carbide_cols if col in df.columns]
        if available_carbide_cols:
            df['carbide_content'] = df[available_carbide_cols].sum(axis=1)
            self.feature_columns.append('carbide_content')
        
        return df

    def _add_processing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add processing-based features"""
        logger.debug("Adding processing features...")
        
        # Extract processing parameters
        proc_cols = [col for col in df.columns if col.startswith('proc_')]
        for col in proc_cols:
            if col in df.columns:
                self.feature_columns.append(col)
        
        # Add normalized temperature (relative to melting point)
        if 'proc_sintering_temperature' in df.columns:
            # Assume typical ceramic melting points around 2000-3000°C
            df['normalized_temperature'] = df['proc_sintering_temperature'] / 2500.0
            self.feature_columns.append('normalized_temperature')
        
        # Add pressure-temperature interaction
        if 'proc_pressure' in df.columns and 'proc_sintering_temperature' in df.columns:
            df['pressure_temp_product'] = df['proc_pressure'] * df['proc_sintering_temperature']
            self.feature_columns.append('pressure_temp_product')
        
        # Add grain size categories
        if 'proc_grain_size' in df.columns:
            df['fine_grain'] = (df['proc_grain_size'] < 1.0).astype(int)
            df['coarse_grain'] = (df['proc_grain_size'] > 10.0).astype(int)
            self.feature_columns.extend(['fine_grain', 'coarse_grain'])
        
        return df

    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add microstructure-based features"""
        logger.debug("Adding microstructure features...")
        
        # Extract microstructure parameters
        micro_cols = [col for col in df.columns if col.startswith('micro_')]
        for col in micro_cols:
            if col in df.columns:
                self.feature_columns.append(col)
        
        # Add porosity categories
        if 'micro_porosity' in df.columns:
            df['low_porosity'] = (df['micro_porosity'] < 0.02).astype(int)
            df['high_porosity'] = (df['micro_porosity'] > 0.1).astype(int)
            self.feature_columns.extend(['low_porosity', 'high_porosity'])
        
        # Add phase distribution encoding
        if 'micro_phase_distribution' in df.columns:
            df['uniform_phase'] = (df['micro_phase_distribution'] == 'uniform').astype(int)
            df['gradient_phase'] = (df['micro_phase_distribution'] == 'gradient').astype(int)
            df['layered_phase'] = (df['micro_phase_distribution'] == 'layered').astype(int)
            self.feature_columns.extend(['uniform_phase', 'gradient_phase', 'layered_phase'])
        
        # Add interface quality encoding
        if 'micro_interface_quality' in df.columns:
            quality_map = {'poor': 1, 'fair': 2, 'good': 3, 'excellent': 4}
            df['interface_quality_numeric'] = df['micro_interface_quality'].map(quality_map).fillna(3)
            self.feature_columns.append('interface_quality_numeric')
        
        return df

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features based on materials science principles"""
        logger.debug("Adding derived features...")
        
        # Density estimation based on composition
        if 'comp_SiC' in df.columns and 'comp_B4C' in df.columns and 'comp_Al2O3' in df.columns:
            # Approximate densities: SiC=3.2, B4C=2.5, Al2O3=4.0 g/cm³
            estimated_density = (
                df['comp_SiC'] * 3.2 + 
                df['comp_B4C'] * 2.5 + 
                df['comp_Al2O3'] * 4.0
            )
            df['estimated_density'] = estimated_density
            self.feature_columns.append('estimated_density')
        
        # Hardness indicator based on composition
        if 'comp_SiC' in df.columns and 'comp_B4C' in df.columns:
            # SiC and B4C are very hard materials
            df['hardness_indicator'] = df['comp_SiC'] * 0.8 + df['comp_B4C'] * 0.9
            self.feature_columns.append('hardness_indicator')
        
        # Toughness indicator (inverse relationship with hardness for ceramics)
        if 'hardness_indicator' in df.columns:
            df['toughness_indicator'] = 1.0 - df['hardness_indicator']
            self.feature_columns.append('toughness_indicator')
        
        # Processing efficiency indicator
        if 'proc_sintering_temperature' in df.columns and 'proc_pressure' in df.columns:
            # Higher temperature and pressure generally improve densification
            df['processing_efficiency'] = (
                (df['proc_sintering_temperature'] / 2000.0) * 0.7 + 
                (df['proc_pressure'] / 100.0) * 0.3
            )
            self.feature_columns.append('processing_efficiency')
        
        # Microstructure quality indicator
        if 'micro_porosity' in df.columns and 'interface_quality_numeric' in df.columns:
            df['microstructure_quality'] = (
                (1.0 - df['micro_porosity']) * 0.6 + 
                (df['interface_quality_numeric'] / 4.0) * 0.4
            )
            self.feature_columns.append('microstructure_quality')
        
        return df