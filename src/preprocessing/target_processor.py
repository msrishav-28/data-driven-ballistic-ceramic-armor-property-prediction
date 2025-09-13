import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
logger = logging.getLogger(name)
class TargetProcessor:
"""
Process and prepare target variables for ceramic armor prediction
"""
def __init__(self):
    self.mechanical_targets = [
        'fracture_toughness',    # MPa·m^0.5
        'vickers_hardness',      # HV/GPa
        'density',               # g/cm³
        'elastic_modulus'        # GPa
    ]
    
    self.ballistic_targets = [
        'v50_velocity',          # m/s
        'penetration_resistance', # Categorical → numerical
        'back_face_deformation', # mm
        'multi_hit_capability'   # Binary
    ]
    
    self.target_ranges = {
        'fracture_toughness': (1.0, 10.0),
        'vickers_hardness': (1000, 4000),
        'density': (2.0, 16.0),
        'elastic_modulus': (200, 700),
        'v50_velocity': (400, 1200),
        'back_face_deformation': (0, 50)
    }

def process_targets(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Process and validate target variables
    
    Returns:
        Processed dataframe and target metadata
    """
    logger.info("Processing target variables...")
    
    # Convert categorical targets
    df = self._convert_categorical_targets(df)
    
    # Normalize target units
    df = self._normalize_units(df)
    
    # Validate target ranges
    df = self._validate_ranges(df)
    
    # Create target metadata
    metadata = self._create_metadata(df)
    
    return df, metadata

def _convert_categorical_targets(self, df: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical ballistic properties to numerical"""
    
    # Penetration resistance scoring
    if 'penetration_resistance' in df.columns:
        resistance_map = {
            'Very Low': 1, 'Low': 2, 'Medium': 3, 
            'High': 4, 'Very High': 5
        }
        df['penetration_resistance_score'] = df['penetration_resistance'].map(resistance_map)
    
    # Multi-hit capability
    if 'multi_hit_capability' in df.columns:
        if df['multi_hit_capability'].dtype == 'object':
            df['multi_hit_capability'] = df['multi_hit_capability'].map({'Yes': 1, 'No': 0})
    
    return df

def _normalize_units(self, df: pd.DataFrame) -> pd.DataFrame:
    """Ensure consistent units across all targets"""
    
    # Hardness: Convert HV to GPa if needed
    if 'vickers_hardness' in df.columns:
        # If values > 100, assume they're in HV and convert to GPa
        if df['vickers_hardness'].median() > 100:
            df['vickers_hardness'] = df['vickers_hardness'] / 100
    
    # Fracture toughness: Ensure MPa·m^0.5
    if 'fracture_toughness' in df.columns:
        # Check if values seem to be in Pa·m^0.5
        if df['fracture_toughness'].median() > 1000:
            df['fracture_toughness'] = df['fracture_toughness'] / 1e6
    
    return df

def _validate_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clip target values to reasonable ranges"""
    
    for target, (min_val, max_val) in self.target_ranges.items():
        if target in df.columns:
            outliers = (df[target] < min_val) | (df[target] > max_val)
            if outliers.any():
                logger.warning(f"Found {outliers.sum()} outliers in {target}")
                df[target] = df[target].clip(min_val, max_val)
    
    return df

def _create_metadata(self, df: pd.DataFrame) -> Dict:
    """Create metadata for target variables"""
    
    metadata = {
        'mechanical_targets': [],
        'ballistic_targets': [],
        'statistics': {}
    }
    
    # Identify available targets
    for target in self.mechanical_targets:
        if target in df.columns and df[target].notna().sum() > 10:
            metadata['mechanical_targets'].append(target)
            metadata['statistics'][target] = {
                'mean': df[target].mean(),
                'std': df[target].std(),
                'min': df[target].min(),
                'max': df[target].max(),
                'count': df[target].notna().sum()
            }
    
    for target in self.ballistic_targets:
        if target in df.columns and df[target].notna().sum() > 10:
            metadata['ballistic_targets'].append(target)
            metadata['statistics'][target] = {
                'mean': df[target].mean() if df[target].dtype != 'object' else None,
                'std': df[target].std() if df[target].dtype != 'object' else None,
                'unique': df[target].nunique(),
                'count': df[target].notna().sum()
            }
    
    logger.info(f"Found {len(metadata['mechanical_targets'])} mechanical targets")
    logger.info(f"Found {len(metadata['ballistic_targets'])} ballistic targets")
    
    return metadata
