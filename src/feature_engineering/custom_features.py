import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy import stats
class CeramicSpecificFeatures:
"""
Custom feature engineering specific to ceramic armor materials
Based on materials science domain knowledge
"""
@staticmethod
def calculate_brittleness_index(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate brittleness indices"""
    # Pugh's modulus ratio
    if 'shear_modulus_vrh' in df.columns and 'bulk_modulus_vrh' in df.columns:
        df['pugh_modulus_ratio'] = df['shear_modulus_vrh'] / df['bulk_modulus_vrh']
        
        # Classification based on Pugh's criterion
        df['pugh_ductile'] = (df['pugh_modulus_ratio'] < 0.571).astype(int)
    
    # Pettifor's Cauchy pressure criterion
    if 'cauchy_pressure' in df.columns:
        df['pettifor_brittle'] = (df['cauchy_pressure'] < 0).astype(int)
    
    return df

@staticmethod
def calculate_hardness_models(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate theoretical hardness using various models"""
    
    # Chen-Niu model
    if 'shear_modulus_vrh' in df.columns:
        g = df['shear_modulus_vrh']
        df['chen_niu_hardness'] = 2 * (g ** 0.585) - 3
    
    # Tian model
    if 'shear_modulus_vrh' in df.columns and 'bulk_modulus_vrh' in df.columns:
        g = df['shear_modulus_vrh']
        k = df['bulk_modulus_vrh']
        df['tian_hardness'] = 0.92 * ((g/k) ** 1.137) * (g ** 0.708)
    
    # Mazhnik-Oganov model
    if 'shear_modulus_vrh' in df.columns and 'bulk_modulus_vrh' in df.columns:
        g = df['shear_modulus_vrh']
        k = df['bulk_modulus_vrh']
        nu = (3*k - 2*g) / (6*k + 2*g)  # Poisson's ratio
        df['mazhnik_oganov_hardness'] = 15.76 * (g ** 0.8)
    
    return df

@staticmethod
def calculate_toughness_predictors(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate fracture toughness predictors"""
    
    # Lawn-Evans-Marshall relation
    if 'youngs_modulus' in df.columns and 'vickers_hardness' in df.columns:
        E = df['youngs_modulus']
        H = df['vickers_hardness']
        df['lem_toughness_predictor'] = 0.016 * np.sqrt(E/H) * (E ** 0.5)
    
    # Rice model
    if 'grain_size' in df.columns and 'youngs_modulus' in df.columns:
        d = df['grain_size']
        E = df['youngs_modulus']
        df['rice_toughness'] = 2 * np.sqrt(2 * E * 1 / (np.pi * d))
    
    return df

@staticmethod
def calculate_ballistic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate ballistic performance indicators"""
    
    # Acoustic impedance
    if 'density' in df.columns and 'youngs_modulus' in df.columns:
        rho = df['density']
        E = df['youngs_modulus']
        c = np.sqrt(E / rho)  # Sound velocity
        df['acoustic_impedance'] = rho * c
    
    # Hugoniot elastic limit (HEL)
    if 'yield_strength' in df.columns and 'poisson_ratio' in df.columns:
        sigma_y = df['yield_strength']
        nu = df['poisson_ratio']
        df['hel_pressure'] = sigma_y * (1 - nu) / (1 - 2*nu)
    
    # Grady-Kipp fragmentation
    if 'fracture_toughness' in df.columns and 'density' in df.columns:
        K_IC = df['fracture_toughness']
        rho = df['density']
        if 'youngs_modulus' in df.columns:
            c = np.sqrt(df['youngs_modulus'] / rho)
            df['grady_fragment_size'] = 24 * K_IC / (rho * c**2)
    
    return df
