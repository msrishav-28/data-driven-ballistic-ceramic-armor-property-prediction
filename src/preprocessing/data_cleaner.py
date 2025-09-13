import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import KNNImputer, SimpleImputer
import logging
from typing import Tuple, List, Optional
logger = logging.getLogger(name)
class DataCleaner:
"""
Comprehensive data cleaning and preprocessing for ceramic materials data
"""
def __init__(self):
    self.scaler = RobustScaler()
    self.imputer = KNNImputer(n_neighbors=5)
    self.feature_columns = []
    self.target_columns = []
    
def clean_and_preprocess(self, df: pd.DataFrame, 
                        target_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Complete data cleaning pipeline
    
    Args:
        df: Raw dataframe
        target_cols: List of target column names
        
    Returns:
        Cleaned and preprocessed dataframe
    """
    logger.info(f"Starting data cleaning for {len(df)} samples...")
    
    # Remove duplicates
    df = self._remove_duplicates(df)
    
    # Handle missing values
    df = self._handle_missing_values(df, target_cols)
    
    # Remove outliers
    df = self._remove_outliers(df, target_cols)
    
    # Normalize features
    df = self._normalize_features(df, target_cols)
    
    # Encode categorical variables
    df = self._encode_categorical(df)
    
    logger.info(f"Cleaning complete. Final dataset: {len(df)} samples")
    
    return df

def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate entries based on material ID or composition"""
    initial_size = len(df)
    
    # Remove exact duplicates
    df = df.drop_duplicates()
    
    # Remove duplicates based on key identifiers
    if 'material_id' in df.columns:
        df = df.drop_duplicates(subset=['material_id'], keep='first')
    
    removed = initial_size - len(df)
    if removed > 0:
        logger.info(f"Removed {removed} duplicate entries")
    
    return df

def _handle_missing_values(self, df: pd.DataFrame, 
                          target_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Handle missing values with appropriate strategies"""
    
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Separate features and targets
    if target_cols:
        feature_cols = [col for col in numeric_cols if col not in target_cols]
    else:
        feature_cols = numeric_cols
    
    # Check missing value percentages
    missing_pct = df[numeric_cols].isnull().sum() / len(df) * 100
    high_missing = missing_pct[missing_pct > 50].index.tolist()
    
    if high_missing:
        logger.warning(f"Dropping columns with >50% missing: {high_missing}")
        df = df.drop(columns=high_missing)
        feature_cols = [col for col in feature_cols if col not in high_missing]
    
    # Impute remaining missing values
    if feature_cols:
        # Use KNN imputation for features
        df[feature_cols] = self.imputer.fit_transform(df[feature_cols])
    
    # Simple imputation for targets (if any missing)
    if target_cols:
        for col in target_cols:
            if col in df.columns and df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
    
    return df

def _remove_outliers(self, df: pd.DataFrame, 
                    target_cols: Optional[List[str]] = None,
                    threshold: float = 3.5) -> pd.DataFrame:
    """Remove outliers using modified Z-score"""
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Don't remove outliers from target columns
    if target_cols:
        check_cols = [col for col in numeric_cols if col not in target_cols]
    else:
        check_cols = numeric_cols
    
    # Calculate modified Z-scores
    for col in check_cols:
        median = df[col].median()
        mad = np.median(np.abs(df[col] - median))
        
        if mad > 0:
            modified_z = 0.6745 * (df[col] - median) / mad
            outliers = np.abs(modified_z) > threshold
            
            n_outliers = outliers.sum()
            if n_outliers > 0 and n_outliers < len(df) * 0.1:  # Remove if <10% are outliers
                df = df[~outliers]
                logger.info(f"Removed {n_outliers} outliers from {col}")
    
    return df

def _normalize_features(self, df: pd.DataFrame, 
                       target_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Normalize feature values using RobustScaler"""
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Don't normalize targets or identifiers
    exclude_cols = ['material_id', 'nsites', 'space_group']
    if target_cols:
        exclude_cols.extend(target_cols)
    
    normalize_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if normalize_cols:
        df[normalize_cols] = self.scaler.fit_transform(df[normalize_cols])
        self.feature_columns = normalize_cols
    
    return df

def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical variables"""
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Skip certain columns
    skip_cols = ['material_id', 'formula', 'composition', 'structure', 'reference']
    encode_cols = [col for col in categorical_cols if col not in skip_cols]
    
    for col in encode_cols:
        if df[col].nunique() < 20:  # Only encode if reasonable number of categories
            # Use one-hot encoding
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[col])
            logger.info(f"Encoded categorical column: {col}")
    
    return df
