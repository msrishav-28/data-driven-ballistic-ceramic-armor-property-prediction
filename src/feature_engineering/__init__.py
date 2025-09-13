"""
Feature Engineering Package for Ceramic Armor ML.

This package provides comprehensive feature engineering capabilities for
converting material composition, processing, and microstructure data
into ML-ready feature vectors.
"""

from .ceramic_features import (
    CeramicFeatureEngineer,
    FeatureMetadata,
    get_feature_engineer
)

__all__ = [
    'CeramicFeatureEngineer',
    'FeatureMetadata', 
    'get_feature_engineer'
]