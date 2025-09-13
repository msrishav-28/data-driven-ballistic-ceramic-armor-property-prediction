from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import joblib
import logging
logger = logging.getLogger(name)
class CeramicArmorPredictor:
"""
Multi-target prediction system for ceramic armor properties
Separate models for mechanical and ballistic properties
"""
def __init__(self, model_type: str = 'xgboost'):
    """
    Initialize predictor with specified model type
    
    Args:
        model_type: 'xgboost', 'lightgbm', 'catboost', or 'ensemble'
    """
    self.model_type = model_type
    self.models = {
        'mechanical': None,
        'ballistic': None
    }
    
    # Optimized hyperparameters for each model type
    self.model_configs = {
        'xgboost': {
            'n_estimators': 1000,
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1
        },
        'lightgbm': {
            'n_estimators': 1200,
            'max_depth': 10,
            'learning_rate': 0.08,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': -1
        },
        'catboost': {
            'iterations': 1000,
            'depth': 8,
            'learning_rate': 0.1,
            'l2_leaf_reg': 3.0,
            'random_strength': 1.0,
            'bagging_temperature': 1.0,
            'random_state': 42,
            'verbose': False
        }
    }
    
    self.feature_importance = {}
    self.training_history = {}
    
def build_models(self, mechanical_targets: int = 4, ballistic_targets: int = 4):
    """
    Build separate models for mechanical and ballistic properties
    
    Args:
        mechanical_targets: Number of mechanical properties to predict
        ballistic_targets: Number of ballistic properties to predict
    """
    logger.info(f"Building {self.model_type} models...")
    
    if self.model_type == 'xgboost':
        base_mechanical = xgb.XGBRegressor(**self.model_configs['xgboost'])
        base_ballistic = xgb.XGBRegressor(**self.model_configs['xgboost'])
        
    elif self.model_type == 'lightgbm':
        base_mechanical = lgb.LGBMRegressor(**self.model_configs['lightgbm'])
        base_ballistic = lgb.LGBMRegressor(**self.model_configs['lightgbm'])
        
    elif self.model_type == 'catboost':
        base_mechanical = cb.CatBoostRegressor(**self.model_configs['catboost'])
        base_ballistic = cb.CatBoostRegressor(**self.model_configs['catboost'])
        
    elif self.model_type == 'ensemble':
        # Use ensemble of multiple algorithms
        base_mechanical = self._create_ensemble_model()
        base_ballistic = self._create_ensemble_model()
    else:
        raise ValueError(f"Unknown model type: {self.model_type}")
    
    # Wrap in MultiOutputRegressor for multiple targets
    self.models['mechanical'] = MultiOutputRegressor(base_mechanical)
    self.models['ballistic'] = MultiOutputRegressor(base_ballistic)
    
    logger.info("Models built successfully")

def _create_ensemble_model(self):
    """Create an ensemble of multiple model types"""
    from sklearn.ensemble import VotingRegressor
    
    models = [
        ('xgb', xgb.XGBRegressor(**self.model_configs['xgboost'])),
        ('lgb', lgb.LGBMRegressor(**self.model_configs['lightgbm'])),
        ('rf', RandomForestRegressor(n_estimators=500, max_depth=10, random_state=42))
    ]
    
    return VotingRegressor(models)

def train(self, X_train: pd.DataFrame, 
         y_mechanical: pd.DataFrame, 
         y_ballistic: pd.DataFrame,
         X_val: Optional[pd.DataFrame] = None,
         y_mechanical_val: Optional[pd.DataFrame] = None,
         y_ballistic_val: Optional[pd.DataFrame] = None):
    """
    Train both mechanical and ballistic property models
    
    Args:
        X_train: Training features
        y_mechanical: Mechanical property targets
        y_ballistic: Ballistic property targets
        X_val: Optional validation features
        y_mechanical_val: Optional validation mechanical targets
        y_ballistic_val: Optional validation ballistic targets
    """
    logger.info("Training models...")
    
    # Train mechanical properties model
    if y_mechanical is not None and len(y_mechanical.columns) > 0:
        logger.info(f"Training mechanical model for {y_mechanical.columns.tolist()}")
        
        if self.model_type in ['xgboost', 'lightgbm'] and X_val is not None:
            # Use early stopping for tree-based models
            eval_set = [(X_val, y_mechanical_val)] if y_mechanical_val is not None else None
            
            # Train each target separately for early stopping
            estimators = []
            for i, col in enumerate(y_mechanical.columns):
                estimator = self.models['mechanical'].estimator.copy()
                if eval_set:
                    estimator.fit(X_train, y_mechanical.iloc[:, i],
                                eval_set=[(X_val, y_mechanical_val.iloc[:, i])],
                                early_stopping_rounds=50, verbose=False)
                else:
                    estimator.fit(X_train, y_mechanical.iloc[:, i])
                estimators.append(estimator)
            self.models['mechanical'].estimators_ = estimators
        else:
            self.models['mechanical'].fit(X_train, y_mechanical)
        
        # Store feature importance
        self._calculate_feature_importance('mechanical', X_train.columns)
    
    # Train ballistic properties model
    if y_ballistic is not None and len(y_ballistic.columns) > 0:
        logger.info(f"Training ballistic model for {y_ballistic.columns.tolist()}")
        
        if self.model_type in ['xgboost', 'lightgbm'] and X_val is not None:
            # Use early stopping for tree-based models
            eval_set = [(X_val, y_ballistic_val)] if y_ballistic_val is not None else None
            
            # Train each target separately
            estimators = []
            for i, col in enumerate(y_ballistic.columns):
                estimator = self.models['ballistic'].estimator.copy()
                if eval_set:
                    estimator.fit(X_train, y_ballistic.iloc[:, i],
                                eval_set=[(X_val, y_ballistic_val.iloc[:, i])],
                                early_stopping_rounds=50, verbose=False)
                else:
                    estimator.fit(X_train, y_ballistic.iloc[:, i])
                estimators.append(estimator)
            self.models['ballistic'].estimators_ = estimators
        else:
            self.models['ballistic'].fit(X_train, y_ballistic)
        
        # Store feature importance
        self._calculate_feature_importance('ballistic', X_train.columns)
    
    logger.info("Training complete")

def predict(self, X: pd.DataFrame, model_type: str = 'both') -> Dict[str, np.ndarray]:
    """
    Generate predictions for mechanical and/or ballistic properties
    
    Args:
        X: Feature matrix
        model_type: 'mechanical', 'ballistic', or 'both'
        
    Returns:
        Dictionary with predictions for each model type
    """
    predictions = {}
    
    if model_type in ['mechanical', 'both'] and self.models['mechanical'] is not None:
        predictions['mechanical'] = self.models['mechanical'].predict(X)
    
    if model_type in ['ballistic', 'both'] and self.models['ballistic'] is not None:
        predictions['ballistic'] = self.models['ballistic'].predict(X)
    
    return predictions

def predict_with_uncertainty(self, X: pd.DataFrame, n_iterations: int = 100) -> Dict:
    """
    Generate predictions with uncertainty quantification
    Using dropout or bootstrap approaches
    """
    predictions = {'mechanical': [], 'ballistic': []}
    
    for _ in range(n_iterations):
        # Add noise to features for uncertainty estimation
        X_noisy = X + np.random.normal(0, 0.01, X.shape)
        
        preds = self.predict(X_noisy)
        for key in preds:
            predictions[key].append(preds[key])
    
    # Calculate mean and std
    results = {}
    for key in predictions:
        if predictions[key]:
            preds_array = np.array(predictions[key])
            results[key] = {
                'mean': preds_array.mean(axis=0),
                'std': preds_array.std(axis=0),
                'lower_95': np.percentile(preds_array, 2.5, axis=0),
                'upper_95': np.percentile(preds_array, 97.5, axis=0)
            }
    
    return results

def _calculate_feature_importance(self, model_name: str, feature_names: List[str]):
    """Calculate and store feature importance"""
    try:
        if hasattr(self.models[model_name], 'estimators_'):
            # Average importance across all estimators
            importances = []
            for estimator in self.models[model_name].estimators_:
                if hasattr(estimator, 'feature_importances_'):
                    importances.append(estimator.feature_importances_)
            
            if importances:
                avg_importance = np.mean(importances, axis=0)
                self.feature_importance[model_name] = dict(zip(feature_names, avg_importance))
                
                # Sort by importance
                self.feature_importance[model_name] = dict(
                    sorted(self.feature_importance[model_name].items(), 
                          key=lambda x: x[1], reverse=True)
                )
    except Exception as e:
        logger.warning(f"Could not calculate feature importance: {str(e)}")

def save_models(self, path: str):
    """Save trained models to disk"""
    model_data = {
        'models': self.models,
        'model_type': self.model_type,
        'feature_importance': self.feature_importance,
        'training_history': self.training_history
    }
    joblib.dump(model_data, path)
    logger.info(f"Models saved to {path}")

def load_models(self, path: str):
    """Load trained models from disk"""
    model_data = joblib.load(path)
    self.models = model_data['models']
    self.model_type = model_data['model_type']
    self.feature_importance = model_data.get('feature_importance', {})
    self.training_history = model_data.get('training_history', {})
    logger.info(f"Models loaded from {path}")