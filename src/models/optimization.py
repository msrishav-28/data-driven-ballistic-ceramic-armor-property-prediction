import optuna
from optuna.samplers import TPESampler
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging
logger = logging.getLogger(name)
class HyperparameterOptimizer:
"""
Bayesian hyperparameter optimization for ceramic property prediction models
"""
def __init__(self, model_type: str = 'xgboost', random_state: int = 42):
    """
    Initialize optimizer
    
    Args:
        model_type: Type of model to optimize ('xgboost', 'lightgbm', 'catboost')
        random_state: Random seed for reproducibility
    """
    self.model_type = model_type
    self.random_state = random_state
    self.best_params = {}
    self.optimization_history = []
    
def optimize_mechanical_model(self, X: pd.DataFrame, y: pd.DataFrame, 
                             n_trials: int = 200,
                             target_r2: float = 0.85) -> Dict:
    """
    Optimize hyperparameters for mechanical property prediction
    Target: R² > 0.85
    
    Args:
        X: Feature matrix
        y: Target matrix (multiple mechanical properties)
        n_trials: Number of optimization trials
        target_r2: Target R² score
        
    Returns:
        Best hyperparameters found
    """
    logger.info(f"Starting hyperparameter optimization for mechanical properties...")
    logger.info(f"Target R² score: {target_r2}")
    
    def objective(trial):
        # Suggest hyperparameters based on model type
        params = self._suggest_params(trial, self.model_type)
        
        # Create model
        if self.model_type == 'xgboost':
            model = xgb.XGBRegressor(**params)
        elif self.model_type == 'lightgbm':
            model = lgb.LGBMRegressor(**params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Perform cross-validation
        kfold = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        # For multi-output, evaluate average R² across all targets
        r2_scores = []
        for i in range(y.shape[1] if len(y.shape) > 1 else 1):
            y_target = y.iloc[:, i] if len(y.shape) > 1 else y
            scores = cross_val_score(model, X, y_target, cv=kfold, scoring='r2')
            r2_scores.append(scores.mean())
        
        avg_r2 = np.mean(r2_scores)
        
        # Log progress
        if trial.number % 20 == 0:
            logger.info(f"Trial {trial.number}: Average R² = {avg_r2:.4f}")
        
        return avg_r2
    
    # Create study with TPE sampler for efficient search
    sampler = TPESampler(seed=self.random_state)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        study_name='mechanical_properties_optimization'
    )
    
    # Optimize
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Store results
    self.best_params['mechanical'] = study.best_params
    self.optimization_history.append({
        'model': 'mechanical',
        'best_value': study.best_value,
        'n_trials': n_trials,
        'best_params': study.best_params
    })
    
    logger.info(f"Optimization complete. Best R² score: {study.best_value:.4f}")
    
    if study.best_value < target_r2:
        logger.warning(f"Target R² of {target_r2} not achieved. Best: {study.best_value:.4f}")
    else:
        logger.info(f"Target R² achieved! Best score: {study.best_value:.4f}")
    
    return study.best_params

def optimize_ballistic_model(self, X: pd.DataFrame, y: pd.DataFrame,
                            n_trials: int = 200,
                            target_r2: float = 0.80) -> Dict:
    """
    Optimize hyperparameters for ballistic property prediction
    Target: R² > 0.80
    
    Args:
        X: Feature matrix
        y: Target matrix (multiple ballistic properties)
        n_trials: Number of optimization trials
        target_r2: Target R² score
        
    Returns:
        Best hyperparameters found
    """
    logger.info(f"Starting hyperparameter optimization for ballistic properties...")
    logger.info(f"Target R² score: {target_r2}")
    
    def objective(trial):
        # Suggest hyperparameters
        params = self._suggest_params(trial, self.model_type)
        
        # Create model
        if self.model_type == 'xgboost':
            model = xgb.XGBRegressor(**params)
        elif self.model_type == 'lightgbm':
            model = lgb.LGBMRegressor(**params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Cross-validation
        kfold = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        # Evaluate across all ballistic targets
        r2_scores = []
        for i in range(y.shape[1] if len(y.shape) > 1 else 1):
            y_target = y.iloc[:, i] if len(y.shape) > 1 else y
            
            # Skip if target has too few samples
            if y_target.notna().sum() < 50:
                continue
            
            scores = cross_val_score(model, X[y_target.notna()], 
                                    y_target[y_target.notna()], 
                                    cv=kfold, scoring='r2')
            r2_scores.append(scores.mean())
        
        avg_r2 = np.mean(r2_scores) if r2_scores else 0
        
        return avg_r2
    
    # Create and run study
    sampler = TPESampler(seed=self.random_state)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        study_name='ballistic_properties_optimization'
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Store results
    self.best_params['ballistic'] = study.best_params
    self.optimization_history.append({
        'model': 'ballistic',
        'best_value': study.best_value,
        'n_trials': n_trials,
        'best_params': study.best_params
    })
    
    logger.info(f"Optimization complete. Best R² score: {study.best_value:.4f}")
    
    if study.best_value < target_r2:
        logger.warning(f"Target R² of {target_r2} not achieved. Best: {study.best_value:.4f}")
    else:
        logger.info(f"Target R² achieved! Best score: {study.best_value:.4f}")
    
    return study.best_params

def _suggest_params(self, trial, model_type: str) -> Dict:
    """Suggest hyperparameters for given model type"""
    
    if model_type == 'xgboost':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 2.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 2.0),
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbosity': 0
        }
    
    elif model_type == 'lightgbm':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 500, 2500),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 2.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 2.0),
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbosity': -1
        }
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
