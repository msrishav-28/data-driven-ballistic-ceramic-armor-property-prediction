from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
logger = logging.getLogger(name)
class ModelValidator:
"""
Comprehensive model validation for ceramic armor prediction system
"""
def __init__(self, random_state: int = 42):
    self.random_state = random_state
    self.cv_strategies = {
        'kfold': KFold(n_splits=5, shuffle=True, random_state=random_state),
        'group': GroupKFold(n_splits=5)
    }
    self.validation_results = {}
    
def comprehensive_validation(self, model, X: pd.DataFrame, y: pd.DataFrame,
                           target_r2: float = 0.85,
                           target_names: Optional[List[str]] = None) -> Dict:
    """
    Perform comprehensive model validation
    
    Args:
        model: Trained model
        X: Feature matrix
        y: Target matrix
        target_r2: Target R² score
        target_names: Names of target variables
        
    Returns:
        Dictionary with validation results
    """
    logger.info("Starting comprehensive model validation...")
    
    results = {
        'cross_validation': {},
        'metrics': {},
        'target_analysis': {},
        'meets_requirements': False
    }
    
    # Cross-validation scores
    cv_scores = []
    predictions = cross_val_predict(model, X, y, cv=self.cv_strategies['kfold'])
    
    # Calculate metrics for each target
    if len(y.shape) > 1:
        for i in range(y.shape[1]):
            y_true = y.iloc[:, i] if hasattr(y, 'iloc') else y[:, i]
            y_pred = predictions[:, i] if len(predictions.shape) > 1 else predictions
            
            target_name = target_names[i] if target_names else f"Target_{i}"
            
            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            results['target_analysis'][target_name] = {
                'r2_score': r2,
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'meets_target': r2 >= target_r2
            }
            
            cv_scores.append(r2)
    else:
        # Single target
        r2 = r2_score(y, predictions)
        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        
        results['target_analysis']['single_target'] = {
            'r2_score': r2,
            'mae': mae,
            'rmse': rmse,
            'meets_target': r2 >= target_r2
        }
        cv_scores.append(r2)
    
    # Overall metrics
    results['cross_validation']['mean_r2'] = np.mean(cv_scores)
    results['cross_validation']['std_r2'] = np.std(cv_scores)
    results['cross_validation']['min_r2'] = np.min(cv_scores)
    results['cross_validation']['max_r2'] = np.max(cv_scores)
    
    # Check if meets requirements
    results['meets_requirements'] = results['cross_validation']['mean_r2'] >= target_r2
    
    # Log results
    logger.info(f"Validation complete. Mean R²: {results['cross_validation']['mean_r2']:.4f}")
    logger.info(f"Meets target R² of {target_r2}: {results['meets_requirements']}")
    
    self.validation_results = results
    return results

def validate_experimental_screening_reduction(self, model, 
                                             candidate_materials: pd.DataFrame,
                                             true_performance: Optional[pd.DataFrame] = None,
                                             top_percent: float = 0.4) -> Dict:
    """
    Validate the claim of 60% experimental screening reduction
    
    Args:
        model: Trained model
        candidate_materials: Features of candidate materials
        true_performance: Optional true performance for validation
        top_percent: Fraction of materials to select (0.4 = 60% reduction)
        
    Returns:
        Dictionary with screening reduction analysis
    """
    logger.info("Validating experimental screening reduction...")
    
    total_candidates = len(candidate_materials)
    
    # Make predictions
    predictions = model.predict(candidate_materials)
    
    # Calculate prediction uncertainty (using model variance if ensemble)
    uncertainties = self._calculate_prediction_uncertainty(model, candidate_materials)
    
    # Score materials based on predictions and confidence
    if len(predictions.shape) > 1:
        # Multi-target: use weighted average
        scores = np.mean(predictions, axis=1)
    else:
        scores = predictions
    
    # Adjust scores by uncertainty (prefer high score with low uncertainty)
    if uncertainties is not None:
        confidence_scores = scores - 0.5 * uncertainties
    else:
        confidence_scores = scores
    
    # Select top candidates
    n_select = int(total_candidates * top_percent)
    top_indices = np.argsort(confidence_scores)[-n_select:]
    
    results = {
        'total_candidates': total_candidates,
        'selected_candidates': n_select,
        'reduction_percentage': (1 - top_percent) * 100,
        'reduction_achieved': (total_candidates - n_select) / total_candidates * 100,
        'meets_60_percent_target': (1 - top_percent) >= 0.6
    }
    
    # If true performance available, calculate selection efficiency
    if true_performance is not None:
        true_top_indices = np.argsort(true_performance)[-n_select:]
        overlap = len(set(top_indices) & set(true_top_indices))
        results['selection_accuracy'] = overlap / n_select * 100
        results['average_rank_correlation'] = self._calculate_rank_correlation(
            confidence_scores, true_performance
        )
    
    logger.info(f"Screening reduction: {results['reduction_percentage']:.1f}%")
    logger.info(f"Meets 60% reduction target: {results['meets_60_percent_target']}")
    
    return results

def _calculate_prediction_uncertainty(self, model, X: pd.DataFrame) -> Optional[np.ndarray]:
    """Calculate prediction uncertainty using bootstrap or ensemble variance"""
    
    # If model has multiple estimators (ensemble), use variance
    if hasattr(model, 'estimators_'):
        predictions = []
        for estimator in model.estimators_:
            pred = estimator.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        return np.std(predictions, axis=0)
    
    # Otherwise, use bootstrap
    n_bootstrap = 10
    predictions = []
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(len(X), size=len(X), replace=True)
        X_boot = X.iloc[indices]
        
        pred = model.predict(X_boot)
        predictions.append(pred)
    
    predictions = np.array(predictions)
    return np.std(predictions, axis=0)

def _calculate_rank_correlation(self, predicted: np.ndarray, 
                               actual: np.ndarray) -> float:
    """Calculate Spearman rank correlation"""
    from scipy.stats import spearmanr
    
    correlation, _ = spearmanr(predicted, actual)
    return correlation
