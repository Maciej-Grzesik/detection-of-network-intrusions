"""
Hyperparameter tuning utilities for HybridIntrusionEnsemble.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score
)
from .hybrid_ensemble import HybridIntrusionEnsemble


def grid_search_weights(
    rf_model,
    ae_model,
    X_train,
    y_train,
    X_val,
    y_val,
    rf_weights=None,
    thresholds=None,
    metric='f1',
    device=None,
    verbose=False
):
    """
    Perform grid search over RF weights and decision thresholds.
    
    Args:
        rf_model: Trained Random Forest model
        ae_model: Trained Autoencoder model
        X_train: Training data for ensemble calibration (normal samples)
        y_train: Training labels (can be None if X_train only contains normal samples)
        X_val: Validation data for evaluation
        y_val: Validation labels
        rf_weights: List of RF weights to try (default: [0.0, 0.1, ..., 1.0])
        thresholds: List of decision thresholds to try (default: [0.3, 0.4, 0.5, 0.6])
        metric: Metric to optimize ('accuracy', 'precision', 'recall', 'f1', 'roc_auc')
        device: Device for PyTorch (default: auto-detect)
        verbose: Whether to print progress
        
    Returns:
        dict: {
            'best_config': {'rf_weight': float, 'threshold': float, 'score': float},
            'all_results': list of all configurations and their scores
        }
    """
    if rf_weights is None:
        rf_weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6]
    
    all_results = []
    best_score = -1
    best_config = None
    
    total_configs = len(rf_weights) * len(thresholds)
    current = 0
    
    for rf_weight in rf_weights:
        ae_weight = 1.0 - rf_weight
        
        for threshold in thresholds:
            current += 1
            
            # Create ensemble with current configuration
            ensemble = HybridIntrusionEnsemble(
                rf_model=rf_model,
                ae_model=ae_model,
                rf_weight=rf_weight,
                ae_weight=ae_weight,
                threshold=threshold,
                device=device
            )
            
            # Calibrate ensemble on training data (normal samples)
            ensemble.fit(X_train)
            
            # Evaluate on validation data
            y_pred = ensemble.predict(X_val)
            y_proba = ensemble.predict_proba(X_val)
            
            # Calculate metrics
            metrics = {
                'rf_weight': rf_weight,
                'ae_weight': ae_weight,
                'threshold': threshold,
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred, zero_division=0),
                'recall': recall_score(y_val, y_pred, zero_division=0),
                'f1_score': f1_score(y_val, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_val, y_proba[:, 1])
            }
            
            all_results.append(metrics)
            
            # Track best configuration
            # Map metric name to dictionary key
            metric_key = 'f1_score' if metric == 'f1' else metric
            score = metrics[metric_key]
            if score > best_score:
                best_score = score
                best_config = {
                    'rf_weight': rf_weight,
                    'ae_weight': ae_weight,
                    'threshold': threshold,
                    'score': score
                }
            
            if verbose:
                print(f"[{current}/{total_configs}] "
                      f"rf_weight={rf_weight:.1f}, threshold={threshold:.1f} → "
                      f"{metric}={score:.4f}")
    
    if verbose:
        print(f"\n✓ Best configuration: {best_config}")
    
    return {
        'best_config': best_config,
        'all_results': all_results
    }
