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
    rf_model, ae_model, X_train, y_train, X_val, y_val,
    rf_weights=None, thresholds=None, metric='f1', device=None, verbose=False
):
    if rf_weights is None:
        rf_weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6]
    
    # Convert multiclass labels to binary (Normal=0, Attack=1) for metrics calculation
    y_val_binary = (y_val != 0).astype(int)
    y_train_binary = (y_train != 0).astype(int)
        
    # 1. Pre-compute everything (The major optimization)
    if verbose: print("Pre-computing model predictions...")
    
    # We just need a dummy ensemble to handle the AE inference/calibration logic
    base_ensemble = HybridIntrusionEnsemble(rf_model, ae_model, device=device)
    base_ensemble.fit(X_train, y_train) # Calibrate on train
    
    # Pre-calculate validation probabilities and errors
    rf_val_prob = rf_model.predict_proba(X_val)[:, 1]
    ae_val_errors = base_ensemble._get_ae_reconstruction_errors(X_val)
    
    # Normalize AE errors using the calibrated min/max
    ae_val_prob = (ae_val_errors - base_ensemble.ae_min_error_) / (
        base_ensemble.ae_max_error_ - base_ensemble.ae_min_error_
    )
    ae_val_prob = np.clip(ae_val_prob, 0.0, 1.0)

    # 2. Run the fast grid search
    all_results = []
    best_score = -1
    best_config = None
    
    metric_key = 'f1_score' if metric == 'f1' else metric
    total_configs = len(rf_weights) * len(thresholds)
    current = 0
    
    for rf_weight in rf_weights:
        ae_weight = 1.0 - rf_weight
        
        # Calculate ensemble probabilities once per weight combo
        weight_sum = rf_weight + ae_weight
        ens_prob = ((rf_weight / weight_sum) * rf_val_prob) + ((ae_weight / weight_sum) * ae_val_prob)
        
        for threshold in thresholds:
            current += 1
            
            # Apply threshold (Vectorized)
            y_pred = (ens_prob >= threshold).astype(int)
            
            # Calculate metrics (using binary labels for multiclass-safe calculation)
            metrics = {
                'rf_weight': rf_weight,
                'ae_weight': ae_weight,
                'threshold': threshold,
                'accuracy': accuracy_score(y_val_binary, y_pred),
                'precision': precision_score(y_val_binary, y_pred, zero_division=0),
                'recall': recall_score(y_val_binary, y_pred, zero_division=0),
                'f1_score': f1_score(y_val_binary, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_val_binary, ens_prob)
            }
            
            all_results.append(metrics)
            
            if metrics[metric_key] > best_score:
                best_score = metrics[metric_key]
                best_config = {k: metrics[k] for k in ['rf_weight', 'ae_weight', 'threshold']}
                best_config['score'] = best_score
                
    if verbose:
        print(f"\n✓ Best configuration: {best_config}")
        
    return {'best_config': best_config, 'all_results': all_results}
