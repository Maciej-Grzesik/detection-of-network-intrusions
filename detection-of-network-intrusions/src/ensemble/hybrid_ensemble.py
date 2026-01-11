import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_array
from typing import Optional, Union


class HybridIntrusionEnsemble(BaseEstimator, ClassifierMixin):
    
    def __init__(
        self,
        rf_model,
        ae_model,
        rf_weight: float = 0.5,
        ae_weight: float = 0.5,
        threshold: float = 0.5,
        device: Optional[str] = None
    ):
        self.rf_model = rf_model
        self.ae_model = ae_model
        self.rf_weight = rf_weight
        self.ae_weight = ae_weight
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Calibration parameters (learned during fit)
        self.ae_min_error_ = None
        self.ae_max_error_ = None
        self.classes_ = np.array([0, 1])
        
    def fit(self, X, y=None):

        X = check_array(X, accept_sparse=False, force_all_finite=True)
        
        # Compute reconstruction errors on calibration set
        ae_errors = self._get_ae_reconstruction_errors(X)
        
        # Use percentiles for robust normalization (avoid outlier influence)
        self.ae_min_error_ = float(np.percentile(ae_errors, 1))
        self.ae_max_error_ = float(np.percentile(ae_errors, 99))
        
        # Handle edge case where all errors are identical
        if self.ae_max_error_ == self.ae_min_error_:
            self.ae_max_error_ = self.ae_min_error_ + 1e-8
            
        return self
    
    def predict_proba(self, X):
        check_is_fitted(self, ['ae_min_error_', 'ae_max_error_'])
        X = check_array(X, accept_sparse=False, force_all_finite=True)
        
        # 1. Get Random Forest probabilities
        rf_proba = self.rf_model.predict_proba(X)
        rf_attack_prob = rf_proba[:, 1]  # Probability of attack class
        
        # 2. Get Autoencoder reconstruction errors
        ae_errors = self._get_ae_reconstruction_errors(X)
        
        # 3. Normalize AE errors to [0, 1] range
        ae_attack_prob = (ae_errors - self.ae_min_error_) / (
            self.ae_max_error_ - self.ae_min_error_
        )
        ae_attack_prob = np.clip(ae_attack_prob, 0.0, 1.0)
        
        # 4. Compute weighted ensemble score
        weight_sum = self.rf_weight + self.ae_weight
        normalized_rf_weight = self.rf_weight / weight_sum
        normalized_ae_weight = self.ae_weight / weight_sum
        
        ensemble_attack_prob = (
            normalized_rf_weight * rf_attack_prob +
            normalized_ae_weight * ae_attack_prob
        )
        
        # 5. Return probabilities for both classes [P(normal), P(attack)]
        proba = np.column_stack([
            1 - ensemble_attack_prob,  # P(normal)
            ensemble_attack_prob       # P(attack)
        ])
        
        return proba
    
    def predict(self, X):
        proba = self.predict_proba(X)
        attack_prob = proba[:, 1]
        
        # Apply threshold
        predictions = (attack_prob >= self.threshold).astype(int)
        
        return predictions
    
    def _get_ae_reconstruction_errors(self, X):
        # Convert to PyTorch tensor
        X_tensor = torch.from_numpy(X).float().to(self.device)
        
        # Ensure model is on correct device and in eval mode
        self.ae_model = self.ae_model.to(self.device)
        self.ae_model.eval()
        
        # Compute reconstruction errors
        with torch.no_grad():
            # Forward pass through autoencoder
            reconstructed = self.ae_model(X_tensor)
            
            # Compute MSE per sample (mean over features)
            mse_errors = torch.mean((reconstructed - X_tensor) ** 2, dim=1)
            
            # Convert back to numpy
            errors = mse_errors.cpu().numpy()
        
        return errors
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        
        Required for sklearn compatibility.
        """
        return {
            'rf_model': self.rf_model,
            'ae_model': self.ae_model,
            'rf_weight': self.rf_weight,
            'ae_weight': self.ae_weight,
            'threshold': self.threshold,
            'device': self.device
        }
    
    def set_params(self, **params):
        """
        Set parameters for this estimator.
        
        Required for sklearn compatibility.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
