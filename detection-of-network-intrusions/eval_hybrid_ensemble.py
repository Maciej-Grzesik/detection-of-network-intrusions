"""
Comprehensive Evaluation Protocol for Hybrid Intrusion Detection System (NIDS).

Evaluates a Hybrid Ensemble (Random Forest + Autoencoder) across three datasets:
- KDD Cup '99
- Netflow V9
- Kitsune

Uses RepeatedStratifiedKFold cross-validation with nested evaluation loops to assess
generalization capabilities on unseen attack types.
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, accuracy_score
)

warnings.filterwarnings('ignore')

# ============================================================================
# IMPORT EXISTING MODEL CLASSES AND FUNCTIONS FROM YOUR CODEBASE
# ============================================================================
# Assuming these are available in your src directory structure:
try:
    from src.autoencoder.autoencoder import Autoencoder
    from src.autoencoder.train import train_autoencoder
    from src.ensemble.hybrid_ensemble import HybridIntrusionEnsemble
    from src.ensemble.tuning import grid_search_weights
except ImportError:
    print("Warning: Could not import from src directory. Ensure src/ is in PYTHONPATH.")
    print("You may need to add the project directory to sys.path.")
    sys.path.insert(0, str(Path(__file__).parent / 'detection-of-network-intrusions'))
    try:
        from src.autoencoder.autoencoder import Autoencoder
        from src.autoencoder.train import train_autoencoder
        from src.ensemble.hybrid_ensemble import HybridIntrusionEnsemble
        from src.ensemble.tuning import grid_search_weights
    except ImportError as e:
        raise ImportError(f"Failed to import required modules: {e}")


# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

DATASET_CONFIG = {
    'KDD': {
        'paths': {
            'data': './data/kddcup.data'
        },
        'attack_types': {
            0: 'Normal',
            1: 'DoS',
            2: 'Probing',
            3: 'R2L',
            4: 'U2R'
        }
    },
    'Netflow': {
        'paths': {
            'train': './data/train_net.csv',
            'test': './data/test_net.csv'
        },
        'attack_types': {
            0: 'Normal',
            1: 'Port_Scanning',
            2: 'DoS',
            3: 'Malware'
        }
    },
    'Kitsune': {
        'paths': {
            'data': './data/kitsune_2M.csv'
        },
        'attack_types': {
            0: 'Benign',
            1: 'Reconnaissance',
            2: 'Man_in_the_Middle',
            3: 'DoS_DDOS',
            4: 'Botnet'
        }
    }
}

RF_HYPERPARAMS = {
    'n_estimators': 100,
    'max_depth': 15,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'n_jobs': -1,
    'random_state': 42,
    'class_weight': 'balanced'
}

AE_HYPERPARAMS = {
    'input_dim': None,  # Will be set dynamically
    'latent_dim': 16,
    'epochs': 15,  # Reduced for faster execution
    'batch_size': 64,  # Increased batch size
    'lr': 1e-3,
    'patience': 3  # Earlier stopping
}

CV_CONFIG = {
    'n_splits': 2,
    'n_repeats': 2,  # Reduced repeats for faster execution
    'random_state': 42
}

# Data sampling config - sample large datasets for faster execution
SAMPLING_CONFIG = {
    'KDD': {'sample_size': 500000, 'random_state': 42},  # Sample 500k from 4.9M
    'Netflow': {'sample_size': 200000, 'random_state': 42},  # Sample 200k
    'Kitsune': {'sample_size': 200000, 'random_state': 42}  # Sample 200k from 2M
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_device() -> str:
    """Get available device (CUDA or CPU)."""
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✓ CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("✓ Using CPU")
    return device


def calculate_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate specificity (True Negative Rate).
    
    Specificity = TN / (TN + FP)
    """
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return float(specificity)
    except Exception:
        return 0.0


def load_and_preprocess_data(dataset_name: str) -> Optional[Tuple[np.ndarray, np.ndarray, Dict]]:
    """
    Load and preprocess data for the specified dataset.
    
    Args:
        dataset_name: One of 'KDD', 'Netflow', 'Kitsune'
    
    Returns:
        Tuple of (X, y, class_mapping) or None if data not found
    """
    config = DATASET_CONFIG.get(dataset_name)
    if not config:
        print(f"Unknown dataset: {dataset_name}")
        return None
    
    try:
        if dataset_name == 'KDD':
            return _load_kdd_data(config)
        elif dataset_name == 'Netflow':
            return _load_netflow_data(config)
        elif dataset_name == 'Kitsune':
            return _load_kitsune_data(config)
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return None


def _load_kdd_data(config: Dict) -> Optional[Tuple[np.ndarray, np.ndarray, Dict]]:
    """Load KDD Cup '99 dataset."""
    data_path = config['paths']['data']
    if not os.path.exists(data_path):
        print(f"KDD data file not found: {data_path}")
        return None
    
    print(f"Loading KDD data from {data_path}...")
    df = pd.read_csv(data_path, header=None)
    
    # Last column is the label
    y_raw = df.iloc[:, -1].values
    
    # Map KDD labels to numeric IDs
    # KDD labels: normal., dos attacks (back., land., neptune., smurf., teardrop.), 
    #             probe (ipsweep., nmap., portsweep., satan.),
    #             r2l (ftp_write., guess_passwd., imap., multihop., phf., spy., warezclient., warezmaster.),
    #             u2r (buffer_overflow., loadmodule., perl., rootkit.)
    
    kdd_label_map = {
        'normal.': 0,
        # DoS attacks
        'back.': 1, 'land.': 1, 'neptune.': 1, 'smurf.': 1, 'teardrop.': 1,
        # Probing attacks
        'ipsweep.': 2, 'nmap.': 2, 'portsweep.': 2, 'satan.': 2,
        # R2L attacks
        'ftp_write.': 3, 'guess_passwd.': 3, 'imap.': 3, 'multihop.': 3,
        'phf.': 3, 'spy.': 3, 'warezclient.': 3, 'warezmaster.': 3,
        # U2R attacks
        'buffer_overflow.': 4, 'loadmodule.': 4, 'perl.': 4, 'rootkit.': 4
    }
    
    # Convert labels to numeric
    y = np.array([kdd_label_map.get(label, -1) for label in y_raw])
    
    # Filter to only known classes
    valid_mask = y >= 0
    df = df[valid_mask]
    y = y[valid_mask]
    
    if len(df) == 0:
        print(f"No valid samples found in KDD data")
        return None
    
    # Drop the label column
    X = df.iloc[:, :-1].values
    
    # Convert to numeric, handling categorical features
    X_numeric = []
    for i in range(X.shape[1]):
        col = X[:, i]
        try:
            # Try to convert directly to float
            col_numeric = col.astype(np.float32)
        except (ValueError, TypeError):
            # If it's categorical, use LabelEncoder
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            col_numeric = le.fit_transform(col).astype(np.float32)
        X_numeric.append(col_numeric)
    
    X = np.column_stack(X_numeric).astype(np.float32)
    
    class_mapping = config['attack_types']
    
    print(f"  Shape: {X.shape}, Classes: {np.unique(y)}")
    return X, y, class_mapping


def _load_netflow_data(config: Dict) -> Optional[Tuple[np.ndarray, np.ndarray, Dict]]:
    """Load Netflow V9 dataset."""
    train_path = config['paths']['train']
    test_path = config['paths']['test']
    
    if not os.path.exists(train_path):
        print(f"Netflow train file not found: {train_path}")
        return None
    if not os.path.exists(test_path):
        print(f"Netflow test file not found: {test_path}")
        return None
    
    print(f"Loading Netflow data from {train_path} and {test_path}...")
    
    # Load both train and test datasets
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # Combine train and test
    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    
    # Extract label from ANOMALY column and features
    label_col = 'ANOMALY'
    y = df[label_col].values
    
    # Drop non-numeric columns: FLOW_ID, ID, and the label column
    cols_to_drop = ['FLOW_ID', 'ID', label_col]
    X_df = df.drop(columns=cols_to_drop)
    
    # Convert categorical columns to numeric using LabelEncoder
    from sklearn.preprocessing import LabelEncoder
    X_numeric = []
    for col in X_df.columns:
        col_data = X_df[col].values
        try:
            # Try to convert directly to float
            col_numeric = col_data.astype(np.float32)
        except (ValueError, TypeError):
            # If it's categorical, use LabelEncoder
            le = LabelEncoder()
            col_numeric = le.fit_transform(col_data).astype(np.float32)
        X_numeric.append(col_numeric)
    
    X = np.column_stack(X_numeric).astype(np.float32)
    
    class_mapping = config['attack_types']
    valid_classes = list(class_mapping.keys())
    mask = np.isin(y, valid_classes)
    X = X[mask]
    y = y[mask]
    
    print(f"  Shape: {X.shape}, Classes: {np.unique(y)}")
    return X, y, class_mapping


def _load_kitsune_data(config: Dict) -> Optional[Tuple[np.ndarray, np.ndarray, Dict]]:
    """Load Kitsune dataset."""
    data_path = config['paths']['data']
    if not os.path.exists(data_path):
        print(f"Kitsune data file not found: {data_path}")
        return None
    
    print(f"Loading Kitsune data from {data_path}...")
    df = pd.read_csv(data_path)
    
    label_col = 'category' if 'category' in df.columns else df.columns[-1]
    y = df[label_col].values
    X = df.drop(columns=[label_col]).values.astype(np.float32)
    
    class_mapping = config['attack_types']
    valid_classes = list(class_mapping.keys())
    mask = np.isin(y, valid_classes)
    X = X[mask]
    y = y[mask]
    
    print(f"  Shape: {X.shape}, Classes: {np.unique(y)}")
    return X, y, class_mapping


# ============================================================================
# CORE EVALUATION PROTOCOL
# ============================================================================

def run_evaluation_protocol(device: str) -> List[Dict]:
    """
    Execute the comprehensive evaluation protocol for the NIDS.
    
    Args:
        device: 'cuda' or 'cpu'
    
    Returns:
        List of result dictionaries
    """
    # Ensure results directory exists
    os.makedirs('./results/', exist_ok=True)
    
    all_results = []
    
    # ========================================================================
    # 1. DATASET LOOP
    # ========================================================================
    for dataset_name in ['KDD', 'Netflow', 'Kitsune']:
        print(f"\n{'='*80}")
        print(f"DATASET: {dataset_name}")
        print(f"{'='*80}")
        
        # Load and preprocess data
        data = load_and_preprocess_data(dataset_name)
        if data is None:
            print(f"⚠ Skipping {dataset_name} - data loading failed")
            continue
        
        X, y, class_mapping = data
        
        # Sample data for faster execution if dataset is large
       # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        print(f"✓ Data loaded: {X.shape}")
        print(f"✓ Classes: {np.unique(y)}")

        # ====================================================================
        # 2. CROSS-VALIDATION LOOP (Split ALL data first to prevent leakage)
        # ====================================================================
        cv = RepeatedStratifiedKFold(
            n_splits=CV_CONFIG['n_splits'],
            n_repeats=CV_CONFIG['n_repeats'],
            random_state=CV_CONFIG['random_state']
        )
        
        fold_counter = 0
        for fold_num, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            fold_counter += 1
            print(f"\n  {'─'*76}")
            print(f"  FOLD {fold_counter}/{CV_CONFIG['n_splits'] * CV_CONFIG['n_repeats']}")
            print(f"  {'─'*76}")
            
            X_train_full, y_train_full = X[train_idx], y[train_idx]
            X_test_full, y_test_full = X[test_idx], y[test_idx]

            # ================================================================
            # 3. SOURCE ATTACK LOOP (Train on Known Attacks)
            # ================================================================
            for source_attack_id, source_attack_name in class_mapping.items():
                if source_attack_id == 0:  # Skip Normal as source attack
                    continue
                
                # Filter TRAINING data to Normal (0) + Source Attack
                train_mask = (y_train_full == 0) | (y_train_full == source_attack_id)
                X_train_fold = X_train_full[train_mask]
                y_train_fold = y_train_full[train_mask]
                
                if len(np.unique(y_train_fold)) < 2:
                    print(f"    ⚠ Insufficient samples for {source_attack_name} in train fold. Skipping...")
                    continue
                
                # --- NEW: Split Train Fold into Train & Validation for Tuning ---
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_fold, y_train_fold, test_size=0.2, 
                    random_state=CV_CONFIG['random_state'], stratify=y_train_fold
                )
                
                print(f"\n    Training on Normal vs {source_attack_name}...")
                
                # --- Train Random Forest ---
                rf_model = RandomForestClassifier(**RF_HYPERPARAMS)
                rf_model.fit(X_train, y_train) # Note: Trained only on the 80% split
                
                # --- Train Autoencoder ---
                normal_mask = (y_train == 0)
                X_ae_train = X_train[normal_mask]
                
                if len(X_ae_train) < 2:
                    print("    ⚠ Insufficient normal samples for AE. Skipping fold...")
                    continue
                
                X_ae_train_tensor = torch.from_numpy(X_ae_train).float()
                ae_train_dataset = TensorDataset(X_ae_train_tensor, X_ae_train_tensor)
                ae_train_loader = DataLoader(ae_train_dataset, batch_size=AE_HYPERPARAMS['batch_size'], shuffle=True)
                
                # Use the normal traffic from the validation split for AE early stopping
                val_normal_mask = (y_val == 0)
                X_ae_val = X_val[val_normal_mask]
                X_ae_val_tensor = torch.from_numpy(X_ae_val).float()
                ae_val_dataset = TensorDataset(X_ae_val_tensor, X_ae_val_tensor)
                ae_val_loader = DataLoader(ae_val_dataset, batch_size=AE_HYPERPARAMS['batch_size'], shuffle=False)
                
                input_dim = X_train.shape[1]
                ae_model = Autoencoder(input_dim=input_dim, latent_dim=AE_HYPERPARAMS['latent_dim'])
                
                ae_model = train_autoencoder(
                    model=ae_model, train_loader=ae_train_loader, val_loader=ae_val_loader,
                    epochs=AE_HYPERPARAMS['epochs'], lr=AE_HYPERPARAMS['lr'], device=device, patience=AE_HYPERPARAMS['patience']
                )
                
                # --- Tuning Phase ---
                print("    ✓ Tuning...", end=" ")
                tuning_result = grid_search_weights(
                    rf_model=rf_model, ae_model=ae_model, X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val, rf_weights=np.linspace(0.0, 1.0, 6), # Note: Uses the 20% validation split
                    thresholds=[0.4, 0.5, 0.6], metric='f1', device=device, verbose=False
                )
                best_config = tuning_result['best_config']
                
                ensemble = HybridIntrusionEnsemble(
                    rf_model=rf_model, ae_model=ae_model, rf_weight=best_config['rf_weight'],
                    ae_weight=best_config['ae_weight'], threshold=best_config['threshold'], device=device
                )
                ensemble.fit(X_ae_train, y=None)
                
                print("✓ Evaluation")
                
                # ============================================================
                # 4. DISCOVERY PHASE (Evaluate on UNSEEN attacks from TEST fold)
                # ============================================================
                unique_test_classes = np.unique(y_test_full)
                
                for unseen_attack_id, unseen_attack_name in class_mapping.items():
                    if unseen_attack_id == 0 or unseen_attack_id == source_attack_id:
                        continue
                        
                    if unseen_attack_id not in unique_test_classes:
                        continue
                    
                    # Filter TEST data to Normal + Unseen Attack (No Leakage!)
                    unseen_mask = (y_test_full == 0) | (y_test_full == unseen_attack_id)
                    X_eval = X_test_full[unseen_mask]
                    y_eval = y_test_full[unseen_mask]
                    
                    if len(X_eval) == 0:
                        continue
                    
                    # Binarize labels
                    y_eval_binary = (y_eval != 0).astype(int)
                    
                    # Get Predictions
                    y_pred_rf = rf_model.predict(X_eval)
                    y_pred_rf_binary = (y_pred_rf != 0).astype(int)
                    
                    # AE Baseline using calibrated probability
                    ae_errors = ensemble._get_ae_reconstruction_errors(X_eval)
                    ae_error_prob = (ae_errors - ensemble.ae_min_error_) / (ensemble.ae_max_error_ - ensemble.ae_min_error_)
                    ae_error_prob = np.clip(ae_error_prob, 0.0, 1.0)
                    y_pred_ae_binary = (ae_error_prob >= best_config['threshold']).astype(int)
                    
                    # Ensemble Predictions
                    y_pred_ensemble = ensemble.predict(X_eval)
                    
                    # Calculate Metrics
                    metrics_rf = _calculate_metrics(y_eval_binary, y_pred_rf_binary)
                    metrics_ae = _calculate_metrics(y_eval_binary, y_pred_ae_binary)
                    metrics_ens = _calculate_metrics(y_eval_binary, y_pred_ensemble)
                    
                    all_results.append({
                        'Dataset': dataset_name, 'Source_Attack': source_attack_name, 'Unseen_Attack': unseen_attack_name,
                        'Fold': fold_counter,
                        'RF_Precision': metrics_rf['precision'], 'RF_Recall': metrics_rf['recall'], 'RF_F1': metrics_rf['f1'], 'RF_Specificity': metrics_rf['specificity'],
                        'AE_Precision': metrics_ae['precision'], 'AE_Recall': metrics_ae['recall'], 'AE_F1': metrics_ae['f1'], 'AE_Specificity': metrics_ae['specificity'],
                        'Ensemble_Precision': metrics_ens['precision'], 'Ensemble_Recall': metrics_ens['recall'], 'Ensemble_F1': metrics_ens['f1'], 'Ensemble_Specificity': metrics_ens['specificity'],
                        'RF_Weight': best_config['rf_weight'], 'AE_Weight': best_config['ae_weight'], 'Decision_Threshold': best_config['threshold']
                    })
        
        # ====================================================================
        # Save results for this dataset
        # ====================================================================
        dataset_results = [r for r in all_results if r['Dataset'] == dataset_name]
        if dataset_results:
            df_results = pd.DataFrame(dataset_results)
            output_file = f"./results/{dataset_name}_evaluation_results.csv"
            df_results.to_csv(output_file, index=False)
            print(f"\n  ✓ {dataset_name} Results saved: {output_file}")
            print(f"    Total evaluations: {len(dataset_results)}")
    
    return all_results


def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive metrics for binary classification."""
    metrics = {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'specificity': calculate_specificity(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred),
    }
    return metrics


def generate_summary_report(all_results: List[Dict]):
    """Generate a summary report of evaluation results."""
    if not all_results:
        print("No results to summarize.")
        return
    
    df = pd.DataFrame(all_results)
    
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY REPORT")
    print(f"{'='*80}")
    
    # Overall statistics by dataset
    for dataset in df['Dataset'].unique():
        df_dataset = df[df['Dataset'] == dataset]
        print(f"\n{dataset} Dataset:")
        print(f"  Total evaluations: {len(df_dataset)}")
        print(f"  Ensemble Average F1: {df_dataset['Ensemble_F1'].mean():.4f} ± {df_dataset['Ensemble_F1'].std():.4f}")
        print(f"  Ensemble Average Recall: {df_dataset['Ensemble_Recall'].mean():.4f} ± {df_dataset['Ensemble_Recall'].std():.4f}")
        print(f"  Ensemble Average Precision: {df_dataset['Ensemble_Precision'].mean():.4f} ± {df_dataset['Ensemble_Precision'].std():.4f}")
    
    # Model comparison
    print(f"\n{'─'*80}")
    print("Model Comparison (across all datasets):")
    print(f"{'─'*80}")
    
    for metric in ['F1', 'Recall', 'Precision', 'Specificity']:
        rf_col = f'RF_{metric}'
        ae_col = f'AE_{metric}'
        ens_col = f'Ensemble_{metric}'
        
        if rf_col in df.columns:
            print(f"\n  {metric}:")
            print(f"    Random Forest: {df[rf_col].mean():.4f} ± {df[rf_col].std():.4f}")
            print(f"    Autoencoder:   {df[ae_col].mean():.4f} ± {df[ae_col].std():.4f}")
            print(f"    Ensemble:      {df[ens_col].mean():.4f} ± {df[ens_col].std():.4f}")
    
    print(f"\n{'='*80}\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("HYBRID INTRUSION DETECTION SYSTEM - COMPREHENSIVE EVALUATION PROTOCOL")
    print("="*80 + "\n")
    
    # Setup
    device = get_device()
    print()
    
    # Run evaluation
    try:
        all_results = run_evaluation_protocol(device)
        
        # Generate summary report
        generate_summary_report(all_results)
        
        # Save comprehensive results
        if all_results:
            df_all = pd.DataFrame(all_results)
            comprehensive_file = "./results/comprehensive_evaluation_results.csv"
            df_all.to_csv(comprehensive_file, index=False)
            print(f"\n✓ Comprehensive results saved: {comprehensive_file}")
            print(f"  Total results: {len(all_results)}")
        
        print("\n✓ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
