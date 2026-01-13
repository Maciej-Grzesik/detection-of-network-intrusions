import pandas as pd
import numpy as np
import random
import problexity as px
import time
import json
import os
from datetime import datetime
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def stratified_with_min(X, y, total_n, min_per_class=50, random_state=42):
    import pandas as _pd
    df = X.copy()
    df['_label'] = y
    counts = df['_label'].value_counts().to_dict()
    props = {k: v / len(df) for k, v in counts.items()}

    desired = {}
    for k, prop in props.items():
        desired[k] = min(counts[k], max(int(round(prop * total_n)), min_per_class))

    total_desired = sum(desired.values())
    if total_desired > total_n:
        surplus = total_desired - total_n
        for k in sorted(desired, key=lambda kk: -desired[kk]):
            if desired[k] > min_per_class and surplus > 0:
                take = min(desired[k] - min_per_class, surplus)
                desired[k] -= take
                surplus -= take
            if surplus == 0:
                break

    parts = []
    for k, n_k in desired.items():
        n_k = int(n_k)
        if n_k <= 0:
            continue
        part = df[df['_label'] == k]
        if len(part) <= n_k:
            parts.append(part)
        else:
            parts.append(part.sample(n=n_k, random_state=random_state))

    sampled = _pd.concat(parts).sample(frac=1, random_state=random_state).reset_index(drop=True)
    y_sampled = sampled['_label'].copy()
    X_sampled = sampled.drop(columns=['_label'])
    return X_sampled, y_sampled

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Get script directory for absolute paths
SCRIPT_DIR = Path(__file__).parent.absolute()

# Configuration
DATA_PATH = SCRIPT_DIR.parent / 'detection-of-network-intrusions' / 'data' / 'train_net.csv'
SAMPLE_SIZE = 10000  # Adjust based on your needs: 10k=fast, 50k=balanced, 100k=slower
MIN_PER_CLASS = 50  # minimum samples per class in the stratified sample
OUTPUT_DIR = SCRIPT_DIR / 'problexity_results'

print("="*60)
print("Kaggle Network Dataset Complexity Analysis")
print("="*60)

# Step 1: Load data
print(f"\n[1/5] Loading data from {DATA_PATH}...")
start_load = time.time()
df = pd.read_csv(DATA_PATH, index_col='ID')
print(f"✓ Loaded {len(df):,} rows in {time.time()-start_load:.1f}s")

# Step 2: Data preprocessing
print("\n[2/5] Preprocessing data...")
start_preprocess = time.time()

attack_map = {
    'Port Scanning': 2,
    'Denial of Service': 1,
    'Malware': 3
}

protocol_map = {
    'tcp': 0, 'udp': 1, 'icmp': 2, 'ipv6': 3,
    'ipv6-icmp': 4, 'gre': 5, 'skip': 6
}

df['PROTOCOL_MAP'] = df['PROTOCOL_MAP'].map(protocol_map)
df['ALERT'] = df['ALERT'].map(attack_map).fillna(0)

# Drop IP addresses
df = df.drop(labels=['IPV4_SRC_ADDR', 'IPV4_DST_ADDR'], axis=1)

print(f"✓ Preprocessing completed in {time.time()-start_preprocess:.1f}s")

# Step 3: Prepare data and sample
print("\n[3/5] Preparing dataset...")
X = df.drop(columns=['ALERT'])
y = df['ALERT']

# Note: do NOT impute the entire dataset. Imputation is applied only to the sampled subset below.

print(f"Full dataset: {len(X):,} samples")
print(f"Class distribution:\n{y.value_counts().sort_index().to_dict()}")

# Stratified sampling with minimum per-class
if len(X) > SAMPLE_SIZE:
    print(f"\nSampling {SAMPLE_SIZE:,} samples (stratified with min_per_class={MIN_PER_CLASS})...")
    start_sample = time.time()
    X_sample, y_sample = stratified_with_min(X, y, total_n=SAMPLE_SIZE, min_per_class=MIN_PER_CLASS, random_state=42)
    print(f"✓ Sampled {len(X_sample):,} samples in {time.time()-start_sample:.1f}s")
    print(f"Sampled class distribution:\n{y_sample.value_counts().sort_index().to_dict()}")

    # Impute missing values only on the sampled data
    na_total_sample = int(X_sample.isna().sum().sum())
    if na_total_sample > 0:
        print(f"Found {na_total_sample:,} missing values in sampled features. Missing per-column:")
        missing_per_col = X_sample.isna().sum()
        print(missing_per_col[missing_per_col > 0].to_dict())

        num_cols = X_sample.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in X_sample.columns if c not in num_cols]

        if num_cols:
            imp = SimpleImputer(strategy='median')
            X_sample[num_cols] = imp.fit_transform(X_sample[num_cols])

        for c in cat_cols:
            X_sample[c] = X_sample[c].fillna(-1)

        print('✓ Missing values imputed on sampled data (numeric=median, categorical=-1)')
else:
    print(f"Dataset is smaller than {SAMPLE_SIZE:,}, using full dataset")
    X_sample, y_sample = X, y

# Step 4: Calculate complexity metrics
print("\n[4/5] Calculating complexity metrics...")
print("This may take 5-15 minutes depending on sample size...")
start_complexity = time.time()

cc = px.ComplexityCalculator()

try:
    cc.fit(X_sample, y_sample)
    elapsed = time.time() - start_complexity
    print(f"✓ Complexity calculation completed in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
except Exception as e:
    print(f"✗ Error during complexity calculation: {e}")
    raise

# Step 5: Save results
print("\n[5/5] Saving results...")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = OUTPUT_DIR / f"kaggle_complexity_{timestamp}.json"
plot_file = OUTPUT_DIR / f"kaggle_complexity_{timestamp}.png"
report_file = OUTPUT_DIR / f"kaggle_complexity_{timestamp}_report.txt"

# Get complexity score and metrics
complexity_score = cc.complexity
metrics_data = cc._metrics()

# Handle metrics - can be list or dict depending on version
if isinstance(metrics_data, list):
    # Get metric names from the calculator
    metric_names = [m.__name__ for m in cc.metrics] if hasattr(cc, 'metrics') else [f"metric_{i}" for i in range(len(metrics_data))]
    metrics_dict = {name: float(v) if hasattr(v, '__float__') else v 
                    for name, v in zip(metric_names, metrics_data)}
else:
    metrics_dict = {k: float(v) if hasattr(v, '__float__') else v 
                    for k, v in metrics_data.items()}

# Generate text report
report_lines = []
report_lines.append("=" * 60)
report_lines.append("COMPLEXITY ANALYSIS REPORT")
report_lines.append("=" * 60)
report_lines.append(f"\nTimestamp: {timestamp}")
report_lines.append(f"\nDataset Information:")
report_lines.append(f"  Total samples: {len(X):,}")
report_lines.append(f"  Sampled samples: {len(X_sample):,}")
report_lines.append(f"  Sample size: {SAMPLE_SIZE:,}")
report_lines.append(f"\nFull Class Distribution:")
for k, v in sorted(y.value_counts().sort_index().to_dict().items()):
    report_lines.append(f"  Class {k}: {v:,} ({v/len(y)*100:.2f}%)")
report_lines.append(f"\nSampled Class Distribution:")
for k, v in sorted(y_sample.value_counts().sort_index().to_dict().items()):
    report_lines.append(f"  Class {k}: {v:,} ({v/len(y_sample)*100:.2f}%)")
report_lines.append(f"\nComplexity Score:")
if isinstance(complexity_score, list):
    report_lines.append(f"  Average: {sum(complexity_score)/len(complexity_score):.6f}")
    for i, score in enumerate(complexity_score):
        report_lines.append(f"  Class {i}: {score:.6f}")
else:
    report_lines.append(f"  {complexity_score}")
report_lines.append(f"\nComplexity Metrics:")
for metric_name, value in metrics_dict.items():
    if isinstance(value, (int, float)):
        report_lines.append(f"  {metric_name}: {value:.6f}")
    else:
        report_lines.append(f"  {metric_name}: {value}")
report_lines.append(f"\nComputation Time: {elapsed:.2f}s ({elapsed/60:.2f} minutes)")
report_lines.append("\n" + "=" * 60)

report_text = '\n'.join(report_lines)

# Save report to file
with open(report_file, 'w') as f:
    f.write(report_text)
print(f"✓ Report saved to: {report_file}")

# Generate and save plot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 10))
try:
    cc.plot(fig, (1, 1, 1))
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Plot saved to: {plot_file}")
except Exception as e:
    print(f"⚠ Warning: Could not save plot: {e}")
    plt.close(fig)

# Prepare JSON results
results = {
    'timestamp': timestamp,
    'dataset_info': {
        'total_samples': int(len(X)),
        'sampled_samples': int(len(X_sample)),
        'sample_size': SAMPLE_SIZE,
        'min_per_class': MIN_PER_CLASS,
        'full_class_distribution': {int(k): int(v) for k, v in y.value_counts().sort_index().to_dict().items()},
        'sampled_class_distribution': {int(k): int(v) for k, v in y_sample.value_counts().sort_index().to_dict().items()},
    },
    'complexity_score': float(complexity_score) if hasattr(complexity_score, '__float__') else str(complexity_score),
    'metrics': metrics_dict,
    'report': report_text,
    'computation_time_seconds': float(elapsed),
    'computation_time_minutes': float(elapsed / 60),
    'output_files': {
        'json': str(results_file),
        'plot': str(plot_file),
        'report': str(report_file)
    }
}

with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ Results saved to: {results_file}")

# Display summary
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(f"Complexity Score: {results['complexity_score']}")
print(f"\nComputation Time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
print(f"\nTop Metrics:")
for metric, value in list(results['metrics'].items())[:10]:
    print(f"  {metric}: {value}")

print("\n" + "="*60)
print("Analysis complete! Check the JSON file for full results.")
print("="*60)
