"""
Visualization Generation Script for NIDS Evaluation Results
============================================================

Generates publication-quality plots for the comprehensive evaluation:
- Baseline generalization (RF blindness on unseen attacks)
- Hybrid ensemble performance improvement
- Normalized confusion matrices for key scenarios
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path

# =====================================================================
# SETUP & CONFIGURATION
# =====================================================================
# Set publication-quality plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

# Create output directory for the manuscript figures
OUTPUT_DIR = './results/manuscript_figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the data
DATA_PATH = './results/comprehensive_evaluation_results.csv'
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Could not find {DATA_PATH}\n"
        f"Please ensure eval_hybrid_ensemble.py has completed successfully."
    )

print(f"Loading evaluation results from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
print(f"  Loaded {len(df)} evaluation records\n")

# Create a summary dataframe averaging the CV folds for the bar charts
metrics_to_average = ['RF_Recall', 'AE_Recall', 'Ensemble_Recall', 
                      'RF_Specificity', 'AE_Specificity', 'Ensemble_Specificity']
df_summary = df.groupby(['Dataset', 'Source_Attack', 'Unseen_Attack'])[metrics_to_average].mean().reset_index()


# =====================================================================
# PLOT 1: BASELINE GENERALIZATION (RANDOM FOREST BLINDNESS)
# =====================================================================
def plot_rf_generalization(dataset_name):
    """
    Generates grouped bar chart showing RF's baseline detection rate
    on unseen attack types trained only on supervised examples.
    """
    df_data = df_summary[df_summary['Dataset'] == dataset_name]
    if df_data.empty:
        print(f"  ⚠ No data for {dataset_name}")
        return
    
    plt.figure(figsize=(13, 6))
    
    sns.barplot(
        data=df_data, 
        x='Unseen_Attack', 
        y='RF_Recall', 
        hue='Source_Attack',
        palette='tab10'
    )
    
    plt.title(
        f'Baseline Generalization: Random Forest on {dataset_name}',
        pad=15, fontweight='bold', fontsize=14
    )
    plt.xlabel('Evaluated On (Unseen Attack Type)', fontweight='bold', fontsize=12)
    plt.ylabel('Detection Rate (Recall)', fontweight='bold', fontsize=12)
    plt.ylim(0, 1.05)
    plt.axhline(0.5, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Random Baseline')
    plt.legend(title='Trained on:', loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=10)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    filename = os.path.join(OUTPUT_DIR, f'{dataset_name}_01_RF_Generalization.pdf')
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✓ Saved: {filename}")


# =====================================================================
# PLOT 2: HYBRID ENSEMBLE LIFT (SIDE-BY-SIDE BARS)
# =====================================================================
def plot_ensemble_comparison(dataset_name):
    """
    Generates grouped bar chart comparing RF (supervised), AE (unsupervised),
    and Ensemble performance on detecting unseen attacks.
    """
    df_data = df_summary[df_summary['Dataset'] == dataset_name]
    if df_data.empty:
        print(f"  ⚠ No data for {dataset_name}")
        return
    
    # Melt the dataframe for Seaborn grouping
    df_melted = pd.melt(
        df_data, 
        id_vars=['Source_Attack', 'Unseen_Attack'], 
        value_vars=['RF_Recall', 'AE_Recall', 'Ensemble_Recall'],
        var_name='Model_Type', 
        value_name='Detection_Rate'
    )
    
    # Clean up labels for the legend
    df_melted['Model_Type'] = df_melted['Model_Type'].replace({
        'RF_Recall': 'RF (Supervised)',
        'AE_Recall': 'AE (Unsupervised)',
        'Ensemble_Recall': 'Hybrid Ensemble'
    })
    
    # Create a combined category for the X-axis
    df_melted['Scenario'] = "Train: " + df_melted['Source_Attack'] + "\nTest: " + df_melted['Unseen_Attack']
    
    plt.figure(figsize=(16, 7))
    
    sns.barplot(
        data=df_melted,
        x='Scenario',
        y='Detection_Rate',
        hue='Model_Type',
        palette=['#3498db', '#e74c3c', '#2ecc71']  # Blue, Red, Green
    )
    
    plt.title(
        f'Zero-Day Detection: Hybrid Architecture on {dataset_name}',
        pad=15, fontweight='bold', fontsize=14
    )
    plt.xlabel('Generalization Scenario', fontweight='bold', fontsize=12)
    plt.ylabel('Detection Rate (Recall)', fontweight='bold', fontsize=12)
    plt.ylim(0, 1.05)
    plt.axhline(0.5, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    plt.legend(title='Architecture:', loc='upper right', fontsize=10)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    
    plt.tight_layout()
    filename = os.path.join(OUTPUT_DIR, f'{dataset_name}_02_Ensemble_Comparison.pdf')
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✓ Saved: {filename}")


# =====================================================================
# PLOT 3: NORMALIZED CONFUSION MATRICES
# =====================================================================
def plot_comparative_confusion_matrix(dataset, source_attack, unseen_attack):
    """
    Generates side-by-side normalized confusion matrices comparing
    RF and Ensemble performance for a specific attack scenario.
    """
    # Filter the raw folds data for the specific scenario
    scenario_df = df[
        (df['Dataset'] == dataset) & 
        (df['Source_Attack'] == source_attack) & 
        (df['Unseen_Attack'] == unseen_attack)
    ]
    
    if scenario_df.empty:
        print(f"  ⚠ Scenario not found: {dataset} | Train: {source_attack} -> Test: {unseen_attack}")
        return
    
    # Average the metrics across all CV folds
    rf_recall = scenario_df['RF_Recall'].mean()
    rf_spec = scenario_df['RF_Specificity'].mean()
    
    ens_recall = scenario_df['Ensemble_Recall'].mean()
    ens_spec = scenario_df['Ensemble_Specificity'].mean()
    
    # Reconstruct Normalized Confusion Matrices
    # Layout: [[True Negative Rate (Specificity), False Positive Rate], 
    #          [False Negative Rate, True Positive Rate (Recall)]]
    cm_rf = np.array([
        [rf_spec, 1 - rf_spec],
        [1 - rf_recall, rf_recall]
    ])
    
    cm_ens = np.array([
        [ens_spec, 1 - ens_spec],
        [1 - ens_recall, ens_recall]
    ])
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        f'{dataset}: Trained on {source_attack} → Tested on {unseen_attack}',
        fontsize=13, fontweight='bold', y=1.02
    )
    
    labels = ['Normal', 'Attack']
    
    # Plot RF Matrix
    sns.heatmap(cm_rf, annot=True, fmt='.1%', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, ax=axes[0], 
                annot_kws={"size": 12, "weight": "bold"}, cbar=False, vmin=0, vmax=1)
    axes[0].set_title('Random Forest (Baseline)', fontsize=12, pad=10, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    
    # Plot Ensemble Matrix
    sns.heatmap(cm_ens, annot=True, fmt='.1%', cmap='Greens', 
                xticklabels=labels, yticklabels=labels, ax=axes[1], 
                annot_kws={"size": 12, "weight": "bold"}, cbar=False, vmin=0, vmax=1)
    axes[1].set_title('Hybrid Ensemble', fontsize=12, pad=10, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    # Format filename safely (replace special chars)
    safe_source = source_attack.replace(" ", "_").replace("/", "_")
    safe_unseen = unseen_attack.replace(" ", "_").replace("/", "_")
    filename = os.path.join(OUTPUT_DIR, f'{dataset}_03_CM_{safe_source}_vs_{safe_unseen}.pdf')
    
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✓ Saved: {filename}")


# =====================================================================
# PLOT 4: PERFORMANCE HEATMAP (ENSEMBLE RECALL ACROSS ALL SCENARIOS)
# =====================================================================
def plot_scenario_heatmap(dataset_name):
    """
    Generates a heatmap showing Ensemble performance across all
    source/unseen attack combinations.
    """
    df_data = df_summary[df_summary['Dataset'] == dataset_name]
    if df_data.empty:
        print(f"  ⚠ No data for {dataset_name}")
        return
    
    # Create pivot table
    pivot = df_data.pivot_table(
        index='Source_Attack',
        columns='Unseen_Attack',
        values='Ensemble_Recall',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(10, 6))
    
    sns.heatmap(
        pivot, annot=True, fmt='.2f', cmap='RdYlGn', 
        vmin=0, vmax=1, cbar_kws={'label': 'Detection Rate (Recall)'},
        annot_kws={"size": 11, "weight": "bold"}
    )
    
    plt.title(
        f'Ensemble Detection Performance Matrix: {dataset_name}',
        pad=15, fontweight='bold', fontsize=13
    )
    plt.xlabel('Unseen Attack Type (Test)', fontweight='bold', fontsize=11)
    plt.ylabel('Training Attack Type', fontweight='bold', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    filename = os.path.join(OUTPUT_DIR, f'{dataset_name}_04_Performance_Heatmap.pdf')
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✓ Saved: {filename}")


# =====================================================================
# MAIN EXECUTION
# =====================================================================
if __name__ == "__main__":
    print("="*70)
    print("VISUALIZATION GENERATION FOR NIDS EVALUATION")
    print("="*70)
    print(f"\nOutput directory: {OUTPUT_DIR}\n")
    
    datasets = df['Dataset'].unique()
    
    # 1. Generate Bar Charts for all datasets
    print("1. Generating Baseline Generalization Charts...")
    for dataset in datasets:
        print(f"\n  {dataset}:")
        plot_rf_generalization(dataset)
    
    # 2. Generate Ensemble Comparison Charts
    print("\n2. Generating Ensemble Comparison Charts...")
    for dataset in datasets:
        print(f"\n  {dataset}:")
        plot_ensemble_comparison(dataset)
    
    # 3. Generate Performance Heatmaps
    print("\n3. Generating Performance Heatmaps...")
    for dataset in datasets:
        print(f"\n  {dataset}:")
        plot_scenario_heatmap(dataset)
    
    # 4. Generate specific Confusion Matrices for key scenarios
    print("\n4. Generating Scenario-Specific Confusion Matrices...")
    
    # Get available scenarios from data
    scenarios = {}
    for dataset in datasets:
        df_dataset = df[df['Dataset'] == dataset]
        unique_combos = df_dataset[['Source_Attack', 'Unseen_Attack']].drop_duplicates().values.tolist()
        scenarios[dataset] = unique_combos
    
    # Plot top interesting scenarios (first 3 per dataset to keep output manageable)
    for dataset, scenario_list in scenarios.items():
        print(f"\n  {dataset}:")
        for source, unseen in scenario_list[:3]:
            plot_comparative_confusion_matrix(dataset, source, unseen)
    
    print("\n" + "="*70)
    print("✓ All visualizations generated successfully!")
    print("="*70)
    print(f"\nFigures saved to: {os.path.abspath(OUTPUT_DIR)}\n")
