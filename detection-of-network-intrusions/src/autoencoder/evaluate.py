import torch
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json


def evaluate_autoencoder(
    model,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    train_errors = []
    with torch.no_grad():
        for xb, _ in train_loader:
            xb = xb.to(device)
            recon = model(xb)
            err = torch.mean((recon - xb) ** 2, dim=1).cpu().numpy()
            train_errors.append(err)
    train_errors = np.concatenate(train_errors)

    test_errors = []
    true_labels = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            recon = model(xb)
            err = torch.mean((recon - xb) ** 2, dim=1).cpu().numpy()
            test_errors.append(err)
            true_labels.append(yb.numpy())

    test_errors = np.concatenate(test_errors)
    true_labels = np.concatenate(true_labels).astype(int)
    
    try:
        auc = roc_auc_score(true_labels, test_errors)
    except Exception:
        auc = float("nan")

    precision, recall, pr_thresholds = precision_recall_curve(true_labels, test_errors)
    print(precision, recall)
    f1_scores = np.where(
        (precision + recall) == 0, 
        0, 
        2 * (precision * recall) / (precision + recall)
    )

    optimal_index = np.argmax(f1_scores)
    
    if optimal_index < len(pr_thresholds):
        threshold = pr_thresholds[optimal_index]
    else:
        threshold = pr_thresholds[-1] if len(pr_thresholds) > 0 else 0.0

    preds = (test_errors >= threshold).astype(int) 

    precision_final = precision_score(true_labels, preds)
    recall_final = recall_score(true_labels, preds)
    f1_final = f1_score(true_labels, preds)

    cm = confusion_matrix(true_labels, preds)
    tn, fp, fn, tp = cm.ravel()
    
    fpr, tpr, roc_thresholds = roc_curve(true_labels, test_errors)
    
    return {
        "threshold": threshold,
        "precision": precision_final,
        "recall": recall_final,
        "f1": f1_final,
        "auc": auc,
        "errors": test_errors,
        "preds": preds,
        "true_labels": true_labels,
        "train_errors": train_errors,
        "cm": cm,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": roc_thresholds,
    }

def plot_results(results, output_dir: str = None, name: str = "evaluation"):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    cm = results["cm"]
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Anomaly"])
    ax.set_yticklabels(["Normal", "Anomaly"])
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, cm[i, j], ha="center", va="center", color="white", fontsize=12
            )
    plt.colorbar(im, ax=ax)

    ax = axes[0, 1]
    test_errors = results["errors"]
    true_labels = results["true_labels"]
    threshold = results["threshold"]

    normal_errors = test_errors[true_labels == 0]
    anomaly_errors = test_errors[true_labels == 1]

    ax.hist(normal_errors, bins=50, alpha=0.7, label="Normal", color="blue")
    ax.hist(anomaly_errors, bins=50, alpha=0.7, label="Anomaly", color="red")
    ax.axvline(
        threshold,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({threshold:.4f})",
    )
    ax.set_xlabel("Reconstruction Error (MSE)")
    ax.set_ylabel("Frequency")
    ax.set_title("Reconstruction Error Distribution")
    ax.legend()
    ax.set_yscale("log")

    ax = axes[1, 0]
    fpr = results["fpr"]
    tpr = results["tpr"]
    auc = results["auc"]

    ax.plot(fpr, tpr, color="blue", lw=2, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.axis("off")
    metrics_text = f"""
    EVALUATION METRICS
    
    Threshold: {results['threshold']:.6f}
    
    Precision: {results['precision']:.4f}
    Recall: {results['recall']:.4f}
    F1 Score: {results['f1']:.4f}
    AUC: {results['auc']:.4f}
    
    CONFUSION MATRIX
    TP: {results['tp']}
    FP: {results['fp']}
    TN: {results['tn']}
    FN: {results['fn']}
    """
    ax.text(
        0.1,
        0.5,
        metrics_text,
        fontsize=11,
        verticalalignment="center",
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    plt.tight_layout()

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = output_path / f"{name}_{timestamp}.png"
        metrics_file = output_path / f"{name}_{timestamp}_metrics.json"

        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {plot_file}")

        metrics_to_save = {
            "timestamp": timestamp,
            "threshold": float(results["threshold"]),
            "precision": float(results["precision"]),
            "recall": float(results["recall"]),
            "f1": float(results["f1"]),
            "auc": float(results["auc"]),
            "tp": int(results["tp"]),
            "fp": int(results["fp"]),
            "tn": int(results["tn"]),
            "fn": int(results["fn"]),
        }
        with open(metrics_file, "w") as f:
            json.dump(metrics_to_save, f, indent=2)
        print(f"Metrics saved to {metrics_file}")
    else:
        plt.show()

    return fig
