import argparse
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from src.autoencoder.load_cores_iot import load_cores_iot
from src.autoencoder.load_netflow import load_netflow
from src.autoencoder.load_kddcup import load_kddcup
from src.autoencoder.dataset import AutoencoderDataset
from src.autoencoder.autoencoder import Autoencoder
from src.autoencoder.evaluate import evaluate_autoencoder, plot_results
from src.autoencoder.train import train_autoencoder
import random
import numpy as np
from pathlib import Path
from datetime import datetime

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def run_experiment(dataset: str, train_label: str | None, device: str, row_limit: int | None = None):
    if dataset == "cores_iot":
        loader = load_cores_iot
        loader_kwargs = {
            "path": "data/dataset3/cores_iot.csv",
            "normal_value": 0.0,
            "train_size": 0.6,
            "val_size": 0.2,
        }
    elif dataset == "netflow":
        loader = load_netflow
        loader_kwargs = {
            "train_path": "data/dataset2/train_net.csv",
            "test_path": "data/dataset2/test_net.csv",
            "train_label": train_label or "None",
            "normal_value": 0.0,
            "train_size": 0.6,
            "val_size": 0.2,
        }
    else:
        loader = load_kddcup
        loader_kwargs = {
            "path": "data/dataset1/kddcup.data",
            "train_label": train_label or "normal.",
            "train_size": 0.6,
            "val_size": 0.2,
            "row_limit": row_limit,
        }

    print(f"Running experiment: dataset={dataset}, train_label={train_label}")
    X_train, X_val, X_test, y_val, y_test, mean, std = loader(**loader_kwargs)

    print(
        f"Shapes ---- Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
    )

    train_loader = DataLoader(
        AutoencoderDataset(X_train), batch_size=256, shuffle=True
    )
    val_loader = DataLoader(
        AutoencoderDataset(X_val), batch_size=256, shuffle=False
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long()
    )
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    input_dim = X_train.shape[1]
    model = Autoencoder(input_dim=input_dim, latent_dim=32)
    print(f"Model created with input_dim={input_dim}")

    print("Training model")
    train_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=30,
        lr=1e-3,
        device=device,
    )

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = models_dir / f"v1_model_{dataset}_{timestamp}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    print("Evaluating model")
    results = evaluate_autoencoder(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
    )

    safe_label = (train_label or "default").replace(" ", "_")
    results_dir = Path("results") / dataset / safe_label
    plot_results(
        results,
        output_dir=str(results_dir),
        name=f"v1_model_{dataset}_{safe_label}",
    )

    print("Results:")
    print(f"  Threshold: {results['threshold']:.6f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  F1 Score: {results['f1']:.4f}")
    print(f"  AUC: {results['auc']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate autoencoder on selected dataset")
    parser.add_argument(
        "--dataset",
        choices=["cores_iot", "netflow", "kddcup"],
        default="cores_iot",
        help="Dataset to use for training and evaluation",
    )
    parser.add_argument(
        "--train-label",
        type=str,
        default=None,
        help=(
            "Label to treat as the training (normal) class."
            " For netflow this maps to ALERT (e.g. 'Denial of Service', 'Port Scanning', 'None')."
            " For kddcup this can be group name (dos/probe/r2l/u2r/normal.) or raw label (e.g. 'neptune.')."
            " Ignored for cores_iot."
        ),
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run all predefined experiments (datasets and train labels) sequentially",
    )
    parser.add_argument(
        "--run-all-dataset",
        choices=["cores_iot", "netflow", "kddcup"],
        help="Run all predefined experiments only for the selected dataset",
    )
    parser.add_argument(
        "--row-limit",
        type=int,
        default=None,
        help="Optional row cap (reservoir sample) for KDDCup to reduce memory usage",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    all_experiments = [
        ("cores_iot", [None]),
        ("netflow", ["Denial of Service", "Port Scanning", "Malware", "None"]),
        ("kddcup", ["normal.", "dos", "probe", "r2l", "u2r"]),
    ]

    if args.run_all:
        for ds, labels in all_experiments:
            for lbl in labels:
                run_experiment(ds, lbl, device, row_limit=args.row_limit)
    elif args.run_all_dataset:
        for ds, labels in all_experiments:
            if ds == args.run_all_dataset:
                for lbl in labels:
                    run_experiment(ds, lbl, device, row_limit=args.row_limit)
                break
    else:
        run_experiment(args.dataset, args.train_label, device, row_limit=args.row_limit)


if __name__ == "__main__":
    main()
