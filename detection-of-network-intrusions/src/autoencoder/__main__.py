import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from src.autoencoder.load_cores_iot import load_cores_iot
from src.autoencoder.dataset import CoresIotDataset
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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    X_train, X_val, X_test, y_val, y_test, mean, std = load_cores_iot(
        path="data/dataset3/cores_iot.csv",
        normal_value=0.0,
        train_size=0.6,
        val_size=0.2,
    )

    print(
        f"Shapes ---- Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    train_loader = DataLoader(
        CoresIotDataset(X_train_scaled), batch_size=256, shuffle=True
    )
    val_loader = DataLoader(
        CoresIotDataset(X_val_scaled), batch_size=256, shuffle=False
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test_scaled).float(), torch.from_numpy(y_test).long()
    )
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    input_dim = X_train_scaled.shape[1]
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
    model_path = models_dir / f"v1_model_{timestamp}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    print("Evaluating model")
    results = evaluate_autoencoder(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
    )

    plot_results(results, output_dir="results", name="v1_model")

    print("Results:")
    print(f"  Threshold: {results['threshold']:.6f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  F1 Score: {results['f1']:.4f}")
    print(f"  AUC: {results['auc']:.4f}")


if __name__ == "__main__":
    main()
