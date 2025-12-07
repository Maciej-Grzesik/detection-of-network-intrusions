import torch
from torch import nn
from torch.utils.data import DataLoader


def train_autoencoder(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
    lr: float = 1e-3,
    device: str = None,
    patience=5,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.L1Loss()

    best_val_loss = float("inf")
    wait = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item() * xb.size(0)
        avg_val_loss = val_loss / len(val_loader.dataset)

        # early stopp
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            wait = 0
            best_model_state = model.state_dict()
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                model.load_state_dict(best_model_state)
                return model

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            msg = f"Epoch {epoch}/{epochs} | train_loss: {avg_train_loss:.6f}"
            if avg_val_loss is not None:
                msg += f" | val_loss: {avg_val_loss:.6f}"
            print(msg)

    return model
