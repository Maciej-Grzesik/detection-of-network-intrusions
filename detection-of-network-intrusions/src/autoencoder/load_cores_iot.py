import csv
import os
from typing import Tuple
import numpy as np
from sklearn.model_selection import train_test_split


def load_cores_iot(
    path: str, normal_value: float = 0.0, train_size: float = 0.6, val_size: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        rows = [r for r in reader if any(cell.strip() for cell in r)]

    if not rows:
        raise ValueError(f"Empty CSV: {path}")

    data_rows = rows

    data = np.array([[float(x) for x in r] for r in data_rows], dtype=np.float32)
    # print(data[1])
    class_idx = -1

    class_idx = data.shape[1] - 1

    y = data[:, class_idx]
    # print(y[1])
    X = np.delete(data, class_idx, axis=1)
    # print(X[1])

    normal_mask = y == float(normal_value)
    X_normal = X[normal_mask]
    X_all = X
    y_all = y

    # train only normal samples
    X_train, _ = train_test_split(X_normal, test_size=(1.0 - train_size), shuffle=True)

    # val,test all samples
    test_fraction = (1.0 - train_size - val_size) / (1.0 - train_size)
    x_temp, x_test, y_temp, y_test = train_test_split(
        X_all, y_all, test_size=test_fraction, shuffle=True
    )

    x_val = x_temp
    y_val = y_temp

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std == 0] = 1.0

    X_train = (X_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    return (
        X_train.astype(np.float32),
        x_val.astype(np.float32),
        x_test.astype(np.float32),
        y_val.astype(np.float32),
        y_test.astype(np.float32),
        mean.astype(np.float32),
        std.astype(np.float32),
    )
