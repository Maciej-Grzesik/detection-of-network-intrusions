import os
import ipaddress
from typing import Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import scipy.sparse as sp
from sklearn.impute import SimpleImputer
import csv

NETFLOW_COLUMNS = [
    "FLOW_ID",
    "PROTOCOL_MAP",
    "L4_SRC_PORT",
    "IPV4_SRC_ADDR",
    "L4_DST_PORT",
    "IPV4_DST_ADDR",
    "FIRST_SWITCHED",
    "FLOW_DURATION_MILLISECONDS",
    "LAST_SWITCHED",
    "PROTOCOL",
    "TCP_FLAGS",
    "TCP_WIN_MAX_IN",
    "TCP_WIN_MAX_OUT",
    "TCP_WIN_MIN_IN",
    "TCP_WIN_MIN_OUT",
    "TCP_WIN_MSS_IN",
    "TCP_WIN_SCALE_IN",
    "TCP_WIN_SCALE_OUT",
    "SRC_TOS",
    "DST_TOS",
    "TOTAL_FLOWS_EXP",
    "MIN_IP_PKT_LEN",
    "MAX_IP_PKT_LEN",
    "TOTAL_PKTS_EXP",
    "TOTAL_BYTES_EXP",
    "IN_BYTES",
    "IN_PKTS",
    "OUT_BYTES",
    "OUT_PKTS",
    "ANALYSIS_TIMESTAMP",
    "ANOMALY",
    "ALERT",
    "ID",
]

CATEGORICAL_FEATURES = {"PROTOCOL_MAP"}
IP_FEATURES = {"IPV4_SRC_ADDR", "IPV4_DST_ADDR"}
DROP_FEATURES = {"ANOMALY", "ALERT", "ID"}


def _safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return np.nan


def _ip_to_int(value) -> int:
    try:
        return int(ipaddress.ip_address(str(value)))
    except Exception:
        return 0


def _load_csv(path: str) -> np.ndarray:
    rows = []
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        for r in reader:
            if not any(cell.strip() for cell in r):
                continue
            rows.append(r)
    if not rows:
        return np.empty((0, 0), dtype=object)
    max_cols = max(len(r) for r in rows)
    normalized = [r + [""] * (max_cols - len(r)) for r in rows]
    return np.array(normalized, dtype=object)


def load_netflow(
    train_path: str,
    test_path: str | None = None,
    train_label: str | None = None,
    normal_value: float = 0.0,
    train_size: float = 0.6,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"File not found: {train_path}")
    if test_path and not os.path.exists(test_path):
        raise FileNotFoundError(f"File not found: {test_path}")

    train_raw = _load_csv(train_path)
    test_raw = _load_csv(test_path) if test_path else None

    def _normalize_columns(arr: np.ndarray, path_label: str) -> tuple[np.ndarray, bool]:
        if arr.shape[1] == len(NETFLOW_COLUMNS):
            return arr, True
        if arr.shape[1] == len(NETFLOW_COLUMNS) - 1:
            alert_col = np.full((arr.shape[0], 1), "None", dtype=object)
            arr = np.column_stack((arr[:, : -1], alert_col, arr[:, -1]))
            return arr, False
        raise ValueError(
            f"Unexpected column count in {path_label}: got {arr.shape[1]}, expected {len(NETFLOW_COLUMNS)} or {len(NETFLOW_COLUMNS)-1} (without ALERT)"
        )

    train_raw, train_has_alert = _normalize_columns(train_raw, train_path)
    if test_raw is not None:
        test_raw, test_has_alert = _normalize_columns(test_raw, test_path)
    else:
        test_has_alert = True

    label_idx_anomaly = NETFLOW_COLUMNS.index("ANOMALY")
    label_idx_alert = NETFLOW_COLUMNS.index("ALERT")
    feature_indices = [i for i, name in enumerate(NETFLOW_COLUMNS) if name not in DROP_FEATURES]
    feature_names = [NETFLOW_COLUMNS[i] for i in feature_indices]

    def split_features_labels(data: np.ndarray, has_alert: bool):
        if train_label is None:
            y_raw = np.array([_safe_float(v) for v in data[:, label_idx_anomaly]], dtype=np.float32)
            y = (y_raw != float(normal_value)).astype(np.float32)
        else:
            if has_alert:
                alerts = data[:, label_idx_alert].astype(str)
                y = (alerts != train_label).astype(np.float32)
            else:
                y_raw = np.array([_safe_float(v) for v in data[:, label_idx_anomaly]], dtype=np.float32)
                y = (y_raw != float(normal_value)).astype(np.float32)

        columns = []
        cat_cols = []
        for idx, name in zip(feature_indices, feature_names):
            col = data[:, idx]
            if name in CATEGORICAL_FEATURES:
                cat_cols.append(col.astype(str))
            elif name in IP_FEATURES:
                columns.append(np.array([_ip_to_int(v) for v in col], dtype=np.float64))
            else:
                columns.append(np.array([_safe_float(v) for v in col], dtype=np.float64))

        X_num = np.column_stack(columns) if columns else np.empty((len(data), 0))
        X_cat = np.column_stack(cat_cols) if cat_cols else None
        return X_num, X_cat, y

    X_num_train, X_cat_train, y_train = split_features_labels(train_raw, train_has_alert)
    if X_cat_train is not None:
        X_cat_train = X_cat_train.astype(object)
    if test_raw is not None:
        X_num_test, X_cat_test, y_test_full = split_features_labels(test_raw, test_has_alert)
        if X_cat_test is not None:
            X_cat_test = X_cat_test.astype(object)
    else:
        X_num_test, X_cat_test, y_test_full = (None, None, None)

    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent") if X_cat_train is not None else None
    encoder = OneHotEncoder(sparse_output=True, handle_unknown="ignore") if X_cat_train is not None else None

    normal_mask_train = y_train == 0.0
    num_imputer.fit(X_num_train[normal_mask_train])
    if cat_imputer is not None and encoder is not None:
        cat_imputer.fit(X_cat_train[normal_mask_train])
        encoder.fit(cat_imputer.transform(X_cat_train[normal_mask_train]))

    n_train = X_num_train.shape[0]
    train_indices = np.arange(n_train)

    normal_indices = train_indices[normal_mask_train]
    X_train_idx, _ = train_test_split(
        normal_indices, test_size=(1.0 - train_size), shuffle=True, random_state=random_state
    )

    if X_num_test is None:
        test_fraction = (1.0 - train_size - val_size) / (1.0 - train_size)
        x_temp_idx, x_test_idx, y_temp, y_test = train_test_split(
            train_indices, y_train, test_size=test_fraction, shuffle=True, random_state=random_state
        )
        x_val_idx = x_temp_idx
        y_val = y_temp
    else:
        _, x_val_idx, _, y_val = train_test_split(
            train_indices, y_train, test_size=val_size, shuffle=True, random_state=random_state
        )
        x_test_idx = None

    num_train_t = num_imputer.transform(X_num_train[X_train_idx])
    mean = num_train_t.mean(axis=0, keepdims=True)
    std = num_train_t.std(axis=0, keepdims=True)
    std[std == 0] = 1.0

    def build_split_from_train(idx_array: np.ndarray):
        num_part = X_num_train[idx_array]
        if X_cat_train is not None:
            cat_part = X_cat_train[idx_array]
        else:
            cat_part = None
        num_t = num_imputer.transform(num_part)
        num_scaled = (num_t - mean) / std
        if cat_part is None or encoder is None or cat_imputer is None:
            return num_scaled
        cat_t = cat_imputer.transform(cat_part)
        cat_enc = encoder.transform(cat_t)
        num_sparse = sp.csr_matrix(num_scaled)
        combined = sp.hstack([num_sparse, cat_enc], format="csr")
        return combined.toarray()

    X_train = build_split_from_train(X_train_idx).astype(np.float32)

    if X_num_test is None:
        x_val = build_split_from_train(x_val_idx).astype(np.float32)
        x_test = build_split_from_train(x_test_idx).astype(np.float32)
        y_test = y_test.astype(np.float32)
    else:
        x_val = build_split_from_train(x_val_idx).astype(np.float32)
        def build_split_from_test():
            num_part = X_num_test
            if X_cat_test is not None:
                cat_part = X_cat_test
            else:
                cat_part = None
            num_t = num_imputer.transform(num_part)
            num_scaled = (num_t - mean) / std
            if cat_part is None or encoder is None or cat_imputer is None:
                return num_scaled
            cat_t = cat_imputer.transform(cat_part)
            cat_enc = encoder.transform(cat_t)
            num_sparse = sp.csr_matrix(num_scaled)
            combined = sp.hstack([num_sparse, cat_enc], format="csr")
            return combined.toarray()

        x_test = build_split_from_test().astype(np.float32)
        y_test = y_test_full.astype(np.float32)

    return (
        X_train.astype(np.float32),
        x_val.astype(np.float32),
        x_test.astype(np.float32),
        y_val.astype(np.float32),
        y_test.astype(np.float32),
        mean.astype(np.float32),
        std.astype(np.float32),
    )
