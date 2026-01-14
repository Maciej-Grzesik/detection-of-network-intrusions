import os
import random
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import scipy.sparse as sp
from sklearn.impute import SimpleImputer

# Column names for KDD Cup 99 dataset
KDD_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
]

CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]

DOS_LABELS = {
    "back",
    "land",
    "neptune",
    "pod",
    "smurf",
    "teardrop",
    "mailbomb",
    "apache2",
    "processtable",
    "udpstorm",
}

PROBE_LABELS = {
    "ipsweep",
    "nmap",
    "portsweep",
    "satan",
    "mscan",
    "saint",
}

R2L_LABELS = {
    "ftp_write",
    "guess_passwd",
    "imap",
    "multihop",
    "phf",
    "spy",
    "warezclient",
    "warezmaster",
    "sendmail",
    "named",
    "snmpgetattack",
    "snmpguess",
    "xlock",
    "xsnoop",
    "worm",
}

U2R_LABELS = {
    "buffer_overflow",
    "loadmodule",
    "perl",
    "rootkit",
    "httptunnel",
    "ps",
    "sqlattack",
    "xterm",
}

GROUP_LABELS = {
    "normal": "normal",
    "dos": "dos",
    "probe": "probe",
    "r2l": "r2l",
    "u2r": "u2r",
}


def _safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return np.nan


def load_kddcup(
    path: str,
    train_label: str = "normal.",
    train_size: float = 0.6,
    val_size: float = 0.2,
    random_state: int = 42,
    row_limit: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    dtypes = {col: np.float32 for col in KDD_COLUMNS}
    for c in CATEGORICAL_COLUMNS + ["label"]:
        dtypes[c] = "category"

    df = pd.read_csv(
        path,
        names=KDD_COLUMNS,
        header=None,
        nrows=row_limit,
        dtype=dtypes,
        na_values=["?", ""],
        low_memory=False,
    )

    X_raw = df.drop(columns=["label"])
    y_raw = df["label"].astype(str).values

    def _normalize_label(label: str) -> str:
        return label.strip().lower().rstrip(".")

    def _label_to_group(label: str) -> str:
        lbl = _normalize_label(label)
        if lbl in GROUP_LABELS:
            return GROUP_LABELS[lbl]
        if lbl in DOS_LABELS:
            return "dos"
        if lbl in PROBE_LABELS:
            return "probe"
        if lbl in R2L_LABELS:
            return "r2l"
        if lbl in U2R_LABELS:
            return "u2r"
        return "normal" if lbl == "normal" else "unknown"

    target_group = _label_to_group(train_label)
    y_groups = np.array([_label_to_group(lab) for lab in y_raw])
    y = (y_groups != target_group).astype(np.float32)  # anomalies = 1

    X_cat = X_raw[CATEGORICAL_COLUMNS].astype(str).to_numpy(dtype=object)
    num_cols = [c for c in X_raw.columns if c not in CATEGORICAL_COLUMNS]
    X_num = X_raw[num_cols].to_numpy(dtype=np.float32)

    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")
    encoder = OneHotEncoder(sparse_output=True, handle_unknown="ignore")

    normal_mask = y == 0.0

    num_imputer.fit(X_num[normal_mask])
    cat_imputer.fit(X_cat[normal_mask])
    encoder.fit(cat_imputer.transform(X_cat[normal_mask]))

    n_samples = X_num.shape[0]
    indices = np.arange(n_samples)

    normal_indices = indices[normal_mask]
    X_train_idx, _ = train_test_split(
        normal_indices, test_size=(1.0 - train_size), shuffle=True, random_state=random_state
    )

    test_fraction = (1.0 - train_size - val_size) / (1.0 - train_size)
    x_temp_idx, x_test_idx, y_temp, y_test = train_test_split(
        indices, y, test_size=test_fraction, shuffle=True, random_state=random_state
    )
    x_val_idx = x_temp_idx
    y_val = y_temp

    num_train_t = num_imputer.transform(X_num[X_train_idx])
    mean = num_train_t.mean(axis=0, keepdims=True)
    std = num_train_t.std(axis=0, keepdims=True)
    std[std == 0] = 1.0

    def build_split(idx_array: np.ndarray) -> np.ndarray:
        num_part = X_num[idx_array]
        cat_part = X_cat[idx_array]
        num_t = num_imputer.transform(num_part)
        num_scaled = (num_t - mean) / std
        cat_t = cat_imputer.transform(cat_part)
        cat_enc = encoder.transform(cat_t)  # sparse matrix
        num_sparse = sp.csr_matrix(num_scaled)
        combined = sp.hstack([num_sparse, cat_enc], format="csr")
        return combined.toarray()

    X_train = build_split(X_train_idx).astype(np.float32)
    x_val = build_split(x_val_idx).astype(np.float32)
    x_test = build_split(x_test_idx).astype(np.float32)

    return (
        X_train.astype(np.float32),
        x_val.astype(np.float32),
        x_test.astype(np.float32),
        y_val.astype(np.float32),
        y_test.astype(np.float32),
        mean.astype(np.float32),
        std.astype(np.float32),
    )
