import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

import problexity as px
from problexity import classification as c


def allocate_quotas(class_counts: dict[int, int], sample_size: int) -> dict[int, int]:
    total = sum(class_counts.values())
    raw = {k: sample_size * v / total for k, v in class_counts.items()}
    q = {k: int(np.floor(v)) for k, v in raw.items()}
    rem = sample_size - sum(q.values())
    order = sorted(raw.keys(), key=lambda k: raw[k] - q[k], reverse=True)
    for k in order[:rem]:
        q[k] += 1
    return q


def _process_chunk_for_sampling(chunk, label_col, quotas, taken, rng):
    chunk[label_col] = chunk[label_col].astype(int)
    picked = []

    for cls, target in quotas.items():
        need = target - taken[cls]
        if need <= 0:
            continue
        sub = chunk[chunk[label_col] == cls]
        if sub.empty:
            continue
        n = min(need, len(sub))
        rs = int(rng.integers(0, 2**31 - 1))
        picked.append(sub.sample(n=n, random_state=rs))
        taken[cls] += n

    return picked


def stratified_fold_from_csv(
    csv_path: Path,
    sample_size: int = 10_000,
    chunksize: int = 200_000,
    random_state: int = 42,
):
    cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    label_col = cols[-1]

    class_counts = Counter()
    for chunk in pd.read_csv(csv_path, usecols=[label_col], chunksize=chunksize):
        vc = chunk[label_col].astype(int).value_counts()
        for cls, cnt in vc.items():
            class_counts[int(cls)] += int(cnt)

    quotas = allocate_quotas(dict(class_counts), sample_size)
    taken = dict.fromkeys(quotas, 0)
    parts = []
    rng = np.random.default_rng(random_state)

    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        picked = _process_chunk_for_sampling(chunk, label_col, quotas, taken, rng)

        if picked:
            parts.append(pd.concat(picked, axis=0))

        if all(taken[k] >= quotas[k] for k in quotas):
            break

    fold = pd.concat(parts, axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return fold, label_col, dict(class_counts), quotas


project_root = Path(__file__).resolve().parents[1]
csv_path = project_root / "data" / "kitsune" / "kitsune_4M.csv"

fold_df, label_col, full_counts, fold_quotas = stratified_fold_from_csv(
    csv_path=csv_path,
    sample_size=10_000,
    chunksize=200_000,
    random_state=42,
)

X = fold_df.drop(columns=[label_col]).to_numpy(dtype=np.float32)
y = fold_df[label_col].to_numpy(dtype=np.int32)

print("full_counts:", full_counts)
print("fold_quotas:", fold_quotas)
print("X:", X.shape, "y:", y.shape, "classes:", np.unique(y))

metrics = [c.n1, c.n2, c.n3, c.n4, c.lsc, c.t1, c.f2, c.f3, c.f4]

cc = px.ComplexityCalculator(metrics=metrics, mode="classification", multiclass_strategy="ovo")
cc.fit(X, y)

print(cc.report())

fig = plt.figure(figsize=(8, 8))
cc.plot(fig, (1, 1, 1))

out_path = project_root / "reports" / "plots" / "problexity_selected_metrics_stratified_10k.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved figure: {out_path}")
