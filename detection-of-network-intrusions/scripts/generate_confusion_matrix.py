#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import numpy as np
import re
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")


def find_metric_files(results_dir: Path, dataset: str):
    base = results_dir / dataset
    if not base.exists():
        return []
    return list(base.rglob("*.json"))


def read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def infer_trained_on(path: Path, data: dict):
    parts = path.parts
    trained = None
    if "results" in parts:
        i = parts.index("results")
        if i + 2 < len(parts):
            trained = parts[i + 2]
    if not trained:
        trained = data.get("trained_on") or data.get("train_label")
    return trained or "unknown"


def build_confusion_matrix(files):
    agg = {}
    def normalize_label(s: str) -> str:
        if s is None:
            return ""
        s = str(s).strip().lower()
        s = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", s)
        return s

    for p in files:
        data = read_json(p)
        if not data:
            continue
        raw_label = data.get("test_on") or data.get("tested_on") or p.parent.name
        test_label = normalize_label(raw_label)

        tp = data.get("tp")
        fp = data.get("fp")
        tn = data.get("tn")
        fn = data.get("fn")

        if tp is None or fp is None or tn is None or fn is None:
            conf = data.get("confusion")
            if conf and isinstance(conf, (list, tuple)) and len(conf) == 2:
                try:
                    tn, fp = conf[0]
                    fn, tp = conf[1]
                except Exception:
                    tn = fp = fn = tp = None
        if any(v is None for v in (tp, fp, tn, fn)):
            continue

        if test_label not in agg:
            agg[test_label] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        agg[test_label]["tp"] += int(tp)
        agg[test_label]["fp"] += int(fp)
        agg[test_label]["tn"] += int(tn)
        agg[test_label]["fn"] += int(fn)

    return agg


def plot_2x2_matrices(agg_counts: dict, classes: list[str], out_path: Path, title: str = "Confusion Matrices"):
    n = len(classes)
    cols = 3
    rows = (n + cols - 1) // cols
    figsize = (cols * 4.5, rows * 4)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for i, cls in enumerate(classes):
        ax = axes[i]
        counts = agg_counts.get(cls)
        if not counts:
            ax.text(0.5, 0.5, f"No data for {cls}", ha="center", va="center")
            ax.set_axis_off()
            continue

        tn = counts.get("tn", 0)
        fp = counts.get("fp", 0)
        fn = counts.get("fn", 0)
        tp = counts.get("tp", 0)

        mat = np.array([[tn, fp], [fn, tp]])

        im = ax.imshow(mat, cmap="Blues", vmin=0)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Normal", "Pred Anomaly"], rotation=45, ha="right")
        ax.set_yticklabels(["True Normal", "True Anomaly"])
        ax.set_title(f"{cls}")

        vmax = mat.max() if mat.size > 0 else 1
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(mat[r, c]), ha="center", va="center", color=("white" if mat[r, c] > vmax / 2 else "black"))

        fig.colorbar(im, ax=ax)

    for j in range(n, len(axes)):
        axes[j].set_axis_off()

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def main():
    p = argparse.ArgumentParser(description="Generate confusion-matrix heatmap(s) from results JSONs")
    p.add_argument("--dataset", default="kddcup", help="dataset folder under results to process")
    p.add_argument("--out", default=None, help="output PNG path (defaults to results/<dataset>_confusion.png)")
    args = p.parse_args()

    files = find_metric_files(RESULTS_DIR, args.dataset)
    agg = build_confusion_matrix(files)

    classes = ["dos", "probe", "r2l", "u2r", "normal"]
    def norm_list(lst):
        return [re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", s.strip().lower()) for s in lst]
    classes = norm_list(classes)

    out = Path(args.out) if args.out else RESULTS_DIR / f"{args.dataset}_5_confusion_matrices.png"
    saved = plot_2x2_matrices(agg, classes, out, title=f"Confusion matrices for {args.dataset}")
    if saved:
        print(f"Saved confusion matrices to {saved}")


if __name__ == "__main__":
    main()
