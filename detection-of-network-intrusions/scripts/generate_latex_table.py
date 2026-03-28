#!/usr/bin/env python3
import json
import os
from pathlib import Path
import glob
import pandas as pd

RESULTS_DIR = Path('results')


def find_metric_files(results_dir: Path):
    files = list(results_dir.rglob('*.json'))
    return files


def extract_row_from_json(path: Path):
    try:
        with open(path, 'r') as fh:
            data = json.load(fh)
    except Exception:
        return None
    parts = path.parts
    dataset = None
    trained_on = None
    if 'results' in parts:
        i = parts.index('results')
        if i + 1 < len(parts):
            dataset = parts[i+1]
        if i + 2 < len(parts):
            trained_on = parts[i+2]
    name = path.stem

    metric_keys = ['precision', 'recall', 'f1', 'auc', 'threshold']
    row = {
        'file': str(path),
        'dataset': dataset or '',
        'trained_on': trained_on or '',
    }
    for k in metric_keys:
        row[k] = data.get(k)
    # also include any top-level metadata fields
    if 'trained_on' in data:
        row['trained_on'] = data['trained_on']
    if 'dataset' in data:
        row['dataset'] = data['dataset']
    return row


def build_dataframe(files):
    rows = []
    for p in files:
        r = extract_row_from_json(Path(p))
        if r:
            rows.append(r)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df['trained_on'] = df['trained_on'].fillna('').astype(str)
    df['dataset'] = df['dataset'].fillna('').astype(str)
    cols = ['dataset', 'trained_on', 'precision', 'recall', 'f1']
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]


def df_to_latex(df: pd.DataFrame, caption: str = 'Results', label: str = 'tab:results') -> str:
    if df.empty:
        return '% No results found'
    fmt_df = df.copy()
    for c in ['precision', 'recall', 'f1']:
        if c in fmt_df.columns:
            fmt_df[c] = fmt_df[c].apply(lambda x: f"{x:.4f}" if isinstance(x, (float, int)) else '')
    latex = fmt_df.to_latex(index=False, caption=caption, label=label, na_rep='')
    return latex


if __name__ == '__main__':
    files = find_metric_files(RESULTS_DIR)
    df = build_dataframe(files)
    tex = df_to_latex(df, caption='Autoencoder results', label='tab:autoencoder_results')
    print(tex)
