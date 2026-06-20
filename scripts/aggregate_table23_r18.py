#!/usr/bin/env python3
"""Aggregate CIFAR-100 ResNet-18 linear/kNN results for Table 2/3 reproduction.

Fixes relative to the earlier helper:
  * Does not group on config.yaml `eval_seeds`.
  * Averages linear probe seeds or kNN split seeds inside each pretrained encoder.
  * Computes confidence intervals across pretrained encoder seeds.
  * Selects hyperparameter configurations by validation metric, preferring complete four-seed configs.
"""
from __future__ import annotations

import argparse
import glob
import json
from math import sqrt
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from scipy.stats import t

LINEAR_HEADER = [
    "seed", "epoch_num", "eval_epochs", "val loss", "val accuracy", "test loss", "test accuracy"
]
EXPECTED_ENCODER_SEEDS = 4
SKIP_CONFIG_KEYS = {"eval_seeds"}
DEFAULT_DISCARD_COLS = {"dataset_dir", "model_path", "run_dir", "device"}


def _safe_config_value(value):
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), sort_keys=True)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return value


def _read_linear_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if {"seed", "val accuracy", "test accuracy"}.issubset(df.columns):
        return df
    raw = pd.read_csv(path, header=None)
    if raw.shape[1] != len(LINEAR_HEADER):
        raise ValueError(f"unexpected shape {raw.shape} for {path}")
    if [str(x).strip() for x in raw.iloc[0].tolist()] == LINEAR_HEADER:
        raw = raw.iloc[1:].reset_index(drop=True)
    raw.columns = LINEAR_HEADER
    return raw


def get_dataframe(base_dir: str | Path, result_type: str = "linear") -> pd.DataFrame:
    base_dir = Path(base_dir)
    files = sorted(glob.glob(str(base_dir / f"**/{result_type}_eval_results_v2.csv"), recursive=True))
    frames = []
    for filename in files:
        path = Path(filename)
        try:
            df = _read_linear_csv(path) if result_type == "linear" else pd.read_csv(path)
        except Exception as exc:
            print(f"[warn] failed to read {path}: {exc}")
            continue
        if df.empty:
            continue
        if "seed" in df.columns:
            df = df.rename(columns={"seed": "eval seed"})
        config_path = path.parent / "config.yaml"
        if not config_path.exists():
            print(f"[warn] missing config.yaml next to {path}")
            continue
        cfg = yaml.safe_load(config_path.read_text()) or {}
        for key, value in cfg.items():
            if key in SKIP_CONFIG_KEYS:
                continue
            df[key] = _safe_config_value(value)
        df["run_dir"] = str(path.parent)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=0, ignore_index=True)
    for col in out.columns:
        if col.startswith(("val", "test")) or col in {"eval seed", "split_seed"}:
            out[col] = pd.to_numeric(out[col], errors="ignore")
    return out


def _ci(vals: np.ndarray, alpha: float = 0.05):
    vals = pd.to_numeric(pd.Series(vals), errors="coerce").dropna().to_numpy(dtype=float)
    n = int(vals.size)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    mean = float(vals.mean())
    std = float(vals.std(ddof=1)) if n > 1 else 0.0
    ci = float(t.ppf(1 - alpha / 2, n - 1) * std / sqrt(n)) if n > 1 else 0.0
    return mean, std, ci, n


def summarize_hierarchical(
    df: pd.DataFrame,
    run_seed_col: str = "seed",
    eval_seed_col: str = "eval seed",
    test_col: str = "test accuracy",
    val_col: str = "val accuracy",
    test_loss_col: str | None = "test loss",
    val_loss_col: str | None = "val loss",
    discard_cols: Iterable[str] = tuple(DEFAULT_DISCARD_COLS),
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    metrics = [test_col, val_col, test_loss_col, val_loss_col]
    metrics = [m for m in metrics if m is not None and m in df.columns]
    excluded = set(metrics) | set(discard_cols) | SKIP_CONFIG_KEYS | {eval_seed_col}
    group_cols = [c for c in df.columns if c not in excluded]
    per_encoder = df.groupby(group_cols, dropna=False, as_index=False)[metrics].mean()
    config_cols = [c for c in group_cols if c != run_seed_col]
    rows = []
    for key, g in per_encoder.groupby(config_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {c: key[i] for i, c in enumerate(config_cols)}
        seeds = sorted(pd.Series(g[run_seed_col]).dropna().astype(str).unique().tolist())
        row["encoder_seed_count"] = len(seeds)
        row["encoder_seeds"] = ",".join(seeds)
        row["is_complete_config"] = len(seeds) >= EXPECTED_ENCODER_SEEDS
        for m in metrics:
            mean, std, ci, n = _ci(g[m].values)
            row[f"{m}_mean"] = mean
            row[f"{m}_std_enc"] = std
            row[f"{m}_n_enc"] = n
            row[f"{m}_ci_halfwidth"] = ci
        rows.append(row)
    return pd.DataFrame(rows)


def choose_best(summary: pd.DataFrame, select_col: str, group_cols: list[str], min_n_col: str | None = None) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows = []
    for keys, g in summary.groupby(group_cols, dropna=False):
        cand = g
        if min_n_col and min_n_col in g.columns:
            complete = g[g[min_n_col] >= EXPECTED_ENCODER_SEEDS]
            if not complete.empty:
                cand = complete
            else:
                print(f"[warn] no complete configs for group={keys}; falling back to all rows")
        rows.append(cand.loc[cand[select_col].idxmax()])
    return pd.DataFrame(rows).reset_index(drop=True)


def summarize_linear(base_dir: str | Path) -> pd.DataFrame:
    df = get_dataframe(base_dir, "linear")
    if df.empty:
        return pd.DataFrame()
    summary = summarize_hierarchical(df)
    return choose_best(
        summary,
        select_col="val accuracy_mean",
        group_cols=["policy", "use_weib_topm", "dataset", "resnet"],
        min_n_col="val accuracy_n_enc",
    )


def summarize_knn(base_dir: str | Path) -> pd.DataFrame:
    df = get_dataframe(base_dir, "knn")
    if df.empty:
        return pd.DataFrame()
    val_cols = [c for c in df.columns if c.startswith("val_knn_")]
    test_cols = {c[len("test_"):]: c for c in df.columns if c.startswith("test_knn_")}
    rows = []
    for val_col in val_cols:
        suffix = val_col[len("val_"):]
        if suffix not in test_cols:
            continue
        test_col = test_cols[suffix]
        ignore = [c for c in val_cols if c != val_col] + [c for c in test_cols.values() if c != test_col]
        sub = df[[c for c in df.columns if c not in ignore]].copy()
        summary = summarize_hierarchical(
            sub, eval_seed_col="split_seed", val_col=val_col, test_col=test_col,
            test_loss_col=None, val_loss_col=None
        )
        best = choose_best(
            summary,
            select_col=f"{val_col}_mean",
            group_cols=["dataset", "resnet", "policy", "use_weib_topm"],
            min_n_col=f"{val_col}_n_enc",
        )
        best["selected_by"] = val_col
        best["selected_for"] = test_col
        rows.append(best)
    return pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame()


def compact_knn(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for keys, g in df.groupby(["dataset", "resnet", "policy", "use_weib_topm"], dropna=False):
        row = dict(zip(["dataset", "resnet", "policy", "use_weib_topm"], keys))
        for _, r in g.iterrows():
            metric = r["selected_for"]
            row[f"{metric}_mean"] = r.get(f"{metric}_mean", np.nan)
            row[f"{metric}_ci_halfwidth"] = r.get(f"{metric}_ci_halfwidth", np.nan)
            row[f"{metric}_n_enc"] = r.get(f"{metric}_n_enc", np.nan)
            for hp in ["select_aic_margin", "select_kappa_rho", "select_kappa_aic"]:
                if hp in r.index:
                    row[f"{metric}_{hp}"] = r.get(hp, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-dir", required=True)
    p.add_argument("--weince-dir", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    lin = pd.concat([summarize_linear(args.baseline_dir), summarize_linear(args.weince_dir)], ignore_index=True)
    lin = lin[(lin["dataset"] == "CIFAR100") & (lin["resnet"] == "resnet18")]
    lin.to_csv(out / "r18_linear_selected_fixed.csv", index=False)
    print("\n[linear]")
    keep = [c for c in ["dataset", "resnet", "policy", "use_weib_topm", "select_aic_margin", "select_kappa_rho", "select_kappa_aic", "test accuracy_mean", "test accuracy_ci_halfwidth", "val accuracy_mean", "val accuracy_ci_halfwidth", "test accuracy_n_enc", "val accuracy_n_enc"] if c in lin.columns]
    print(lin[keep].to_string(index=False))

    knn = pd.concat([summarize_knn(args.baseline_dir), summarize_knn(args.weince_dir)], ignore_index=True)
    knn = knn[(knn["dataset"] == "CIFAR100") & (knn["resnet"] == "resnet18")]
    knn.to_csv(out / "r18_knn_selected_long_fixed.csv", index=False)
    compact = compact_knn(knn)
    compact.to_csv(out / "r18_knn_selected_compact_fixed.csv", index=False)
    print("\n[kNN compact]")
    print(compact.to_string(index=False))


if __name__ == "__main__":
    main()
