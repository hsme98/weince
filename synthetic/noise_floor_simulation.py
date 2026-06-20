#!/usr/bin/env python3
"""Synthetic noise floor simulation for the WEINCE paper.

This script reproduces the toy experiment showing that a correctly specified
Weibit link removes the nonvanishing misspecification error floor left by a
restricted translation-coordinate softmax when the data are generated from an
endpoint-shortfall law.

It writes:
  - noise_floor.png
  - noise_floor_results.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z = z - np.max(z, axis=axis, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=axis, keepdims=True)


def shortfall_probs(scores: np.ndarray, beta: float = 1.0, endpoint: float = 1.0) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    q = endpoint - scores
    if np.any(q <= 0):
        raise ValueError("All scores must lie strictly below the endpoint.")
    w = q ** (-beta)
    return w / w.sum()


def softmax_probs_from_theta(theta: float | np.ndarray, features: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    features = np.asarray(features, dtype=float)
    return softmax(theta[..., None] * features[None, :], axis=-1)


def softmax_mean(theta: float | np.ndarray, features: np.ndarray) -> np.ndarray:
    probs = softmax_probs_from_theta(theta, features)
    return probs @ features


def invert_softmax_mean(target_mean: np.ndarray, features: np.ndarray, lo: float = -100.0, hi: float = 100.0) -> np.ndarray:
    """Invert theta -> E_theta[features] for a one-parameter softmax family."""
    target_mean = np.asarray(target_mean, dtype=float)
    features = np.asarray(features, dtype=float)
    out = np.empty_like(target_mean, dtype=float)
    fmin = features.min()
    fmax = features.max()

    clipped = np.clip(target_mean, fmin + 1e-10, fmax - 1e-10)
    for idx, target in np.ndenumerate(clipped):
        a, b = lo, hi
        for _ in range(100):
            mid = 0.5 * (a + b)
            m = float(np.asarray(softmax_mean(mid, features)).squeeze())
            if m < target:
                a = mid
            else:
                b = mid
        out[idx] = 0.5 * (a + b)
    return out


def run_simulation(seed: int = 20260113, n_reps: int = 12_000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # A bounded score vector and a shortfall data-generating law.
    scores = np.array([0.15, 0.45, 0.70, 0.90])
    beta_true = 0.8
    p_star = shortfall_probs(scores, beta=beta_true)

    # Endpoint coordinate, reference candidate, and top candidate.
    x_coord = -np.log1p(-scores)
    ref_idx = 0
    top_idx = int(np.argmax(scores))
    x_gap = float(x_coord[top_idx] - x_coord[ref_idx])
    s_gap = float(scores[top_idx] - scores[ref_idx])
    true_log_odds = beta_true * x_gap

    # Pseudo-true softmax slope in the raw score coordinate.
    mu_s = float(np.dot(p_star, scores))
    theta_soft_pseudo = float(invert_softmax_mean(np.array([mu_s]), scores)[0])
    q_soft_pseudo = softmax_probs_from_theta(theta_soft_pseudo, scores)[0]
    mu_q = float(np.dot(q_soft_pseudo, scores))
    A = float(np.dot(q_soft_pseudo, (scores - mu_q) ** 2))
    B = float(np.dot(p_star, (scores - mu_q) ** 2))
    godambe_floor = B / (A ** 2)

    # Correctly specified Weibit Fisher floor in the endpoint coordinate.
    mu_x = float(np.dot(p_star, x_coord))
    fisher_x = float(np.dot(p_star, (x_coord - mu_x) ** 2))
    weibit_floor_log_odds = (x_gap ** 2) / fisher_x

    softmax_pseudo_log_odds = theta_soft_pseudo * s_gap
    softmax_bias_floor = (softmax_pseudo_log_odds - true_log_odds) ** 2

    rows: list[dict[str, float]] = []
    for n in [100, 200, 500, 1_000, 2_000, 5_000, 10_000]:
        counts = rng.multinomial(n, p_star, size=n_reps)
        empirical = counts / n
        mean_s = empirical @ scores
        mean_x = empirical @ x_coord

        theta_soft_hat = invert_softmax_mean(mean_s, scores)
        beta_weibit_hat = invert_softmax_mean(mean_x, x_coord)

        lambda_soft_hat = theta_soft_hat * s_gap
        lambda_weibit_hat = beta_weibit_hat * x_gap

        rows.append(
            {
                "n": float(n),
                "mse_lambda_weibit": float(np.mean((lambda_weibit_hat - true_log_odds) ** 2)),
                "mse_lambda_softmax": float(np.mean((lambda_soft_hat - true_log_odds) ** 2)),
                "weibit_theory_c_over_n": float(weibit_floor_log_odds / n),
                "softmax_bias_floor": float(softmax_bias_floor),
                "true_log_odds": float(true_log_odds),
                "softmax_pseudo_log_odds": float(softmax_pseudo_log_odds),
                "softmax_godambe_floor": float(godambe_floor),
                "weibit_fisher_floor_log_odds": float(weibit_floor_log_odds),
            }
        )
    return pd.DataFrame(rows)


def make_plot(df: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(df["n"], df["mse_lambda_weibit"], marker="o", label="Weibit MSE on true log-odds")
    ax.plot(df["n"], df["mse_lambda_softmax"], marker="o", label="softmax MSE on true log-odds")
    ax.plot(df["n"], df["weibit_theory_c_over_n"], linestyle="--", label="Weibit theory ~ c / n")
    ax.axhline(float(df["softmax_bias_floor"].iloc[0]), linestyle="--", label="softmax bias floor")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Correctly specified Weibit removes the nonvanishing error floor")
    ax.set_xlabel("sample size n")
    ax.set_ylabel("MSE for true top-vs-reference log-odds")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260113)
    parser.add_argument("--n-reps", type=int, default=12_000)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = run_simulation(seed=args.seed, n_reps=args.n_reps)
    df.to_csv(args.out_dir / "noise_floor_results.csv", index=False)
    make_plot(df, args.out_dir / "noise_floor.png")
    print(f"Wrote {args.out_dir / 'noise_floor_results.csv'}")
    print(f"Wrote {args.out_dir / 'noise_floor.png'}")


if __name__ == "__main__":
    main()
