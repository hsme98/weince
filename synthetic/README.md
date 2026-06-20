# Synthetic noise floor simulation

This folder contains the synthetic experiment used to illustrate the statistical consequence of using a misspecified top-one link.

The data are generated from an endpoint-shortfall, or Weibit, winner law. The target is the true top-vs-reference log-odds. A correctly specified Weibit estimator has the expected Fisher-style `1 / n` error decay, while a restricted translation-coordinate softmax converges to a pseudo-true parameter and keeps a nonvanishing bias floor.

<p align="center">
  <img src="noise_floor.png" width="640" alt="Synthetic noise floor simulation">
</p>

## Files

```text
noise_floor_simulation.ipynb  # executed notebook with the full derivation and experiments
noise_floor_simulation.py     # standalone script that regenerates the final plot
noise_floor.png               # final figure used in the paper and README
```

## Run

From the repository root:

```bash
python synthetic/noise_floor_simulation.py
```

This writes:

```text
synthetic/noise_floor_results.csv
synthetic/noise_floor.png
```

The script uses only NumPy, pandas, and Matplotlib.
