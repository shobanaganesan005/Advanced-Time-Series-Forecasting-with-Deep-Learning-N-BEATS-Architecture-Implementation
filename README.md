# Advanced Time Series Forecasting with N-BEATS

This project implements the N-BEATS (Neural Basis Expansion Analysis for Interpretable Time Series Forecasting) architecture
for long‑horizon forecasting on a synthetic multi‑variate time series dataset, and compares it against a SARIMA baseline.
It also includes an ablation study disabling trend/seasonality stacks.

## Contents

- `src/data_generator.py` – generate a synthetic multi‑variate dataset with level, trend, seasonality and noise.
- `src/nbeats.py` – core N‑BEATS model implementation in PyTorch.
- `src/train_nbeats.py` – end‑to‑end training/evaluation pipeline, including basic hyperparameter search.
- `src/baseline_sarima.py` – classical SARIMA baseline using `statsmodels`.
- `src/evaluate.py` – common metrics (MAE, RMSE, sMAPE) and plotting helpers.
- `data/synthetic_timeseries.csv` – example generated dataset (for quick experiments).
- `report/report.md` – project report and analysis write‑up.

## Quick Start

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Regenerate / customise dataset (optional)**

```bash
python src/data_generator.py --n-series 200 --timesteps 300 --output data/synthetic_timeseries.csv
```

3. **Train N‑BEATS**

```bash
python src/train_nbeats.py \
  --data-path data/synthetic_timeseries.csv \
  --input-length 48 \
  --forecast-horizon 24 \
  --max-epochs 30
```

4. **Run SARIMA baseline (per‑series)**

```bash
python src/baseline_sarima.py \
  --data-path data/synthetic_timeseries.csv \
  --series-idx 0 \
  --forecast-horizon 24
```

Results (metrics tables and plots) are written to the `outputs/` directory.

## Notes

- The implementation is intentionally educational: the model and training loop are written from scratch with comments.
- Hyperparameter search is a simple random/grid search over a few key parameters (learning rate, stack size, etc.).
- GPU is **optional** but recommended; enable it with `--device cuda` if available.
