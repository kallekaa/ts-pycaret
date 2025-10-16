# PyCaret Time Series Experiment Playground

This project wraps the PyCaret Time Series module in a small, scriptable workflow so you can benchmark multiple forecasting models, capture their evaluation statistics, and persist visuals for the top performers. By default, it runs against PyCaret's classic airline passengers dataset, but you can point it at any CSV with a univariate series.

## Environment Setup

1. Use Python 3.11 (matching `.python-version`).  
2. Create a virtual environment (example: `python -m venv .venv` then activate it).  
3. Install dependencies:

   ```bash
   pip install -e .
   ```

   If you use [uv](https://github.com/astral-sh/uv), you can instead run `uv sync`.

## Running the Demo

From the project root:

```bash
python main.py --forecast-horizon 24 --folds 3 --top-n 3 --output-dir artifacts
```

Key options:

- `--forecast-horizon`: future periods to predict (default 24).  
- `--folds`: expanding-window cross-validation folds (default 3).  
- `--top-n`: how many best models to analyse and plot (default 3).  
- `--turbo`: add this flag to run PyCaret's turbo mode if you need quicker experimentation.  
- `--seasonal-period`: manually hint a seasonal period when auto-detection struggles.  
- `--sort-metric`: choose the PyCaret leaderboard metric (default `MASE`).

## Using Your Own Data

```bash
python main.py \
  --csv-path data/sales.csv \
  --time-column date \
  --target-column revenue \
  --frequency MS \
  --forecast-horizon 12
```

- `--csv-path`: points to a CSV containing the time series.  
- `--time-column`: optional datetime column; if supplied the script sorts and indexes by it.  
- `--target-column`: choose the numeric column to forecast (defaults to the first numeric column).  
- `--frequency`: enforce a pandas frequency (e.g. `D`, `MS`, `Q`). Leave unset to keep the raw index.

## What Gets Saved

All outputs land in the directory passed via `--output-dir` (defaults to `artifacts/`):

- `leaderboard.csv`: complete PyCaret leaderboard with every evaluated model.  
- `top_models_summary.csv` / `.json`: the subset of rows for the top *N* models.  
- `plots/*.png`: forecast charts for the top *N* models, annotated with their numeric metrics.  
- `metrics/*.json`: combined leaderboard metrics and descriptive statistics (`count`, `mean`, `std`, etc.) for each model's forecasted values.  
- `predictions/*.csv`: raw forecasts produced by each top model.

Each plot is generated with Matplotlib via PyCaret's `plot_model` helper, then annotated with metrics such as MASE, RMSE, and SMAPE so you can compare models at a glance.

## Next Steps

- Swap in your own dataset and tweak cross-validation folds to match its cadence.  
- Persist the serialized models with `pycaret.time_series.save_model` if you need to deploy the finalists.  
- Automate scheduled runs (e.g. daily with new data) by wrapping `main.py` in your orchestration tool of choice.
