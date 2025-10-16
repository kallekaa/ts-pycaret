from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pycaret.datasets import get_data
from pycaret.time_series import TSForecastingExperiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a PyCaret time series experiment and persist the results."
    )
    data_source = parser.add_mutually_exclusive_group()
    data_source.add_argument(
        "--dataset-name",
        default="airline",
        help="Name of a PyCaret sample dataset to load (default: airline).",
    )
    data_source.add_argument(
        "--csv-path",
        type=Path,
        help="Path to a CSV file containing the time series.",
    )
    parser.add_argument(
        "--time-column",
        help=(
            "Name of the datetime column. Required when using --csv-path with a datetime index."
        ),
    )
    parser.add_argument(
        "--target-column",
        help=(
            "Name of the target column. Defaults to the first numeric column in the CSV."
        ),
    )
    parser.add_argument(
        "--frequency",
        help="Optional pandas frequency string to enforce on the datetime index.",
    )
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        default=24,
        help="Number of future periods to forecast (default: 24).",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=3,
        help="Number of cross-validation folds (default: 3).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of top models to report and visualise (default: 3).",
    )
    parser.add_argument(
        "--seasonal-period",
        type=int,
        help="Optional seasonal period hint passed to PyCaret.",
    )
    parser.add_argument(
        "--session-id",
        type=int,
        default=123,
        help="Random seed for reproducibility (default: 123).",
    )
    parser.add_argument(
        "--sort-metric",
        default="MASE",
        help="Metric used to sort the leaderboard (default: MASE).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory where experiment outputs will be saved (default: artifacts).",
    )
    parser.add_argument(
        "--turbo",
        action="store_true",
        help=(
            "Enable PyCaret turbo mode for faster comparisons (may skip some models)."
        ),
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def slugify(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def load_series_from_csv(
    path: Path,
    time_column: Optional[str],
    target_column: Optional[str],
    frequency: Optional[str],
) -> pd.Series:
    df = pd.read_csv(path)
    if time_column:
        if time_column not in df.columns:
            raise ValueError(f"Time column '{time_column}' not found in {path}.")
        df[time_column] = pd.to_datetime(df[time_column], errors="coerce")
        if df[time_column].isna().any():
            raise ValueError(
                f"Column '{time_column}' contains non-datetime values after parsing."
            )
        df = df.sort_values(time_column)
        df = df.set_index(time_column)
        if frequency:
            df = df.asfreq(frequency)
    if target_column:
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in {path}.")
        series = df[target_column]
    else:
        numeric_cols = df.select_dtypes(include="number").columns
        if not numeric_cols.any():
            raise ValueError("No numeric columns found to use as the target series.")
        series = df[numeric_cols[0]]
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    series.name = series.name or "target"
    return series


def load_time_series(
    dataset_name: Optional[str],
    csv_path: Optional[Path],
    time_column: Optional[str],
    target_column: Optional[str],
    frequency: Optional[str],
) -> pd.Series:
    if csv_path:
        logging.info("Loading time series from %s", csv_path)
        return load_series_from_csv(csv_path, time_column, target_column, frequency)
    dataset = dataset_name or "airline"
    logging.info("Loading PyCaret sample dataset '%s'", dataset)
    series = get_data(dataset)
    if isinstance(series, pd.DataFrame):
        if series.shape[1] != 1:
            raise ValueError(
                f"Dataset '{dataset}' returned multiple columns. "
                "Please specify --target-column to select one."
            )
        series = series.iloc[:, 0]
    series.name = series.name or dataset
    return series


def ensure_output_dirs(base_dir: Path) -> dict[str, Path]:
    outputs = {
        "base": base_dir,
        "plots": base_dir / "plots",
        "metrics": base_dir / "metrics",
        "predictions": base_dir / "predictions",
    }
    for path in outputs.values():
        path.mkdir(parents=True, exist_ok=True)
    return outputs


def extract_numeric_metrics(row: pd.Series) -> dict[str, float]:
    metrics = {
        k: float(v)
        for k, v in row.items()
        if isinstance(v, (int, float)) and pd.notna(v)
    }
    return metrics


def series_descriptive_stats(series: pd.Series) -> dict[str, float]:
    stats = series.describe(include="all")
    numeric_stats = {}
    for key in stats.index:
        value = stats[key]
        if isinstance(value, (int, float)):
            numeric_stats[str(key)] = float(value)
    return numeric_stats


def plot_forecast(
    history: pd.Series,
    forecast: pd.Series,
    model_label: str,
    metrics: dict[str, float],
    forecast_stats: dict[str, float],
) -> plt.Figure:
    history = history.sort_index()
    forecast = forecast.sort_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    history.plot(ax=ax, label="History", color="#1f77b4")
    forecast.plot(ax=ax, label="Forecast", color="#ff7f0e")

    try:
        cutoff = history.index[-1]
        ax.axvline(cutoff, linestyle="--", color="gray", linewidth=1, label="Forecast start")
    except Exception:
        logging.debug("Unable to draw forecast cutoff marker.", exc_info=True)

    ax.set_title(f"{model_label} Forecast")
    ax.set_xlabel("Time")
    ax.set_ylabel(history.name or "value")
    ax.legend()

    summary_lines = []
    if metrics:
        summary_lines.append("Metrics:")
        for name, value in sorted(metrics.items()):
            summary_lines.append(f"  {name}: {value:,.3f}")
    if forecast_stats:
        selected_keys = ["mean", "std", "min", "max"]
        available = [k for k in selected_keys if k in forecast_stats]
        if available:
            summary_lines.append("Forecast stats:")
            for key in available:
                summary_lines.append(f"  {key}: {forecast_stats[key]:,.3f}")

    if summary_lines:
        ax.text(
            0.02,
            0.98,
            "\n".join(summary_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

    fig.tight_layout()
    return fig


def run_experiment(args: argparse.Namespace) -> None:
    outputs = ensure_output_dirs(args.output_dir)
    data = load_time_series(
        args.dataset_name, args.csv_path, args.time_column, args.target_column, args.frequency
    )
    logging.info("Fetched %d observations for the experiment.", len(data))

    exp = TSForecastingExperiment()
    exp.setup(
        data=data,
        fh=args.forecast_horizon,
        fold=args.folds,
        session_id=args.session_id,
        seasonal_period=args.seasonal_period,
        n_jobs=-1,
        verbose=False,
    )
    logging.info("Comparing models (%s metric)...", args.sort_metric)
    comparison = exp.compare_models(
        n_select=max(1, args.top_n),
        sort=args.sort_metric,
        turbo=args.turbo,
        errors="raise",
    )
    models = comparison if isinstance(comparison, list) else [comparison]
    leaderboard = exp.pull()
    leaderboard_path = outputs["base"] / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    logging.info("Leaderboard written to %s", leaderboard_path)

    top_rows = leaderboard.head(len(models)).copy()
    top_summary_path = outputs["base"] / "top_models_summary.csv"
    top_rows.to_csv(top_summary_path, index=False)

    with (outputs["base"] / "top_models_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(top_rows.to_dict(orient="records"), fp, indent=2)

    for idx, ((_, leaderboard_row), model) in enumerate(zip(top_rows.iterrows(), models), start=1):
        model_label = str(leaderboard_row.get("Model", f"model_{idx}"))
        label_slug = slugify(model_label)
        logging.info("Processing model #%d: %s", idx, model_label)

        finalized_model = exp.finalize_model(model)
        forecast = exp.predict_model(finalized_model, fh=args.forecast_horizon)
        if isinstance(forecast, pd.DataFrame):
            forecast_series = (
                forecast["y_pred"] if "y_pred" in forecast.columns else forecast.iloc[:, 0]
            )
        else:
            forecast_series = forecast

        prediction_path = outputs["predictions"] / f"{idx:02d}_{label_slug}_forecast.csv"
        if isinstance(forecast, pd.DataFrame):
            forecast.to_csv(prediction_path, index=True)
        else:
            forecast_series.to_frame(name="y_pred").to_csv(prediction_path, index=True)

        metrics = extract_numeric_metrics(leaderboard_row)
        forecast_stats = series_descriptive_stats(forecast_series)
        combined_stats = {
            "leaderboard_metrics": metrics,
            "forecast_descriptive_stats": forecast_stats,
        }
        stats_path = outputs["metrics"] / f"{idx:02d}_{label_slug}_stats.json"
        with stats_path.open("w", encoding="utf-8") as fp:
            json.dump(combined_stats, fp, indent=2)

        fig = plot_forecast(
            history=data,
            forecast=forecast_series,
            model_label=model_label,
            metrics=metrics,
            forecast_stats=forecast_stats,
        )
        fig_path = outputs["plots"] / f"{idx:02d}_{label_slug}_forecast.png"
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logging.info("Saved forecast plot to %s", fig_path)


def main():
    configure_logging()
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
