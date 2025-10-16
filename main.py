from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from pycaret.datasets import get_data
from pycaret.time_series import TSForecastingExperiment


def parse_args() -> argparse.Namespace:
    today = date.today()
    default_start = date(today.year - 2, 1, 1).isoformat()
    default_end = date(today.year - 1, 12, 31).isoformat()

    parser = argparse.ArgumentParser(
        description="Run a PyCaret time series experiment and persist the results."
    )
    data_source = parser.add_mutually_exclusive_group()
    data_source.add_argument(
        "--dataset-name",
        default=None,
        help="Name of a PyCaret sample dataset to load.",
    )
    data_source.add_argument(
        "--csv-path",
        type=Path,
        help="Path to a CSV file containing the time series.",
    )
    data_source.add_argument(
        "--yfinance-ticker",
        default="GOOG",
        help="Ticker symbol to download daily closes from Yahoo Finance via yfinance.",
    )
    parser.add_argument(
        "--time-column",
        help=(
            "Name of the datetime column. Required when using --csv-path with a datetime index."
        ),
    )
    parser.add_argument(
        "--yfinance-start",
        default=default_start,
        help="Optional start date (YYYY-MM-DD) when fetching data with --yfinance-ticker.",
    )
    parser.add_argument(
        "--yfinance-end",
        default=default_end,
        help="Optional end date (YYYY-MM-DD) when fetching data with --yfinance-ticker.",
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
        default=30,
        help="Number of future periods to forecast (default: 30).",
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
        default=5,
        help="Number of top models to report and visualise (default: 5).",
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


def configure_logging(log_path: Path) -> None:
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    log_path.write_text("", encoding="utf-8")
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    stream_handler = logging.StreamHandler()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[file_handler, stream_handler],
    )


def slugify(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def resolve_series_name(*series_like: Optional[pd.Series | pd.DataFrame]) -> str:
    for obj in series_like:
        if obj is None:
            continue
        if isinstance(obj, pd.Series) and obj.name:
            return str(obj.name)
        if isinstance(obj, pd.DataFrame) and obj.columns.size == 1:
            col = obj.columns[0]
            if isinstance(col, str):
                return col
    return "value"


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


def load_series_from_yfinance(
    ticker: str,
    start: Optional[str],
    end: Optional[str],
    frequency: Optional[str],
) -> pd.Series:
    params: dict[str, object] = {"interval": "1d", "progress": False, "auto_adjust": False}
    if start:
        params["start"] = start
    if end:
        params["end"] = end
    if "start" not in params and "end" not in params:
        params["period"] = "max"
    data = yf.download(ticker, **params)
    if data.empty or "Close" not in data.columns:
        raise ValueError(f"No closing price data returned for ticker '{ticker}'.")
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        if close.shape[1] == 1:
            close = close.iloc[:, 0]
        else:
            raise ValueError(
                "Expected a single Close series but received multiple columns. "
                "Please request a single ticker."
            )
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.name = f"{ticker.upper()}_close"
    if frequency:
        try:
            close = close.asfreq(frequency)
        except Exception as exc:  # pragma: no cover
            logging.warning("Could not enforce frequency %s: %s", frequency, exc)
        close = close.ffill()
    else:
        inferred = pd.infer_freq(close.index)
        if inferred:
            close = close.asfreq(inferred)
            close = close.ffill()
        else:
            close = close.resample("B").ffill()
    return close


def load_time_series(
    dataset_name: Optional[str],
    csv_path: Optional[Path],
    yfinance_ticker: Optional[str],
    yfinance_start: Optional[str],
    yfinance_end: Optional[str],
    time_column: Optional[str],
    target_column: Optional[str],
    frequency: Optional[str],
) -> pd.Series:
    if yfinance_ticker:
        logging.info(
            "Downloading daily close prices for %s from Yahoo Finance.", yfinance_ticker
        )
        return load_series_from_yfinance(
            yfinance_ticker,
            start=yfinance_start,
            end=yfinance_end,
            frequency=frequency,
        )
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
        "logs": base_dir / "logs",
    }
    for key, path in outputs.items():
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
    test: Optional[pd.Series] = None,
    test_predictions: Optional[pd.Series] = None,
) -> plt.Figure:
    history = history.sort_index()
    forecast = forecast.sort_index()
    combined_actuals = history

    fig, ax = plt.subplots(figsize=(10, 5))
    history.plot(ax=ax, label="Train", color="#1f77b4")

    if test is not None:
        test = test.sort_index()
        test.plot(ax=ax, label="Test", color="#2ca02c")
        combined_actuals = pd.concat([history, test]).sort_index()
    else:
        combined_actuals = history

    if test_predictions is not None:
        test_predictions = test_predictions.sort_index()
        test_predictions.plot(
            ax=ax,
            label="Test prediction",
            color="#d62728",
            linestyle="--",
        )

    forecast.plot(ax=ax, label="Forecast", color="#ff7f0e")

    try:
        cutoff = combined_actuals.index[-1]
        ax.axvline(cutoff, linestyle="--", color="gray", linewidth=1, label="Forecast start")
    except Exception:
        logging.debug("Unable to draw forecast cutoff marker.", exc_info=True)

    series_name = history.name or (test.name if test is not None else forecast.name) or "value"
    ax.set_title(f"{model_label} Forecast")
    ax.set_xlabel("Time")
    ax.set_ylabel(series_name)
    ax.legend(loc="upper left", fontsize="small")

    summary_lines = []
    if metrics:
        desired_metrics = ["MAPE", "RMSE", "R2"]
        normalized = {name.upper(): (name, value) for name, value in metrics.items()}
        selected = [
            normalized[key]
            for key in desired_metrics
            if key in normalized
        ]
        if selected:
            summary_lines.append("Metrics:")
            for name, value in selected:
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
            0.5,
            0.98,
            "\n".join(summary_lines),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

    fig.tight_layout()
    return fig


def run_experiment(
    args: argparse.Namespace,
    outputs: dict[str, Path],
    run_timestamp: str,
    series_slug: str,
) -> None:
    data = load_time_series(
        args.dataset_name,
        args.csv_path,
        args.yfinance_ticker,
        args.yfinance_start,
        args.yfinance_end,
        args.time_column,
        args.target_column,
        args.frequency,
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
    train_series_raw = exp.get_config("y_train")
    test_series_raw = exp.get_config("y_test")

    def _normalize_series(raw):
        if raw is None:
            return None
        if isinstance(raw, pd.Series):
            return raw
        if isinstance(raw, pd.DataFrame):
            if raw.shape[1] == 1:
                return raw.iloc[:, 0]
            numeric_cols = raw.select_dtypes(include="number").columns
            if len(numeric_cols) > 0:
                logging.warning(
                    "Multiple columns detected; proceeding with the first numeric column '%s'.",
                    numeric_cols[0],
                )
                return raw[numeric_cols[0]]
            logging.warning(
                "Expected a single column time series but received %s columns; unable to select one.",
                raw.shape[1],
            )
            return None
        logging.warning("Unsupported time series type %s; skipping conversion.", type(raw))
        return None

    train_series = _normalize_series(train_series_raw)
    test_series = _normalize_series(test_series_raw)
    if train_series is None:
        train_series = _normalize_series(data) or data
    series_name = resolve_series_name(train_series, test_series, data)
    if isinstance(train_series, pd.Series):
        train_series = train_series.rename(series_name)
    if isinstance(test_series, pd.Series):
        test_series = test_series.rename(series_name)

    logging.info("Comparing models (%s metric)...", args.sort_metric)
    comparison = exp.compare_models(
        n_select=max(1, args.top_n),
        sort=args.sort_metric,
        turbo=args.turbo,
        errors="raise",
    )
    models = comparison if isinstance(comparison, list) else [comparison]
    leaderboard = exp.pull()
    identifier = f"{run_timestamp}_{series_slug}"
    leaderboard_path = outputs["base"] / f"{identifier}_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    logging.info("Leaderboard written to %s", leaderboard_path)

    top_rows = leaderboard.head(len(models)).copy()
    top_summary_path = outputs["base"] / f"{identifier}_top_models_summary.csv"
    top_rows.to_csv(top_summary_path, index=False)

    with (outputs["base"] / f"{identifier}_top_models_summary.json").open(
        "w", encoding="utf-8"
    ) as fp:
        json.dump(top_rows.to_dict(orient="records"), fp, indent=2)

    for idx, ((_, leaderboard_row), model) in enumerate(zip(top_rows.iterrows(), models), start=1):
        model_label = str(leaderboard_row.get("Model", f"model_{idx}"))
        label_slug = slugify(model_label)
        file_stem = f"{identifier}_{idx:02d}_{label_slug}"
        logging.info("Processing model #%d: %s", idx, model_label)

        finalized_model = exp.finalize_model(model)
        forecast = exp.predict_model(finalized_model, fh=args.forecast_horizon)
        if isinstance(forecast, pd.DataFrame):
            forecast_series = (
                forecast["y_pred"] if "y_pred" in forecast.columns else forecast.iloc[:, 0]
            )
        else:
            forecast_series = forecast

        test_predictions = None
        if test_series is not None:
            try:
                test_input = test_series.to_frame(name=series_name)
                test_pred_df = exp.predict_model(finalized_model, data=test_input)
                if isinstance(test_pred_df, pd.DataFrame):
                    if "y_pred" in test_pred_df.columns:
                        test_predictions = test_pred_df["y_pred"]
                    else:
                        test_predictions = test_pred_df.iloc[:, 0]
                elif isinstance(test_pred_df, pd.Series):
                    test_predictions = test_pred_df
                if test_predictions is not None:
                    test_predictions = pd.Series(test_predictions)
                    if len(test_predictions) != len(test_series):
                        test_predictions = test_predictions.reindex(test_series.index)
                    else:
                        test_predictions.index = test_series.index
                    test_predictions.name = "y_pred"
            except Exception:
                logging.warning("Unable to generate test predictions for %s.", model_label, exc_info=True)
                test_predictions = None

        prediction_path = outputs["predictions"] / f"{file_stem}_forecast.csv"
        if isinstance(forecast, pd.DataFrame):
            forecast.to_csv(prediction_path, index=True)
        else:
            forecast_series.to_frame(name="y_pred").to_csv(prediction_path, index=True)

        if test_predictions is not None:
            test_pred_path = outputs["predictions"] / f"{file_stem}_test_predictions.csv"
            test_predictions.to_frame(name="y_pred").to_csv(test_pred_path, index=True)

        metrics = extract_numeric_metrics(leaderboard_row)
        forecast_stats = series_descriptive_stats(forecast_series)
        combined_stats = {
            "leaderboard_metrics": metrics,
            "forecast_descriptive_stats": forecast_stats,
        }
        stats_path = outputs["metrics"] / f"{file_stem}_stats.json"
        with stats_path.open("w", encoding="utf-8") as fp:
            json.dump(combined_stats, fp, indent=2)

        fig = plot_forecast(
            history=train_series,
            forecast=forecast_series,
            model_label=model_label,
            metrics=metrics,
            forecast_stats=forecast_stats,
            test=test_series,
            test_predictions=test_predictions,
        )
        fig_path = outputs["plots"] / f"{file_stem}_forecast.png"
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logging.info("Saved forecast plot to %s", fig_path)


def main():
    args = parse_args()
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    series_slug = slugify(args.yfinance_ticker or args.dataset_name or "series")
    outputs = ensure_output_dirs(args.output_dir)
    log_path = outputs["logs"] / f"{run_timestamp}_{series_slug}_run.log"
    configure_logging(log_path)
    logging.info("Starting experiment for '%s' with timestamp %s", series_slug, run_timestamp)
    run_experiment(args, outputs, run_timestamp, series_slug)


if __name__ == "__main__":
    main()
