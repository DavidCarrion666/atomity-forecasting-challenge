"""Evaluation pipeline: compares all registered baselines via cross-validation.

Usage
-----
    python -m cwfc.evaluate --data challenge_data

Output (stdout):
    A ranked table of baselines sorted by cross-validated MAE.

Output (files, optional via --output-dir):
    results_<target>.csv   — per-fold metrics for each baseline

Overlapping validation windows and non-independence
---------------------------------------------------
In rolling-origin evaluation (step=1, horizon=24, min_train=48, n_total=110):
- 39 folds are evaluated, yielding 39 × 24 = 936 prediction-actual pairs.
- However, these 936 pairs cover only 62 distinct timestamps (indices 48 to 109).
- Observations in indices 71 to 86 appear in up to 24 distinct validation windows.
- Consequently, the 936 evaluation pairs are NOT independent observations.
- Aggregated metrics (MAE, RMSE, sMAPE) pool all pairs, but serial correlation
  is present both across folds and along the forecast horizon.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cwfc.baselines import BASELINE_REGISTRY
from cwfc.data import load_data, TRAIN_OBS
from cwfc.metrics import mae, rmse, smape, normalized_mae
from cwfc.validation import expanding_window_splits, cross_validate
from cwfc.models import (
    temporal_regression_forecast,
    arima_forecast,
    hist_gradient_boosting_forecast,
)

TARGETS = ["aggregate_cpu_utilization", "active_vm_count"]
HORIZON = 24        # 24 steps = 2 hours at 5-min cadence
MIN_TRAIN = 48      # 4 hours minimum training window
CV_STEP = 1         # rolling-origin (every step)

CANDIDATE_REGISTRY = {
    "temporal_regression": temporal_regression_forecast,
    "arima_200": lambda train, horizon: arima_forecast(train, horizon, order=(2, 0, 0)),
    "arima_100": lambda train, horizon: arima_forecast(train, horizon, order=(1, 0, 0)),
    "arima_101": lambda train, horizon: arima_forecast(train, horizon, order=(1, 0, 1)),
    "arima_201": lambda train, horizon: arima_forecast(train, horizon, order=(2, 0, 1)),
    "hist_gradient_boosting": hist_gradient_boosting_forecast,
}


def _aggregate_fold_results(fold_results: list[dict]) -> dict:
    """Compute aggregate metrics across folds.

    Notes
    -----
    Concatenating predictions across folds pools 936 evaluation pairs that cover
    62 unique timestamps and are not statistically independent due to rolling-window
    overlap.
    """
    all_true = np.concatenate([r["y_true"] for r in fold_results])
    all_pred = np.concatenate([r["y_pred"] for r in fold_results])
    return {
        "mae": mae(all_true, all_pred),
        "rmse": rmse(all_true, all_pred),
        "smape": smape(all_true, all_pred),
        "n_folds": len(fold_results),
    }


def compute_fold_metrics(fold_results: list[dict]) -> pd.DataFrame:
    """Compute MAE, RMSE, and sMAPE for each individual cross-validation fold.

    Parameters
    ----------
    fold_results:
        Output of :func:`cwfc.validation.cross_validate`.

    Returns
    -------
    pd.DataFrame
        Columns: fold, n_train, horizon, mae, rmse, smape.
    """
    records = []
    for r in fold_results:
        y_t = r["y_true"]
        y_p = r["y_pred"]
        records.append(
            {
                "fold": r.get("fold", len(records)),
                "n_train": r.get("n_train", len(y_t)),
                "horizon": r.get("horizon", len(y_t)),
                "mae": mae(y_t, y_p),
                "rmse": rmse(y_t, y_p),
                "smape": smape(y_t, y_p),
            }
        )
    return pd.DataFrame(records)


def compute_horizon_metrics(fold_results: list[dict]) -> pd.DataFrame:
    """Compute forecast error metrics stratified by prediction horizon step (1 to H).

    Parameters
    ----------
    fold_results:
        Output of :func:`cwfc.validation.cross_validate`.

    Returns
    -------
    pd.DataFrame
        Columns: horizon_step, mae, rmse, smape, n_folds.
        horizon_step is 1-indexed (1 to H).
    """
    if not fold_results:
        return pd.DataFrame(columns=["horizon_step", "mae", "rmse", "smape", "n_folds"])

    horizon = len(fold_results[0]["y_true"])
    records = []
    for h in range(horizon):
        y_true_h = np.array([r["y_true"][h] for r in fold_results])
        y_pred_h = np.array([r["y_pred"][h] for r in fold_results])
        records.append(
            {
                "horizon_step": h + 1,
                "mae": mae(y_true_h, y_pred_h),
                "rmse": rmse(y_true_h, y_pred_h),
                "smape": smape(y_true_h, y_pred_h),
                "n_folds": len(fold_results),
            }
        )
    return pd.DataFrame(records)


def compute_fold_dispersion(fold_results: list[dict]) -> dict[str, float]:
    """Compute dispersion and summary statistics of MAE across folds.

    Parameters
    ----------
    fold_results:
        Output of :func:`cwfc.validation.cross_validate`.

    Returns
    -------
    dict[str, float]
        Keys: mean, std, median, min, max, iqr.
    """
    if not fold_results:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "iqr": 0.0}

    fold_maes = np.array([mae(r["y_true"], r["y_pred"]) for r in fold_results])
    q75, q25 = np.percentile(fold_maes, [75, 25])
    return {
        "mean": float(np.mean(fold_maes)),
        "std": float(np.std(fold_maes, ddof=1)) if len(fold_maes) > 1 else 0.0,
        "median": float(np.median(fold_maes)),
        "min": float(np.min(fold_maes)),
        "max": float(np.max(fold_maes)),
        "iqr": float(q75 - q25),
    }


def evaluate_models(
    series: np.ndarray,
    target_name: str,
    models: dict | None = None,
    horizon: int = HORIZON,
    min_train: int = MIN_TRAIN,
    step: int = CV_STEP,
) -> pd.DataFrame:
    """Cross-validate a dictionary of models and return a summary DataFrame.

    Parameters
    ----------
    series:
        1-D array of training observations (length = TRAIN_OBS = 110).
    target_name:
        Label for display purposes (e.g., "aggregate_cpu_utilization").
    models:
        Dict of model_name -> forecaster_callable. Defaults to BASELINE_REGISTRY.
    horizon, min_train, step:
        Passed directly to :func:`~cwfc.validation.expanding_window_splits`.

    Returns
    -------
    pd.DataFrame
        Columns: rank, target, model, mae, rmse, smape, n_folds, nmae_vs_persistence.
        Sorted by MAE ascending (best first).
    """
    if models is None:
        models = BASELINE_REGISTRY

    splits = expanding_window_splits(
        n_total=len(series),
        horizon=horizon,
        min_train=min_train,
        step=step,
    )
    if not splits:
        raise RuntimeError(
            f"No CV splits generated for target '{target_name}'. "
            f"Check min_train ({min_train}) and series length ({len(series)})."
        )

    # Always evaluate persistence if not present so nmae_vs_persistence can be calculated
    models_to_run = dict(models)
    if "persistence" not in models_to_run and "persistence" in BASELINE_REGISTRY:
        models_to_run["persistence"] = BASELINE_REGISTRY["persistence"]

    rows = []
    pers_mae = None

    for name, forecaster in models_to_run.items():
        fold_results = cross_validate(series, forecaster, splits)
        agg = _aggregate_fold_results(fold_results)
        agg["model"] = name
        rows.append(agg)
        if name == "persistence":
            pers_mae = agg["mae"]

    df = pd.DataFrame(rows)[["model", "mae", "rmse", "smape", "n_folds"]]

    # Filter out persistence if it was injected only for normalization and not requested
    if "persistence" not in models:
        df = df[df["model"].isin(models.keys())].copy()

    # Add normalized MAE relative to persistence
    if pers_mae is not None and pers_mae > 0:
        df["nmae_vs_persistence"] = df["mae"] / pers_mae
    else:
        df["nmae_vs_persistence"] = np.nan

    df = df.sort_values("mae").reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    df.insert(1, "target", target_name)
    return df


def evaluate_baselines(
    series: np.ndarray,
    target_name: str,
    horizon: int = HORIZON,
    min_train: int = MIN_TRAIN,
    step: int = CV_STEP,
) -> pd.DataFrame:
    """Cross-validate every baseline and return a summary DataFrame."""
    return evaluate_models(
        series=series,
        target_name=target_name,
        models=BASELINE_REGISTRY,
        horizon=horizon,
        min_train=min_train,
        step=step,
    )


def print_results(df: pd.DataFrame) -> None:
    """Pretty-print evaluation results to stdout."""
    target = df["target"].iloc[0]
    print(f"\n{'-' * 70}")
    print(f"  TARGET: {target}")
    print(f"  CV folds: {df['n_folds'].iloc[0]}  |  horizon: {HORIZON} steps ({HORIZON * 5} min)")
    print(f"{'-' * 70}")

    display_cols = ["rank", "model", "mae", "rmse", "smape"]
    if "nmae_vs_persistence" in df.columns:
        display_cols.append("nmae_vs_persistence")

    fmt = {
        "mae": "{:.4f}".format,
        "rmse": "{:.4f}".format,
        "smape": "{:.2f}%".format,
        "nmae_vs_persistence": "{:.3f}".format,
    }
    print(df[display_cols].to_string(index=False, formatters=fmt))


def print_detailed_diagnostics(
    series: np.ndarray,
    target_name: str,
    models: dict | None = None,
) -> None:
    """Print fold dispersion and horizon error analysis for models."""
    if models is None:
        models = BASELINE_REGISTRY

    splits = expanding_window_splits(len(series), horizon=HORIZON, min_train=MIN_TRAIN, step=CV_STEP)
    print(f"\n{'=' * 70}")
    print(f"  DETAILED DIAGNOSTICS: {target_name}")
    print(f"{'=' * 70}")

    print("\n--- MAE Dispersion Across Folds ---")
    disp_rows = []
    for name, forecaster in models.items():
        res = cross_validate(series, forecaster, splits)
        disp = compute_fold_dispersion(res)
        disp["model"] = name
        disp_rows.append(disp)
    disp_df = pd.DataFrame(disp_rows)[["model", "mean", "std", "median", "min", "max", "iqr"]]
    disp_df = disp_df.sort_values("mean").reset_index(drop=True)
    fmt = {col: "{:.4f}".format for col in ["mean", "std", "median", "min", "max", "iqr"]}
    print(disp_df.to_string(index=False, formatters=fmt))

    print("\n--- MAE by Forecast Horizon (selected steps) ---")
    # Show top model, candidates (if present), and persistence & mean
    top_models = [disp_df["model"].iloc[0]]
    for candidate in ["temporal_regression", "mean", "persistence"]:
        if candidate in models and candidate not in top_models:
            top_models.append(candidate)

    for name in top_models:
        forecaster = models[name]
        res = cross_validate(series, forecaster, splits)
        h_df = compute_horizon_metrics(res)
        h_sample = h_df[h_df["horizon_step"].isin([1, 2, 3, 6, 12, 18, 24])]
        print(f"\n  Model: {name}")
        print(h_sample[["horizon_step", "mae", "rmse", "smape"]].to_string(
            index=False,
            formatters={"mae": "{:.4f}".format, "rmse": "{:.4f}".format, "smape": "{:.2f}%".format}
        ))


def run(
    data_dir: Path,
    output_dir: Path | None = None,
    detailed: bool = False,
    include_candidates: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load data, evaluate models, print results.

    Parameters
    ----------
    data_dir:
        Directory with challenge data files.
    output_dir:
        If provided, saves per-target result CSVs here.
    detailed:
        If True, also print fold dispersion and horizon error profiles.
    include_candidates:
        If True, evaluates candidate models alongside baselines.

    Returns
    -------
    dict[str, pd.DataFrame]
        Maps target name → results DataFrame.
    """
    challenge = load_data(data_dir)
    all_results: dict[str, pd.DataFrame] = {}

    models_to_run = dict(BASELINE_REGISTRY)
    if include_candidates:
        models_to_run.update(CANDIDATE_REGISTRY)

    for target in TARGETS:
        series = challenge.train[target].to_numpy(dtype=float)
        df = evaluate_models(series, target_name=target, models=models_to_run)
        print_results(df)
        if detailed:
            print_detailed_diagnostics(series, target_name=target, models=models_to_run)
        all_results[target] = df

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / f"results_{target}.csv"
            df.to_csv(out_path, index=False)
            print(f"  -> saved {out_path}")

    return all_results


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate models via time-aware cross-validation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        required=True,
        metavar="DIR",
        help="Path to challenge_data/ directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="If set, save result CSVs to this directory.",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="If set, display fold dispersion and horizon error analysis.",
    )
    parser.add_argument(
        "--candidates",
        action="store_true",
        help="If set, include candidate models alongside baselines.",
    )
    args = parser.parse_args()
    run(
        Path(args.data),
        Path(args.output_dir) if args.output_dir else None,
        detailed=args.detailed,
        include_candidates=args.candidates,
    )


if __name__ == "__main__":
    _cli()
