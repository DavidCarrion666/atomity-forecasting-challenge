"""Tests for cwfc.evaluate module: aggregation, baseline evaluation, and diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cwfc.baselines import BASELINE_REGISTRY
from cwfc.evaluate import (
    _aggregate_fold_results,
    evaluate_baselines,
    compute_fold_metrics,
    compute_horizon_metrics,
    compute_fold_dispersion,
)


# ── _aggregate_fold_results tests (F-06) ──────────────────────────────────────

def test_aggregate_fold_results_exact_math():
    """Verify pooled MAE, RMSE, sMAPE calculation across multiple folds."""
    fold_results = [
        {
            "fold": 0,
            "y_true": np.array([10.0, 20.0]),
            "y_pred": np.array([12.0, 19.0]),  # errors: +2, -1
        },
        {
            "fold": 1,
            "y_true": np.array([30.0, 40.0]),
            "y_pred": np.array([30.0, 42.0]),  # errors: 0, +2
        },
    ]
    # All true: [10, 20, 30, 40], all pred: [12, 19, 30, 42]
    # Abs errors: [2, 1, 0, 2] -> sum = 5, mean = 1.25
    # Sq errors: [4, 1, 0, 4] -> sum = 9, mean = 2.25 -> sqrt = 1.5
    res = _aggregate_fold_results(fold_results)
    assert res["n_folds"] == 2
    assert pytest.approx(res["mae"], rel=1e-6) == 1.25
    assert pytest.approx(res["rmse"], rel=1e-6) == 1.5
    assert res["smape"] > 0


def test_aggregate_fold_results_perfect_predictions():
    """Verify zero error on perfect predictions."""
    fold_results = [
        {
            "fold": 0,
            "y_true": np.array([5.0, 15.0]),
            "y_pred": np.array([5.0, 15.0]),
        }
    ]
    res = _aggregate_fold_results(fold_results)
    assert res["mae"] == 0.0
    assert res["rmse"] == 0.0
    assert res["smape"] == 0.0
    assert res["n_folds"] == 1


# ── evaluate_baselines tests (F-06) ───────────────────────────────────────────

def test_evaluate_baselines_constant_series():
    """On a flat series, baselines like persistence and mean should achieve 0 MAE."""
    series = np.full(110, 25.0)
    df = evaluate_baselines(series, target_name="constant_target", horizon=10, min_train=30, step=5)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(BASELINE_REGISTRY)

    expected_cols = ["rank", "target", "model", "mae", "rmse", "smape", "n_folds", "nmae_vs_persistence"]
    for col in expected_cols:
        assert col in df.columns

    assert df["target"].iloc[0] == "constant_target"
    assert df["rank"].tolist() == list(range(1, len(df) + 1))

    # Persistence on constant series has 0 error
    pers_row = df[df["model"] == "persistence"].iloc[0]
    assert pers_row["mae"] == 0.0
    assert pers_row["rmse"] == 0.0


def test_evaluate_baselines_ranking_order():
    """Verify that returned DataFrame is sorted ascending by MAE."""
    np.random.seed(42)
    series = np.sin(np.linspace(0, 10, 110)) + 10.0
    df = evaluate_baselines(series, target_name="synthetic_sin")

    assert df["mae"].is_monotonic_increasing
    assert df["rank"].iloc[0] == 1
    assert df["rank"].iloc[-1] == len(BASELINE_REGISTRY)


def test_evaluate_baselines_too_short_raises():
    """If series length < min_train + horizon, it should raise RuntimeError."""
    series = np.arange(20.0)
    with pytest.raises(RuntimeError):
        evaluate_baselines(series, target_name="short", horizon=24, min_train=48)


# ── Evaluation diagnostics tests (fold, horizon, dispersion) ──────────────────

def test_compute_fold_metrics():
    """Verify per-fold metric breakdown."""
    fold_results = [
        {"fold": 0, "n_train": 48, "horizon": 2, "y_true": np.array([1.0, 2.0]), "y_pred": np.array([1.5, 2.5])},
        {"fold": 1, "n_train": 49, "horizon": 2, "y_true": np.array([3.0, 4.0]), "y_pred": np.array([3.0, 5.0])},
    ]
    df = compute_fold_metrics(fold_results)
    assert len(df) == 2
    assert list(df.columns) == ["fold", "n_train", "horizon", "mae", "rmse", "smape"]
    assert pytest.approx(df["mae"].iloc[0], rel=1e-6) == 0.5
    assert pytest.approx(df["mae"].iloc[1], rel=1e-6) == 0.5


def test_compute_horizon_metrics():
    """Verify per-horizon step error calculation."""
    fold_results = [
        {"y_true": np.array([10.0, 20.0]), "y_pred": np.array([11.0, 24.0])},
        {"y_true": np.array([10.0, 20.0]), "y_pred": np.array([13.0, 22.0])},
    ]
    df = compute_horizon_metrics(fold_results)
    assert len(df) == 2
    assert list(df.columns) == ["horizon_step", "mae", "rmse", "smape", "n_folds"]
    assert df["horizon_step"].tolist() == [1, 2]
    # Step 1: errors |11-10|=1, |13-10|=3 -> mean = 2.0
    assert pytest.approx(df["mae"].iloc[0], rel=1e-6) == 2.0
    # Step 2: errors |24-20|=4, |22-20|=2 -> mean = 3.0
    assert pytest.approx(df["mae"].iloc[1], rel=1e-6) == 3.0


def test_compute_fold_dispersion():
    """Verify dispersion metrics (mean, std, median, min, max, iqr)."""
    fold_results = [
        {"y_true": np.array([1.0]), "y_pred": np.array([2.0])},  # mae = 1.0
        {"y_true": np.array([1.0]), "y_pred": np.array([3.0])},  # mae = 2.0
        {"y_true": np.array([1.0]), "y_pred": np.array([4.0])},  # mae = 3.0
    ]
    disp = compute_fold_dispersion(fold_results)
    assert pytest.approx(disp["mean"]) == 2.0
    assert pytest.approx(disp["median"]) == 2.0
    assert pytest.approx(disp["min"]) == 1.0
    assert pytest.approx(disp["max"]) == 3.0
    assert pytest.approx(disp["std"]) == 1.0
