"""Tests for the expanding-window cross-validation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cwfc.validation import (
    expanding_window_splits,
    cross_validate,
    IndexSplit,
    rolling_origin_splits,
)
from cwfc.baselines import persistence_forecast


# ── expanding_window_splits ───────────────────────────────────────────────────

def test_expanding_window_splits_count():
    # n_total=10, horizon=3, min_train=4, step=1
    # valid train_end values: 4,5,6,7  (4+3=7 ≤ 10; 8+3=11 > 10 is actually 7+3=10 ≤ 10 → ok; 8+3=11 > 10)
    splits = expanding_window_splits(n_total=10, horizon=3, min_train=4, step=1)
    # train_end in {4,5,6,7}: 4 splits
    assert len(splits) == 4


def test_expanding_window_no_overlap_with_holdout():
    # Simulates the challenge: 110 train obs, 24 horizon
    splits = expanding_window_splits(n_total=110, horizon=24, min_train=48, step=1)
    for sp in splits:
        assert sp.val_end <= 110, "Validation extends into holdout!"
        assert sp.train_end <= sp.val_start, "Train/val overlap!"


def test_expanding_window_monotonic_train():
    splits = expanding_window_splits(n_total=50, horizon=5, min_train=10, step=2)
    train_ends = [s.train_end for s in splits]
    assert train_ends == sorted(train_ends), "Train ends not monotonically increasing"


def test_expanding_window_all_windows_correct_size():
    splits = expanding_window_splits(n_total=30, horizon=5, min_train=10, step=1)
    for sp in splits:
        assert sp.n_val == 5
        assert sp.n_train >= 10


def test_expanding_window_invalid_params():
    with pytest.raises(ValueError):
        expanding_window_splits(n_total=20, horizon=0, min_train=5)
    with pytest.raises(ValueError):
        expanding_window_splits(n_total=20, horizon=5, min_train=0)
    with pytest.raises(ValueError):
        expanding_window_splits(n_total=20, horizon=5, min_train=5, step=0)


def test_expanding_window_empty_when_impossible():
    # min_train + horizon > n_total → no valid splits
    splits = expanding_window_splits(n_total=10, horizon=8, min_train=5, step=1)
    assert len(splits) == 0, "Should return empty list when no valid split fits"


# ── cross_validate ────────────────────────────────────────────────────────────

def test_cross_validate_shapes():
    series = np.arange(50.0)
    splits = expanding_window_splits(n_total=50, horizon=5, min_train=10, step=5)
    results = cross_validate(series, persistence_forecast, splits)
    for r in results:
        assert r["y_true"].shape == (5,)
        assert r["y_pred"].shape == (5,)


def test_cross_validate_persistence_on_constant_series():
    # Persistence on a constant series → zero error
    series = np.full(60, 9.5)
    splits = expanding_window_splits(n_total=60, horizon=5, min_train=10, step=1)
    results = cross_validate(series, persistence_forecast, splits)
    for r in results:
        np.testing.assert_allclose(r["y_true"], r["y_pred"])


def test_cross_validate_fold_indices_are_consistent():
    series = np.arange(30.0)
    splits = expanding_window_splits(n_total=30, horizon=4, min_train=8, step=2)
    results = cross_validate(series, persistence_forecast, splits)
    for r, sp in zip(results, splits):
        expected_true = series[sp.val_start: sp.val_end]
        np.testing.assert_array_equal(r["y_true"], expected_true)


# ── rolling_origin_splits (backward compat) ───────────────────────────────────

def test_rolling_origin_splits_are_temporal():
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=20, freq="h", tz="UTC")
    })
    splits = rolling_origin_splits(frame, horizon=4, min_train_size=8, step=4)
    assert len(splits) == 3
    assert splits[0].train_end < splits[0].validation_start
    assert splits[0].validation_end < splits[1].validation_start


# ── get_validation_overlap (overlap analysis) ─────────────────────────────────

def test_get_validation_overlap():
    from cwfc.validation import get_validation_overlap

    splits = expanding_window_splits(n_total=110, horizon=24, min_train=48, step=1)
    overlap_df = get_validation_overlap(splits)

    # 39 splits of horizon 24 evaluate obs indices 48 to 109 (62 unique observations)
    assert len(overlap_df) == 62
    assert overlap_df["observation_index"].min() == 48
    assert overlap_df["observation_index"].max() == 109

    # Total evaluations across all folds = 39 * 24 = 936
    assert overlap_df["count"].sum() == 39 * 24

    # The first observation (48) is only evaluated once (fold 0, horizon step 1)
    obs_48 = overlap_df[overlap_df["observation_index"] == 48].iloc[0]
    assert obs_48["count"] == 1
    assert obs_48["min_horizon"] == 1
    assert obs_48["max_horizon"] == 1

    # Middle observations (e.g. 71 to 86) appear in 24 validation windows
    obs_75 = overlap_df[overlap_df["observation_index"] == 75].iloc[0]
    assert obs_75["count"] == 24
    assert obs_75["min_horizon"] == 1
    assert obs_75["max_horizon"] == 24
