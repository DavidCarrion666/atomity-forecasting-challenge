"""Tests for the baseline forecasting models."""

from __future__ import annotations

import numpy as np
import pytest

from cwfc.baselines import (
    mean_baseline,
    persistence_forecast,
    drift_forecast,
    moving_average_forecast,
    exp_smoothing_forecast,
    seasonal_naive_forecast,
    BASELINE_REGISTRY,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

FLAT_SERIES = np.array([5.0] * 20)           # constant series → easy to reason about
RAMP_SERIES = np.arange(1.0, 21.0)           # strictly increasing by 1
SIMPLE = np.array([1.0, 2.0, 3.0, 4.0, 5.0])


# ── mean_baseline ─────────────────────────────────────────────────────────────

def test_mean_baseline_flat_series():
    pred = mean_baseline(FLAT_SERIES, horizon=5)
    np.testing.assert_allclose(pred, 5.0)


def test_mean_baseline_returns_correct_shape():
    pred = mean_baseline(SIMPLE, horizon=10)
    assert pred.shape == (10,)


def test_mean_baseline_empty_raises():
    with pytest.raises(ValueError):
        mean_baseline([], horizon=5)


# ── persistence_forecast ──────────────────────────────────────────────────────

def test_persistence_returns_last_value():
    pred = persistence_forecast(SIMPLE, horizon=3)
    np.testing.assert_allclose(pred, 5.0)


def test_persistence_correct_shape():
    pred = persistence_forecast(FLAT_SERIES, horizon=24)
    assert pred.shape == (24,)


def test_persistence_empty_raises():
    with pytest.raises(ValueError):
        persistence_forecast([], horizon=1)


# ── drift_forecast ────────────────────────────────────────────────────────────

def test_drift_flat_series_equals_persistence():
    # Drift on a constant series → mean_diff = 0 → same as persistence
    pred = drift_forecast(FLAT_SERIES, horizon=5)
    np.testing.assert_allclose(pred, FLAT_SERIES[-1])


def test_drift_ramp_series():
    # Ramp series: last=20, mean_diff=1 → [21, 22, 23]
    pred = drift_forecast(RAMP_SERIES, horizon=3)
    np.testing.assert_allclose(pred, [21.0, 22.0, 23.0])


def test_drift_needs_min_two_points():
    with pytest.raises(ValueError):
        drift_forecast([5.0], horizon=1)


# ── moving_average_forecast ───────────────────────────────────────────────────

def test_moving_average_flat_series():
    pred = moving_average_forecast(FLAT_SERIES, horizon=5, window=6)
    np.testing.assert_allclose(pred, 5.0)


def test_moving_average_window_clamp():
    # window larger than series → uses all available history
    pred = moving_average_forecast(SIMPLE, horizon=4, window=100)
    expected_level = SIMPLE.mean()
    np.testing.assert_allclose(pred, expected_level)


def test_moving_average_empty_raises():
    with pytest.raises(ValueError):
        moving_average_forecast([], horizon=1, window=3)


# ── exp_smoothing_forecast ────────────────────────────────────────────────────

def test_exp_smoothing_flat_series():
    pred = exp_smoothing_forecast(FLAT_SERIES, horizon=5, alpha=0.3)
    np.testing.assert_allclose(pred, 5.0, atol=1e-8)


def test_exp_smoothing_alpha_1_equals_persistence():
    # alpha=1 → level always = latest observation → equals persistence
    pred = exp_smoothing_forecast(SIMPLE, horizon=3, alpha=1.0)
    np.testing.assert_allclose(pred, SIMPLE[-1])


def test_exp_smoothing_invalid_alpha():
    with pytest.raises(ValueError):
        exp_smoothing_forecast(SIMPLE, horizon=3, alpha=0.0)
    with pytest.raises(ValueError):
        exp_smoothing_forecast(SIMPLE, horizon=3, alpha=1.5)


# ── seasonal_naive_forecast ───────────────────────────────────────────────────

def test_seasonal_naive_repeats_pattern():
    # Season length 3, horizon 6 → repeats last 3 values twice
    series = np.array([10.0, 20.0, 30.0, 1.0, 2.0, 3.0])
    pred = seasonal_naive_forecast(series, horizon=6, season_length=3)
    np.testing.assert_allclose(pred, [1.0, 2.0, 3.0, 1.0, 2.0, 3.0])


def test_seasonal_naive_not_enough_history():
    with pytest.raises(ValueError):
        seasonal_naive_forecast([1.0, 2.0], horizon=3, season_length=24)


# ── BASELINE_REGISTRY ─────────────────────────────────────────────────────────

def test_registry_all_run_without_error():
    series = np.random.default_rng(42).uniform(5.0, 15.0, size=110)
    for name, fn in BASELINE_REGISTRY.items():
        pred = fn(series, 24)
        assert pred.shape == (24,), f"{name}: expected shape (24,), got {pred.shape}"
        assert np.all(np.isfinite(pred)), f"{name}: forecast contains non-finite values"


def test_registry_keys_are_strings():
    for k in BASELINE_REGISTRY:
        assert isinstance(k, str)
