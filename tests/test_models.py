"""Tests for cwfc.models module."""

from __future__ import annotations

import numpy as np
import pytest

from cwfc.models import (
    fit_ridge_linear_regression,
    temporal_regression_forecast,
    arima_forecast,
    _build_hist_gradient_boosting_training_data,
    hist_gradient_boosting_forecast,
)


def test_fit_ridge_exact():
    # Simple 2D problem: y = 2.0 + 3.0 * x
    X = np.array([
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 3.0],
        [1.0, 4.0],
    ])
    y = np.array([5.0, 8.0, 11.0, 14.0])

    w_ols = fit_ridge_linear_regression(X, y, alpha=0.0)
    assert pytest.approx(w_ols[0], rel=1e-5) == 2.0
    assert pytest.approx(w_ols[1], rel=1e-5) == 3.0

    # With positive alpha, non-intercept slope is shrunk towards 0
    w_ridge = fit_ridge_linear_regression(X, y, alpha=10.0)
    assert w_ridge[1] < w_ols[1]


def test_temporal_regression_flat_series():
    series = np.full(50, 10000.0)
    preds = temporal_regression_forecast(series, horizon=24)
    assert len(preds) == 24
    assert np.all(preds == 10000.0)


def test_temporal_regression_shape_and_range():
    np.random.seed(42)
    series = np.random.uniform(5.0, 15.0, size=60)
    preds = temporal_regression_forecast(series, horizon=24, clip_min=0.0, clip_max=100.0)
    assert len(preds) == 24
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 100.0)


def test_temporal_regression_short_history():
    series = np.array([10.0, 12.0])
    preds = temporal_regression_forecast(series, horizon=5, lags=(1, 2, 3), window=3)
    assert len(preds) == 5
    assert np.all(preds == 12.0)


# ── ARIMA tests ───────────────────────────────────────────────────────────────

def test_arima_forecast_flat_series():
    series = np.full(50, 10000.0)
    preds = arima_forecast(series, horizon=24, order=(2, 0, 0))
    assert len(preds) == 24
    assert np.all(preds == 10000.0)


def test_arima_forecast_shape_and_finite():
    np.random.seed(42)
    series = np.random.uniform(8.0, 11.0, size=50)
    preds = arima_forecast(series, horizon=24, order=(2, 0, 0))
    assert len(preds) == 24
    assert np.all(np.isfinite(preds))
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 100.0)


def test_arima_forecast_short_history():
    series = np.array([10.0, 12.0])
    preds = arima_forecast(series, horizon=5, order=(2, 0, 0))
    assert len(preds) == 5
    assert np.all(preds == 12.0)


def test_arima_forecast_different_orders():
    series = np.linspace(8.0, 10.0, 48)
    for order in [(1, 0, 0), (2, 0, 0)]:
        preds = arima_forecast(series, horizon=24, order=order)
        assert len(preds) == 24
        assert np.all(np.isfinite(preds))


# ── HistGradientBoosting tests ───────────────────────────────────────────────

def test_hist_gradient_boosting_forecast_shape_and_finite():
    series = 9.5 + 0.3 * np.sin(np.arange(60) / 4)
    preds = hist_gradient_boosting_forecast(series, horizon=24)
    assert preds.shape == (24,)
    assert np.all(np.isfinite(preds))


def test_hist_gradient_boosting_forecast_constant_series():
    preds = hist_gradient_boosting_forecast(np.full(48, 10.0), horizon=24)
    np.testing.assert_array_equal(preds, np.full(24, 10.0))


def test_hist_gradient_boosting_forecast_invalid_inputs():
    with pytest.raises(ValueError, match="one-dimensional"):
        hist_gradient_boosting_forecast(np.ones((4, 2)), horizon=3)
    with pytest.raises(ValueError, match="finite"):
        hist_gradient_boosting_forecast(np.array([1.0, 2.0, np.nan]), horizon=3)
    with pytest.raises(ValueError, match="horizon"):
        hist_gradient_boosting_forecast(np.arange(5.0), horizon=0)


def test_hist_gradient_boosting_features_use_only_prior_observations():
    series = np.arange(12.0)
    X, y = _build_hist_gradient_boosting_training_data(series)
    changed_future = series.copy()
    changed_future[8:] += 1000.0
    X_changed, y_changed = _build_hist_gradient_boosting_training_data(changed_future)

    # Features for targets at indices 3..7 are unaffected by later observations.
    np.testing.assert_array_equal(X[:5], X_changed[:5])
    np.testing.assert_array_equal(y[:5], y_changed[:5])
    np.testing.assert_array_equal(X[0], [2.0, 1.0, 0.0, 1.5, 1.0, 1.0])
