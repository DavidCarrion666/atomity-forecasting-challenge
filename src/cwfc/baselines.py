"""Baseline forecasting models for the cloud workload forecasting challenge.

All forecasters share the same interface:

    forecast(train, horizon) -> np.ndarray  (shape: (horizon,))

where
- ``train``   is a 1-D array-like of historical target values (float).
- ``horizon`` is the number of steps to predict (int, ≥ 1).

This makes it trivial to loop over baselines in the evaluation pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Shared type alias ──────────────────────────────────────────────────────────
ArrayLike = np.ndarray | list | pd.Series


def _to_array(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=float)


# ── Baseline implementations ──────────────────────────────────────────────────

def mean_baseline(train: ArrayLike, horizon: int) -> np.ndarray:
    """Predict the training mean for every future step.

    Simple but useful as a floor — if persistence beats this, persistence is
    a sensible starting point.
    """
    y = _to_array(train)
    if len(y) == 0:
        raise ValueError("train must not be empty")
    return np.full(horizon, y.mean())


def persistence_forecast(train: ArrayLike, horizon: int) -> np.ndarray:
    """Carry the last observed value forward for all ``horizon`` steps.

    Supported by ACF(1) ≈ 0.66 — the strongest autocorrelation signal in the
    training data.
    """
    y = _to_array(train)
    if len(y) == 0:
        raise ValueError("train must not be empty")
    return np.full(horizon, y[-1])


def drift_forecast(train: ArrayLike, horizon: int) -> np.ndarray:
    """Extrapolate the average per-step change observed in training.

    Forecast at step h:  y_last + h * mean_diff

    This captures a slow linear trend if one exists.  With a near-flat series
    the drift will be negligible, making this a slight generalisation of
    persistence.
    """
    y = _to_array(train)
    if len(y) < 2:
        raise ValueError("drift requires at least 2 training observations")
    mean_diff = (y[-1] - y[0]) / (len(y) - 1)
    steps = np.arange(1, horizon + 1)
    return y[-1] + mean_diff * steps


def moving_average_forecast(train: ArrayLike, horizon: int, window: int = 12) -> np.ndarray:
    """Predict the rolling mean of the last ``window`` observations.

    Default ``window=12`` corresponds to a 1-hour lookback at 5-min cadence.
    The same value is forecast for every step in the horizon (flat projection).

    Parameters
    ----------
    window:
        Number of most-recent training observations to average.
        Clamped to ``len(train)`` if insufficient history exists.
    """
    y = _to_array(train)
    if len(y) == 0:
        raise ValueError("train must not be empty")
    w = min(window, len(y))
    level = y[-w:].mean()
    return np.full(horizon, level)


def exp_smoothing_forecast(train: ArrayLike, horizon: int, alpha: float = 0.3) -> np.ndarray:
    """Single exponential smoothing (SES) — flat forecast equal to the smoothed level.

    Level update:  L_t = alpha * y_t + (1 - alpha) * L_{t-1}

    The forecast for all future steps equals the final level L_T.
    alpha = 0.3 is a reasonable starting point for a moderately persistent
    series; it can be tuned by minimising MSE on a validation fold.

    Parameters
    ----------
    alpha:
        Smoothing parameter in (0, 1].  Higher values react faster to recent
        changes; lower values produce a smoother (inertial) forecast.
    """
    y = _to_array(train)
    if len(y) == 0:
        raise ValueError("train must not be empty")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    level = y[0]
    for val in y[1:]:
        level = alpha * val + (1 - alpha) * level
    return np.full(horizon, level)


def seasonal_naive_forecast(train: ArrayLike, horizon: int, season_length: int = 24) -> np.ndarray:
    """Repeat the pattern from ``season_length`` steps ago.

    Default ``season_length=24`` repeats observations from 2 hours ago
    (24 × 5 min).  The ACF analysis found ACF(24) ≈ 0, so this baseline is
    expected to perform poorly here — but it is required by the assignment and
    serves as a useful negative control.

    Parameters
    ----------
    season_length:
        Number of steps in one seasonal cycle.
    """
    y = _to_array(train)
    if season_length <= 0:
        raise ValueError("season_length must be positive")
    if len(y) < season_length:
        raise ValueError(
            f"seasonal_naive requires at least {season_length} observations; "
            f"got {len(y)}"
        )
    seasonal_block = y[-season_length:]
    return np.array([float(seasonal_block[i % season_length]) for i in range(horizon)])


# ── Registry: name → callable ─────────────────────────────────────────────────
# Each entry maps a human-readable model name to a function with the signature
#   fn(train: ArrayLike, horizon: int) -> np.ndarray
# Add new models here to include them automatically in the evaluation pipeline.

BASELINE_REGISTRY: dict[str, callable] = {
    "mean": mean_baseline,
    "persistence": persistence_forecast,
    "drift": drift_forecast,
    "moving_average_12": lambda train, h: moving_average_forecast(train, h, window=12),
    "moving_average_6": lambda train, h: moving_average_forecast(train, h, window=6),
    "exp_smoothing_03": lambda train, h: exp_smoothing_forecast(train, h, alpha=0.3),
    "exp_smoothing_05": lambda train, h: exp_smoothing_forecast(train, h, alpha=0.5),
    "seasonal_naive_24": lambda train, h: seasonal_naive_forecast(train, h, season_length=24),
}


# ── Legacy adapter (backwards-compat with existing code) ─────────────────────
# The original baselines.py used a DataFrame-based interface.  These wrappers
# preserve compatibility if any existing code imports them.

def _persistence_compat(train: pd.DataFrame, template: pd.DataFrame, target: str) -> pd.Series:
    last_value = float(train.iloc[-1][target])
    return pd.Series(last_value, index=template.index, dtype=float)


def _seasonal_naive_compat(
    train: pd.DataFrame, template: pd.DataFrame, target: str, season_length: int
) -> pd.Series:
    values = train[target].astype(float).to_numpy()
    if len(values) < season_length:
        raise ValueError("not enough history for seasonal-naive forecast")
    forecast = [float(values[-season_length + (i % season_length)]) for i in range(len(template))]
    return pd.Series(forecast, index=template.index, dtype=float)
