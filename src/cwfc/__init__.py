"""Cloud Workload Forecasting Challenge utilities.

Public API
----------
- :mod:`cwfc.data`       — data loading and target construction
- :mod:`cwfc.baselines`  — baseline forecasting models
- :mod:`cwfc.metrics`    — MAE, RMSE, sMAPE, normalized MAE
- :mod:`cwfc.validation` — time-aware cross-validation splits
- :mod:`cwfc.evaluate`   — baseline evaluation pipeline
- :mod:`cwfc.predict`    — final prediction generation (CLI entry point)
"""

from cwfc.data import load_data, ChallengeData, TRAIN_OBS, HOLD_OBS
from cwfc.baselines import (
    mean_baseline,
    persistence_forecast,
    drift_forecast,
    moving_average_forecast,
    exp_smoothing_forecast,
    seasonal_naive_forecast,
    BASELINE_REGISTRY,
)
from cwfc.metrics import mae, rmse, smape, normalized_mae
from cwfc.validation import (
    expanding_window_splits,
    cross_validate,
    IndexSplit,
    TimeSplit,
    rolling_origin_splits,
    get_validation_overlap,
)
from cwfc.models import (
    fit_ridge_linear_regression,
    temporal_regression_forecast,
    arima_forecast,
    hist_gradient_boosting_forecast,
)

__all__ = [
    "load_data",
    "ChallengeData",
    "TRAIN_OBS",
    "HOLD_OBS",
    "mean_baseline",
    "persistence_forecast",
    "drift_forecast",
    "moving_average_forecast",
    "exp_smoothing_forecast",
    "seasonal_naive_forecast",
    "BASELINE_REGISTRY",
    "mae",
    "rmse",
    "smape",
    "normalized_mae",
    "expanding_window_splits",
    "cross_validate",
    "IndexSplit",
    "TimeSplit",
    "rolling_origin_splits",
    "get_validation_overlap",
    "fit_ridge_linear_regression",
    "temporal_regression_forecast",
    "arima_forecast",
    "hist_gradient_boosting_forecast",
]
