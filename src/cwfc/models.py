"""Statistical and machine learning forecasting models for CWFC.

All models adhere to the forecaster protocol:
    Callable[[np.ndarray, int], np.ndarray]
where:
    train: 1-D array of historical observations.
    horizon: number of forward steps to predict.
    return: 1-D array of length `horizon`.

Leakage protocol:
    Feature engineering, transformations, and fitting MUST be performed
    strictly on the provided `train` array.
"""

from __future__ import annotations

import numpy as np


def fit_ridge_linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """Fit a linear regression with L2 (Ridge) penalty on non-intercept coefficients.

    Parameters
    ----------
    X:
        Design matrix of shape (n_samples, n_features), where column 0 is the intercept.
    y:
        Target vector of shape (n_samples,).
    alpha:
        L2 regularization strength. If alpha == 0, solves standard OLS.

    Returns
    -------
    np.ndarray
        Weight vector of shape (n_features,).
    """
    p = X.shape[1]
    if alpha <= 0.0:
        weights, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        return weights

    penalty = alpha * np.eye(p)
    penalty[0, 0] = 0.0  # Do not penalize the intercept term
    return np.linalg.solve(X.T @ X + penalty, X.T @ y)


def temporal_regression_forecast(
    train: np.ndarray,
    horizon: int,
    lags: tuple[int, ...] = (1, 2, 3),
    window: int = 3,
    alpha: float = 1.0,
    clip_min: float = 0.0,
    clip_max: float = 100.0,
) -> np.ndarray:
    """Autoregressive Ridge regression using lags, moving average, and local slope.

    Features constructed at each step t:
    1. Intercept (bias).
    2. Autoregressive lags: y_{t-l} for l in `lags`.
    3. Moving average: mean(y_{t-window:t}).
    4. Local slope / trend: (y_{t-1} - y_{t-window}) / (window - 1).

    Multi-step predictions are generated iteratively (recursive forecasting).
    At each step h, the newly predicted value is appended to the historical
    sequence to update lags, rolling averages, and local slope.

    Parameters
    ----------
    train:
        1-D array of past observations. Must have length > max(max(lags), window).
    horizon:
        Number of steps ahead to forecast.
    lags:
        Tuple of integer lag orders (e.g. (1, 2, 3)).
    window:
        Window size for rolling mean and local trend estimation.
    alpha:
        L2 regularization strength for Ridge regression.
    clip_min, clip_max:
        Value clipping bounds to enforce physical domain constraints.

    Returns
    -------
    np.ndarray
        1-D array of predictions of length `horizon`.
    """
    train = np.asarray(train, dtype=float)
    n = len(train)
    max_lag = max(max(lags), window)

    if n <= max_lag + 1:
        # Fallback to last value if history is too short to fit features
        return np.full(horizon, train[-1] if n > 0 else 0.0)

    # If the training series is constant (e.g. active_vm_count = 10,000), return constant
    if np.all(train == train[0]):
        return np.full(horizon, train[0])

    # ── 1. Build design matrix strictly from past train data ─────────────────
    X_rows = []
    y_targets = []
    denom_trend = float(window - 1) if window > 1 else 1.0

    for t in range(max_lag, n):
        row = [1.0]  # Intercept
        for l in lags:
            row.append(train[t - l])
        row.append(np.mean(train[t - window:t]))
        row.append((train[t - 1] - train[t - window]) / denom_trend)
        X_rows.append(row)
        y_targets.append(train[t])

    X = np.array(X_rows, dtype=float)
    y = np.array(y_targets, dtype=float)

    # ── 2. Fit Ridge regression inside the fold ──────────────────────────────
    weights = fit_ridge_linear_regression(X, y, alpha=alpha)

    # ── 3. Recursive multi-step generation ───────────────────────────────────
    history = list(train)
    predictions = []

    for _ in range(horizon):
        t_curr = len(history)
        row = [1.0]
        for l in lags:
            row.append(history[t_curr - l])
        row.append(np.mean(history[t_curr - window:t_curr]))
        row.append((history[t_curr - 1] - history[t_curr - window]) / denom_trend)

        pred = float(np.dot(row, weights))
        pred = np.clip(pred, clip_min, clip_max)
        predictions.append(pred)
        history.append(pred)

    return np.asarray(predictions, dtype=float)


def arima_forecast(
    train: np.ndarray,
    horizon: int,
    order: tuple[int, int, int] = (2, 0, 0),
    trend: str | None = "c",
    clip_min: float = 0.0,
    clip_max: float = 100.0,
) -> np.ndarray:
    """Forecast future observations using an ARIMA model.

    Parameters
    ----------
    train:
        1-D array of historical observations.
    horizon:
        Number of steps ahead to forecast.
    order:
        (p, d, q) order specification for ARIMA.
    trend:
        Deterministic trend parameter for statsmodels ARIMA (e.g. 'c' for constant, None).
    clip_min, clip_max:
        Value clipping bounds to enforce domain boundaries.

    Returns
    -------
    np.ndarray
        1-D array of predictions of length `horizon`.
    """
    from statsmodels.tsa.arima.model import ARIMA
    import warnings

    train = np.asarray(train, dtype=float)
    n = len(train)
    p, d, q = order
    min_required = p + d + q + 2

    if n <= min_required:
        return np.full(horizon, train[-1] if n > 0 else 0.0)

    # Constant series (e.g. active_vm_count = 10,000)
    if np.all(train == train[0]):
        return np.full(horizon, train[0])

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(train, order=order, trend=trend)
            fitted = model.fit()
            preds = fitted.forecast(steps=horizon)
            preds_arr = np.asarray(preds, dtype=float)
            if not np.all(np.isfinite(preds_arr)):
                return np.full(horizon, np.mean(train))
            return np.clip(preds_arr, clip_min, clip_max)
    except Exception:
        # Fallback to mean baseline if numerical optimization fails entirely
        return np.full(horizon, np.mean(train))


def _hist_gradient_boosting_feature_row(history: np.ndarray) -> list[float]:
    """Build one autoregressive feature row from observations strictly in history."""
    recent = history[-3:]
    return [
        float(recent[-1]),
        float(recent[-2]),
        float(recent[-3]),
        float(np.mean(recent[-2:])),
        float(np.mean(recent)),
        float((recent[-1] - recent[-3]) / 2.0),
    ]


def _build_hist_gradient_boosting_training_data(
    train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one-step training samples whose features precede each target."""
    X = np.array(
        [_hist_gradient_boosting_feature_row(train[:t]) for t in range(3, len(train))],
        dtype=float,
    )
    y = train[3:].astype(float, copy=True)
    return X, y


def hist_gradient_boosting_forecast(
    train: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Forecast recursively with a small HistGradientBoostingRegressor.

    The six features for each target at time ``t`` are lags 1, 2, and 3,
    trailing means over 2 and 3 observations, and the mean local change from
    lag 3 to lag 1. Every training feature uses only observations before its
    target. In recursive prediction, each predicted value is appended to the
    history and is then used to recalculate the next step's lags, means, and
    trend.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    train = np.asarray(train, dtype=float)
    if train.ndim != 1:
        raise ValueError("train must be a one-dimensional array")
    if len(train) == 0:
        raise ValueError("train must not be empty")
    if not np.all(np.isfinite(train)):
        raise ValueError("train must contain only finite values")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if len(train) < 4:
        return np.full(horizon, train[-1], dtype=float)
    if np.all(train == train[0]):
        return np.full(horizon, train[0], dtype=float)

    X, y = _build_hist_gradient_boosting_training_data(train)
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=50,
        max_leaf_nodes=3,
        min_samples_leaf=5,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=0,
    )
    model.fit(X, y)

    history = list(train)
    predictions = []
    for _ in range(horizon):
        row = np.asarray([_hist_gradient_boosting_feature_row(np.asarray(history))])
        prediction = float(model.predict(row)[0])
        predictions.append(prediction)
        history.append(prediction)

    return np.asarray(predictions, dtype=float)
