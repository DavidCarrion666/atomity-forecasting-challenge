"""Tests for final prediction generation and its leakage boundary."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cwfc import predict
from cwfc.baselines import BASELINE_REGISTRY
from cwfc.data import ChallengeData, HOLD_OBS, TRAIN_OBS


class TimestampOnlyHoldout:
    """Expose holdout timestamps and reject access to any target column."""

    def __init__(self, timestamps):
        self.timestamps = np.asarray(timestamps)
        self.accessed_columns: list[str] = []

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, column):
        self.accessed_columns.append(column)
        if column != "timestamp":
            raise AssertionError(f"Holdout target access is forbidden: {column}")
        return pd.Series(self.timestamps)


@pytest.fixture
def challenge_data():
    train_timestamps = np.arange(TRAIN_OBS) * 300
    holdout_timestamps = np.arange(TRAIN_OBS, TRAIN_OBS + HOLD_OBS) * 300
    train = pd.DataFrame(
        {
            "timestamp": train_timestamps,
            "active_vm_count": np.full(TRAIN_OBS, 10_000),
            "aggregate_cpu_utilization": 9.0 + np.sin(np.arange(TRAIN_OBS) / 5),
        }
    )
    holdout = TimestampOnlyHoldout(holdout_timestamps)
    return ChallengeData(
        train=train,
        holdout=holdout,
        vm_metadata=pd.DataFrame(),
    )


def test_generate_predictions_selects_arima_200_and_respects_output_contract(
    monkeypatch,
    challenge_data,
):
    observed = {}

    def fake_arima_forecast(train, horizon, order, trend):
        observed["train"] = train.copy()
        observed["horizon"] = horizon
        observed["order"] = order
        observed["trend"] = trend
        return np.linspace(-1.0, 101.0, horizon)

    monkeypatch.setattr(predict, "load_data", lambda _: challenge_data)
    monkeypatch.setattr(predict, "arima_forecast", fake_arima_forecast)

    result = predict.generate_predictions(
        data_dir=Path("."),
        cpu_model_name="arima_2_0_0",
    )

    assert observed["order"] == (2, 0, 0)
    assert observed["trend"] == "c"
    assert observed["horizon"] == HOLD_OBS
    np.testing.assert_array_equal(
        observed["train"],
        challenge_data.train["aggregate_cpu_utilization"].to_numpy(),
    )
    assert challenge_data.holdout.accessed_columns == ["timestamp"]

    assert len(result) == HOLD_OBS
    assert list(result.columns) == [
        "timestamp",
        "active_vm_count",
        "aggregate_cpu_utilization",
    ]
    np.testing.assert_array_equal(
        result["timestamp"].to_numpy(),
        np.arange(TRAIN_OBS, TRAIN_OBS + HOLD_OBS) * 300,
    )
    assert result["timestamp"].is_monotonic_increasing
    assert result["timestamp"].is_unique
    assert not result.isna().any().any()
    assert np.isfinite(
        result[["active_vm_count", "aggregate_cpu_utilization"]].to_numpy(dtype=float)
    ).all()
    assert (result["active_vm_count"] >= 0).all()
    assert result["active_vm_count"].eq(10_000).all()
    assert result["aggregate_cpu_utilization"].between(0, 100).all()
    assert result["aggregate_cpu_utilization"].iloc[0] == 0.0
    assert result["aggregate_cpu_utilization"].iloc[-1] == 100.0


def test_generate_predictions_defaults_to_arima_200(monkeypatch, challenge_data):
    selected = {}

    def fake_arima_forecast(train, horizon, order, trend):
        selected.update(order=order, trend=trend)
        return np.full(horizon, 9.5)

    monkeypatch.setattr(predict, "load_data", lambda _: challenge_data)
    monkeypatch.setattr(predict, "arima_forecast", fake_arima_forecast)

    predict.generate_predictions(data_dir=Path("."))

    assert selected == {"order": (2, 0, 0), "trend": "c"}


def test_prediction_registry_contains_original_models_and_requested_candidates():
    assert set(BASELINE_REGISTRY).issubset(predict.PREDICTION_MODEL_REGISTRY)
    assert {
        "temporal_regression",
        "arima_2_0_0",
    }.issubset(predict.PREDICTION_MODEL_REGISTRY)
    assert "hist_gradient_boosting" not in predict.PREDICTION_MODEL_REGISTRY


def test_generate_predictions_rejects_changed_active_vm_invariant(
    monkeypatch,
    challenge_data,
):
    challenge_data.train.loc[0, "active_vm_count"] = 9_999
    monkeypatch.setattr(predict, "load_data", lambda _: challenge_data)

    with pytest.raises(ValueError, match="active_vm_count"):
        predict.generate_predictions(data_dir=Path("."))


def test_generate_predictions_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unknown model"):
        predict.generate_predictions(
            data_dir=Path("."),
            cpu_model_name="unknown",
        )
