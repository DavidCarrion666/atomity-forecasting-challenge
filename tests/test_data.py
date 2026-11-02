"""Integration tests for cwfc.data.load_data and ChallengeData invariants."""

from __future__ import annotations

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from cwfc.data import load_data, ChallengeData, TRAIN_OBS, HOLD_OBS, TOTAL_OBS, INTERVAL_SECONDS

DATA_DIR = Path("challenge_data")


@pytest.fixture(scope="module")
def challenge():
    if not (DATA_DIR / "vm_cpu_readings.csv.gz").exists():
        pytest.skip("Real dataset not found in challenge_data/")
    return load_data(DATA_DIR)


def test_challenge_data_structure(challenge):
    """Verify ChallengeData container type and fields."""
    assert isinstance(challenge, ChallengeData)
    assert hasattr(challenge, "train")
    assert hasattr(challenge, "holdout")
    assert hasattr(challenge, "vm_metadata")

    # Invariant F-03: holdout target values must NOT be accessible in ChallengeData
    assert not hasattr(challenge, "targets"), "ChallengeData should not expose 'targets' attribute with holdout values"


def test_train_invariants(challenge):
    """Verify invariants for training portion (first 110 observations)."""
    train = challenge.train
    assert isinstance(train, pd.DataFrame)
    assert len(train) == TRAIN_OBS, f"Expected {TRAIN_OBS} training rows, got {len(train)}"

    expected_cols = ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]
    assert list(train.columns) == expected_cols

    # No missing values
    assert not train.isna().any().any(), "Training data contains NaNs"

    # Monotonicity and cadence
    ts = train["timestamp"].to_numpy()
    assert (np.diff(ts) == INTERVAL_SECONDS).all(), f"Cadence should be strictly {INTERVAL_SECONDS}s"

    # Range and validity
    assert (train["active_vm_count"] >= 0).all()
    assert (train["active_vm_count"] == 10000).all(), "All 10,000 VMs expected active in train"
    assert train["aggregate_cpu_utilization"].between(0.0, 100.0).all()


def test_holdout_invariants(challenge):
    """Verify invariants for holdout portion (last 24 observations, F-03)."""
    holdout = challenge.holdout
    assert isinstance(holdout, pd.DataFrame)
    assert len(holdout) == HOLD_OBS, f"Expected {HOLD_OBS} holdout rows, got {len(holdout)}"

    # F-03: holdout must contain EXCLUSIVELY the timestamp column
    assert list(holdout.columns) == ["timestamp"], (
        f"holdout must contain exclusively ['timestamp'], got {list(holdout.columns)}"
    )
    assert "active_vm_count" not in holdout.columns
    assert "aggregate_cpu_utilization" not in holdout.columns

    # No missing timestamps
    assert not holdout["timestamp"].isna().any()

    # Monotonicity and cadence
    ts = holdout["timestamp"].to_numpy()
    assert (np.diff(ts) == INTERVAL_SECONDS).all(), f"Cadence should be strictly {INTERVAL_SECONDS}s"

    # Temporal continuity between train and holdout
    train_last_ts = challenge.train["timestamp"].iloc[-1]
    holdout_first_ts = holdout["timestamp"].iloc[0]
    assert holdout_first_ts - train_last_ts == INTERVAL_SECONDS, (
        f"Holdout should start exactly {INTERVAL_SECONDS}s after train ends"
    )


def test_metadata_invariants(challenge):
    """Verify vm_metadata loading and contents."""
    meta = challenge.vm_metadata
    assert isinstance(meta, pd.DataFrame)
    assert "vm_id" in meta.columns
    assert len(meta) == 10000


def test_load_data_missing_dir(tmp_path):
    """Verify error on non-existent directory."""
    with pytest.raises(Exception):
        load_data(tmp_path / "non_existent_dir")
