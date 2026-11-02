"""Generate the final predictions.csv for submission.

Usage
-----
    python -m cwfc.predict --data challenge_data --output predictions.csv [--model <name>]

The script:
1. Loads the raw data and constructs aggregate targets.
2. Uses the first 110 observations as training data.
3. Fits the selected model on the full training window.
4. Generates 24 forecasts (2 hours).
5. Maps predictions to the exact holdout timestamps.
6. Writes predictions.csv in the required schema:
       timestamp, active_vm_count, aggregate_cpu_utilization

Model selection
---------------
By default the script uses ARIMA(2,0,0) for aggregate_cpu_utilization and the
constant 10000 for active_vm_count, after checking that every training value
confirms this invariant.

Pass ``--model <name>`` to override the CPU model. Available names include all
registered baselines, ``temporal_regression``, and ``arima_2_0_0``.

Leakage contract
----------------
- :func:`cwfc.data.load_data` returns target values for only the first 110
  timestamps and a timestamp-only frame for the final 24.
- Model fitting receives only the 110-row training series.
- Holdout target values are never used; only holdout timestamps enter the CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cwfc.baselines import BASELINE_REGISTRY
from cwfc.data import load_data, TRAIN_OBS, HOLD_OBS
from cwfc.models import arima_forecast, temporal_regression_forecast


ACTIVE_VM_DEFAULT = 10_000  # constant — all VMs observed at every training timestamp


PREDICTION_MODEL_REGISTRY = {
    **BASELINE_REGISTRY,
    "temporal_regression": temporal_regression_forecast,
    "arima_2_0_0": lambda train, horizon: arima_forecast(
        train,
        horizon,
        order=(2, 0, 0),
        trend="c",
    ),
}


def generate_predictions(
    data_dir: Path,
    cpu_model_name: str = "arima_2_0_0",
) -> pd.DataFrame:
    """Fit model on train, forecast holdout, return submission DataFrame.

    Parameters
    ----------
    data_dir:
        Directory containing the challenge CSV files.
    cpu_model_name:
        Key into :data:`PREDICTION_MODEL_REGISTRY` to use for
        ``aggregate_cpu_utilization``. Defaults to ARIMA(2,0,0).

    Returns
    -------
    pd.DataFrame
        24 rows × 3 columns: timestamp, active_vm_count, aggregate_cpu_utilization.
    """
    if cpu_model_name not in PREDICTION_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{cpu_model_name}'. "
            f"Available: {sorted(PREDICTION_MODEL_REGISTRY)}"
        )

    challenge = load_data(data_dir)
    train = challenge.train
    holdout = challenge.holdout

    if len(train) != TRAIN_OBS:
        raise ValueError(f"Expected {TRAIN_OBS} training rows, got {len(train)}")
    if len(holdout) != HOLD_OBS:
        raise ValueError(f"Expected {HOLD_OBS} holdout timestamps, got {len(holdout)}")
    if not train["timestamp"].is_monotonic_increasing or not train["timestamp"].is_unique:
        raise ValueError("Training timestamps must be strictly ordered and unique")

    train_timestamps = train["timestamp"].to_numpy()
    holdout_timestamps = holdout["timestamp"].to_numpy()
    if pd.isna(holdout_timestamps).any():
        raise ValueError("Holdout timestamps must not contain null values")
    if not pd.Index(holdout_timestamps).is_monotonic_increasing:
        raise ValueError("Holdout timestamps must be ordered")
    if pd.Index(holdout_timestamps).has_duplicates:
        raise ValueError("Holdout timestamps must be unique")
    if not train_timestamps[-1] < holdout_timestamps[0]:
        raise ValueError("Training timestamps must precede holdout timestamps")

    # Keep the constant forecast only while the training invariant holds.
    active_vm_train = train["active_vm_count"].to_numpy(dtype=float)
    if not np.all(np.isfinite(active_vm_train)):
        raise ValueError("Training active_vm_count must contain only finite values")
    if not np.all(active_vm_train == ACTIVE_VM_DEFAULT):
        raise ValueError(
            "The constant active_vm_count forecast requires all training values "
            f"to equal {ACTIVE_VM_DEFAULT}"
        )
    active_vm_pred = np.full(HOLD_OBS, ACTIVE_VM_DEFAULT)

    # load_data exposes target values for training timestamps only.
    cpu_series = train["aggregate_cpu_utilization"].to_numpy(dtype=float)
    if not np.all(np.isfinite(cpu_series)):
        raise ValueError("Training aggregate_cpu_utilization must contain only finite values")

    forecaster = PREDICTION_MODEL_REGISTRY[cpu_model_name]
    cpu_pred = forecaster(cpu_series, HOLD_OBS)
    cpu_pred = np.asarray(cpu_pred, dtype=float)
    if cpu_pred.shape != (HOLD_OBS,):
        raise ValueError(
            f"Model '{cpu_model_name}' must return {HOLD_OBS} predictions, "
            f"got shape {cpu_pred.shape}"
        )
    if not np.all(np.isfinite(cpu_pred)):
        raise ValueError(f"Model '{cpu_model_name}' returned non-finite predictions")

    cpu_pred = np.clip(cpu_pred, 0.0, 100.0)

    submission = pd.DataFrame(
        {
            "timestamp": holdout_timestamps,
            "active_vm_count": active_vm_pred.astype(int),
            "aggregate_cpu_utilization": cpu_pred,
        }
    )

    expected_columns = [
        "timestamp",
        "active_vm_count",
        "aggregate_cpu_utilization",
    ]
    if list(submission.columns) != expected_columns:
        raise ValueError(f"Submission columns must be {expected_columns}")
    if len(submission) != HOLD_OBS:
        raise ValueError(f"Expected {HOLD_OBS} predictions, got {len(submission)}")
    if submission.isna().any().any():
        raise ValueError("Submission must not contain null values")
    if not submission["timestamp"].is_monotonic_increasing:
        raise ValueError("Submission timestamps are not sorted")
    if submission["timestamp"].nunique() != HOLD_OBS:
        raise ValueError("Submission timestamps contain duplicates")
    numeric_predictions = submission[
        ["active_vm_count", "aggregate_cpu_utilization"]
    ].to_numpy(dtype=float)
    if not np.all(np.isfinite(numeric_predictions)):
        raise ValueError("Submission predictions must be finite")
    if not (submission["active_vm_count"] >= 0).all():
        raise ValueError("active_vm_count must be non-negative")
    if not submission["aggregate_cpu_utilization"].between(0, 100).all():
        raise ValueError("aggregate_cpu_utilization must be in [0, 100]")

    return submission


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Generate predictions.csv for the holdout period.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        required=True,
        metavar="DIR",
        help="Path to challenge_data/ directory.",
    )
    parser.add_argument(
        "--output",
        default="predictions.csv",
        metavar="FILE",
        help="Output path for predictions.csv.",
    )
    parser.add_argument(
        "--model",
        default="arima_2_0_0",
        metavar="NAME",
        help=(
            "Model to use for aggregate_cpu_utilization. "
            f"Options: {sorted(PREDICTION_MODEL_REGISTRY)}"
        ),
    )
    args = parser.parse_args()

    submission = generate_predictions(Path(args.data), cpu_model_name=args.model)

    out_path = Path(args.output)
    submission.to_csv(out_path, index=False)

    print(f"Wrote {len(submission)} predictions to {out_path}")
    print(submission.to_string(index=False))


if __name__ == "__main__":
    _cli()
