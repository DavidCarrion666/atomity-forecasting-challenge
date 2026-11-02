"""Data loading and target construction for the cloud workload forecasting challenge.

Leakage protocol
----------------
- The full panel is used ONLY to build the target time series (one row per
  timestamp).
- No target values from the holdout period (final 24 timestamps) are used
  to fit, select, or tune any model.
- Features derived from training timestamps must only look backwards.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import pandas as pd

# Challenge constants
TRAIN_OBS: int = 110
HOLD_OBS: int = 24
TOTAL_OBS: int = TRAIN_OBS + HOLD_OBS  # 134
INTERVAL_SECONDS: int = 300  # 5 minutes


@dataclass(frozen=True)
class ChallengeData:
    """Container for the split targets and raw panel."""

    train: pd.DataFrame    # first 110 rows (timestamp, active_vm_count, aggregate_cpu_utilization)
    holdout: pd.DataFrame  # last 24 rows (timestamp only — targets sealed from pipeline)
    vm_metadata: pd.DataFrame


def load_data(data_dir: str | Path) -> ChallengeData:
    """Load raw CSVs, build aggregate targets, and apply the 110/24 split.

    Parameters
    ----------
    data_dir:
        Directory containing ``vm_cpu_readings.csv.gz`` and
        ``vm_metadata.csv.gz``.

    Returns
    -------
    ChallengeData
        Named container with ``train``, ``holdout``, and ``vm_metadata``
        DataFrames.  ``holdout`` contains ONLY the ``timestamp`` column to
        prevent any leakage of real evaluation values into the pipeline.

    Notes
    -----
    - ``active_vm_count``: number of distinct VMs present at each timestamp
      (each VM counted at most once per timestamp).
    - ``aggregate_cpu_utilization``: mean of all ``avg_cpu`` readings at each
      timestamp.  Missing readings at a given timestamp are excluded from the
      mean (implicit NaN-safe behaviour of ``pandas.mean``).
    """
    data_dir = Path(data_dir)

    cpu = pd.read_csv(data_dir / "vm_cpu_readings.csv.gz")
    meta = pd.read_csv(data_dir / "vm_metadata.csv.gz")

    unique_timestamps = sorted(cpu["timestamp"].unique())
    n = len(unique_timestamps)
    if n != TOTAL_OBS:
        raise ValueError(
            f"Expected {TOTAL_OBS} unique timestamps, found {n}. "
            "Check the data files."
        )

    # ── Build aggregate target series for training period only ───────────────
    train_timestamps = set(unique_timestamps[:TRAIN_OBS])
    cpu_train = cpu[cpu["timestamp"].isin(train_timestamps)]

    train = (
        cpu_train.groupby("timestamp", sort=True)
        .agg(
            active_vm_count=("vm_id", "nunique"),
            aggregate_cpu_utilization=("avg_cpu", "mean"),
        )
        .reset_index()
    )
    train = train.sort_values("timestamp").reset_index(drop=True)

    # ── Holdout period: timestamps only (targets sealed) ─────────────────────
    holdout = pd.DataFrame(
        {"timestamp": unique_timestamps[TRAIN_OBS:]}
    )

    return ChallengeData(
        train=train,
        holdout=holdout,
        vm_metadata=meta,
    )
