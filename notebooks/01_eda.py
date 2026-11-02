"""Exploratory analysis for the cloud workload forecasting challenge.

Leakage protocol
----------------
- Panel integrity (row counts, cadence, join keys) may use the full release.
- Target dynamics, autocorrelation, segments, peaks, and any statistic that
  could influence model choice use ONLY the first 110 timestamps.
- The final 24 timestamps are not loaded into those summaries.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "challenge_data"
FIG = Path(__file__).resolve().parent / "figures"
TRAIN_OBS = 110
HOLD_OBS = 24


def acf(values: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    x = x - x.mean()
    denom = np.dot(x, x)
    out = [1.0]
    for lag in range(1, max_lag + 1):
        out.append(float(np.dot(x[:-lag], x[lag:]) / denom) if denom else 0.0)
    return np.asarray(out)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    cpu = pd.read_csv(DATA / "vm_cpu_readings.csv.gz")
    meta = pd.read_csv(DATA / "vm_metadata.csv.gz")

    timestamps = np.sort(cpu["timestamp"].unique())
    assert len(timestamps) == TRAIN_OBS + HOLD_OBS
    train_ts = set(timestamps[:TRAIN_OBS].tolist())
    cpu_train = cpu[cpu["timestamp"].isin(train_ts)].copy()

    targets = (
        cpu.groupby("timestamp", sort=True)
        .agg(
            active_vm_count=("vm_id", "nunique"),
            aggregate_cpu_utilization=("avg_cpu", "mean"),
        )
        .reset_index()
    )
    train = targets.iloc[:TRAIN_OBS].copy()
    y = train["aggregate_cpu_utilization"].to_numpy(dtype=float)

    print("=== PROTOCOL ===")
    print("train timestamps", int(train["timestamp"].min()), "->", int(train["timestamp"].max()), "n", len(train))
    print("holdout sealed after", int(timestamps[TRAIN_OBS - 1]))

    print("\n=== PANEL (full release, quality only) ===")
    print("cpu rows", len(cpu), "vms", cpu["vm_id"].nunique(), "timestamps", cpu["timestamp"].nunique())
    print("active_vm_count unique", targets["active_vm_count"].unique().tolist())

    print("\n=== TRAIN TARGET ===")
    print(pd.Series(y).describe(percentiles=[0.05, 0.5, 0.95]).to_string())
    print("range", float(y.min()), float(y.max()), "std", float(y.std(ddof=1)))
    lag1 = np.abs(np.diff(y))
    print("mean |delta| 5min", float(lag1.mean()), "max |delta|", float(lag1.max()))
    persistence_mae = float(np.mean(np.abs(y[1:] - y[:-1])))
    mean_mae = float(np.mean(np.abs(y - y.mean())))
    print("in-sample persistence MAE (y_t vs y_{t-1})", persistence_mae)
    print("in-sample mean-baseline MAE", mean_mae)

    acf_vals = acf(y, 36)
    print("\n=== ACF (train) ===")
    for lag in [1, 2, 3, 6, 12, 24, 36]:
        print(f"lag {lag:2d} ({lag * 5} min)", float(acf_vals[lag]))

    spread = cpu_train.assign(spread=cpu_train["max_cpu"] - cpu_train["min_cpu"])
    vm_stats = (
        cpu_train.groupby("vm_id")
        .agg(
            mean_avg=("avg_cpu", "mean"),
            p95_avg=("avg_cpu", lambda s: s.quantile(0.95)),
            mean_max=("max_cpu", "mean"),
        )
        .reset_index()
    )
    mean_spread = spread.groupby("vm_id", as_index=False)["spread"].mean().rename(columns={"spread": "mean_spread"})
    vm_stats = vm_stats.merge(mean_spread, on="vm_id").merge(meta, on="vm_id", how="left")
    total_cpu = vm_stats["mean_avg"].sum()
    ordered = vm_stats.sort_values("mean_avg", ascending=False)
    shares = {
        "top_1pct": ordered.head(100)["mean_avg"].sum() / total_cpu,
        "top_5pct": ordered.head(500)["mean_avg"].sum() / total_cpu,
        "top_10pct": ordered.head(1000)["mean_avg"].sum() / total_cpu,
    }
    print("\n=== VM CONCENTRATION (train means) ===")
    print("VM mean_avg describe")
    print(vm_stats["mean_avg"].describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_string())
    print("share of fleet mean CPU", {k: float(v) for k, v in shares.items()})
    print("VMs with mean_avg > 50%", int((vm_stats["mean_avg"] > 50).sum()))
    print("mean intra-interval max-min spread", float(spread["spread"].mean()))

    print("\n=== SEGMENTS (train) ===")
    for col in ["vm_category", "virtual_core_bucket", "memory_gb_bucket"]:
        g = vm_stats.groupby(col)["mean_avg"].agg(["count", "mean", "median"])
        print(col)
        print(g.sort_values("mean", ascending=False).to_string())
        print()

    peak_cut = y.mean() + 2 * y.std(ddof=1)
    peaks = train.loc[y >= peak_cut, ["timestamp", "aggregate_cpu_utilization"]]
    print("=== PEAKS train mean+2sd ===", float(peak_cut))
    print(peaks.to_string(index=False))

    hours = (train["timestamp"] / 3600).to_numpy()
    hour_bin = np.floor(hours).astype(int)
    by_hour = pd.DataFrame({"hour": hour_bin, "cpu": y}).groupby("hour")["cpu"].mean()
    print("\n=== MEAN AGG CPU BY ELAPSED HOUR (train) ===")
    print(by_hour.to_string())

    # Figures
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train["timestamp"] / 3600, y, lw=1.4, label="aggregate_cpu_utilization")
    ax.axhline(y.mean(), color="C1", ls="--", lw=1, label="train mean")
    ax.set_xlabel("Elapsed hours from trace start")
    ax.set_ylabel("Mean VM avg_cpu (%)")
    ax.set_title("Fleet-mean CPU — train window only (110 x 5 min)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "train_aggregate_cpu.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    lags = np.arange(0, 37)
    ax.bar(lags, acf_vals, width=0.8)
    ax.axhline(1.96 / np.sqrt(len(y)), color="C1", ls="--", lw=1, label="~95% white-noise bound")
    ax.axhline(-1.96 / np.sqrt(len(y)), color="C1", ls="--", lw=1)
    ax.set_xlabel("Lag (5-minute steps)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("ACF of aggregate_cpu_utilization (train)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "train_acf.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(vm_stats["mean_avg"], bins=60, color="C0")
    ax.set_xlabel("Per-VM mean avg_cpu over train (%)")
    ax.set_ylabel("VM count")
    ax.set_title("Cross-section of VM mean CPU (train)")
    fig.tight_layout()
    fig.savefig(FIG / "vm_mean_cpu_hist.png", dpi=120)
    plt.close(fig)

    cat = vm_stats.groupby("vm_category")["mean_avg"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(cat.index.astype(str), cat.values)
    ax.set_ylabel("Mean of VM-level mean avg_cpu (%)")
    ax.set_title("Train CPU by vm_category")
    fig.tight_layout()
    fig.savefig(FIG / "cpu_by_category.png", dpi=120)
    plt.close(fig)

    print("\nSaved figures in", FIG)


if __name__ == "__main__":
    main()
