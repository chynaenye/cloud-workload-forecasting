"""Load the data and build the forecasting targets."""
from pathlib import Path

import pandas as pd

TRAIN_LEN = 110
HORIZON = 24
TARGETS = ["active_vm_count", "aggregate_cpu_utilization"]


def load_data(data_dir="data"):
    """Read the CPU readings and VM metadata from data_dir."""
    data_dir = Path(data_dir)
    cpu = pd.read_csv(data_dir / "vm_cpu_readings.csv.gz")
    meta = pd.read_csv(data_dir / "vm_metadata.csv.gz")
    return cpu, meta


def build_targets(cpu):
    """One row per timestamp: distinct active VMs and mean avg_cpu."""
    valid = cpu.dropna(subset=["avg_cpu"]).drop_duplicates(["vm_id", "timestamp"])
    return (
        valid.groupby("timestamp")
        .agg(
            active_vm_count=("vm_id", "nunique"),
            aggregate_cpu_utilization=("avg_cpu", "mean"),
        )
        .sort_index()
    )


def split_target(target, train_len=TRAIN_LEN, horizon=HORIZON):
    """Chronological split: first train_len rows train, last horizon rows holdout."""
    if len(target) < train_len + horizon:
        raise ValueError("series is too short for the requested split")
    train = target.iloc[:train_len]
    holdout = target.iloc[train_len : train_len + horizon]
    return train, holdout
