"""Produce predictions.csv for the final 24 timestamps."""
import argparse

import numpy as np
import pandas as pd

from src.data import HORIZON, build_targets, load_data, split_target
from src.models import clip_count, clip_cpu
from src.train import fit_forecast

COLUMNS = ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]


def build_predictions(train, holdout_index):
    """Forecast from `train` only. The holdout contributes timestamps, never values."""
    forecast = fit_forecast(train, len(holdout_index))
    return pd.DataFrame(
        {
            "timestamp": holdout_index,
            "active_vm_count": clip_count(forecast["active_vm_count"]),
            "aggregate_cpu_utilization": clip_cpu(forecast["aggregate_cpu_utilization"]),
        }
    )


def check_predictions(df, horizon=HORIZON):
    """Raise ValueError if the file breaks the submission rules."""
    if list(df.columns) != COLUMNS:
        raise ValueError(f"columns must be {COLUMNS}")
    if len(df) != horizon:
        raise ValueError(f"expected {horizon} rows, got {len(df)}")
    if not df["timestamp"].is_monotonic_increasing or df["timestamp"].duplicated().any():
        raise ValueError("timestamps must be increasing and unique")
    if not np.isfinite(df[COLUMNS[1:]].to_numpy(dtype=float)).all():
        raise ValueError("values must be finite")
    if (df["active_vm_count"] < 0).any():
        raise ValueError("active_vm_count must be >= 0")
    if not df["aggregate_cpu_utilization"].between(0, 100).all():
        raise ValueError("aggregate_cpu_utilization must be in [0, 100]")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="challenge_data")
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args(argv)

    cpu, _ = load_data(args.data)
    train, holdout = split_target(build_targets(cpu))
    predictions = build_predictions(train, holdout.index)
    check_predictions(predictions)
    predictions.to_csv(args.output, index=False)
    print(f"Wrote {len(predictions)} rows to {args.output}")


if __name__ == "__main__":
    main()