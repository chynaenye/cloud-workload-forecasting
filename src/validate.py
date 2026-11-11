"""Time-aware validation: rolling-origin evaluation on the training window only."""
import argparse

import numpy as np
import pandas as pd

from src.data import HORIZON, TARGETS, build_targets, load_data, split_target
from src.models import MODELS


def mae(actual, pred):
    return np.mean(np.abs(actual - pred))


def rmse(actual, pred):
    return np.sqrt(np.mean((actual - pred) ** 2))


def relative_improvement(model_error, baseline_error):
    """0.20 means 20% better than the baseline. NaN if the baseline error is 0."""
    if baseline_error == 0:
        return np.nan
    return 1 - model_error / baseline_error


def rolling_origin(series, forecast_fn, h=HORIZON, first_origin=60, step=6):
    """Forecast h steps from several origins, using only data before each origin."""
    actual, pred, steps = [], [], []
    for origin in range(first_origin, len(series) - h + 1, step):
        history = series.iloc[:origin]
        actual.append(series.iloc[origin : origin + h].values)
        pred.append(forecast_fn(history, h))
        steps.append(np.arange(1, h + 1))
    return np.concatenate(actual), np.concatenate(pred), np.concatenate(steps)


def compare_models(train, models=MODELS):
    """MAE and RMSE for every model on every target."""
    rows = []
    for target in TARGETS:
        for name, fn in models.items():
            a, p, _ = rolling_origin(train[target], fn)
            rows.append({"target": target, "model": name, "MAE": mae(a, p), "RMSE": rmse(a, p)})
    return pd.DataFrame(rows)


def safety_margin(series, forecast_fn, q=0.9):
    """q-th quantile of validation errors (actual - forecast)."""
    a, p, _ = rolling_origin(series, forecast_fn)
    return float(np.quantile(a - p, q))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="challenge_data")
    args = parser.parse_args(argv)

    cpu, _ = load_data(args.data)
    train, _ = split_target(build_targets(cpu))  # holdout is never touched here
    print(compare_models(train).round(4).to_string(index=False))


if __name__ == "__main__":
    main()