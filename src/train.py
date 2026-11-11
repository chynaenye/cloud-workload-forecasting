"""Fit the final models on the training window only and forecast the horizon."""
import argparse

from src.data import HORIZON, TARGETS, build_targets, load_data, split_target
from src.models import FINAL_MODELS


def fit_forecast(train, horizon=HORIZON, models=FINAL_MODELS):
    """Return {target: array of `horizon` forecasts} using only `train`."""
    return {target: models[target](train[target], horizon) for target in TARGETS}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    args = parser.parse_args(argv)

    cpu, _ = load_data(args.data)
    train, _ = split_target(build_targets(cpu))
    for target, values in fit_forecast(train).items():
        print(target, values[:3].round(4), "...")


if __name__ == "__main__":
    main()