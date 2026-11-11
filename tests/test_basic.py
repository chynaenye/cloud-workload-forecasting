"""Smoke tests for the whole pipeline. Run with: python -m pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src.data import HORIZON, TRAIN_LEN, build_targets, split_target
from src.models import MODELS
from src.predict import build_predictions, check_predictions, main
from src.validate import mae, relative_improvement, rmse, rolling_origin

N_VMS, N_STEPS = 20, 134


@pytest.fixture
def cpu():
    """Small fake dataset with the same shape rules as the real one."""
    rng = np.random.default_rng(0)
    ts = np.repeat(np.arange(N_STEPS) * 300, N_VMS)
    vm = np.tile([f"vm{i}" for i in range(N_VMS)], N_STEPS)
    avg = rng.uniform(1, 20, size=len(ts))
    return pd.DataFrame(
        {"timestamp": ts, "vm_id": vm, "min_cpu": avg * 0.5, "max_cpu": avg * 1.5, "avg_cpu": avg}
    )


def test_targets_and_split(cpu):
    target = build_targets(cpu)
    assert len(target) == N_STEPS
    assert (target["active_vm_count"] == N_VMS).all()
    train, holdout = split_target(target)
    assert len(train) == TRAIN_LEN and len(holdout) == HORIZON
    assert train.index.max() < holdout.index.min()  # chronological


def test_duplicates_counted_once(cpu):
    target = build_targets(pd.concat([cpu, cpu.iloc[:5]]))
    assert (target["active_vm_count"] == N_VMS).all()


@pytest.mark.parametrize("name", list(MODELS))
def test_models_return_h_finite_values(cpu, name):
    train, _ = split_target(build_targets(cpu))
    pred = MODELS[name](train["aggregate_cpu_utilization"], HORIZON)
    assert len(pred) == HORIZON and np.isfinite(pred).all()


def test_rolling_origin_only_sees_the_past(cpu):
    series = build_targets(cpu)["aggregate_cpu_utilization"].iloc[:TRAIN_LEN]
    seen = []

    def spy(history, h):
        seen.append(len(history))
        return np.zeros(h)

    rolling_origin(series, spy)
    assert max(seen) + HORIZON <= len(series)  # forecast window never leaves the data


def test_metrics():
    a, p = np.array([1.0, 2.0, 3.0]), np.array([1.0, 3.0, 5.0])
    assert mae(a, p) == pytest.approx(1.0)
    assert rmse(a, p) == pytest.approx(np.sqrt(5 / 3))
    assert relative_improvement(0.5, 1.0) == pytest.approx(0.5)
    assert np.isnan(relative_improvement(0.0, 0.0))


def test_holdout_values_do_not_change_predictions(cpu):
    """Leakage check: changing the holdout data must not change the forecast."""
    train, holdout = split_target(build_targets(cpu))
    original = build_predictions(train, holdout.index)

    changed = cpu.copy()
    last = sorted(changed["timestamp"].unique())[-HORIZON:]
    changed.loc[changed["timestamp"].isin(last), "avg_cpu"] *= 3
    train2, holdout2 = split_target(build_targets(changed))
    pd.testing.assert_frame_equal(original, build_predictions(train2, holdout2.index))


def test_end_to_end_writes_valid_file(cpu, tmp_path):
    cpu.to_csv(tmp_path / "vm_cpu_readings.csv.gz", index=False)
    pd.DataFrame({"vm_id": cpu["vm_id"].unique()}).to_csv(tmp_path / "vm_metadata.csv.gz", index=False)
    out = tmp_path / "predictions.csv"

    main(["--data", str(tmp_path), "--output", str(out)])

    df = pd.read_csv(out)
    check_predictions(df)
    expected = sorted(cpu["timestamp"].unique())[-HORIZON:]
    assert df["timestamp"].tolist() == expected


def test_check_predictions_catches_bad_file():
    bad = pd.DataFrame({"timestamp": [0], "active_vm_count": [1], "aggregate_cpu_utilization": [5]})
    with pytest.raises(ValueError):
        check_predictions(bad)