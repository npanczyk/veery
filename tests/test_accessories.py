import pytest
from veery.accessories import *
from conftest import DummyDisturbanceDistribution
import pandas as pd


def test_check_spec_percent():
    history = pd.DataFrame({"Actual": [2, 2, 2], "Desired": [1, 1, 1]})
    threshold = 0.5
    check = check_spec(history, threshold, percent=True)
    assert check == False


def test_check_spec_raw_deviation():
    history = pd.DataFrame({"Actual": [0, 1, 0], "Desired": [1, 1, 1]})
    threshold = 1
    check = check_spec(history, threshold, percent=True)
    assert check == False


def test_robustness():
    actual = [1, 1, 1]
    desired = [5, 4, 5]
    threshold = 1
    assert -3 == robustness(actual, desired, threshold, percent=False)


def test_robustness_percent():
    actual = [1, 1, 1]
    desired = [2, 1, 2]
    threshold = 0.5
    assert 0 == robustness(actual, desired, threshold, percent=True)


def test_rollout_from_snapshot_matches_env_method(bakery, dummy_model):
    snap = bakery.get_snapshot()
    # fresh distributions so both rollouts see the same disturbance sequence
    expected = bakery.rollout_from_snapshot(
        dummy_model, snap, DummyDisturbanceDistribution(xs_values=(0, 1))
    )
    actual, desired = rollout_from_snapshot(
        bakery,
        dummy_model,
        snap,
        DummyDisturbanceDistribution(xs_values=(0, 1)),
        get_actual=lambda env: env.cakes,
    )
    assert actual == expected[0]
    assert desired == expected[1]


def test_rollout_from_snapshot_runs_to_runtime(
    bakery, dummy_model, dummy_disturbance_dist
):
    bakery.step(action=1)
    snap = bakery.get_snapshot()
    actual, desired = rollout_from_snapshot(
        bakery,
        dummy_model,
        snap,
        dummy_disturbance_dist,
        get_actual=lambda env: env.cakes,
    )
    # starts at t=1, so 4 steps remain before runtime=5
    assert len(actual) == len(desired) == 4
    assert bakery.time == bakery.runtime
