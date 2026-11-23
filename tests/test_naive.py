import pytest
from pathlib import Path
from veery.naive import *
import numpy as np


def test_direct_falsify(bakery, dummy_disturbance_dist, dummy_model):
    results_dict = direct_falsify(
        environment=bakery,
        model=dummy_model,
        disturbance_dist=dummy_disturbance_dist,
        m=10,
        threshold=1,
        percent=False,
        get_history=None,
        seed=None,
    )
    assert np.all(results_dict["passed"]) == True


def test_direct_falsify_failing(bakery, failing_disturbance_dist, dummy_model):
    results_dict = direct_falsify(
        environment=bakery,
        model=dummy_model,
        disturbance_dist=failing_disturbance_dist,
        m=10,
        threshold=1,
        percent=False,
        get_history=None,
        seed=None,
    )
    assert not np.all(results_dict["passed"]) == True


def test_direct_falsify_cost(bakery, failing_disturbance_dist, dummy_model):
    m = 10
    results_dict = direct_falsify(
        environment=bakery,
        model=dummy_model,
        disturbance_dist=failing_disturbance_dist,
        m=m,
        threshold=1,
        percent=False,
    )
    cost = results_dict["cost"]
    failed = ~results_dict["passed"]
    # one disturbance is sampled per step
    assert cost["env_steps"] == sum(len(xs) for xs in results_dict["disturbances"])
    assert cost["rollouts"] == m
    assert cost["n_failures"] == failed.sum()
    assert cost["first_failure_steps"] == results_dict["cumulative_steps"][failed][0]
    assert cost["wall_time_s"] > 0
    assert results_dict["p_fail"] == failed.mean()
