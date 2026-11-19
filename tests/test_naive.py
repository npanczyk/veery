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
