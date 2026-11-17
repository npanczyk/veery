"""Specification checks (ψ) and robustness (ρ), following Kochenderfer et al.,
Algorithms for Validation (MIT Press, 2026), Ch. 3."""

import numpy as np


def check_spec(history, threshold, percent=True):
    """Checks if a rollout's power trajectory stays within 3% (relative) of desired power.

    Args:
        history (pd.DataFrame): rollout history with 'Actual' and 'Desired' columns
        threshold (float): specficiation value for percent difference limit
        percent (bool): True to use a percent deviation from the "Desired" column in history. Otherwise uses threshold as a flat deviation from "Desired"
    Returns:
        bool: True if actual power never deviates >=3% (relative) from desired power
    """
    if percent:
        dev = np.abs((history["Actual"] - history["Desired"]) / history["Desired"])
    else:
        dev = np.abs(history["Actual"] - history["Desired"])
    return (dev < threshold).all()


def percent_diff(actual, desired):
    return abs(actual - desired) / desired


def robustness(actual, desired, threshold, percent=True):
    """Calculates the robustness of a trajectory defined by the actual power and the desired power for all timesteps. This is based on signal temporal logic.

    Args:
        actual (array): Actual achieved output
        desired (array): Desired output at corresponding times to actual
        threshold (float): specification as a deviation from profile
        percent (bool): True if threshold is a fractional deviation from desired. Otherwise threshold is a flat deviation, which must be used when desired can be 0.
    """
    # if we're at a terminal node, we won't get power arrays, so return infinity (we're going to take a min of this rho and the path rho, so path rho will always get picked if we're at a terminal node)
    if type(actual) != list:
        actual = np.array([actual])
    if type(desired) != list:
        desired = np.array([desired])
    if len(actual) == 0:
        return np.inf
    diffs = abs(np.array(actual) - np.array(desired))
    if percent:
        diffs = diffs / np.array(desired)
    rho = np.min(threshold - diffs)
    return rho


def rollout_from_snapshot(env, model, snap, disturbance_distribution, get_actual):
    """Runs a rollout of a model from a snapshot until the episode ends. Environments can call this from their own rollout_from_snapshot method, which MCTS uses to score nodes.

    Args:
        env (object): environment implementing set_snapshot, step (gymnasium-style return), profile, state, time, and runtime
        model (object): model with an SB3-style predict(observation, deterministic) method
        snap (dict): snapshot of the environment state from env.get_snapshot()
        disturbance_distribution (DisturbanceDistribution): sampled once per step
        get_actual (function): takes env and returns the actual output at the current state, compared against env.profile(env.time)
    Returns:
        tuple: actual output over rollout, desired output over rollout
    """
    observation = env.set_snapshot(snap)
    actual = []
    desired = []
    done = False
    while env.time < env.runtime and not done:
        action, _ = model.predict(observation, deterministic=True)
        x = disturbance_distribution.sample(state=env.state, action=action)
        observation, _, terminated, truncated, _ = env.step(action=action, disturbance=x)
        actual.append(get_actual(env))
        desired.append(float(env.profile(env.time)))
        done = terminated or truncated
    return actual, desired
