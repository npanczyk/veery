"""Direct falsification (naive sampling), following Kochenderfer et al.,
Algorithms for Validation (MIT Press, 2026), Ch. 4."""

import time
import numpy as np
from tqdm import tqdm
from veery.accessories import CostTracker


def direct_falsify(
    environment,
    model,
    disturbance_dist,
    m,
    threshold,
    percent=True,
    get_history=None,
    seed=None,
):
    """Runs m rollouts with disturbances sampled from the nominal distribution and checks each against the specification. A rollout ends at the first step that violates the specification.

    Args:
        environment (object): environment implementing reset, get_snapshot, step, and state_robustness (same interface MCTS uses)
        model (object): model with an SB3-style predict(observation, deterministic) method
        disturbance_dist (DisturbanceDistribution): sampled once per step
        m (int): number of rollouts
        threshold (float): specification as a deviation from the profile
        percent (bool, optional): True if threshold is a fractional deviation from the profile, False for a flat deviation. Defaults to True.
        get_history (function, optional): takes the environment after each rollout and returns what to keep as that rollout's history (e.g. lambda env: env.render(save=False)). Defaults to None, which keeps no histories.
        seed (int, optional): seeds the disturbance distribution and the environment's first reset (environment.reset(seed=seed)), so the same seed gives the same m rollouts. Defaults to None, which leaves both unseeded.

    Returns:
        dict:
            "passed" (np.ndarray[bool]): True where the rollout passed the specification
            "robustness" (np.ndarray[float]): minimum robustness over each rollout (for failures, the robustness at the failing step)
            "loglikelihoods" (np.ndarray[float]): log-likelihood of each rollout's disturbances, up to and including the failing step
            "disturbances" (list[list[Disturbance]]): disturbances sampled at each step of each rollout
            "histories" (list): output of get_history for each rollout, or None
            "p_fail" (float): estimated probability of failure
            "std_err" (float): standard error of p_fail
            "cost" (dict): attributes of the run's CostTracker (env_steps, rollouts, wall_time_s, n_failures, first_failure_steps)
            "cumulative_steps" (np.ndarray[int]): env_steps used by the end of each rollout
    """
    rhos = np.zeros(m)
    loglikelihoods = np.zeros(m)
    disturbances = []
    histories = [] if get_history is not None else None
    cost = CostTracker()
    cumulative_steps = np.zeros(m, dtype=int)

    if seed is not None:
        disturbance_dist.seed(seed)

    t0 = time.perf_counter()
    for i in tqdm(range(m)):
        # seed only the first reset; later resets continue the environment's random stream (gym convention)
        if seed is not None and i == 0:
            environment.reset(seed=seed)
        else:
            environment.reset()
        observation = environment.get_snapshot()["observation"]
        xs = []
        rho = np.inf
        done = False
        # stop at the first failure, so the loglikelihood only counts the path up to it (same as path_logp in MCTS)
        while environment.time < environment.runtime and not done and rho > 0:
            action, _ = model.predict(observation, deterministic=True)
            x = disturbance_dist.sample(state=environment.state, action=action)
            observation, _, terminated, truncated, _ = environment.step(
                action=action, disturbance=x
            )
            cost.env_steps += 1
            xs.append(x)
            rho = min(rho, environment.state_robustness(threshold, percent=percent))
            done = terminated or truncated

        rhos[i] = rho
        cost.rollouts += 1
        cumulative_steps[i] = cost.env_steps
        if rho <= 0:
            cost.record_failure()
        # sum over all three components (xo, xa, xs) and every step
        loglikelihoods[i] = sum(np.sum(disturbance_dist.logpdf(x)) for x in xs)
        disturbances.append(xs)
        if get_history is not None:
            histories.append(get_history(environment))
    cost.wall_time_s = time.perf_counter() - t0

    # a rollout fails if its robustness is not positive, the same as check_spec
    passed = rhos > 0
    failed = rhos <= 0
    prob_fail = failed.mean()
    return {
        "passed": passed,
        "robustness": rhos,
        "loglikelihoods": loglikelihoods,
        "disturbances": disturbances,
        "histories": histories,
        "p_fail": float(prob_fail),
        "std_err": float(np.sqrt(prob_fail * (1 - prob_fail) / m)),
        "cost": vars(cost),
        "cumulative_steps": cumulative_steps,
    }
