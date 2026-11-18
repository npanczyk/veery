import numpy as np
from pathlib import Path
import stable_baselines3 as sb3
from veery.direct_falsification import direct_falsify
import env
import profiles
import numpy as np
from veery import viz
from veery import fuzzing
import disturbances
from tqdm import tqdm
import helpers


def rollout(
    model: sb3.PPO,
    env_type,
    env_kwargs,
    disturbance_distribution,
    results_folder,
    save_histories=False,
):
    """Run one deterministic rollout of `model` on `env` to termination.

    Standard gym inference loop: reset, then step with the model's
    greedy action until the episode ends. `env.render()` is called
    once, after the loop — this assumes `render()` produces a
    post-hoc summary (e.g. a plot from accumulated history) rather
    than a per-frame live view.

    Args:
        model: Trained SB3 model with a `.predict()` method.
        env: Gym environment instance (not vectorized).
        disturbance_distribution (DisturbanceDistribution object from fuzzing.py): holds disturbance distributions for observations, actions, and states
        results_folder (str): where to save the histories and aux info
        save_histories (bool, optional): decides whether to save histories from render as a csv. Defaults to True.
    """
    test_env = env_type(**env_kwargs, save_dir=results_folder)
    obs, _ = test_env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        # get disturbances for the step
        x = disturbance_distribution.sample(test_env.state, action)
        obs, _, terminated, truncated, _ = test_env.step(action=action, disturbance=x)
        done = terminated or truncated

    if save_histories:
        return test_env.render(save=True)
    # CONSIDER DELETING THIS ELSE STATEMENT
    else:
        return test_env.render(save=False)


def run_rollouts(test_profile, model_name, run_name, m, p_sig=0.01, threshold=0.03):
    """Runs m rollouts with direct falsification, saves the trajectories and checks spec compliance.

    Args:
        test_profile (): from profiles
        model_name (str): model name inside /models folder
        run_name (str): name of folder inside /results to save the results
        m (int): number of rollouts
        threshold (float): fractional deviation from desired power allowed by the spec

    Returns:
        histories (list[pd.DataFrame]), results (np.ndarray[bool]): trajectories and pass/fail per rollout
    """
    _, testing_kwargs = profiles.get_profile(test_profile)
    results_root = Path("results")
    results_folder = results_root / run_name
    results_folder.mkdir(parents=True, exist_ok=True)
    plots_folder = results_folder / "figures"
    plots_folder.mkdir(parents=True, exist_ok=True)

    # set up disturbance distribution for the test
    disturbance_dist = fuzzing.DisturbanceDistribution(
        Do=disturbances.Do(sigma_p=p_sig, sigma_drum=0.005),
        Da=disturbances.Da(sigma_dtheta=0),
        Ds=disturbances.Ds(),
    )

    model = helpers.load_trained_model(model_name)
    test_env = env.HolosMulti(**testing_kwargs, save_dir=results_folder)

    out = direct_falsify(
        environment=test_env,
        model=model,
        disturbance_dist=disturbance_dist,
        m=m,
        threshold=threshold,
        get_history=lambda e: e.render(save=False),
    )
    histories = out["histories"]
    results = ~out["failed"]  # True where the rollout passed

    viz.plot_rollouts(histories, results, save_dir=plots_folder)
    viz.plot_robustness(
        histories[-1], threshold=threshold, time=None, ax=None, save_dir=plots_folder
    )

    return histories, results


def overlapping_rollouts(test_profile, model_name, run_name):
    """_summary_

    Args:
        test_profile (_type_): _description_
        model_name (_type_): _description_
        run_name (_type_): _description_
    """
    histories_list = []
    results_list = []
    sigmas = [0.01, 0.005, 0.0025, 0.00125]
    colors = ["cyan", "magenta", "blue", "tomato"]
    for sig in sigmas:
        h, r = run_rollouts(
            test_profile=test_profile,
            model_name=model_name,
            run_name=run_name,
            m=5,
            p_sig=sig,
        )
        histories_list.append(h)
        results_list.append(r)

    viz.plot_overlapping_rollouts(
        histories_list,
        results_list,
        sigmas,
        colors,
        threshold=0.03,
        save_as=f"results/{run_name}/figures/overlapping.png",
        output_name="Fraction of Total Power, [-]",
    )
    return


if __name__ == "__main__":
    run_rollouts(
        test_profile="test", model_name="default", run_name="demo", m=1, p_sig=0.01
    )
    overlapping_rollouts(test_profile="test", model_name="default", run_name="demo")
