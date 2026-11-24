"""Tunes MCTS hyperparameters on the Holos test profile with Optuna, minimizing the number of environment steps it takes to find the first failure.

Run from experiments/holos (load_trained_model uses relative paths). Change the settings by editing the defaults of main().
"""

import json
from functools import partial
from pathlib import Path

import numpy as np
import optuna
from veery.fuzzing import DisturbanceDistribution
from veery.mcts import run_MCTS
import disturbances
import env
import profiles
from helpers import load_trained_model

# same setup as holos_tutorial.ipynb
PROFILE = "test"
MODEL_NAME = "exp__seed14"
THRESHOLD = 0.03
# current run_MCTS defaults, enqueued as trial 0 so the tuned values can be compared against them
BASELINE = {
    "c": 0.5,
    "k": 1,
    "alpha": 0.25,
    "lambda": 0.01,
    "max_children": 3,
    "n_rollouts": 3,
}


def make_disturbance_dist():
    """Disturbance distribution from holos_tutorial.ipynb (observation noise only)."""
    return DisturbanceDistribution(
        Do=disturbances.Do(sigma_p=0.005, sigma_drum=0.005),
        Da=disturbances.Da(sigma_dtheta=0),
        Ds=disturbances.Ds(
            sigma_nr=0,
            sigma_precursors=0,
            sigma_Tf=0,
            sigma_Tm=0,
            sigma_Tc=0,
            sigma_xe=0,
            sigma_i=0,
        ),
    )


def objective(trial, model, results_path, iterations, seeds, no_fail_penalty):
    """Mean env steps to the first failure over seeds for one set of hyperparameters.

    Args:
        trial (optuna.Trial): Trial that suggests the hyperparameters
        model (sb3 model): Controller being falsified
        results_path (Path): Folder for this study's MCTS results
        iterations (int): MCTS iteration budget per seed
        seeds (tuple): Seeds to run MCTS with
        no_fail_penalty (float): A seed that finds no failure costs env_steps * no_fail_penalty

    Returns:
        float: mean env steps to the first failure
    """
    hyperparams = {
        "c": trial.suggest_float("c", 0.01, 5, log=True),
        "k": trial.suggest_float("k", 0.5, 5),
        "alpha": trial.suggest_float("alpha", 0.1, 0.9),
        "lambda": trial.suggest_float("lambda", 1e-4, 1, log=True),
        "max_children": trial.suggest_int("max_children", 2, 20),
        "n_rollouts": trial.suggest_int("n_rollouts", 1, 5),
    }
    _, testing_kwargs = profiles.get_profile(name=PROFILE, max_failed_drums=0)

    steps, n_found = [], 0
    for seed in seeds:
        run_path = results_path / f"trial_{trial.number}" / f"seed_{seed}"
        run_MCTS(
            environment=env.HolosMulti(**testing_kwargs),
            model=model,
            disturbance_dist=make_disturbance_dist(),
            results_path=run_path,
            threshold=THRESHOLD,
            iterations=iterations,
            hyperparams=hyperparams,
            percent=True,
            seed=seed,
            model_name=MODEL_NAME,
            extra_config={"profile": PROFILE, "trial": trial.number},
            stop_at_first_failure=True,
            save_tree=False,
        )
        with open(run_path / "cost.json") as f:
            cost = json.load(f)
        if cost["first_failure_steps"] is not None:
            steps.append(cost["first_failure_steps"])
            n_found += 1
        else:
            # censored: we only know the first failure takes more than env_steps, so penalize the miss
            steps.append(cost["env_steps"] * no_fail_penalty)

    trial.set_user_attr("steps_per_seed", steps)
    trial.set_user_attr("n_found", n_found)
    return float(np.mean(steps))


def main(
    n_trials=50,
    iterations=500,
    seeds=(0, 1, 2),
    study_name="mcts_ttff",
    no_fail_penalty=2,
):
    """Runs (or resumes) an Optuna study and saves trials.csv and best_hyperparams.json to results/tuning/<study_name>.

    Args:
        n_trials (int, optional): Number of trials to add to the study. Defaults to 50.
        iterations (int, optional): MCTS iteration budget per seed. Defaults to 500.
        seeds (tuple, optional): Seeds each trial runs MCTS with. Defaults to (0, 1, 2).
        study_name (str, optional): Name of the study and its results folder. Defaults to "mcts_ttff".
        no_fail_penalty (float, optional): A seed that finds no failure costs env_steps * no_fail_penalty. Defaults to 2.

    Returns:
        optuna.Study: the finished study
    """
    results_path = Path("results/tuning") / study_name
    results_path.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=f"sqlite:///results/tuning/{study_name}.db",
        load_if_exists=True,
    )
    # seed by the number of trials so far, so a resumed study doesn't repeat the samples it already tried
    study.sampler = optuna.samplers.TPESampler(seed=len(study.trials))
    # only enqueue the baseline for a new study, otherwise resuming would rerun it
    if len(study.trials) == 0:
        study.enqueue_trial(BASELINE)

    model = load_trained_model(MODEL_NAME)
    study.optimize(
        partial(
            objective,
            model=model,
            results_path=results_path,
            iterations=iterations,
            seeds=seeds,
            no_fail_penalty=no_fail_penalty,
        ),
        n_trials=n_trials,
    )

    study.trials_dataframe().to_csv(results_path / "trials.csv", index=False)
    with open(results_path / "best_hyperparams.json", "w") as f:
        json.dump(
            {
                "hyperparams": study.best_params,
                "mean_steps_to_first_failure": study.best_value,
                "steps_per_seed": study.best_trial.user_attrs["steps_per_seed"],
                "iterations": iterations,
                "seeds": list(seeds),
            },
            f,
            indent=4,
        )

    baseline = study.trials[0]
    print(f"Baseline (trial 0): {baseline.value:.1f} steps to first failure")
    print(
        f"Best (trial {study.best_trial.number}): {study.best_value:.1f} steps to first failure"
    )
    print(f"Best hyperparams: {study.best_params}")
    return study


if __name__ == "__main__":
    main()
