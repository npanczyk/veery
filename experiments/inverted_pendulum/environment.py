import numpy as np
from scipy.stats import norm
from helpers import clamp
import pandas as pd
from veery import accessories
import time
from pathlib import Path
from tqdm import tqdm


class PendulumController:
    """Proportional Controller for the inverted pendulum system"""

    def __init__(self):
        self.alpha = np.array(
            [[-15, -8]]
        )  # (1 x 2) gain matrix for proportional control

    def predict(self, observation, deterministic=True):
        """Prediction method to match SB3 interface.

        Args:
            observation (dict): with keys "theta" and "omega"
            deterministic (bool, optional): unused, just needed to match SB3 formatting for MCTS. Defaults to True.

        Returns:
            torque, None: action and placeholder for MCTS
        """
        observation_matrix = np.array([[observation["theta"]], [observation["omega"]]])
        torque = self.alpha @ observation_matrix
        return torque[0][0], None


class InvertedPendulum:
    def __init__(self, m, l, dt, episode_length, save_dir):
        self.m = m  # mass of the pendulum
        self.l = l  # length of the pendulum
        self.dt = dt  # time step in seconds
        self.episode_length = episode_length  # in seconds
        # check the path and make sure it's valid
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.runtime = episode_length * dt - dt
        self.g = 9.81  # m/s**2, gravitational constant
        self.w_max = 8.0  # maximum angular velocity
        self.a_max = 2.0  # maximum applicable torque
        # define a constant profile so that we can calculate robustness
        self.profile = lambda t: 0
        self.rng = np.random.default_rng()  # unseeded until reset(seed=...)

    def reset(self, seed=None):
        # reseed only when given a seed; otherwise continue the current random stream (gym convention)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.time = 0
        theta0 = self.rng.uniform(-np.pi / 16, np.pi / 16)
        w0 = self.rng.uniform(-1.0, 1.0)
        # np.float64 keeps .item()/.copy() working on observations (as when state was an array)
        self.state = {
            "theta": np.float64(theta0),
            "omega": np.float64(w0),
            "runtime": self.runtime,
        }
        # do not fuzz the initial conditions
        xo = {"theta": 0, "omega": 0}
        observation = self._get_observation(xo)
        self._log_history(observation=observation, init=True)
        self._last_observation = observation
        return observation

    def _get_observation(self, xo):
        # additive noise sensor
        # grab disturbance distribution from init and sample
        # s = np.array([[s[0]], [s[1]]])
        # return s + xo
        return {
            "theta": self.state["theta"] + xo["theta"],
            "omega": self.state["omega"] + xo["omega"],
        }

    def _log_history(self, observation, init=False):
        if init:
            self.history = [
                [
                    self.time,
                    observation["theta"].item(),  # observed angle
                    self.profile(self.time),  # desired angle
                    self.state["theta"],  # actual angle
                ]
            ]
        else:
            self.history.append(
                [
                    self.time,
                    observation["theta"].item(),  # observed angle
                    self.profile(self.time),  # desired angle
                    self.state["theta"],  # actual angle
                ]
            )
            return

    def step(self, action, disturbance):
        """Gets values for the next state.

        Args:
            disturbance (Disturbance): noise value for the step
        Returns:
            observation, None, terminated, truncated, None: matching gymnasium output format for compatibility with MCTS
        """
        # fuzz the action
        action = action + disturbance.xa["torque"]
        # grab the state disturbance
        xs = disturbance.xs
        theta, omega = self.state["theta"], self.state["omega"]
        dt, g, m, l = self.dt, self.g, self.m, self.l
        action = clamp(action, -1 * self.a_max, self.a_max)
        omega = (
            omega + ((3 * g / (2 * l)) * np.sin(theta) + (3 * action) / (m * l**2)) * dt
        )
        theta = theta + omega * dt
        omega = clamp(omega, -1 * self.w_max, self.w_max)
        # update time
        self.time += self.dt
        # update state
        self.state["theta"] = np.float64(theta + xs["theta"])
        self.state["omega"] = np.float64(omega + xs["omega"])
        # observe the state
        observation = self._get_observation(disturbance.xo)
        self._log_history(observation=observation, init=False)
        # set the last observation to the new observation
        self._last_observation = observation

        # make sure we haven't exceeded time
        truncated = False
        if self.time > self.runtime:
            truncated = True
        return observation, None, False, truncated, None

    def get_snapshot(self):
        return {
            "time": self.time,
            "runtime": self.state["runtime"],
            "state": dict(self.state),
            "observation": {k: v.copy() for k, v in self._last_observation.items()},
            "last_history_row": list(self.history[-1]),
        }

    def set_snapshot(self, snap):
        self.time = snap["time"]
        self.state = dict(snap["state"])
        self.history = [list(snap["last_history_row"])]
        self._last_observation = {k: v.copy() for k, v in snap["observation"].items()}
        # re-generate the observation dict here to prevent future corruptions from modifying an attribute
        return {k: v.copy() for k, v in snap["observation"].items()}

    def rollout_from_snapshot(self, model, snap, disturbance_distribution):
        return accessories.rollout_from_snapshot(
            self,
            model,
            snap,
            disturbance_distribution,
            get_actual=lambda env: env.state["theta"],
        )

    def state_robustness(self, threshold, percent=True):
        return accessories.robustness(
            self.state["theta"], self.profile(self.time), threshold, percent=percent
        )

    def render(self, save=True):
        history_arr = np.array(self.history)
        cols = ["Time", "Observed", "Desired", "Actual"]
        df = pd.DataFrame(history_arr, columns=cols)
        df["Diff"] = (df["Actual"] - df["Desired"]) * 100
        timestr = time.strftime("%Y%m%d-%H%M%S")
        save_path = self.save_dir / f"run_history_{timestr}.csv"
        if save:
            df.to_csv(save_path, index=False)
        return df

    def rollout(self, model, disturbance_distribution, save=False):
        self.reset()
        observation = self._last_observation
        done = False
        while not done:
            action, _ = model.predict(observation)
            x = disturbance_distribution.sample()
            observation, _, terminated, truncated, _ = self.step(
                action=action, disturbance=x
            )
            done = terminated or truncated
        if save:
            self.render(save=True)

        return self.render(save=False)

    def run_rollouts(self, model, disturbance_distribution, m, psi):
        """Runs m rollouts and saves the results.

        Args:
            disturbance_distribution (disturbance distribution)
            m (int): Number of rollouts to run.
            psi (function): Specification to check if rollout was success or failure. Should return an array of bools.
        """
        histories = []
        results = np.zeros(m, dtype=bool)
        for i in tqdm(range(m)):
            history = self.rollout(model, disturbance_distribution)
            histories.append(history)
            results[i] = psi(history, threshold=np.pi / 4)

        return histories, results
