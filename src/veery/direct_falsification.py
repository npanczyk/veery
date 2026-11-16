
def direct_falsify(rollout, psi, system, timesteps, m):
    """Performs direct falsification. Not yet implemented.

    Args:
        rollout (func): Function that runs one rollout and yields a trajectory
        psi (func): Function which returns a bool depending on if a trajectory met a specification
        system (obj): Object containing the agent, environment, and sensor models
        timesteps (int): Number of timesteps in each rollout
        m (int): Number of rollouts

    Returns:
        NumPy array: array of bools showing which trajectories failed
    """
    raise NotImplementedError("direct_falsify is not yet implemented")
