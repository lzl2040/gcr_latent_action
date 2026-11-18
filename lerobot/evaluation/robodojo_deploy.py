"""Simulator-side rollouts; imported without any policy/PyTorch dependencies."""


def _with_executed_action(observation, action, enabled):
    return {**observation, "_executed_action": action} if enabled else observation


def eval_one_episode(TASK_ENV, model_client):  # noqa: N803
    task_env = TASK_ENV
    use_history = model_client.call(func_name="reset") is True
    last_action = None
    while not task_env.is_episode_end():
        model_client.call(
            func_name="update_obs",
            obs=_with_executed_action(task_env.get_obs(), last_action, use_history),
        )
        actions = model_client.call(func_name="get_action")
        if not actions:
            raise ValueError("Policy returned an empty action chunk.")
        for step, action in enumerate(actions):
            task_env.take_action(action)
            last_action = action
            if task_env.is_episode_end():
                break
            # Preserve benchmark video sampling at every control step, including inside chunks.
            if step + 1 < len(actions):
                observation = task_env.get_obs()
                if use_history:
                    model_client.call(
                        func_name="update_obs", obs=_with_executed_action(observation, action, True)
                    )


def eval_one_episode_batch(TASK_ENV, model_client):  # noqa: N803
    task_env = TASK_ENV
    use_history = model_client.call(func_name="reset") is True
    last_actions = {}
    while not task_env.is_episode_end():
        indices = task_env.get_running_env_idx_list()
        observations = [
            _with_executed_action(obs, last_actions.get(index), use_history)
            for index, obs in zip(indices, task_env.get_obs_batch(indices), strict=True)
        ]
        model_client.call(func_name="update_obs_batch", obs=observations)
        actions = model_client.call(func_name="get_action_batch", obs=indices)
        if len(actions) != len(indices) or not actions or not actions[0]:
            raise ValueError("Policy returned an empty or incorrectly sized action batch.")
        horizon = len(actions[0])
        if any(len(chunk) != horizon for chunk in actions):
            raise ValueError("All parallel environments must receive the same chunk length.")
        for step in range(horizon):
            task_env.take_action_batch([chunk[step] for chunk in actions], indices)
            last_actions = {index: chunk[step] for index, chunk in zip(indices, actions, strict=True)}
            if task_env.is_episode_end():
                break
            running = set(task_env.get_running_env_idx_list())
            active = [index for index, env_idx in enumerate(indices) if env_idx in running]
            actions = [actions[index] for index in active]
            indices = [indices[index] for index in active]
            if step + 1 < horizon:
                observations = task_env.get_obs_batch(indices)
                if use_history and indices:
                    model_client.call(
                        func_name="update_obs_batch",
                        obs=[
                            _with_executed_action(obs, last_actions[index], True)
                            for index, obs in zip(indices, observations, strict=True)
                        ],
                    )
