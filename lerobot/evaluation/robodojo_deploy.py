"""Simulator-side rollouts; imported without any policy/PyTorch dependencies."""


def eval_one_episode(TASK_ENV, model_client):  # noqa: N803
    task_env = TASK_ENV
    model_client.call(func_name="reset")
    while not task_env.is_episode_end():
        model_client.call(func_name="update_obs", obs=task_env.get_obs())
        actions = model_client.call(func_name="get_action")
        if not actions:
            raise ValueError("Policy returned an empty action chunk.")
        for step, action in enumerate(actions):
            task_env.take_action(action)
            if task_env.is_episode_end():
                break
            # Preserve benchmark video sampling at every control step, including inside chunks.
            if step + 1 < len(actions):
                task_env.get_obs()


def eval_one_episode_batch(TASK_ENV, model_client):  # noqa: N803
    task_env = TASK_ENV
    model_client.call(func_name="reset")
    while not task_env.is_episode_end():
        indices = task_env.get_running_env_idx_list()
        model_client.call(func_name="update_obs_batch", obs=task_env.get_obs_batch(indices))
        actions = model_client.call(func_name="get_action_batch", obs=indices)
        if len(actions) != len(indices) or not actions or not actions[0]:
            raise ValueError("Policy returned an empty or incorrectly sized action batch.")
        horizon = len(actions[0])
        if any(len(chunk) != horizon for chunk in actions):
            raise ValueError("All parallel environments must receive the same chunk length.")
        for step in range(horizon):
            task_env.take_action_batch([chunk[step] for chunk in actions], indices)
            if task_env.is_episode_end():
                break
            running = set(task_env.get_running_env_idx_list())
            active = [index for index, env_idx in enumerate(indices) if env_idx in running]
            actions = [actions[index] for index in active]
            indices = [indices[index] for index in active]
            if step + 1 < horizon:
                task_env.get_obs_batch(indices)
