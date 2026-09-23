"""Reusable command-line evaluator for checkpoints produced by this repository."""

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Normal


TASKS = {
    "quad": {"env": "AnymalC-Reach-v1", "damages": ("none", "goal_shift", "broken_leg", "slippery_floor", "stuck_joint", "slippery_floor_easy", "goal_shift_easy")},
    "manip": {"env": "PushCube-v1", "damages": ("none", "goal_shift", "slippery_cube", "slippery_table")},
    "humanoid": {"env": "UnitreeG1TransportBox-v1", "damages": ("none", "arm_hand_impairment")},
}


def parse_args(default_task=None):
    parser = argparse.ArgumentParser(description="Evaluate a saved streaming policy and report its success rate.")
    parser.add_argument("--task", choices=TASKS, default=default_task or "quad")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint from a training run.")
    parser.add_argument("--env-name", help="Override the task's default environment.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--layernorm", action="store_true")
    parser.add_argument("--cbp", action="store_true")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of using policy means.")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--damage-type", default="none", help="Optional damage; valid values depend on --task.")
    parser.add_argument("--goal-offset-y", type=float, help="Override the task-specific goal shift on the y-axis.")
    parser.add_argument("--friction", type=float, default=-1.8)
    parser.add_argument("--shoulder-elbow-gain", type=float, default=0.14)
    parser.add_argument("--finger-gain", type=float, default=0.07)
    parser.add_argument("--stuck-joint-idxs", type=int, nargs="+", default=[14, 15, 21, 24])
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.damage_type not in TASKS[args.task]["damages"]:
        parser.error(f"{args.damage_type!r} is not valid for {args.task}; choose from " + ", ".join(TASKS[args.task]["damages"]))
    if not Path(args.checkpoint).is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    return args


def task_components(task):
    # Importing the training module also registers this repository's custom envs.
    if task == "quad":
        from custom_envs.quadruped_reach import QuadrupedReachEnv  # noqa: F401
        from stream_ac_training_adapt_quad import StreamAC, ToNumpyWrapper
        env_class = None
    elif task == "manip":
        from custom_envs.push_cube import PushCubeEnv  # noqa: F401
        from stream_ac_training_adapt_manip import StreamAC, ToNumpyWrapper
        env_class = PushCubeEnv
    else:
        import mani_skill.envs  # noqa: F401 - registers the built-in humanoid env
        from stream_ac_training_adapt_humanoid import StreamAC, ToNumpyWrapper
        env_class = None
    return StreamAC, ToNumpyWrapper, env_class


def remap_sequential_checkpoint_keys(state_dict):
    """Map batch-PPO ``nn.Sequential`` keys to the shared MLP architecture."""
    prefix_map = {
        "actor_mean.0.": "actor_mean.fc1.", "actor_mean.2.": "actor_mean.fc2.",
        "actor_mean.4.": "actor_mean.fc3.", "actor_mean.6.": "actor_mean.fc4.",
        "critic.0.": "critic.fc1.", "critic.2.": "critic.fc2.",
        "critic.4.": "critic.fc3.", "critic.6.": "critic.value.",
    }
    remapped, changed = {}, False
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in prefix_map.items():
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix):]
                changed = True
                break
        remapped[new_key] = value
    return remapped, changed


def build_env(args, to_numpy, env_class=None):
    env_kwargs = dict(num_envs=1, obs_mode="state", max_episode_steps=args.max_episode_steps, reward_mode="normalized_dense", render_mode="human" if args.render else None)
    if args.task == "humanoid":
        env_kwargs["sim_backend"] = "physx_cpu"
    # ManiSkill already registers its own PushCube-v1. Instantiate our custom
    # class directly so goal-shift/friction hooks are guaranteed to be present.
    if env_class is not None and args.env_name is None:
        env_kwargs.pop("max_episode_steps")
        env = gym.wrappers.TimeLimit(env_class(**env_kwargs), args.max_episode_steps)
    else:
        env = gym.make(args.env_name or TASKS[args.task]["env"], **env_kwargs)
    env = to_numpy(env)
    env = gym.wrappers.FlattenObservation(env)
    return gym.wrappers.ClipAction(env)


def load_agent(env, args, stream_ac):
    agent = stream_ac(n_obs=env.observation_space.shape[0], n_actions=env.action_space.shape[0], hidden_size=args.hidden_size, cbp=args.cbp, layernorm=args.layernorm)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    state_dict, remapped = remap_sequential_checkpoint_keys(state_dict)
    agent.load_state_dict(state_dict, strict=not (args.cbp or args.layernorm))
    if remapped:
        print("Remapped sequential PPO checkpoint keys.")
    agent.eval()
    return agent


def apply_environment_damage(env, args):
    target = env.unwrapped
    if args.damage_type in {"goal_shift", "goal_shift_easy"}:
        if args.task == "manip":
            # This is the paper's PushCube goal shift, not the quadruped offset.
            offset = -0.15 if args.goal_offset_y is None else args.goal_offset_y
            target.set_goal_offset(torch.tensor([0.0, offset, 0.0], device=target.device))
        else:
            offset = 2.4 if args.damage_type == "goal_shift_easy" else (3.0 if args.goal_offset_y is None else args.goal_offset_y)
            target.set_goal_offset(0, offset)
    elif args.damage_type in {"slippery_floor", "slippery_floor_easy"}:
        friction = -1.7 if args.damage_type == "slippery_floor_easy" else args.friction
        target.change_friction(friction, friction)
    elif args.damage_type == "slippery_cube":
        target.set_cube_friction(args.friction, args.friction)
    elif args.damage_type == "slippery_table":
        target.set_table_friction(args.friction, args.friction)


def apply_action_damage(action, args):
    action = np.array(action, copy=True)
    if args.damage_type == "broken_leg":
        action[..., [2, 6, 10]] = 0.0
    elif args.damage_type == "stuck_joint":
        action[..., 10] = 0.0
    elif args.damage_type == "arm_hand_impairment":
        action[..., [2, 4, 6, 8, 10]] *= args.shoulder_elbow_gain
        action[..., [14, 15, 16, 20, 21, 22, 24]] *= args.finger_gain
        action[..., [i for i in args.stuck_joint_idxs if 0 <= i < action.shape[-1]]] = 0.0
    return action


def policy_action(agent, observation, stochastic):
    observation = torch.as_tensor(observation, dtype=torch.float32)
    with torch.no_grad():
        action = agent.actor_mean(observation)
        if stochastic:
            action = Normal(action, torch.exp(agent.actor_logstd.expand_as(action))).sample()
    return action.cpu().numpy()


def main(default_task=None):
    args = parse_args(default_task)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    stream_ac, to_numpy, env_class = task_components(args.task)
    env = build_env(args, to_numpy, env_class)
    try:
        agent = load_agent(env, args, stream_ac)
        apply_environment_damage(env, args)
        returns, successes = [], []
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + episode)
            episode_return, done = 0.0, False
            while not done:
                action = apply_action_damage(policy_action(agent, observation, args.stochastic), args)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += float(np.asarray(reward).squeeze())
                done = bool(np.asarray(terminated).squeeze() or np.asarray(truncated).squeeze())
                if args.render:
                    env.render()
            returns.append(episode_return)
            successes.append(int(np.asarray(info.get("success", 0)).squeeze()))
            print(f"Episode {episode + 1}/{args.episodes}: return={episode_return:.3f}, success={successes[-1]}")
        print(f"Mean return: {np.mean(returns):.3f}")
        print(f"Success rate: {np.mean(successes):.3%} ({sum(successes)}/{len(successes)})")
    finally:
        env.close()


if __name__ == "__main__":
    main()
