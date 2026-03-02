import argparse

import gymnasium as gym
import mani_skill.envs  # noqa: F401 - needed to register envs
import numpy as np
import torch
from mani_skill.utils.wrappers.record import RecordEpisode

from stream_ac_transportbox_training import (
    StreamAC,
    ToNumpyWrapper,
    remap_sequential_checkpoint_keys,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen TransportBox policy under joint impairment."
    )
    parser.add_argument("--env_name", type=str, default="UnitreeG1TransportBox-v1")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_episode_steps", type=int, default=200)
    parser.add_argument("--sim_backend", type=str, default="cpu")
    parser.add_argument("--render", action="store_true")

    # Agent config (keep aligned with stream_ac_transportbox_training defaults).
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--optimizer", type=str, default="AdaptiveObGD")
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lamda", type=float, default=0.8)
    parser.add_argument("--kappa_policy", type=float, default=3.0)
    parser.add_argument("--kappa_value", type=float, default=2.0)
    parser.add_argument("--cbp", action="store_true")
    parser.add_argument("--layernorm", action="store_true")

    # Medium right-arm/right-hand impairment scenario.
    parser.add_argument("--apply_damage", action="store_true", default=True)
    parser.add_argument(
        "--no_damage",
        action="store_false",
        dest="apply_damage",
        help="Disable impairment and evaluate nominal policy.",
    )
    parser.add_argument("--shoulder_elbow_gain", type=float, default=0.1)
    parser.add_argument("--finger_gain", type=float, default=0.1)
    parser.add_argument("--stuck_joint_idx", type=int, default=None)
    parser.add_argument(
        "--stuck_joint_idxs",
        type=int,
        nargs="+",
        default=[24],
        help="One or more stuck joint indices, e.g. --stuck_joint_idxs 15 21 24",
    )

    return parser.parse_args()


def build_env(args):
    render_mode = "human" if args.render else None
    env = gym.make(
        args.env_name,
        num_envs=1,
        obs_mode="state",
        render_mode="rgb_array",
        max_episode_steps=args.max_episode_steps,
        reward_mode="normalized_dense",
        sim_backend=args.sim_backend,
    )
    env = ToNumpyWrapper(env)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    # env = RecordEpisode(env, output_dir=f"videos/", save_trajectory=False, max_steps_per_video=1000, video_fps=30)
    return env


def build_agent(env, args):
    agent = StreamAC(
        n_obs=env.observation_space.shape[0],
        n_actions=env.action_space.shape[0],
        hidden_size=args.hidden_size,
        lr=args.lr,
        gamma=args.gamma,
        lamda=args.lamda,
        kappa_policy=args.kappa_policy,
        kappa_value=args.kappa_value,
        cbp=args.cbp,
        layernorm=args.layernorm,
        optimizer=args.optimizer,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    state_dict, remapped = remap_sequential_checkpoint_keys(state_dict)
    if remapped:
        print("Remapped PPO sequential checkpoint keys to StreamAC MLP keys")
    agent.load_state_dict(state_dict, strict=not (args.cbp or args.layernorm))
    agent.eval()
    return agent


def apply_medium_damage(action, shoulder_elbow_gain, finger_gain, stuck_joint_idxs):
    # G1 upper-body action indices (right side):
    # shoulder/elbow: 2,4,6,8,10 ; fingers: 14,15,16,20,21,22,24
    shoulder_elbow_idx = [2, 4, 6, 8, 10]
    finger_idx = [14, 15, 16, 20, 21, 22, 24]
    action[..., shoulder_elbow_idx] *= shoulder_elbow_gain
    action[..., finger_idx] *= finger_gain
    for idx in stuck_joint_idxs:
        action[..., idx] = 0.0
    return action


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = build_env(args)

    if args.stuck_joint_idx is not None:
        # Backward-compatible single-index flag.
        args.stuck_joint_idxs = [args.stuck_joint_idx]

    # De-duplicate and keep only valid action indices for safety.
    action_dim = env.action_space.shape[0]
    args.stuck_joint_idxs = sorted(
        {idx for idx in args.stuck_joint_idxs if 0 <= idx < action_dim}
    )
    if not args.stuck_joint_idxs:
        raise ValueError("No valid stuck joint indices provided.")

    agent = build_agent(env, args)

    returns = []
    successes = []

    for ep in range(args.eval_episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        episode_return = 0.0

        while not done:
            with torch.no_grad():
                action = agent.sample_action(obs)

            action = np.array(action, copy=True)
            print(action.shape)
            if args.apply_damage:
                action = apply_medium_damage(
                    action,
                    shoulder_elbow_gain=args.shoulder_elbow_gain,
                    finger_gain=args.finger_gain,
                    stuck_joint_idxs=args.stuck_joint_idxs,
                )

            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(np.asarray(reward).squeeze())
            done = bool(np.asarray(terminated).squeeze() or np.asarray(truncated).squeeze())

            if args.render:
                env.render()

        success = info.get("success", 0)
        success = int(np.asarray(success).squeeze())
        returns.append(episode_return)
        successes.append(success)
        print(
            f"Episode {ep + 1}/{args.eval_episodes}: return={episode_return:.4f}, success={success}"
        )

    print("==== Summary ====")
    print(f"Mean return: {np.mean(returns):.4f}")
    print(f"Success rate: {np.mean(successes):.4f}")

    env.close()


if __name__ == "__main__":
    main()
