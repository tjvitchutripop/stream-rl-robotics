import torch
import numpy as np
from cleanrl.cleanrl.sac_continuous_action import make_env
# from stable_baselines3.common.env_util import make_vec_env
from streaming_drl.normalization_wrappers import NormalizeObservation

# from sac_pretraining import Actor
from sac_baseline import Actor
import gymnasium as gym


def evaluate(env, agent):
    torch.manual_seed(42)
    episode_count = 0

    while episode_count < 10:
        s, _ = env.reset(seed=42)
        episode_rewards = []
        episode_reward = 0.0
        done = False
        while not done:
            s = torch.tensor(s, dtype=torch.float32)
            a, _, _ = agent.get_action(s)
            a = a.detach().cpu().numpy()
            s, reward, terminated, truncated, info = env.step(a)
            episode_reward += reward
            done = terminated or truncated
            if done:
                episode_rewards.append(episode_reward)
        episode_count += 1

    mean_reward = np.mean(episode_rewards)

    return mean_reward

if __name__ == "__main__":
    env = gym.make("Ant-v4")
    # env = gym.wrappers.FlattenObservation(env)
    # env = gym.wrappers.ClipAction(env)
    env = NormalizeObservation(env)
    agent = Actor(env)
    # Load actor of trained SAC agent
    agent.load_state_dict(torch.load("agent-2.pt")["actor_state_dict"])
    mean_returns = evaluate(env, agent)
    print("Average returns:", mean_returns)