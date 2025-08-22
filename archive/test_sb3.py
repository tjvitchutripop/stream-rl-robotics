import torch
from stable_baselines3 import SAC
import gymnasium as gym
import numpy as np

def load_model_from_zip(model, zip_path):
    model.load_state_dict(torch.load(zip_path))
    return model


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
            a, _ = agent.predict(s)
            s, reward, terminated, truncated, info = env.step(a)
            episode_reward += reward
            done = terminated or truncated
            if done:
                episode_rewards.append(episode_reward)
        episode_count += 1

    mean_reward = np.mean(episode_rewards)

    return mean_reward

policy = SAC.load("/home/tj/Documents/apollo-streaming-rl/weights/sac_Ant-v4_seed0/best/best_model.zip", device="cpu")
env = gym.make("Ant-v4")
mean_returns = evaluate(env, policy)
print("Average returns:", mean_returns)
