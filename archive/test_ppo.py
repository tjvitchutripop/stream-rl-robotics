import gymnasium as gym
import mani_skill.envs

from ppo_stream_pretrain import Agent
from stream_ac_training import StreamAC
import torch
import numpy as np
from mani_skill.utils.wrappers.record import RecordEpisode
from stream_ac_training import ToNumpyWrapper
from streaming_drl.normalization_wrappers import NormalizeObservation, ScaleReward
from mani_skill.utils import gym_utils


env = gym.make("AnymalC-Reach-v1", num_envs=1, obs_mode="state", render_mode="rgb_array")
max_episode_steps = gym_utils.find_max_episode_steps_value(env)
print(f"Max episode steps: {max_episode_steps}")
breakpoint()
env = ToNumpyWrapper(env)
env = gym.wrappers.FlattenObservation(env)
env = gym.wrappers.RecordEpisodeStatistics(env)
env = gym.wrappers.ClipAction(env)
# env = ScaleReward(env, gamma=0.99)
# env = NormalizeObservation(env)

# Load the pre-trained model
# agent = Agent(env)
agent = StreamAC(n_obs=env.observation_space.shape[0], n_actions=env.action_space.shape[0], hidden_size=256, lr=1.0, gamma=0.99, lamda=0.8, kappa_policy=3.0, kappa_value=2.0)
agent.load_state_dict(torch.load("/home/tj/Documents/apollo-streaming-rl/runs/AnymalC-Reach-v1__ppo_stream_pretrain__1__1752786867/final_ckpt.pt"))

returns = []
successes = []
s, _ = env.reset(seed=42)
episode_count = 0
while episode_count < 10:
    s, info = env.reset()
    done = False
    while not done:
        # s = torch.tensor(s, dtype=torch.float32)
        a = agent.sample_action(s)
        # a = agent.get_action(s, deterministic=True)

        s_prime, r, terminated, truncated, info = env.step(a)
        s = s_prime
        done = terminated or truncated
        if done:
            print(info)
            successes.append(int(info['success']))
            returns.append(info['episode']['r'])
            print(f"Episode {episode_count + 1} finished with return: {info['episode']['r']}, success: {info['success']}")
            episode_count += 1
            s, _ = env.reset()
            # episode_return = 0

success_rate = np.mean(successes)
print(f"Average return: {np.mean(returns)}, Success rate: {success_rate}")