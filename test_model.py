import gymnasium as gym
import mani_skill.envs
import torch.nn as nn

from ppo_stream_pretrain_obgd import Agent
# from stream_ac_training import StreamAC
import torch
import numpy as np
from mani_skill.utils.wrappers.record import RecordEpisode
# from stream_ac_training import ToNumpyWrapper
from streaming_drl.normalization_wrappers import NormalizeObservation, ScaleReward
from mani_skill.utils import gym_utils

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        sparse_init(m.weight, sparsity=0.9)
        m.bias.data.fill_(0.0)


env = gym.make("AnymalC-Reach-v1", num_envs=1, obs_mode="state", render_mode="rgb_array")
# env = RecordEpisode(env, output_dir=f"videos/", save_trajectory=False, max_steps_per_video=200, video_fps=30)
max_episode_steps = gym_utils.find_max_episode_steps_value(env)
print(f"Max episode steps: {max_episode_steps}")
# env = ToNumpyWrapper(env)
# env = gym.wrappers.FlattenObservation(env)
# env = gym.wrappers.RecordEpisodeStatistics(env)
# env = gym.wrappers.ClipAction(env)
# env = ScaleReward(env, gamma=0.99)
# env = NormalizeObservation(env)

# Load the pre-trained model
agent = Agent(env)
# agent = StreamAC(n_obs=env.observation_space.shape[0], n_actions=env.action_space.shape[0], hidden_size=256, lr=1.0, gamma=0.99, lamda=0.8, kappa_policy=3.0, kappa_value=2.0)
agent.load_state_dict(torch.load("obgd_ppo_pretrain.pt")["model_state_dict"])
# agent.load_state_dict(torch.load("weights/AdaptiveObGD_brokenleg_backleft/seed_0.pth"))
successes = []
# env.change_friction(-1.8, -1.8)
s, _ = env.reset(seed=42)
env.set_goal_offset(0,2.4)
episode_count = 0
while episode_count < 50:
    s, info = env.reset()
    done = False
    while not done:
        s = torch.tensor(s, dtype=torch.float32)
        a = agent.get_action(s, deterministic=True)
        a = a.detach().cpu().numpy() #* np.array([1,1,1,1,1,1,1,1,1,1,0,1]) # 2 joint stuck # np.array([1,1,0,1,1,1,0,1,1,1,0,1]) # Back Left Leg 
        # a = agent.sample_action(s)

        s_prime, r, terminated, truncated, info = env.step(a)
        s = s_prime
        done = terminated or truncated
        env.render()
        if done:
            successes.append(int(info['success']))
            print(f"Episode {episode_count + 1} finished with success: {info['success']}")
            episode_count += 1
            s, _ = env.reset()
            # episode_return = 0

success_rate = np.mean(successes)
print(f"Success rate: {success_rate}")