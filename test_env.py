import gymnasium as gym
import mani_skill.envs
import torch.nn as nn
from mani_skill.envs.tasks.quadruped.quadruped_joystick import UnitreeGo2JoystickEnv

from ppo_stream_pretrain_obgd import Agent
# from stream_ac_training import StreamAC
import torch
import numpy as np
from mani_skill.utils.wrappers.record import RecordEpisode
# from stream_ac_training import ToNumpyWrapper
from mani_skill.utils import gym_utils

env = gym.make("PickCubeXArm7Gripper-v1", num_envs=1, control_mode="pd_joint_delta_pos", obs_mode="state", render_mode="rgb_array")
env = RecordEpisode(env, output_dir=f"videos/test_env/", save_trajectory=False, max_steps_per_video=1000, video_fps=30)

successes = []
s, _ = env.reset(seed=42)

episode_count = 0
while episode_count < 1:
    s, info = env.reset()
    done = False
    while not done:
        s = torch.tensor(s, dtype=torch.float32)
        # Get random action
        # a = env.action_space.sample()
        a = np.zeros(env.action_space.shape)    
        s_prime, r, terminated, truncated, info = env.step(a)
        # print(s_prime)
        s = s_prime
        done = terminated or truncated
        env.render()
        if done:
            print(f"Episode {episode_count + 1} finished")
            episode_count += 1
            s, _ = env.reset()

