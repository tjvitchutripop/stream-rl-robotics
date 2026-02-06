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

env = gym.make("PickCube-v1", robot_uids="xarm7_gripper", num_envs=1, control_mode="pd_joint_delta_pos", obs_mode="state", render_mode="human")

successes = []
s, _ = env.reset(seed=42)

episode_count = 0
while episode_count < 100000:
    s, info = env.reset()
    done = False
    while not done:
        s = torch.tensor(s, dtype=torch.float32)
        # Get random action
        a = torch.zeros((1, env.action_space.shape[0]))
        s_prime, r, terminated, truncated, info = env.step(a)
        print(r)
        s = s_prime
        done = terminated or truncated
        env.render()
        if done:
            print(f"Episode {episode_count + 1} finished")
            episode_count += 1
            s, _ = env.reset()

