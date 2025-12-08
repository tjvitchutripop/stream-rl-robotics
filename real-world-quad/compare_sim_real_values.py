import gymnasium as gym
import mani_skill.envs
import torch.nn as nn

from ppo_stream_pretrain_obgd import Agent
# from stream_ac_training import StreamAC
import torch
import numpy as np
from mani_skill.utils.wrappers.record import RecordEpisode
# from stream_ac_training import ToNumpyWrapper
from mani_skill.utils import gym_utils

# load the actions from npy
actions = np.load("unitree_go2_reach_action_old.npy")
joint_positions = np.load("joint_state_history.npy")
sim_joint_positions = np.load("unitree_go2_reach_joint_positions_old.npy")

print("Loaded actions shape:", actions.shape)
print("Loaded joint positions shape:", joint_positions.shape)
print("Loaded simulated joint positions shape:", sim_joint_positions.shape)

mses = []
error_1s = []
error_2s = []

for i in range(len(joint_positions)):
    target_pos = actions[i] + np.array([ 0.0000,  0.0000,  0.0000,  0.0000,  0.9000,  0.9000,  0.9000,  0.9000, -1.8000, -1.8000, -1.8000, -1.8000])
    if i == len(joint_positions) - 1:
        break
    error_1 = joint_positions[i+1] - target_pos
    error_1s.append(np.abs(error_1).mean())
    error_2 = sim_joint_positions[i+1] - target_pos
    error_2s.append(np.abs(error_2).mean())
    compare_errors = np.abs(error_1) - np.abs(error_2)
    mse = np.mean(compare_errors)
    mses.append(mse)

    print(f"Step {i}, MSE between joint positions and target positions: {mse}, Error difference: {compare_errors}")
    # print(f"Target positions: {target_pos}, Actual positions: {joint_positions[i+1]}")


average_mse = np.mean(mses)
print(f"Average MSE over all steps: {average_mse}")
average_error_1 = np.mean(np.abs(error_1s), axis=0)
average_error_2 = np.mean(np.abs(error_2s), axis=0)
print(f"Average absolute error between joint positions and target positions: {average_error_1}")
print(f"Average absolute error between simulated joint positions and target positions: {average_error_2}")