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

test = np.load("unitree_go2_joystick_actions.npy")
print("Test load shape:", test.shape)
test2 = np.load("unitree_go2_joystick_joint_positions.npy")
print("Test2 load shape:", test2.shape)
breakpoint()

# load the actions from npy
sim_actions = np.load("unitree_go2_reach_actions_new.npy").squeeze()
joint_positions = np.load("joint_state_history_new.npy")
joint_velocities = np.load("joint_vel_history_new.npy")
sim_joint_positions = np.load("unitree_go2_reach_joint_positions_new.npy").squeeze()
sim_observations = np.load("sim_observations.npy").squeeze()

print("Loaded actions shape:", sim_actions.shape)
print("Loaded joint positions shape:", joint_positions.shape)
print("Loaded joint velocities shape:", joint_velocities.shape)
print("Loaded simulated joint positions shape:", sim_joint_positions.shape)
print("Loaded simulated observations shape:", sim_observations.shape)

env = gym.make("UnitreeGo2-Reach-v1", num_envs=1, obs_mode="state", render_mode="rgb_array", control_mode="pd_joint_delta_pos")
agent = Agent(env)
agent.load_state_dict(torch.load("/home/tj/Documents/stream-rl-plasticity/runs/UnitreeGo2-Reach-v1__ppo_stream_pretrain_adam__1__1759763896/ckpt_1226.pt")["model_state_dict"])

mses = []

for i in range(len(joint_positions)):
    obs = np.concatenate([joint_positions[i], joint_velocities[i], sim_observations[i][-4:]])
    obs = torch.tensor(obs, dtype=torch.float32)
    a = agent.get_action(obs, deterministic=True)
    if i == len(joint_positions) - 1:
        break
    compare_actions = (sim_actions[i] - a.detach().cpu().numpy())**2
    mse = np.mean(compare_actions)
    mses.append(mse)

    print(f"Step {i}, MSE between joint positions and target positions: {mse}, Error difference: {compare_actions}")
    # print(f"Target positions: {target_pos}, Actual positions: {joint_positions[i+1]}")


average_mse = np.mean(mses)
print(f"Average MSE over all steps: {average_mse}")