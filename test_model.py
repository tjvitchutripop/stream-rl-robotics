import gymnasium as gym
import mani_skill.envs
import torch.nn as nn

from ppo_stream_pretrain import Agent
# from stream_ac_training import StreamAC
import torch
import numpy as np
from mani_skill.utils.wrappers.record import RecordEpisode
# from stream_ac_training import ToNumpyWrapper
from mani_skill.utils import gym_utils
from mani_skill.envs.tasks.quadruped.quadruped_joystick import UnitreeGo2JoystickEnv

env = gym.make("PickCubeXArm7Gripper-v1", num_envs=1, obs_mode="state", render_mode="rgb_array", control_mode="pd_joint_delta_pos", max_episode_steps=200, sim_backend="physx_cuda")
env = RecordEpisode(env, output_dir=f"videos/", save_trajectory=False, max_steps_per_video=1000, video_fps=30)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

action_space_low, action_space_high = torch.from_numpy(env.single_action_space.low).to(device), torch.from_numpy(env.single_action_space.high).to(device)
print("Action space low:", action_space_low)
print("Action space high:", action_space_high)
def clip_action(action: torch.Tensor):
    return torch.clamp(action.detach(), action_space_low, action_space_high)

# Load the pre-trained model
agent = Agent(env).to(device)
agent.load_state_dict(torch.load("/home/tj/Documents/stream-rl-plasticity/runs/PickCubeXArm7Gripper-v1__ppo_stream_pretrain__1__1770747661/final_ckpt.pt")["model_state_dict"])

# agent.load_state_dict(torch.load("/home/tj/Documents/stream-rl-plasticity/runs/UnitreeGo2-Joystick-v1__ppo_stream_pretrain_adam__1__1761100143/ckpt_276.pt")["model_state_dict"])
# agent.load_state_dict(torch.load("/home/tj/Documents/stream-rl-plasticity/runs/UnitreeGo2-Reach-v1__ppo_stream_pretrain_adam__1__1760984938/ckpt_1326.pt")["model_state_dict"])

# agent.load_state_dict(torch.load("weights/AdaptiveObGD_brokenleg_backleft/seed_0.pth"))
successes = []
# env.change_friction(-1.8, -1.8)
# env.change_object("025_mug")
s, _ = env.reset(seed=42)
# env.set_goal_offset(0,3)
# env.set_goal_offset(0,2.4)
actions = []
joint_positions = []
height_over_time = []   
observations = []

episode_count = 0
while episode_count < 1:
    s, info = env.reset()
    done = False
    while not done:
        joint_positions.append(env.agent.get_state()["robot_qpos"].cpu().numpy())
        observations.append(s)
        s = torch.tensor(s, dtype=torch.float32).to(device)
        a = agent.get_action(s, deterministic=False)
        a = clip_action(a)
        a = a.detach().cpu().numpy() 
        actions.append(a)
        s_prime, r, terminated, truncated, info = env.step(a)
        s = s_prime
        done = terminated or truncated
        env.render()
        if terminated or truncated:
            if "success" in info:
                successes.append(info["success"].item())
            elif "is_success" in info:
                successes.append(info["is_success"].item())
            else:
                successes.append(0)
            print(f"Episode {episode_count + 1} finished. Success: {successes[-1]}")
            episode_count += 1
            s, _ = env.reset()
            # episode_return = 0

success_rate = np.mean(successes)
print(f"Success rate: {success_rate}")
# Save actions to a file
np.save("pick_cube_actions.npy", np.array(actions))
np.save("pick_cube_joint_positions.npy", np.array(joint_positions))
# np.save("sim_observations.npy", np.array(observations))

# Plot height over time
# import matplotlib.pyplot as plt
# plt.plot(height_over_time)
# plt.xlabel('Timestep')
# plt.ylabel('Height')
# plt.title('Robot Height Over Time')
# plt.savefig('unitree_go2_reach_height_over_time.png')
