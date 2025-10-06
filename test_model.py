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

env = gym.make("UnitreeGo2-Reach-v1", num_envs=1, obs_mode="state", render_mode="rgb_array", control_mode="pd_joint_delta_pos")
env = RecordEpisode(env, output_dir=f"videos/", save_trajectory=False, max_steps_per_video=200, video_fps=30)
max_episode_steps = gym_utils.find_max_episode_steps_value(env)
print(f"Max episode steps: {max_episode_steps}")

# Load the pre-trained model
agent = Agent(env)
agent.load_state_dict(torch.load("/home/tj/Documents/stream-rl-plasticity/runs/UnitreeGo2-Reach-v1__ppo_stream_pretrain_adam__1__1759763896/ckpt_1201.pt")["model_state_dict"])
# agent.load_state_dict(torch.load("weights/AdaptiveObGD_brokenleg_backleft/seed_0.pth"))
successes = []
# env.change_friction(-1.8, -1.8)
s, _ = env.reset(seed=42)
print(s.shape)
# env.set_goal_offset(0,3)
# env.set_goal_offset(0,2.4)
print(env.agent._control_freq)
print(env.agent.robot.get_qpos())
actions = []
joint_positions = []
height_over_time = []   

episode_count = 0
while episode_count < 1:
    s, info = env.reset()
    print("Robot current height:", env.agent.robot.pose.p[:,2])
    done = False
    while not done:
        s = torch.tensor(s, dtype=torch.float32)
        height_over_time.append(env.agent.robot.pose.p[:,2].item())
        print(s)
        a = agent.get_action(s, deterministic=True)
        a = a.detach().cpu().numpy() #* np.array([1,1,0,1,1,1,0,1,1,1,0,1])#* np.array([1,1,1,1,1,1,1,1,1,1,0,1]) # 2 joint stuck # np.array([1,1,0,1,1,1,0,1,1,1,0,1]) # Back Left Leg 
        actions.append(a)
        print(a)
        joint_positions.append(env.agent.robot.get_qpos().numpy().copy())
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
# Save actions to a file
np.save("unitree_go2_reach_actions.npy", np.array(actions))
np.save("unitree_go2_reach_joint_positions.npy", np.array(joint_positions))

# Plot height over time
import matplotlib.pyplot as plt
plt.plot(height_over_time)
plt.xlabel('Timestep')
plt.ylabel('Height')
plt.title('Robot Height Over Time')
plt.savefig('unitree_go2_reach_height_over_time.png')
