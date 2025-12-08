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
from mani_skill.envs.tasks.quadruped.quadruped_joystick import UnitreeGo2JoystickEnv


env = gym.make("AnymalC-Reach-v1", num_envs=1, obs_mode="state", render_mode="rgb_array", control_mode="pd_joint_delta_pos", max_episode_steps=200)
env = RecordEpisode(env, output_dir=f"videos/", save_trajectory=False, max_steps_per_video=1000, video_fps=30)

action_space_low, action_space_high = torch.from_numpy(env.single_action_space.low), torch.from_numpy(env.single_action_space.high)
print("Action space low:", action_space_low)
print("Action space high:", action_space_high)
def clip_action(action: torch.Tensor):
    return torch.clamp(action.detach(), action_space_low, action_space_high)

starting_pos = np.array([0.0000, 0.0000, 0.0000, 0.0000, 0.9000, 0.9000, 0.9000, 0.9000, -1.8000, -1.8000, -1.8000, -1.8000])

# Load the pre-trained model
agent = Agent(env)
# agent.load_state_dict(torch.load("/home/tj/Documents/stream-rl-plasticity/runs/PickCube-v1__ppo_stream_pretrain_adam__1__1762210724/final_ckpt.pt")["model_state_dict"])
agent.load_state_dict(torch.load("/home/tj/Documents/stream-rl-plasticity/pretrained-models/anymalc-reach/obgd_ppo_pretrain.pt")["model_state_dict"])
# agent.load_state_dict(torch.load("/home/tj/Documents/stream-rl-plasticity/runs/UnitreeGo2-Joystick-v1__ppo_stream_pretrain_adam__1__1761100143/ckpt_276.pt")["model_state_dict"])
# agent.load_state_dict(torch.load("/home/tj/Documents/stream-rl-plasticity/runs/UnitreeGo2-Reach-v1__ppo_stream_pretrain_adam__1__1760984938/ckpt_1326.pt")["model_state_dict"])
# print("Model architecture:", agent )
# agent.load_state_dict(torch.load("weights/AdaptiveObGD_brokenleg_backleft/seed_0.pth"))
successes = []
# env.change_friction(-1.8, -1.8)
# env.change_object("025_mug")
s, _ = env.reset(seed=42)
# env.set_goal_offset(0,3)
# env.set_goal_offset(0,2.4)

episode_count = 0
returns = []
episode_return = 0  
while episode_count < 10:
    s, info = env.reset()
    done = False
    while not done:
        s = torch.tensor(s, dtype=torch.float32)
        a = agent.get_action(s, deterministic=False)
        a = clip_action(a)
        a = a.detach().cpu().numpy() #* np.array([1,1,0,1,1,1,0,1,1,1,0,1])#* np.array([1,1,1,1,1,1,1,1,1,1,0,1]) # 2 joint stuck # np.array([1,1,0,1,1,1,0,1,1,1,0,1]) # Back Left Leg 
        s_prime, r, terminated, truncated, info = env.step(a)
        episode_return += r
        s = s_prime
        done = terminated or truncated
        env.render()
        if done:
            print("done")
            # successes.append(int(info['success']))
            print(f"Episode {episode_count + 1} finished ")
            episode_count += 1
            s, _ = env.reset()
            print(f"Return: {episode_return}")
            returns.append(episode_return)
            episode_return = 0
avg_return = np.mean(returns[-10:]) if len(returns) >=10 else np.mean(returns)
print(f"Average Return over last 10 episodes: {avg_return}")

