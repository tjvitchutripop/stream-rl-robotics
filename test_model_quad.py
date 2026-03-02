import gymnasium as gym
import mani_skill.envs
import torch.nn as nn

from ppo_stream_pretrain_adam import Agent
from stream_ac_training import StreamAC
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
# agent.load_state_dict(torch.load("/home/tj/Documents/stream-rl-plasticity/pretrained-models/anymalc-reach/adam_ppo_pretrain_final.pt")["model_state_dict"])
agent.load_state_dict(torch.load("/home/tj/Documents/stream-rl-plasticity/select_models/goal-shift_adaptiveObGD/seed_1.pth"))
# agent = StreamAC(env, hidden_size=256, optimizer="Adam", lr=3e-4, gamma=0.99, cbp=False, layernorm=False)
successes = []
# env.change_friction(-1.8, -1.8)
# env.change_object("025_mug")
s, _ = env.reset(seed=42)
env.set_goal_offset(0,3.0)
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
        a = a.detach().cpu().numpy() 
        # a = a.detach().cpu().numpy() * np.array([1,1,0,1,1,1,0,1,1,1,0,1])
        s_prime, r, terminated, truncated, info = env.step(a)
        episode_return += r
        s = s_prime
        done = terminated or truncated
        env.render()
        if done:
            print("done")
            successes.append(int(info['success']))
            print(f"Episode {episode_count + 1} finished ")
            episode_count += 1
            s, _ = env.reset()
            print(f"Return: {episode_return}")
            returns.append(episode_return)
            episode_return = 0
avg_return = np.mean(returns[-10:]) if len(returns) >=10 else np.mean(returns)
print(f"Average Return over last 10 episodes: {avg_return}")
print(f"Success rate: {np.mean(successes)}")

