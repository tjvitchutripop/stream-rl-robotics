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
#from mani_skill.envs.tasks.quadruped.quadruped_joystick import UnitreeGo2JoystickEnv


env = gym.make("PushCube-v1", num_envs=1, obs_mode="state", render_mode="rgb_array", control_mode="pd_joint_delta_pos", max_episode_steps=200)
#env = RecordEpisode(env, output_dir=f"videos/", save_trajectory=False, max_steps_per_video=1000, video_fps=30)

action_space_low, action_space_high = torch.from_numpy(env.single_action_space.low), torch.from_numpy(env.single_action_space.high)
print("Action space low:", action_space_low)
print("Action space high:", action_space_high)
def clip_action(action: torch.Tensor):
    return torch.clamp(action.detach(), action_space_low, action_space_high)

starting_pos = np.array([0.0000, 0.0000, 0.0000, 0.0000, 0.9000, 0.9000, 0.9000, 0.9000, -1.8000, -1.8000, -1.8000, -1.8000])

# Load the pre-trained model
agent = Agent(env)
agent.load_state_dict(torch.load("/home/alyssa/stream-rl-plasticity/runs/PushCube-v1__ppo_stream_pretrain__1__1771944863/ckpt_1501.pt")["model_state_dict"])
#agent.load_state_dict(torch.load("weights/stream_ac_PushCube-v1_1772138537/seed_0_best_damage.pth")["model_state_dict"])
# agent.load_state_dict(
#     torch.load(
#         "weights/stream_ac_PushCube-v1_1772097741/seed_0_best_damage.pth",
#         map_location="cpu"
#     )
# )

# agent.load_state_dict(torch.load("/home/tj/Documents/stream-rl-plasticity/runs/UnitreeGo2-Joystick-v1__ppo_stream_pretrain_adam__1__1761100143/ckpt_276.pt")["model_state_dict"])
# agent.load_state_dict(torch.load("/home/tj/Documents/stream-rl-plasticity/runs/UnitreeGo2-Reach-v1__ppo_stream_pretrain_adam__1__1760984938/ckpt_1326.pt")["model_state_dict"])

# agent.load_state_dict(torch.load("weights/AdaptiveObGD_brokenleg_backleft/seed_0.pth"))
successes = []
# env.change_friction(-1.8, -1.8)
# env.change_object("025_mug")
s, _ = env.reset(seed=42)
# default_static, default_dynamic = env.unwrapped.get_table_friction()
# print(f"Captured original friction: {default_static}")
# default_goal = env.unwrapped.get_goal_position()
# print(f"Captured original goal: {default_goal}")
# env.set_goal_offset(0,3)
# env.set_goal_offset(0,2.4)
actions = []
joint_positions = []
height_over_time = []   
observations = []

episode_count = 0
goal_offset = torch.tensor([0.0, -0.15, 0.0] , device=env.unwrapped.device)
env.unwrapped.set_goal_offset(goal_offset)
while episode_count < 100:
    s, info = env.reset()
    # env.unwrapped.set_table_friction(-100.0, -100.0)
    # env.unwrapped.set_cube_friction(-100.0, -100.0)
    # env.unwrapped.set_goal_position(
    #     torch.tensor([0.26, 0.19, 0.0010], device=env.unwrapped.device)
    # )

    done = False
    while not done:
        # env.unwrapped.set_table_friction(-100.0, -100.0)
        # env.unwrapped.set_cube_friction(-100.0, -100.0)
        # env.unwrapped.set_goal_position(
        #     torch.tensor([0.26, 0.19, 0.0010], device=env.unwrapped.device)
        # )
        #env.unwrapped.set_goal_offset(goal_offset)

        observations.append(s)
        s = torch.tensor(s, dtype=torch.float32)
        a = agent.get_action(s, deterministic=False)
        a = clip_action(a)
        a = a.detach().cpu().numpy() #* np.array([1,1,0,1,1,1,0,1,1,1,0,1])#* np.array([1,1,1,1,1,1,1,1,1,1,0,1]) # 2 joint stuck # np.array([1,1,0,1,1,1,0,1,1,1,0,1]) # Back Left Leg 
        actions.append(a)
        joint_positions.append(env.agent.robot.qpos)
        s_prime, r, terminated, truncated, info = env.step(a)
        s = s_prime
        done = terminated or truncated
        env.render()
        if done:
            print("done")
            successes.append(int(info['success']))
            print(f"Episode {episode_count + 1} finished ")
            episode_count += 1
            s, _ = env.reset()

success_rate = np.mean(successes)
print(f"Success rate: {success_rate}")
