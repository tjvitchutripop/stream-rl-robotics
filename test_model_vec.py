import gymnasium as gym
import mani_skill.envs
import torch.nn as nn
from ppo_stream_pretrain import Agent
import torch
import numpy as np
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.utils import gym_utils

# Use the same backend as training: physx_cuda
# Use the same max_episode_steps as training: 50 (standard for PickCube-v1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("PickCubeXArm7Gripper-v1", num_envs=1, obs_mode="state", render_mode="rgb_array", control_mode="pd_joint_delta_pos", max_episode_steps=50, sim_backend="physx_cuda")
env = RecordEpisode(env, output_dir=f"videos/", save_trajectory=False, max_steps_per_video=1000, video_fps=30)

action_space_low, action_space_high = torch.from_numpy(env.single_action_space.low).to(device), torch.from_numpy(env.single_action_space.high).to(device)
print("Action space low:", action_space_low)
print("Action space high:", action_space_high)

def clip_action(action: torch.Tensor):
    return torch.clamp(action.detach(), action_space_low, action_space_high)

# starting_pos = np.array([0.0000, 0.0000, 0.0000, 0.0000, 0.9000, 0.9000, 0.9000, 0.9000, -1.8000, -1.8000, -1.8000, -1.8000])

# Load the pre-trained model
agent = Agent(env).to(device)
checkpoint = torch.load("/home/tj/Documents/stream-rl-plasticity/runs/PickCubeXArm7Gripper-v1__ppo_stream_pretrain__1__1770747661/final_ckpt.pt")
agent.load_state_dict(checkpoint["model_state_dict"])

successes_at_end = []
successes_once = []

s, _ = env.reset(seed=42)
actions = []
joint_positions = []
height_over_time = []   
observations = []

episode_count = 0
MAX_EPISODES = 10

while episode_count < MAX_EPISODES:
    s, info = env.reset()
    done = False
    has_succeeded_once = False
    
    while not done:
        # s is likely a numpy array or torch tensor depending on backend
        if isinstance(s, np.ndarray):
            s = torch.from_numpy(s).float().to(device)
        elif isinstance(s, torch.Tensor):
            s = s.float().to(device)
            
        observations.append(s.cpu().numpy()) # Store as numpy for compatibility if needed
        
        # Add batch dimension if needed (obs_mode="state" usually returns (obs_dim,) for single env)
        if s.ndim == 1:
            s = s.unsqueeze(0)

        with torch.no_grad():
            # Evaluation typically uses deterministic=True
            a = agent.get_action(s, deterministic=True)
            a = clip_action(a)
        
        # If simulation is on GPU (physx_cuda), keep action as tensor on device
        # If simulation is on CPU, convert to numpy
        if env.unwrapped.sim_config.sim_backend == "physx_cpu":
            action_input = a.detach().cpu().numpy().flatten()
        else:
            action_input = a # Keep as tensor, ManiSkill handles it
            
        s_prime, r, terminated, truncated, info = env.step(action_input)
        
        # Handle observation for next step
        s = s_prime
        
        done = terminated or truncated
        # env.render() # Render can slow down loop, uncomment if needed

        
        # Track success_once
        if info.get("success", False):
            has_succeeded_once = True
            
        if done:
            # Check success at end
            success_at_end = info.get("success", False)
            successes_at_end.append(1.0 if success_at_end else 0.0)
            successes_once.append(1.0 if has_succeeded_once else 0.0)
            
            print(f"Episode {episode_count + 1} finished. Success (End): {success_at_end}, Success (Once): {has_succeeded_once}")
            episode_count += 1

print(f"Average Success Rate (At End): {np.mean(successes_at_end) * 100:.2f}%")
print(f"Average Success Rate (Once): {np.mean(successes_once) * 100:.2f}%")

# Save actions to a file
# np.save("unitree_go2_joystick_actions.npy", np.array(actions))
# np.save("unitree_go2_joystick_joint_positions.npy", np.array(joint_positions))
# np.save("sim_observations.npy", np.array(observations))

