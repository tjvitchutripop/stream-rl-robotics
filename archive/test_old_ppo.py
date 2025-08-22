import gymnasium as gym
import mani_skill.envs
from ppo import Agent
import torch
from mani_skill.utils.wrappers.record import RecordEpisode


env = gym.make("AnymalC-Reach-v1", num_envs=1, obs_mode="state", render_mode="rgb_array")
env = RecordEpisode(env, output_dir=f"videos/", save_trajectory=False, max_steps_per_video=200, video_fps=30)

# Load the pre-trained model
agent = Agent(env)
agent.load_state_dict(torch.load("/home/tj/Documents/ManiSkill/examples/baselines/ppo/runs/AnymalC-Reach-v1__ppo__1__1752777685/final_ckpt.pt"))

for episode in range(10):
    obs, info = env.reset()
    done = False
    while not done:
        action = agent.get_action(obs, deterministic=True)
        action = action.detach().cpu().numpy()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    print(f"Episode {episode + 1} finished")