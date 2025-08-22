import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle, argparse
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
# import gymnasium_robotics
from gymnasium.wrappers import RecordVideo
import torch.nn.functional as F
from torch.distributions import Normal
from streaming_drl.optim import ObGD as Optimizer
from streaming_drl.sparse_init import sparse_init
from streaming_drl.normalization_wrappers import NormalizeObservation, ScaleReward
from streaming_drl.time_wrapper import AddTimeInfo
from trac_optimizer import start_trac
import wandb    
import time
import moviepy.editor as mp
import glob
import mani_skill.envs
from ppo_stream_pretrain import Agent
# from gym_envs import make_lift_env

class ToNumpyWrapper(gym.Wrapper):
  def reset(self, **kwargs):
    obs, info = self.env.reset(**kwargs)
    return self._to_numpy(obs), info
  def step(self, action):
    obs, reward, terminated, truncated, info = self.env.step(action)
    return self._to_numpy(obs), self._to_numpy(reward), self._to_numpy(terminated), self._to_numpy(truncated), self._to_numpy(info)
  def _to_numpy(self, data):
    if isinstance(data, torch.Tensor):
      return data.cpu().numpy()
    elif isinstance(data, dict):
      return {k: self._to_numpy(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
      return type(data)(self._to_numpy(v) for v in data)
    else:
      return data

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        sparse_init(m.weight, sparsity=0.9)
        m.bias.data.fill_(0.0)

# class Actor(nn.Module):
#     def __init__(self, n_obs=11, n_actions=3, hidden_size=128):
#         super(Actor, self).__init__()
#         self.fc_layer = nn.Linear(n_obs, hidden_size)
#         self.hidden_layer = nn.Linear(hidden_size, hidden_size)
#         self.linear_mu = nn.Linear(hidden_size, n_actions)
#         self.linear_std = nn.Linear(hidden_size, n_actions)
#         self.apply(initialize_weights)

#     def forward(self, x):
#         x = self.fc_layer(x)
#         x = F.layer_norm(x, x.size())
#         x = F.leaky_relu(x)
#         x = self.hidden_layer(x)
#         x = F.layer_norm(x, x.size())
#         x = F.leaky_relu(x)
#         mu = self.linear_mu(x)
#         pre_std = self.linear_std(x)
#         std = F.softplus(pre_std)
#         return mu, std

# class Critic(nn.Module):
#     def __init__(self, n_obs=11, hidden_size=128):
#         super(Critic, self).__init__()
#         self.fc_layer   = nn.Linear(n_obs, hidden_size)
#         self.hidden_layer  = nn.Linear(hidden_size, hidden_size)
#         self.linear_layer  = nn.Linear(hidden_size, 1)
#         self.apply(initialize_weights)

#     def forward(self, x):
#         x = self.fc_layer(x)
#         x = F.layer_norm(x, x.size())
#         x = F.leaky_relu(x)
#         x = self.hidden_layer(x)      
#         x = F.layer_norm(x, x.size())
#         x = F.leaky_relu(x)
#         return self.linear_layer(x)

class StreamAC(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=128, lr=1.0, gamma=0.99, lamda=0.8, kappa_policy=3.0, kappa_value=2.0):
        super(StreamAC, self).__init__()
        self.gamma = gamma
        # self.actor_mean = nn.Sequential(
        #     nn.Linear(n_obs, 256),
        #     nn.LayerNorm(256),
        #     nn.LeakyReLU(),
        #     nn.Linear(256, 256),
        #     nn.LayerNorm(256),
        #     nn.LeakyReLU(),
        #     nn.Linear(256, 256),
        #     nn.LayerNorm(256),
        #     nn.LeakyReLU(),
        #     nn.Linear(256, np.prod(n_actions)),
        # )
        # self.actor_logstd = nn.Parameter(torch.ones(1, np.prod(n_actions)) * -0.5)
        # self.critic = nn.Sequential(
        #     nn.Linear(n_obs, 256),
        #     nn.LayerNorm(256),
        #     nn.LeakyReLU(),
        #     nn.Linear(256, 256),
        #     nn.LayerNorm(256),
        #     nn.LeakyReLU(),
        #     nn.Linear(256, 256),
        #     nn.LayerNorm(256),
        #     nn.LeakyReLU(),
        #     nn.Linear(256, 1),
        # )
        self.actor_mean = nn.Sequential(
            nn.Linear(n_obs, 256),
            # nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, 256),
            # nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, 256),
            # nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, np.prod(n_actions)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, np.prod(n_actions)) * -0.5)
        self.critic = nn.Sequential(
            nn.Linear(n_obs, 256),
            # nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, 256),
            # nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, 256),
            # nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )
        # self.optimizer_policy = Optimizer(list(self.actor_mean.parameters()) + [self.actor_logstd], lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_policy)
        # self.optimizer_value = Optimizer(self.critic.parameters(), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_value)

        self.optimizer = start_trac(log_file='logs/trac.text', Base=torch.optim.Adam)(
            list(self.actor_mean.parameters()) + [self.actor_logstd] + list(self.critic.parameters()),
            lr=3e-4,
            eps=1e-5
        )

    def pi(self, x):
        mu = self.actor_mean(x)
        mu = mu.unsqueeze(0)
        log_std = self.actor_logstd.expand_as(mu)
        std = torch.exp(log_std)
        return mu, std

    def v(self, x):
        return self.critic(x)

    def sample_action(self, s):
        x = torch.from_numpy(s).float()
        mu, std = self.pi(x)
        dist = Normal(mu, std)
        return dist.sample().numpy()

    def update_params(self, s, a, r, s_prime, done, entropy_coeff, overshooting_info=False):
        done_mask = 0 if done else 1
        s, a, r, s_prime, done_mask = torch.tensor(np.array(s), dtype=torch.float), torch.tensor(np.array(a)), \
                                         torch.tensor(np.array(r)), torch.tensor(np.array(s_prime), dtype=torch.float), \
                                         torch.tensor(np.array(done_mask), dtype=torch.float)

        v_s, v_prime = self.v(s), self.v(s_prime)
        td_target = r + self.gamma * v_prime * done_mask
        delta = td_target - v_s

        mu, std = self.pi(s)
        dist = Normal(mu, std)

        log_prob_pi = -(dist.log_prob(a)).sum()
        value_output = -v_s
        entropy_pi = -entropy_coeff * dist.entropy().sum() * torch.sign(delta).item()
        # self.optimizer_value.zero_grad()
        # self.optimizer_policy.zero_grad()
        self.optimizer.zero_grad()
        value_output.backward()
        (log_prob_pi + entropy_pi).backward()
        # self.optimizer_policy.step(delta.item(), reset=done)
        # self.optimizer_value.step(delta.item(), reset=done)
        # nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
        self.optimizer.step()

        if overshooting_info:
            v_s, v_prime = self.v(s), self.v(s_prime)
            td_target = r + self.gamma * v_prime * done_mask
            delta_bar = td_target - v_s
            if torch.sign(delta_bar * delta).item() == -1:
                print("Overshooting Detected!")


class StreamACRunner:
    def __init__(
        self,
        env_name,
        seed=0,
        hidden_size=128,
        lr=1.0,
        gamma=0.99,
        lamda=0.8,
        entropy_coeff=0.01,
        kappa_policy=3.0,
        kappa_value=2.0,
        total_steps=100_000,
        eval_frequency=10_000,
        eval_episodes=50,
        debug=False,
        wandb_log=False,
        overshooting_info=False,
        render=False,
        max_episode_steps=200,
        save_video=False
    ):
        self.env_name = env_name
        self.seed = seed
        self.hidden_size = hidden_size
        self.lr = lr
        self.gamma = gamma
        self.lamda = lamda
        self.entropy_coeff = entropy_coeff
        self.kappa_policy = kappa_policy
        self.kappa_value = kappa_value
        self.total_steps = total_steps
        self.eval_frequency = eval_frequency
        self.eval_episodes = eval_episodes
        self.debug = debug
        self.wandb_log = wandb_log
        self.overshooting_info = overshooting_info
        self.render = render
        self.max_episode_steps = max_episode_steps
        self.save_video = save_video
        
        self.agent = None
        self.env = None
        self.log_file = None
        self.eval_log_file = None
        self.returns = []
        self.term_time_steps = []

        self.model_name = "stream_ac"
        
    def create_logs(self):
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        log_file = os.path.join(log_dir, f"{self.env_name}-training_log_seed_{self.seed}_hidden_size{self.hidden_size}_lr{self.lr}_gamma{self.gamma}_lamda{self.lamda}_entropy{self.entropy_coeff}.txt")
        open(log_file, 'w').close()

        eval_log_file = os.path.join(log_dir, f"{self.env_name}-eval_log_seed_{self.seed}_hidden_size{self.hidden_size}_lr{self.lr}_gamma{self.gamma}_lamda{self.lamda}_entropy{self.entropy_coeff}.txt")
        open(eval_log_file, 'w').close()

        if self.wandb_log:
            wandb.init(
                entity="apollo-lab",
                project=f"stream-ac-test",
                config={
                    "env_name": self.env_name,
                    "seed": self.seed,
                    "hidden_size": self.hidden_size,
                    "learning_rate": self.lr,
                    "gamma": self.gamma,
                    "lambda": self.lamda,
                    "total_steps": self.total_steps,
                    "entropy_coeff": self.entropy_coeff,
                    "kappa_policy": self.kappa_policy,
                    "kappa_value": self.kappa_value,
                    "eval_frequency": self.eval_frequency,
                    "eval_episodes": self.eval_episodes,
                },
                name=f"{self.env_name}_seed{self.seed}_hidden_size{self.hidden_size}_lr{self.lr}_gamma{self.gamma}_lamda{self.lamda}_entropy{self.entropy_coeff}"
            )

        self.log_file = log_file
        self.eval_log_file = eval_log_file
        
    def setup_environment(self):
        render_mode = "human" if self.render else None
        env = gym.make(self.env_name, num_envs=1, render_mode=render_mode, max_episode_steps=self.max_episode_steps)
        env = self.wrap_environment(env)
        return env
    
    def wrap_environment(self, env):
        """Base environment wrappers. Can be overridden by subclasses."""
        env = ToNumpyWrapper(env)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        # env = ScaleReward(env, gamma=self.gamma)
        # env = NormalizeObservation(env)
        # env = AddTimeInfo(env)
        return env
    
    def create_agent(self, env):
        agent = StreamAC(
            n_obs=env.observation_space.shape[0], 
            n_actions=env.action_space.shape[0], 
            hidden_size=self.hidden_size, 
            lr=self.lr, 
            gamma=self.gamma, 
            lamda=self.lamda, 
            kappa_policy=self.kappa_policy, 
            kappa_value=self.kappa_value
        )
        return agent
    
    def save_model_and_stats(self):
        # Save training data
        save_dir = f"results/data_stream_ac_{self.env.spec.id}_hidden_size{self.hidden_size}_lr{self.lr}_gamma{self.gamma}_lamda{self.lamda}_entropy_coeff{self.entropy_coeff}"
        os.makedirs(save_dir, exist_ok=True)  
        with open(os.path.join(save_dir, f"seed_{self.seed}.pkl"), "wb") as f:
            pickle.dump((self.returns, self.term_time_steps, self.env_name), f)

        # Save model weights
        save_dir = f"weights/stream_ac_{self.env.spec.id}_hidden_size{self.hidden_size}_lr{self.lr}_gamma{self.gamma}_lamda{self.lamda}_entropy_coeff{self.entropy_coeff}"
        os.makedirs(save_dir, exist_ok=True)  
        torch.save(self.agent.state_dict(), os.path.join(save_dir, f"seed_{self.seed}.pth"))
        
        # Save env stats
        reward_wrapper = self.env
        while not isinstance(reward_wrapper, ScaleReward) and hasattr(reward_wrapper, 'env'):
            reward_wrapper = reward_wrapper.env
            
        obs_wrapper = self.env
        while not isinstance(obs_wrapper, NormalizeObservation) and hasattr(obs_wrapper, 'env'):
            obs_wrapper = obs_wrapper.env

        reward_stats = reward_wrapper.reward_stats
        obs_stats = obs_wrapper.obs_stats
        with open(os.path.join(save_dir, f"stats_data_{self.seed}.pkl"), "wb") as f:
            pickle.dump((reward_stats, obs_stats), f)

        # Log final model to wandb
        if self.wandb_log:
            wandb.save(os.path.join(save_dir, f"seed_{self.seed}.pth"))
            wandb.finish()
        
        return save_dir
    
    def evaluate(self):
        torch.manual_seed(self.seed)

        returns = []
        successes = []
        s, _ = self.env.reset(seed=self.seed)
        episode_count = 0

        while episode_count < self.eval_episodes:
            a = self.agent.sample_action(s)
            s_prime, r, terminated, truncated, info = self.env.step(a)
            s = s_prime
            if terminated or truncated:
                episode_return = info['episode']['r']
                if isinstance(episode_return, (list, np.ndarray)):
                    episode_return = episode_return[0]
                returns.append(episode_return)

                is_success = info['success']
                if isinstance(is_success, (list, np.ndarray)):
                    is_success = is_success[0]
                successes.append(int(is_success))

                episode_count += 1
                s, _ = self.env.reset()

        success_rate = np.mean(successes)
        return returns, success_rate
    
    def train(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        self.create_logs()
        self.env = self.setup_environment()
        self.agent = self.create_agent(self.env)
        self.agent.load_state_dict(torch.load("/home/tj/Documents/apollo-streaming-rl/runs/AnymalC-Reach-v1__ppo_stream_pretrain_obgd__1__1755726847/ckpt_226.pt")) # prev /home/tj/Documents/apollo-streaming-rl/runs/AnymalC-Reach-v1__ppo_stream_pretrain__1__1752786867/final_ckpt.pt
        if self.debug:
            print(f"seed: {self.seed}", f"env: {self.env.spec.id}")

        self.returns = []
        self.term_time_steps = []
        
        s, _ = self.env.reset(seed=self.seed)
        episode_count = 0

        start_time = time.time()
        converged = False
        
        for t in range(1, self.total_steps + 1):
            # Run evaluation
            if t % self.eval_frequency == 0:
                eval_returns, success_rate = self.evaluate()
                mean_return = np.mean(eval_returns)

                if self.wandb_log:
                    wandb.log({
                        "eval/mean_return": mean_return,
                        "eval/success_rate": success_rate,
                        "eval/episode": t // self.eval_frequency,
                    })
                
                if success_rate > 0.90 and not converged:
                    converged = True
                    elapsed_time = time.time() - start_time
                    with open(self.eval_log_file, 'a') as f:
                        f.write(f"Model converged after {elapsed_time:.2f} seconds\n")
                
                with open(self.eval_log_file, 'a') as f:
                    f.write(f"Mean Eval Episodic Return: {mean_return}, Success Rate: {success_rate}, Eval Number: {t // self.eval_frequency}\n")

            a = self.agent.sample_action(s)
            s_prime, r, terminated, truncated, info = self.env.step(a)
            self.agent.update_params(s, a, r, s_prime, terminated or truncated, self.entropy_coeff, self.overshooting_info)
            s = s_prime

            if terminated or truncated:
                episode_return = info['episode']['r']
                if isinstance(episode_return, (list, np.ndarray)):
                    episode_return = episode_return[0]
                
                if self.wandb_log:
                    wandb.log({
                        "train/episode_return": episode_return,
                        "train/episode": episode_count,
                        "timestep": t
                    })
                
                if self.debug:
                    with open(self.log_file, 'a') as f:
                        f.write(f"Episodic Return: {episode_return}, Time Step {t}\n")

                self.returns.append(episode_return)
                self.term_time_steps.append(t)
                terminated, truncated = False, False
                s, _ = self.env.reset()
                episode_count += 1
        
        self.env.close()
        
        with open(self.eval_log_file, 'a') as f:
            f.write(f"Total training time: {time.time() - start_time:.2f} seconds\n")

        # Save model, stats and data
        self.save_model_and_stats()
    
    def test(self, num_episodes=10):
        torch.manual_seed(self.seed)
        render_mode = "human" if self.render else None
        self.env = self.setup_environment()
        self.agent = self.create_agent(self.env)

        sanitized_env_name = self.env_name.replace("/", "-").replace("\\", "-")
        video_subdir_name = sanitized_env_name + f"_{self.model_name}"
        video_main_dir = "videos"
        video_subdir = os.path.join(video_main_dir, video_subdir_name)
        os.makedirs(video_subdir, exist_ok=True)

        env = gym.make(self.env_name, num_envs=1, render_mode=render_mode, max_episode_steps=self.max_episode_steps)
        env = ToNumpyWrapper(env)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)

        # Load model weights
        save_dir = f"weights/stream_ac_{self.env_name}_hidden_size{self.hidden_size}_lr{self.lr}_gamma{self.gamma}_lamda{self.lamda}_entropy_coeff{self.entropy_coeff}"
        
        stats_path = os.path.join(save_dir, f"stats_data_{self.seed}.pkl")
        if not os.path.exists(stats_path):
            print(f"Warning: Stats file not found at {stats_path}. Normalization might not be accurate.")
            class DummyStats: pass
            reward_stats = DummyStats()
            reward_stats.mean = np.array([0.0])
            reward_stats.var = np.array([1.0])
            reward_stats.count = 1
            reward_stats.p = 0.99
            
            obs_stats = DummyStats()
            obs_shape = self.agent.policy_net.fc_layer.in_features if self.agent else 10
            obs_shape = self.agent.policy_net.fc_layer.in_features if self.agent else 10
            obs_stats.mean = np.zeros(obs_shape)
            obs_stats.var = np.ones(obs_shape)
            obs_stats.count = 1
            obs_stats.p = 0.99
        else:
            reward_stats, obs_stats = pickle.load(open(stats_path, "rb"))
        
        # env = ScaleReward(env, gamma=self.gamma)
        env.reward_stats.mean = reward_stats.mean
        env.reward_stats.var = reward_stats.var
        env.reward_stats.count = reward_stats.count
        if hasattr(reward_stats, 'p'):
            env.reward_stats.p = reward_stats.p

        # env = NormalizeObservation(env)
        env.obs_stats.mean = obs_stats.mean
        env.obs_stats.var = obs_stats.var
        env.obs_stats.count = obs_stats.count
        if hasattr(obs_stats, 'p'):
            env.obs_stats.p = obs_stats.p

        # env = AddTimeInfo(env)

        if self.agent is None:
            self.agent = self.create_agent(env)
            model_path = os.path.join(save_dir, f"seed_{self.seed}.pth")
            self.agent.load_state_dict(torch.load("/home/tj/Documents/apollo-streaming-rl/runs/AnymalC-Reach-v1__ppo_stream_pretrain__1__1752786867/final_ckpt.pt"))
        else:
            model_path = os.path.join(save_dir, f"seed_{self.seed}.pth")
            self.agent.load_state_dict(torch.load("/home/tj/Documents/apollo-streaming-rl/runs/AnymalC-Reach-v1__ppo_stream_pretrain__1__1752786867/final_ckpt.pt"))


        returns = []
        s, _ = env.reset(seed=self.seed)
        episode_count = 0
        gif_recorded_this_run = False

        while episode_count < num_episodes:
            if self.save_video and not gif_recorded_this_run:
                print(f"Recording first test episode for GIF in {video_subdir}...")
                record_env = gym.make(self.env_name, render_mode="rgb_array", max_episode_steps=self.max_episode_steps)
                record_env = gym.wrappers.FlattenObservation(record_env)
                record_env = gym.wrappers.ClipAction(record_env)

                # record_env = ScaleReward(record_env, gamma=self.gamma)
                record_env.reward_stats.mean = reward_stats.mean
                record_env.reward_stats.var = reward_stats.var
                record_env.reward_stats.count = reward_stats.count
                if hasattr(reward_stats, 'p'):
                    record_env.reward_stats.p = reward_stats.p

                # record_env = NormalizeObservation(record_env)
                record_env.obs_stats.mean = obs_stats.mean
                record_env.obs_stats.var = obs_stats.var
                record_env.obs_stats.count = obs_stats.count
                if hasattr(obs_stats, 'p'):
                    record_env.obs_stats.p = obs_stats.p
                
                # record_env = AddTimeInfo(record_env)

                temp_video_name_prefix = f"test_episode_{self.seed}_temp_video"
                video_recorder_env = RecordVideo(
                    record_env,
                    video_folder=video_subdir,
                    name_prefix=temp_video_name_prefix,
                    episode_trigger=lambda ep_id: ep_id == 0,
                    disable_logger=True
                )
                
                s_rec, _ = video_recorder_env.reset(seed=self.seed)
                done_rec = False
                while not done_rec:
                    a_rec = self.agent.sample_action(s_rec)
                    s_prime_rec, _, terminated_rec, truncated_rec, _ = video_recorder_env.step(a_rec)
                    s_rec = s_prime_rec
                    done_rec = terminated_rec or truncated_rec
                
                video_recorder_env.close()
                record_env.close()

                mp4_filename = f"{temp_video_name_prefix}-episode-0.mp4"
                mp4_path = os.path.join(video_subdir, mp4_filename)
                gif_filename = f"test_episode_{self.seed}.gif"
                gif_path = os.path.join(video_subdir, gif_filename)

                if os.path.exists(mp4_path):
                    try:
                        clip = mp.VideoFileClip(mp4_path)
                        clip.write_gif(gif_path, fps=15)
                        clip.close()
                        os.remove(mp4_path) # Clean up MP4
                        print(f"Saved test episode GIF to {gif_path}")
                    except Exception as e:
                        print(f"Error converting MP4 to GIF: {e}. MP4 kept at {mp4_path}")
                else:
                    print(f"Warning: MP4 video not found at {mp4_path}. Cannot create GIF.")
                gif_recorded_this_run = True

            a = self.agent.sample_action(s)
            s_prime, r, terminated, truncated, info = env.step(a)
            s = s_prime

            if terminated or truncated:
                episode_return = info['episode']['r']
                if isinstance(episode_return, (list, np.ndarray)):
                    episode_return = episode_return[0]
                print(f"Episode {episode_count + 1} finished with reward: {episode_return}")
                returns.append(episode_return)
                episode_count += 1
                s, _ = env.reset()
        
        env.close()
        print(f"\nAverage return over {num_episodes} episodes: {np.mean(returns):.2f}")
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stream AC(λ)')
    parser.add_argument('--env_name', type=str, default='AnymalC-Reach-v1')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lamda', type=float, default=0.8)
    parser.add_argument('--total_steps', type=int, default=1_600_000)
    parser.add_argument('--entropy_coeff', type=float, default=0.01)
    parser.add_argument('--kappa_policy', type=float, default=3.0)
    parser.add_argument('--kappa_value', type=float, default=2.0)
    parser.add_argument('--eval_frequency', type=int, default=10_000)
    parser.add_argument('--eval_episodes', type=int, default=50)
    parser.add_argument('--debug', action='store_true', default=True)
    parser.add_argument('--wandb_log', action='store_true', help='Enable logging to Weights & Biases', default=True)
    parser.add_argument('--overshooting_info', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train')
    parser.add_argument('--save_video', action='store_true', help='Enable video recording during testing', default=False)
    # Add Ant-specific arguments
    parser.add_argument('--do_damage', action='store_true', default=True)
    parser.add_argument('--damage_start_step', type=int, default=500_000)
    parser.add_argument('--damage_steps', type=int, default=100_000, help='Steps between damage events (Ant-v5 only)')
    parser.add_argument('--damage_type', type=str, default='broken_leg',
                        choices=['broken_leg', 'weak_joint', 'stuck_joint', 'noisy_joint'],
                        help='Type of damage to apply (Ant-v5 only)')
    args = parser.parse_args()

    runner = StreamACRunner(
        env_name=args.env_name,
        seed=args.seed,
        hidden_size=args.hidden_size,
        lr=args.lr,
        gamma=args.gamma,
        lamda=args.lamda,
        total_steps=args.total_steps,
        entropy_coeff=args.entropy_coeff,
        kappa_policy=args.kappa_policy,
        kappa_value=args.kappa_value,
        eval_frequency=args.eval_frequency,
        eval_episodes=args.eval_episodes,
        debug=args.debug,
        wandb_log=args.wandb_log,
        overshooting_info=args.overshooting_info,
        render=args.render,
        save_video=args.save_video
    )
    
    if args.mode == 'train':
        runner.train()
    else:
        runner.test()
