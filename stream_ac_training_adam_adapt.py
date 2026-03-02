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
from streaming_drl.optim import ObGD, AdaptiveObGD
from streaming_drl.sparse_init import sparse_init
from streaming_drl.normalization_wrappers import NormalizeObservation, ScaleReward
from streaming_drl.time_wrapper import AddTimeInfo
import wandb    
import time
import moviepy.editor as mp
import glob
import mani_skill.envs
from ppo_stream_pretrain import Agent
# from gym_envs import make_lift_env
from model import ActorMean, Critic
from interpretability import MLPWandBLogger

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

class StreamAC(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=128, lr=1.0, gamma=0.99, lamda=0.8, kappa_policy=3.0, kappa_value=2.0, cbp=False, layernorm=False, optimizer="AdaptiveObGD"):
        super(StreamAC, self).__init__()
        self.optimizer = optimizer
        self.gamma = gamma
        if cbp:
            self.actor_mean = ActorMeanCBP(n_obs, n_actions, hidden_size, replacement_rate=1e-5, maturity_threshold=1000)
            self.critic = CriticCBP(n_obs, hidden_size, replacement_rate=1e-5, maturity_threshold=1000)
        elif layernorm:
            self.actor_mean = ActorMeanLN(n_obs, n_actions, hidden_size)
            self.critic = CriticLN(n_obs, hidden_size)
        else:
            self.actor_mean = ActorMean(n_obs, n_actions, hidden_size)
            self.critic = Critic(n_obs, hidden_size)
        self.actor_logstd = nn.Parameter(torch.ones(1, np.prod(n_actions)) * -0.5)
        if self.optimizer == "AdaptiveObGD":
            self.optimizer_policy = AdaptiveObGD(list(self.actor_mean.parameters()) + [self.actor_logstd], lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_policy)
            self.optimizer_value = AdaptiveObGD(self.critic.parameters(), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_value)
        elif self.optimizer == "ObGD":
            self.optimizer_policy = ObGD(list(self.actor_mean.parameters()) + [self.actor_logstd], lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_policy)
            self.optimizer_value = ObGD(self.critic.parameters(), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_value)
        elif self.optimizer == "Adam":
            self.optimizer_policy = torch.optim.Adam(
                list(self.actor_mean.parameters()) + [self.actor_logstd], 
                lr=lr, eps=1e-4
            )
            self.optimizer_value = torch.optim.Adam(
                self.critic.parameters(), 
                lr=lr, eps=1e-4
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
        # ----- Convert to tensors -----
        done_mask = 0 if done else 1
        s = torch.as_tensor(s, dtype=torch.float32)
        a = torch.as_tensor(a)
        r = torch.as_tensor(r)
        s_prime = torch.as_tensor(s_prime, dtype=torch.float32)
        done_mask = torch.as_tensor(done_mask, dtype=torch.float32)

        # ----- Critic -----
        v_s = self.v(s)
        v_prime = self.v(s_prime)

        td_target = r + self.gamma * v_prime * done_mask
        delta = td_target - v_s                                       # <-- TD error

        # ---- Critic loss: 1/2 δ² (semi-gradient TD update) ----
        critic_loss = 0.5 * delta.pow(2)

        # ---- Actor ----
        mu, std = self.pi(s)
        dist = Normal(mu, std)

        log_prob = dist.log_prob(a).sum()
        entropy = dist.entropy().sum()

        # Advantage ≈ TD error
        actor_loss = -(log_prob * delta.detach())                    # <-- delta now used correctly!

        # Adaptive entropy regularization
        entropy_term = -entropy_coeff * entropy * torch.sign(delta).item()

        # Total actor loss:
        actor_total = actor_loss + entropy_term

        # ---- Backprop ----
        self.actor_mean.zero_grad()
        self.critic.zero_grad()

        critic_loss.backward()
        actor_total.backward()

        torch.nn.utils.clip_grad_norm_(list(self.actor_mean.parameters()) + [self.actor_logstd], 1.0)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)

        # ---- Step ----
        self.optimizer_policy.step()
        self.optimizer_value.step()

        # ---- Logging ----
        wandb.log({
            "train/log_prob_pi": log_prob.item(),
            "train/value": v_s.item(),
            "train/td_target": td_target.item(),
            "train/delta": delta.item(),
            "train/entropy": entropy.item(),
            "train/entropy_coeff": entropy_coeff,
            "train/critic_loss": critic_loss.item(),
            "train/actor_loss": actor_loss.item(),
        })


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
        save_video=False,
        do_damage=False,
        damage_type='stuck_joint',
        damage_start_step=0,
        damage_steps=100_000,
        cbp=False,
        layernorm=False,
        optimizer="AdaptiveObGD",
        checkpoint="",
        interpretability=False,
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

        self.do_damage = do_damage
        self.damage_type = damage_type
        self.damage_start_step = damage_start_step
        self.damage_steps = damage_steps
        self.damage_ongoing = False

        self.cbp = cbp
        self.layernorm = layernorm
        self.optimizer = optimizer
        self.checkpoint = checkpoint
        
        self.agent = None
        self.env = None
        self.log_file = None
        self.eval_log_file = None
        self.interpretability = interpretability
        self.returns = []
        self.term_time_steps = []

        self.model_name = "stream_ac"
        self.start_time = int(time.time())
        
    def create_logs(self):
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        log_file = os.path.join(log_dir, f"{self.env_name}-training_{self.damage_type}_{self.optimizer}_cbp={self.cbp}_seed_{self.seed}.txt")
        open(log_file, 'w').close()

        eval_log_file = os.path.join(log_dir, f"{self.env_name}-eval_{self.damage_type}_{self.optimizer}_cbp={self.cbp}_seed_{self.seed}.txt")
        open(eval_log_file, 'w').close()

        if self.wandb_log:
            wandb.init(
                entity="apollo-lab",
                project=f"stream-rl-robotics",
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
                    "cbp": self.cbp, 
                    "layernorm": self.layernorm,
                    "optimizer": self.optimizer,
                    "checkpoint": self.checkpoint,
                    "damage_start_step": self.damage_start_step,
                    "do_damage": self.do_damage,
                    "damage_type": self.damage_type,
                },
                name=f"{self.env_name}_{self.damage_type}_{self.optimizer}_cbp={self.cbp}_ln={self.layernorm}_seed_{self.seed}",
                save_code=True
            )

        self.log_file = log_file
        self.eval_log_file = eval_log_file
        
    def setup_environment(self):
        render_mode = "human" if self.render else None
        env = gym.make(self.env_name, num_envs=1, render_mode=render_mode, max_episode_steps=self.max_episode_steps, reward_mode="normalized_dense")
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
            kappa_value=self.kappa_value,
            cbp=self.cbp,
            layernorm=self.layernorm,
            optimizer=self.optimizer,
        )
        if self.wandb_log and self.interpretability:
            self.logger = MLPWandBLogger(agent, log_interval=1000, activation_fn='tanh')
        return agent
    
    def save_model_and_stats(self):
        # Save training data
        save_dir = f"results/stream_ac_{self.env_name}_{self.start_time}"
        os.makedirs(save_dir, exist_ok=True)  
        with open(os.path.join(save_dir, f"seed_{self.seed}.pkl"), "wb") as f:
            pickle.dump((self.returns, self.term_time_steps, self.env_name), f)

        # Save model weights
        save_dir = f"weights/stream_ac_{self.env_name}_{self.start_time}"
        os.makedirs(save_dir, exist_ok=True)  
        torch.save(self.agent.state_dict(), os.path.join(save_dir, f"seed_{self.seed}.pth"))

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
            if self.do_damage and self.damage_ongoing:
                if self.damage_type == 'broken_leg':
                    a = a * np.array([1,1,0,1,1,1,0,1,1,1,0,1]) # Back Left Leg
                elif self.damage_type == 'stuck_joint':
                    a = a * np.array([1,1,1,1,1,1,1,1,1,1,0,1]) # One Joint Stuck
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
        checkpoint = torch.load(self.checkpoint)
        if self.cbp is True or self.layernorm is True:
            self.agent.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            self.agent.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if self.debug:
            print(f"seed: {self.seed}", f"env: {self.env.spec.id}")

        self.returns = []
        self.term_time_steps = []

        if self.do_damage and self.damage_type == "goal_shift" and self.damage_start_step==0:
            self.env.set_goal_offset(0,3.0)
            self.damage_ongoing = True
            wandb.log({"goal_shift": 1})
        if self.do_damage and self.damage_type == "goal_shift_easy" and self.damage_start_step==0:
            self.env.set_goal_offset(0,2.4)
            self.damage_ongoing = True
            wandb.log({"goal_shift": 1})
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
            if self.do_damage:
                if t >= self.damage_start_step and (t - self.damage_start_step) < self.damage_steps:
                    if self.damage_ongoing == False and self.damage_type == "slippery_floor":
                        self.env.change_friction(-1.8, -1.8)
                    if self.damage_ongoing == False and self.damage_type == "slippery_floor_easy":
                        self.env.change_friction(-1.7, -1.7)
                    if self.damage_type == 'goal_shift' or self.damage_type == 'goal_shift_easy':
                        wandb.log({"goal_shift": 1})
                    else:
                        self.damage_ongoing = True
                    if self.damage_type == 'slippery_floor' or self.damage_type == 'slippery_floor_easy':
                        wandb.log({"slippery_floor": 1})
                    elif self.damage_type == 'broken_leg':
                        pred_a = a.copy()
                        a = a * np.array([1,1,0,1,1,1,0,1,1,1,0,1]) # Back Left Leg
                        wandb.log({"damaged_leg": 0})
                    elif self.damage_type == 'stuck_joint':
                        a = a * np.array([1,1,1,1,1,1,1,1,1,1,0,1]) # One Joint Stuck
                        wandb.log({"damaged_joint": 0})
                else:
                    if self.damage_ongoing == True and (self.damage_type == "slippery_floor" or self.damage_type == "slippery_floor_easy"):
                        self.env.change_friction(0.3, 0.3) 
                    self.damage_ongoing = False
                    if self.damage_type == 'goal_shift' or self.damage_type == 'goal_shift_easy':
                        wandb.log({"goal_shift": 0})
                    if self.damage_type == 'slippery_floor' or self.damage_type == 'slippery_floor_easy':
                        wandb.log({"slippery_floor": 0})
                    elif self.damage_type == 'broken_leg':
                        wandb.log({"damaged_leg": -1})  
                    elif self.damage_type == 'stuck_joint':
                        wandb.log({"damaged_joint": -1})

            s_prime, r, terminated, truncated, info = self.env.step(a)
            if self.wandb_log and self.interpretability:
                self.logger.log(t)
            if self.damage_ongoing and self.damage_type == 'broken_leg':
                self.agent.update_params(s, pred_a, r, s_prime, terminated or truncated, self.entropy_coeff, self.overshooting_info)
            else:
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
                if self.do_damage and (self.damage_type == "goal_shift" or self.damage_type == "goal_shift_easy"):
                    if t+1 >= self.damage_start_step and (t+1 - self.damage_start_step) < self.damage_steps:
                        if self.damage_ongoing == False and self.damage_type == "goal_shift":
                            self.env.set_goal_offset(0,3.0)
                            self.damage_ongoing = True
                        if self.damage_ongoing == False and self.damage_type == "goal_shift_easy":
                            self.env.set_goal_offset(0,2.4)
                            self.damage_ongoing = True
                    else:
                        if self.damage_ongoing == True and (self.damage_type == "goal_shift" or self.damage_type == "goal_shift_easy"):
                            self.env.set_goal_offset(0,0)
                            self.damage_ongoing = False
                s, _ = self.env.reset()
                episode_count += 1
        
        self.env.close()
        
        with open(self.eval_log_file, 'a') as f:
            f.write(f"Total training time: {time.time() - start_time:.2f} seconds\n")

        # Save model, stats and data
        self.save_model_and_stats()

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stream AC(λ)')
    parser.add_argument('--env_name', type=str, default='AnymalC-Reach-v1')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-8)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lamda', type=float, default=0.8)
    parser.add_argument('--total_steps', type=int, default=2_000_000)
    parser.add_argument('--entropy_coeff', type=float, default=0.0)
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
    parser.add_argument('--cbp', action='store_true', default=False)
    parser.add_argument('--layernorm', action='store_true', default=False)
    parser.add_argument('--optimizer', type=str, default="Adam")
    parser.add_argument('--checkpoint', type=str, default="pretrained-models/anymalc-reach/adam_ppo_pretrain_final.pt")
    parser.add_argument('--interpretability', action='store_true', default=False)
    parser.add_argument('--do_damage', action='store_true', default=True)
    parser.add_argument('--damage_start_step', type=int, default=500_000)
    parser.add_argument('--damage_steps', type=int, default=2_000_000, help='Steps between damage events')
    parser.add_argument('--damage_type', type=str, default='broken_leg',
                        choices=['broken_leg', 'stuck_joint', 'slippery_floor', 'slippery_floor_easy', 'goal_shift', 'goal_shift_easy'],
                        help='Type of damage to apply')
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
        save_video=args.save_video,
        do_damage=args.do_damage,
        damage_type=args.damage_type,   
        damage_start_step=args.damage_start_step,
        damage_steps=args.damage_steps,
        cbp=args.cbp,
        layernorm=args.layernorm,
        optimizer=args.optimizer,
        checkpoint=args.checkpoint
    )
    
    if args.mode == 'train':
        runner.train()
    else:
        runner.test()