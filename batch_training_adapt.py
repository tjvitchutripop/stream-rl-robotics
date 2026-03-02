import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle, argparse
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
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
from model import ActorMean, Critic, ActorMeanLN, CriticLN, ActorMeanCBP, CriticCBP

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
    def __init__(self, n_obs=11, n_actions=3, hidden_size=128, lr=1.0, gamma=0.99, gae_lambda=0.95, kappa_policy=3.0, kappa_value=2.0, cbp=False, layernorm=False, optimizer="AdaptiveObGD",
                 update_epochs=10, num_minibatches=32, clip_coef=0.2, norm_adv=True, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, target_kl=None):
        super(StreamAC, self).__init__()
        self.optimizer_name = optimizer
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # PPO specific params
        self.update_epochs = update_epochs
        self.num_minibatches = num_minibatches
        self.clip_coef = clip_coef
        self.norm_adv = norm_adv
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

        if cbp:
            self.critic = CriticCBP(n_obs, hidden_size, replacement_rate=1e-5, maturity_threshold=1000)
            self.actor_mean = ActorMeanCBP(n_obs, n_actions, hidden_size, replacement_rate=1e-5, maturity_threshold=1000)
        elif layernorm:
            self.critic = CriticLN(n_obs, hidden_size)
            self.actor_mean = ActorMeanLN(n_obs, n_actions, hidden_size)
        else:
            self.critic = Critic(n_obs, hidden_size)
            self.actor_mean = ActorMean(n_obs, n_actions, hidden_size)
            
        self.actor_logstd = nn.Parameter(torch.ones(1, np.prod(n_actions)) * -0.5)
        
        if self.optimizer_name == "Adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-5)
        elif self.optimizer_name == "AdaptiveObGD":
            self.optimizer_policy = AdaptiveObGD(list(self.actor_mean.parameters()) + [self.actor_logstd], lr=lr, gamma=gamma, lamda=gae_lambda, kappa=kappa_policy)
            self.optimizer_value = AdaptiveObGD(self.critic.parameters(), lr=lr, gamma=gamma, lamda=gae_lambda, kappa=kappa_value)
        elif self.optimizer_name == "ObGD":
            self.optimizer_policy = ObGD(list(self.actor_mean.parameters()) + [self.actor_logstd], lr=lr, gamma=gamma, lamda=gae_lambda, kappa=kappa_policy)
            self.optimizer_value = ObGD(self.critic.parameters(), lr=lr, gamma=gamma, lamda=gae_lambda, kappa=kappa_value)

    def get_action_and_value(self, x, action=None):
        is_batch = len(x.shape) > 1
        x_tensor = torch.from_numpy(x).float()
        if not is_batch:
            x_tensor = x_tensor.unsqueeze(0)

        action_mean = self.actor_mean(x_tensor)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action_tensor = probs.sample()
        else:
            action_tensor = torch.from_numpy(action).float()
            if not is_batch:
                action_tensor = action_tensor.unsqueeze(0)
        
        log_prob = probs.log_prob(action_tensor).sum(1)
        entropy = probs.entropy().sum(1)
        value = self.critic(x_tensor)

        if not is_batch:
            return action_tensor.squeeze(0).numpy(), log_prob, entropy, value
        else:
            return action_tensor, log_prob, entropy, value

    def get_value(self, x):
        x = torch.from_numpy(x).float().unsqueeze(0)
        return self.critic(x)

    def update_params(self, obs_buf, act_buf, logp_buf, rew_buf, term_buf, trunc_buf, val_buf, trunc_val_buf, next_obs):
        obs_buf = torch.tensor(np.array(obs_buf), dtype=torch.float32)
        act_buf = torch.tensor(np.array(act_buf), dtype=torch.float32)
        logp_buf = torch.tensor(logp_buf, dtype=torch.float32).flatten()
        rew_buf = torch.tensor(np.array(rew_buf), dtype=torch.float32).flatten()
        term_buf = torch.tensor(np.array(term_buf), dtype=torch.float32).flatten()
        trunc_buf = torch.tensor(np.array(trunc_buf), dtype=torch.bool).flatten()
        val_buf = torch.tensor(val_buf, dtype=torch.float32).flatten()
        trunc_val_buf = torch.tensor(trunc_val_buf, dtype=torch.float32).flatten()
        
        batch_size = len(rew_buf)
        
        with torch.no_grad():
            next_value = self.get_value(next_obs).flatten().item()
            advantages = torch.zeros(batch_size, dtype=torch.float32)
            lastgaelam = 0
            for t in reversed(range(batch_size)):
                nextnonterminal = 1.0 - float(term_buf[t] or trunc_buf[t])
                
                if t == batch_size - 1:
                    if trunc_buf[t]:
                        nextvalues = trunc_val_buf[t]
                    else:
                        nextvalues = next_value 
                else:
                    if trunc_buf[t]:
                        nextvalues = trunc_val_buf[t]
                    else:
                        nextvalues = val_buf[t + 1]

                delta = rew_buf[t] + self.gamma * nextvalues * nextnonterminal - val_buf[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + val_buf

        b_obs = obs_buf
        b_logprobs = logp_buf
        b_actions = act_buf
        b_advantages = advantages
        b_returns = returns
        b_values = val_buf

        minibatch_size = max(1, batch_size // self.num_minibatches)
        
        self.train()
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = min(start + minibatch_size, batch_size)
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.get_action_and_value(b_obs[mb_inds].numpy(), b_actions[mb_inds].numpy())
                newvalue = newvalue.flatten()
                    
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]
                
                if self.target_kl is not None and approx_kl > self.target_kl:
                    break

                mb_advantages = b_advantages[mb_inds]
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                mb_returns = b_returns[mb_inds]
                v_loss_unclipped = (newvalue - mb_returns) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -self.clip_coef, self.clip_coef)
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                if self.optimizer_name == "Adam":
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                else: 
                    self.optimizer_value.zero_grad()
                    self.optimizer_policy.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                    self.optimizer_value.step(delta=1.0, reset=False) 
                    self.optimizer_policy.step(delta=1.0, reset=False)

            if self.target_kl is not None and approx_kl > self.target_kl:
                break
        self.eval()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        if wandb.run:
            wandb.log({
                "train/policy_loss": pg_loss.item(),
                "train/value_loss": v_loss.item(),
                "train/entropy_loss": entropy_loss.item(),
                "train/approx_kl": approx_kl.item(),
                "train/clip_fraction": np.mean(clipfracs),
                "train/explained_variance": explained_var,
            })

class StreamACRunner:
    def __init__(self, env_name, seed=0, hidden_size=128, lr=1.0, gamma=0.99, gae_lambda=0.95, entropy_coeff=0.01, kappa_policy=3.0, kappa_value=2.0, total_steps=100_000, eval_frequency=10_000, eval_episodes=50, debug=False, wandb_log=False, overshooting_info=False, render=False, max_episode_steps=200, save_video=False, do_damage=False, damage_type='stuck_joint', damage_start_step=0, damage_steps=100_000, cbp=False, layernorm=False, optimizer="AdaptiveObGD", checkpoint="", num_steps=2048, update_epochs=10, num_minibatches=32, clip_coef=0.2, vf_coef=0.5, norm_adv=True, max_grad_norm=0.5, target_kl=None):
        self.env_name = env_name; self.seed = seed; self.hidden_size = hidden_size; self.lr = lr; self.gamma = gamma; self.gae_lambda = gae_lambda; self.entropy_coeff = entropy_coeff; self.kappa_policy = kappa_policy; self.kappa_value = kappa_value; self.total_steps = total_steps; self.eval_frequency = eval_frequency; self.eval_episodes = eval_episodes; self.debug = debug; self.wandb_log = wandb_log; self.overshooting_info = overshooting_info; self.render = render; self.max_episode_steps = max_episode_steps; self.save_video = save_video; self.do_damage = do_damage; self.damage_type = damage_type; self.damage_start_step = damage_start_step; self.damage_steps = damage_steps; self.damage_ongoing = False; self.cbp = cbp; self.layernorm = layernorm; self.optimizer = optimizer; self.checkpoint = checkpoint; self.num_steps = num_steps; self.update_epochs = update_epochs; self.num_minibatches = num_minibatches; self.clip_coef = clip_coef; self.vf_coef = vf_coef; self.norm_adv = norm_adv; self.max_grad_norm = max_grad_norm; self.target_kl = target_kl
        self.agent = None; self.env = None; self.log_file = None; self.eval_log_file = None; self.returns = []; self.term_time_steps = []
        self.model_name = "ppo_adapt"; self.start_time = int(time.time())

    def create_logs(self):
        log_dir = "logs"
        if not os.path.exists(log_dir): os.makedirs(log_dir)
        self.log_file = os.path.join(log_dir, f"{self.env_name}-training_{self.damage_type}_{self.optimizer}_cbp={self.cbp}_seed_{self.seed}.txt")
        self.eval_log_file = os.path.join(log_dir, f"{self.env_name}-eval_{self.damage_type}_{self.optimizer}_cbp={self.cbp}_seed_{self.seed}.txt")
        open(self.log_file, 'w').close(); open(self.eval_log_file, 'w').close()
        if self.wandb_log:
            config = self.__dict__.copy(); del config['agent'], config['env'], config['log_file'], config['eval_log_file']
            wandb.init(entity="apollo-lab", project=f"stream-rl-robotics", config=config, name=f"{self.env_name}_{self.damage_type}_batch_ppo_no-warm_seed_{self.seed}", save_code=True)
        
    def setup_environment(self):
        render_mode = "human" if self.render else None
        env = gym.make(self.env_name, num_envs=1, render_mode=render_mode, max_episode_steps=self.max_episode_steps, reward_mode="normalized_dense", obs_mode="state", control_mode="pd_joint_delta_pos")
        env = self.wrap_environment(env)
        return env
    
    def wrap_environment(self, env):
        env = ToNumpyWrapper(env)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        return env
    
    def create_agent(self, env):
        agent = StreamAC(n_obs=env.observation_space.shape[0], n_actions=env.action_space.shape[0], hidden_size=self.hidden_size, lr=self.lr, gamma=self.gamma, gae_lambda=self.gae_lambda, kappa_policy=self.kappa_policy, kappa_value=self.kappa_value, cbp=self.cbp, layernorm=self.layernorm, optimizer=self.optimizer, update_epochs=self.update_epochs, num_minibatches=self.num_minibatches, clip_coef=self.clip_coef, norm_adv=self.norm_adv, ent_coef=self.entropy_coeff, vf_coef=self.vf_coef, max_grad_norm=self.max_grad_norm, target_kl=self.target_kl)
        return agent
    
    def save_model_and_stats(self):
        save_dir = f"results/{self.model_name}_{self.env_name}_{self.start_time}"
        os.makedirs(save_dir, exist_ok=True)  
        with open(os.path.join(save_dir, f"seed_{self.seed}.pkl"), "wb") as f:
            pickle.dump((self.returns, self.term_time_steps, self.env_name), f)
        save_dir = f"weights/{self.model_name}_{self.env_name}_{self.start_time}"
        os.makedirs(save_dir, exist_ok=True)  
        torch.save(self.agent.state_dict(), os.path.join(save_dir, f"seed_{self.seed}.pth"))
        if self.wandb_log:
            wandb.save(os.path.join(save_dir, f"seed_{self.seed}.pth"))
            wandb.finish()
        return save_dir
    
    def evaluate(self):
        torch.manual_seed(self.seed)
        self.agent.eval()
        returns, successes = [], []
        s, _ = self.env.reset(seed=self.seed)
        episode_count = 0
        while episode_count < self.eval_episodes:
            with torch.no_grad():
                a, _, _, _ = self.agent.get_action_and_value(s)
            if self.do_damage and self.damage_ongoing:
                if self.damage_type == 'broken_leg':
                    a = a * np.array([1,1,0,1,1,1,0,1,1,1,0,1])
                elif self.damage_type == 'stuck_joint':
                    a = a * np.array([1,1,1,1,1,1,1,1,1,1,0,1])
            s_prime, r, terminated, truncated, info = self.env.step(a)
            s = s_prime
            if terminated or truncated:
                episode_return = info['episode']['r']
                if isinstance(episode_return, (list, np.ndarray)): episode_return = episode_return[0]
                returns.append(episode_return)
                is_success = info.get('success', False)
                if isinstance(is_success, (list, np.ndarray)): is_success = is_success[0]
                successes.append(int(is_success))
                episode_count += 1
                s, _ = self.env.reset()
        success_rate = np.mean(successes) if successes else 0.0
        return returns, success_rate
    
    def train(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        self.create_logs()
        self.env = self.setup_environment()
        self.agent = self.create_agent(self.env)
        if self.checkpoint:
            checkpoint = torch.load(self.checkpoint)
            self.agent.load_state_dict(checkpoint["model_state_dict"], strict=True)
        
        s, _ = self.env.reset(seed=self.seed)
        episode_count = 0
        start_time = time.time()
        
        for global_step in range(0, self.total_steps, self.num_steps):
            if global_step > 0 and (global_step // self.num_steps) % (self.eval_frequency // self.num_steps) == 0:
                eval_returns, success_rate = self.evaluate()
                mean_return = np.mean(eval_returns) if eval_returns else 0.0
                if self.wandb_log:
                    wandb.log({"eval/mean_return": mean_return, "eval/success_rate": success_rate, "global_step": global_step})
                with open(self.eval_log_file, 'a') as f:
                    f.write(f"Mean Eval Episodic Return: {mean_return}, Success Rate: {success_rate}, Step: {global_step}\n")

            if self.do_damage and self.damage_type == "goal_shift" and self.damage_start_step==0:
                self.env.set_goal_offset(0,3.0); self.damage_ongoing = True; wandb.log({"goal_shift": 1})
            if self.do_damage and self.damage_type == "goal_shift_easy" and self.damage_start_step==0:
                self.env.set_goal_offset(0,2.4); self.damage_ongoing = True; wandb.log({"goal_shift": 1})
            s, _ = self.env.reset(seed=self.seed)

            obs_buf, act_buf, logp_buf, rew_buf, term_buf, trunc_buf, val_buf, trunc_val_buf = [], [], [], [], [], [], [], []
            
            for step in range(self.num_steps):
                current_step = global_step + step
                with torch.no_grad():
                    a, logp, _, val = self.agent.get_action_and_value(s)

                obs_buf.append(s); act_buf.append(a); logp_buf.append(logp.item()); val_buf.append(val.item())

                if self.do_damage:
                    if current_step >= self.damage_start_step and (current_step - self.damage_start_step) < self.damage_steps:
                        if self.damage_ongoing == False and self.damage_type == "slippery_floor": self.env.change_friction(-1.8, -1.8)
                        if self.damage_ongoing == False and self.damage_type == "slippery_floor_easy": self.env.change_friction(-1.7, -1.7)
                        if self.damage_type in ['goal_shift', 'goal_shift_easy']: wandb.log({"goal_shift": 1})
                        else: self.damage_ongoing = True
                        if self.damage_type in ['slippery_floor', 'slippery_floor_easy']: wandb.log({"slippery_floor": 1})
                        elif self.damage_type == 'broken_leg': a = a * np.array([1,1,0,1,1,1,0,1,1,1,0,1]); wandb.log({"damaged_leg": 0})
                        elif self.damage_type == 'stuck_joint': a = a * np.array([1,1,1,1,1,1,1,1,1,1,0,1]); wandb.log({"damaged_joint": 0})
                    else:
                        if self.damage_ongoing == True and self.damage_type in ["slippery_floor", "slippery_floor_easy"]: self.env.change_friction(0.3, 0.3) 
                        self.damage_ongoing = False
                        if self.damage_type in ['goal_shift', 'goal_shift_easy']: wandb.log({"goal_shift": 0})
                        if self.damage_type in ['slippery_floor', 'slippery_floor_easy']: wandb.log({"slippery_floor": 0})
                        elif self.damage_type == 'broken_leg': wandb.log({"damaged_leg": -1})  
                        elif self.damage_type == 'stuck_joint': wandb.log({"damaged_joint": -1})
                
                s_prime, r, terminated, truncated, info = self.env.step(a)
                done = terminated or truncated

                rew_buf.append(r)
                term_buf.append(terminated)
                trunc_buf.append(truncated)

                if truncated:
                    with torch.no_grad():
                        val_s_prime = self.agent.get_value(s_prime).item()
                else:
                    val_s_prime = 0.0
                trunc_val_buf.append(val_s_prime)

                s = s_prime

                if done:
                    episode_return = info['episode']['r'][0]
                    if self.wandb_log:
                        wandb.log({"train/episode_return": episode_return, "train/episode": episode_count, "global_step": current_step})
                    if self.debug:
                        with open(self.log_file, 'a') as f: f.write(f"Episodic Return: {episode_return}, Step {current_step}\n")
                    self.returns.append(episode_return); self.term_time_steps.append(current_step)

                    if self.do_damage and self.damage_type in ["goal_shift", "goal_shift_easy"]:
                        if current_step+1 >= self.damage_start_step and (current_step+1 - self.damage_start_step) < self.damage_steps:
                            if self.damage_ongoing == False and self.damage_type == "goal_shift": self.env.set_goal_offset(0,3.0); self.damage_ongoing = True
                            if self.damage_ongoing == False and self.damage_type == "goal_shift_easy": self.env.set_goal_offset(0,2.4); self.damage_ongoing = True
                        else:
                            if self.damage_ongoing == True and self.damage_type in ["goal_shift", "goal_shift_easy"]: self.env.set_goal_offset(0,0); self.damage_ongoing = False
                    s, _ = self.env.reset()
                    episode_count += 1
            
            self.agent.update_params(obs_buf, act_buf, logp_buf, rew_buf, term_buf, trunc_buf, val_buf, trunc_val_buf, s)

        self.env.close()
        with open(self.eval_log_file, 'a') as f: f.write(f"Total training time: {time.time() - start_time:.2f} seconds\n")
        self.save_model_and_stats()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPO-style Adaptation Script')
    parser.add_argument('--env_name', type=str, default='AnymalC-Reach-v1')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--total_steps', type=int, default=1_500_000)
    parser.add_argument('--eval_frequency', type=int, default=10_000)
    parser.add_argument('--eval_episodes', type=int, default=50)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--wandb_log', action='store_true', default=True)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--checkpoint', type=str, default="pretrained-models/anymalc-reach/adam_ppo_pretrain_final.pt")

    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--optimizer', type=str, default="Adam", choices=["AdaptiveObGD", "ObGD", "Adam", "FastTrac"])
    parser.add_argument('--lr', type=float, default=3e-6)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--cbp', action='store_true', default=False)
    parser.add_argument('--layernorm', action='store_true', default=False)

    parser.add_argument('--num_steps', type=int, default=2048)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--num_minibatches', type=int, default=8)
    parser.add_argument('--update_epochs', type=int, default=4)
    parser.add_argument('--norm_adv', action='store_true', default=True)
    parser.add_argument('--clip_coef', type=float, default=0.2)
    parser.add_argument('--ent_coef', type=float, default=0.0)
    parser.add_argument('--vf_coef', type=float, default=0.5)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--target_kl', type=float, default=0.015)

    parser.add_argument('--kappa_policy', type=float, default=3.0)
    parser.add_argument('--kappa_value', type=float, default=2.0)

    parser.add_argument('--do_damage', action='store_true', default=True)
    parser.add_argument('--damage_start_step', type=int, default=0)
    parser.add_argument('--damage_steps', type=int, default=1_500_000)
    parser.add_argument('--damage_type', type=str, default='broken_leg', choices=['broken_leg', 'stuck_joint', 'slippery_floor', 'slippery_floor_easy', 'goal_shift', 'goal_shift_easy'])
    
    args = parser.parse_args()
    args.entropy_coeff = args.ent_coef

    runner = StreamACRunner(
        env_name=args.env_name, seed=args.seed, hidden_size=args.hidden_size, lr=args.lr, gamma=args.gamma, gae_lambda=args.gae_lambda, total_steps=args.total_steps, entropy_coeff=args.entropy_coeff, kappa_policy=args.kappa_policy, kappa_value=args.kappa_value, eval_frequency=args.eval_frequency, eval_episodes=args.eval_episodes, debug=args.debug, wandb_log=args.wandb_log, render=args.render, do_damage=args.do_damage, damage_type=args.damage_type, damage_start_step=args.damage_start_step, damage_steps=args.damage_steps, cbp=args.cbp, layernorm=args.layernorm, optimizer=args.optimizer, checkpoint=args.checkpoint, num_steps=args.num_steps, update_epochs=args.update_epochs, num_minibatches=args.num_minibatches, clip_coef=args.clip_coef, vf_coef=args.vf_coef, norm_adv=args.norm_adv, max_grad_norm=args.max_grad_norm, target_kl=args.target_kl
    )
    runner.train()