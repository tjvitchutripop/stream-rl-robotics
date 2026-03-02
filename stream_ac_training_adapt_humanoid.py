import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(ROOT_DIR)
for _p in (ROOT_DIR, PARENT_DIR):
    if _p not in sys.path:
        sys.path.append(_p)

import pickle, argparse
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym
# import gymnasium_robotics
from gymnasium.wrappers import RecordVideo
import torch.nn.functional as F
from torch.distributions import Normal
try:
    from streaming_drl.optim import ObGD, AdaptiveObGD
    from streaming_drl.sparse_init import sparse_init
    from streaming_drl.normalization_wrappers import NormalizeObservation, ScaleReward
    from streaming_drl.time_wrapper import AddTimeInfo
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Could not import 'streaming_drl'. Ensure the repository contains "
        "'streaming_drl/' and run from project root (or set PYTHONPATH to project root)."
    ) from exc
import wandb    
import time
try:
    import moviepy.editor as mp
except ModuleNotFoundError:
    mp = None
import glob
import mani_skill.envs
# from gym_envs import make_lift_env
from model import ActorMean, Critic, ActorMeanLN, CriticLN, ActorMeanCBP, CriticCBP
try:
    from interpretability import MLPWandBLogger
except ModuleNotFoundError:
    MLPWandBLogger = None

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


def remap_sequential_checkpoint_keys(state_dict):
    """Map PPO nn.Sequential-style keys to model.py MLP keys."""
    prefix_map = {
        "actor_mean.0.": "actor_mean.fc1.",
        "actor_mean.2.": "actor_mean.fc2.",
        "actor_mean.4.": "actor_mean.fc3.",
        "actor_mean.6.": "actor_mean.fc4.",
        "critic.0.": "critic.fc1.",
        "critic.2.": "critic.fc2.",
        "critic.4.": "critic.fc3.",
        "critic.6.": "critic.value.",
    }
    remapped = {}
    changed = False
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in prefix_map.items():
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix):]
                changed = True
                break
        remapped[new_key] = value
    return remapped, changed


class StreamAC(nn.Module):
    def __init__(self, n_obs=11, n_actions=3, hidden_size=128, lr=1.0, gamma=0.99, lamda=0.8, kappa_policy=3.0, kappa_value=2.0, cbp=False, layernorm=False, optimizer="AdaptiveObGD"):
        super(StreamAC, self).__init__()
        self.optimizer = optimizer
        self.gamma = gamma
        if cbp:
            self.actor_mean = ActorMeanCBP(n_obs, n_actions, hidden_size, replacement_rate=1e-5, maturity_threshold=1000, decay_rate=0.99)
            self.critic = CriticCBP(n_obs, hidden_size, replacement_rate=1e-5, maturity_threshold=1000, decay_rate=0.99)
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
        elif self.optimizer == "FastTrac":
            self.optimizer_policy = start_trac(log_file='logs/trac.text', Base=AdaptiveObGD)(
                list(self.actor_mean.parameters()) + [self.actor_logstd] + list(self.critic.parameters()),
                lr=3e-4,
                eps=1e-5
            )
            self.optimizer_value = start_trac(log_file='logs/trac.text', Base=AdaptiveObGD)(
                list(self.critic.parameters()),
                lr=3e-5,
                eps=1e-5
            )
        elif self.optimizer == "Adam":
            self.optimizer_policy = torch.optim.Adam(list(self.actor_mean.parameters()) + [self.actor_logstd], lr=3e-4, eps=1e-5)
            self.optimizer_value = torch.optim.Adam(self.critic.parameters(), lr=3e-4, eps=1e-5)
            # self.optimizer_trac = start_trac(log_file='logs/trac.text', Base=torch.optim.Adam)(
            #     list(self.actor_mean.parameters()) + [self.actor_logstd] + list(self.critic.parameters()),
            #     lr=3e-4,
            #     eps=1e-5
            # )

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

    def set_kappa(self, kappa_policy: float | None = None, kappa_value: float | None = None):
        """Update ObGD/AdaptiveObGD kappa values at runtime."""
        policy_updated = False
        value_updated = False
        if kappa_policy is not None and hasattr(self, "optimizer_policy"):
            for group in self.optimizer_policy.param_groups:
                if "kappa" in group:
                    group["kappa"] = float(kappa_policy)
                    policy_updated = True
        if kappa_value is not None and hasattr(self, "optimizer_value"):
            for group in self.optimizer_value.param_groups:
                if "kappa" in group:
                    group["kappa"] = float(kappa_value)
                    value_updated = True
        return policy_updated, value_updated

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
        # if self.optimizer == "FastTrac":
        #     self.optimizer_trac.zero_grad()
        # else:
        self.optimizer_value.zero_grad()
        self.optimizer_policy.zero_grad()
        value_output.backward()
        (log_prob_pi + entropy_pi).backward()
        if self.optimizer == "Adam":
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
            self.optimizer_policy.step()
            self.optimizer_value.step()
        else:
            self.optimizer_policy.step(delta.item(), reset=done)
            self.optimizer_value.step(delta.item(), reset=done)
        metrics = {
            "train/log_prob_pi": log_prob_pi.item(),
            "train/value": v_s.item(),
            "train/td_target": td_target.item(),
            "train/delta": delta.item(),
            "train/entropy": dist.entropy().sum().item(),
            "train/entropy_coeff": entropy_coeff,
        }

        if overshooting_info:
            v_s, v_prime = self.v(s), self.v(s_prime)
            td_target = r + self.gamma * v_prime * done_mask
            delta_bar = td_target - v_s
            if torch.sign(delta_bar * delta).item() == -1:
                print("Overshooting Detected!")
        return metrics


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
        damage_type='arm_hand_impairment',
        damage_start_step=500_000,
        damage_steps=1_500_000,
        arm_shoulder_elbow_gain=0.14,
        arm_finger_gain=0.07,
        arm_stuck_joint_idxs=None,
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
        self.arm_shoulder_elbow_gain = arm_shoulder_elbow_gain
        self.arm_finger_gain = arm_finger_gain
        if arm_stuck_joint_idxs is None:
            arm_stuck_joint_idxs = [14, 15, 21, 24]
        self.arm_stuck_joint_idxs = sorted({int(i) for i in arm_stuck_joint_idxs})
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
        self.best_adaptation_success_rate = -float("inf")
        self.best_adaptation_step = None
        self.best_adaptation_ckpt_path = None
        self.weights_save_dir = f"weights/stream_ac_{self.env_name}_{self.start_time}"
        self.results_save_dir = f"results/stream_ac_{self.env_name}_{self.start_time}"
        
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
                    "arm_shoulder_elbow_gain": self.arm_shoulder_elbow_gain,
                    "arm_finger_gain": self.arm_finger_gain,
                    "arm_stuck_joint_idxs": self.arm_stuck_joint_idxs,
                },
                name=f"{self.env_name}_{self.damage_type}_{self.optimizer}_cbp={self.cbp}_ln={self.layernorm}_seed_{self.seed}",
                save_code=True
            )

        self.log_file = log_file
        self.eval_log_file = eval_log_file

    def log_metrics(self, metrics: dict, step: int | None = None):
        if not self.wandb_log:
            return
        payload = dict(metrics)
        if step is not None:
            payload.setdefault("timestep", step)
            wandb.log(payload, step=step)
        else:
            wandb.log(payload)
        
    def setup_environment(self):
        env_kwargs = dict(
            num_envs=1,
            max_episode_steps=self.max_episode_steps,
            reward_mode="normalized_dense",
            sim_backend="physx_cpu",
        )
        if self.render:
            env_kwargs.update(render_mode="human")
        else:
            # Keep headless execution without forcing a specific render backend.
            env_kwargs.update(render_mode=None)
        # TransportBox runs are expected to use state observations.
        if "TransportBox" in self.env_name:
            env_kwargs.update(obs_mode="state")
        env = gym.make(self.env_name, **env_kwargs)
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
        if self.wandb_log and self.interpretability and MLPWandBLogger is not None:
            self.logger = MLPWandBLogger(agent, log_interval=1000, activation_fn='silu')
        elif self.wandb_log and self.interpretability and MLPWandBLogger is None:
            print("interpretability logging disabled: interpretability module not found")
        return agent

    def maybe_save_best_adaptation_checkpoint(self, step: int, success_rate: float, mean_return: float):
        """Save checkpoint with best eval success after adaptation starts."""
        if not self.do_damage or step < self.damage_start_step:
            return
        if (step - self.damage_start_step) >= self.damage_steps:
            return
        if success_rate <= self.best_adaptation_success_rate:
            return

        os.makedirs(self.weights_save_dir, exist_ok=True)
        ckpt_path = os.path.join(self.weights_save_dir, f"best_adaptation_seed_{self.seed}.pt")
        payload = {
            "model_state_dict": self.agent.state_dict(),
            "step": int(step),
            "eval_success_rate": float(success_rate),
            "eval_mean_return": float(mean_return),
            "seed": int(self.seed),
            "env_name": self.env_name,
            "damage_type": self.damage_type,
        }
        torch.save(payload, ckpt_path)

        self.best_adaptation_success_rate = float(success_rate)
        self.best_adaptation_step = int(step)
        self.best_adaptation_ckpt_path = ckpt_path

        print(
            f"Saved new best adaptation checkpoint at step {step} "
            f"(success_rate={success_rate:.4f}) -> {ckpt_path}"
        )
        self.log_metrics(
            {
                "eval/best_adaptation_success_rate": float(success_rate),
                "eval/best_adaptation_step": int(step),
            },
            step=step,
        )

    def apply_arm_hand_impairment(self, action: np.ndarray):
        # Right-arm joints in UnitreeG1UpperBody action ordering:
        # shoulder/elbow indices: 2,4,6,8,10
        # right-finger indices: 14,15,16,20,21,22,24
        shoulder_elbow_idx = [2, 4, 6, 8, 10]
        finger_idx = [14, 15, 16, 20, 21, 22, 24]
        damaged_action = np.array(action, copy=True)
        damaged_action[..., shoulder_elbow_idx] *= self.arm_shoulder_elbow_gain
        damaged_action[..., finger_idx] *= self.arm_finger_gain
        for idx in self.arm_stuck_joint_idxs:
            if 0 <= idx < damaged_action.shape[-1]:
                damaged_action[..., idx] = 0.0
        return damaged_action
    
    def save_model_and_stats(self):
        # Save training data
        os.makedirs(self.results_save_dir, exist_ok=True)
        with open(os.path.join(self.results_save_dir, f"seed_{self.seed}.pkl"), "wb") as f:
            pickle.dump((self.returns, self.term_time_steps, self.env_name), f)

        # Save model weights
        os.makedirs(self.weights_save_dir, exist_ok=True)
        final_ckpt_path = os.path.join(self.weights_save_dir, f"seed_{self.seed}.pth")
        torch.save(self.agent.state_dict(), final_ckpt_path)
        
        # Save env stats
        # reward_wrapper = self.env
        # while not isinstance(reward_wrapper, ScaleReward) and hasattr(reward_wrapper, 'env'):
        #     reward_wrapper = reward_wrapper.env
            
        # obs_wrapper = self.env
        # while not isinstance(obs_wrapper, NormalizeObservation) and hasattr(obs_wrapper, 'env'):
        #     obs_wrapper = obs_wrapper.env

        # reward_stats = reward_wrapper.reward_stats
        # obs_stats = obs_wrapper.obs_stats
        # with open(os.path.join(save_dir, f"stats_data_{self.seed}.pkl"), "wb") as f:
        #     pickle.dump((reward_stats, obs_stats), f)

        # Log final model to wandb
        if self.wandb_log:
            wandb.save(final_ckpt_path)
            if self.best_adaptation_ckpt_path and os.path.exists(self.best_adaptation_ckpt_path):
                wandb.save(self.best_adaptation_ckpt_path)
            wandb.finish()

        return self.weights_save_dir
    
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
                    # a = a * np.array([0,1,1,1,0,1,1,1,0,1,1,1]) # Front Right
                    a = a * np.array([1,1,0,1,1,1,0,1,1,1,0,1]) # Back Left Leg
                elif self.damage_type == 'stuck_joint':
                    a = a * np.array([1,1,1,1,1,1,1,1,1,1,0,1]) # One Joint Stuck
                elif self.damage_type == 'arm_hand_impairment':
                    a = self.apply_arm_hand_impairment(a)
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
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        state_dict, remapped = remap_sequential_checkpoint_keys(state_dict)
        if remapped:
            print("Remapped PPO sequential checkpoint keys to StreamAC MLP keys")
        if self.cbp is True or self.layernorm is True:
            self.agent.load_state_dict(state_dict, strict=False)
        else:
            self.agent.load_state_dict(state_dict, strict=True)
        # self.agent.optimizer_policy.load_state_dict(checkpoint["optimizer_policy_state_dict"])
        # self.agent.optimizer_value.load_state_dict(checkpoint["optimizer_value_state_dict"])
        if self.debug:
            print(f"seed: {self.seed}", f"env: {self.env.spec.id}")

        self.returns = []
        self.term_time_steps = []

        if self.do_damage and self.damage_type == "goal_shift" and self.damage_start_step==0:
            self.env.set_goal_offset(0,3.0)
            self.damage_ongoing = True
            self.log_metrics({"goal_shift": 1}, step=0)
        if self.do_damage and self.damage_type == "goal_shift_easy" and self.damage_start_step==0:
            self.env.set_goal_offset(0,2.4)
            self.damage_ongoing = True
            self.log_metrics({"goal_shift": 1}, step=0)
        s, _ = self.env.reset(seed=self.seed)
        episode_count = 0

        start_time = time.time()
        converged = False
        
        for t in range(1, self.total_steps + 1):
            # Run evaluation
            if t % self.eval_frequency == 0:
                eval_returns, success_rate = self.evaluate()
                mean_return = np.mean(eval_returns)

                self.log_metrics(
                    {
                        "eval/mean_return": mean_return,
                        "eval/success_rate": success_rate,
                        "eval/episode": t // self.eval_frequency,
                    },
                    step=t,
                )
                
                if success_rate > 0.90 and not converged:
                    converged = True
                    elapsed_time = time.time() - start_time
                    with open(self.eval_log_file, 'a') as f:
                        f.write(f"Model converged after {elapsed_time:.2f} seconds\n")
                
                with open(self.eval_log_file, 'a') as f:
                    f.write(f"Mean Eval Episodic Return: {mean_return}, Success Rate: {success_rate}, Eval Number: {t // self.eval_frequency}\n")

                self.maybe_save_best_adaptation_checkpoint(
                    step=t,
                    success_rate=success_rate,
                    mean_return=mean_return,
                )

            pred_a = self.agent.sample_action(s)
            a = np.array(pred_a, copy=True)
            damage_spike = 1 if (self.do_damage and t == self.damage_start_step) else 0
            if self.do_damage:
                if t >= self.damage_start_step and (t - self.damage_start_step) < self.damage_steps:
                    if self.damage_ongoing == False and self.damage_type == "slippery_floor":
                        self.env.change_friction(-1.8, -1.8)
                    if self.damage_ongoing == False and self.damage_type == "slippery_floor_easy":
                        self.env.change_friction(-1.7, -1.7)
                    if self.damage_type == 'goal_shift' or self.damage_type == 'goal_shift_easy':
                        self.log_metrics({"goal_shift": 1}, step=t)
                    else:
                        self.damage_ongoing = True
                    if self.damage_type == 'slippery_floor' or self.damage_type == 'slippery_floor_easy':
                        self.log_metrics({"slippery_floor": 1}, step=t)
                    elif self.damage_type == 'broken_leg':
                        # a = a * np.array([0,1,1,1,0,1,1,1,0,1,1,1]) # Front Right
                        a = a * np.array([1,1,0,1,1,1,0,1,1,1,0,1]) # Back Left Leg
                        self.log_metrics({"damaged_leg": 0}, step=t)
                    elif self.damage_type == 'stuck_joint':
                        a = a * np.array([1,1,1,1,1,1,1,1,1,1,0,1]) # One Joint Stuck
                        self.log_metrics({"damaged_joint": 0}, step=t)
                    elif self.damage_type == 'arm_hand_impairment':
                        a = self.apply_arm_hand_impairment(a)
                        self.log_metrics({"arm_hand_impairment": 1}, step=t)
                else:
                    if self.damage_ongoing == True and (self.damage_type == "slippery_floor" or self.damage_type == "slippery_floor_easy"):
                        self.env.change_friction(0.3, 0.3) 
                    self.damage_ongoing = False
                    if self.damage_type == 'goal_shift' or self.damage_type == 'goal_shift_easy':
                        self.log_metrics({"goal_shift": 0}, step=t)
                    if self.damage_type == 'slippery_floor' or self.damage_type == 'slippery_floor_easy':
                        self.log_metrics({"slippery_floor": 0}, step=t)
                    elif self.damage_type == 'broken_leg':
                        self.log_metrics({"damaged_leg": -1}, step=t)
                    elif self.damage_type == 'stuck_joint':
                        self.log_metrics({"damaged_joint": -1}, step=t)
                    elif self.damage_type == 'arm_hand_impairment':
                        self.log_metrics({"arm_hand_impairment": 0}, step=t)

            # Damage monitor charts: a binary active signal and a one-step spike at onset.
            self.log_metrics(
                {
                    "damage/active": int(self.damage_ongoing),
                    "damage/kick_in_spike": damage_spike,
                },
                step=t,
            )

            s_prime, r, terminated, truncated, info = self.env.step(a)
            if self.wandb_log and self.interpretability:
                self.logger.log(t)
            update_action = a
            if self.damage_ongoing and self.damage_type in {'broken_leg', 'stuck_joint', 'arm_hand_impairment'}:
                # Update with intended policy action, not actuator-distorted action.
                update_action = pred_a
            train_metrics = self.agent.update_params(
                s,
                update_action,
                r,
                s_prime,
                terminated or truncated,
                self.entropy_coeff,
                self.overshooting_info,
            )
            self.log_metrics(train_metrics, step=t)
            s = s_prime

            if terminated or truncated:
                episode_return = info['episode']['r']
                if isinstance(episode_return, (list, np.ndarray)):
                    episode_return = episode_return[0]
                
                episode_metrics = {
                    "train/episode_return": episode_return,
                    "train/episode": episode_count,
                }
                episode_success = info.get("success", None)
                if episode_success is not None:
                    if isinstance(episode_success, (list, np.ndarray)):
                        episode_success = episode_success[0]
                    episode_metrics["train/success"] = int(episode_success)
                episode_length = info.get("episode", {}).get("l", None)
                if episode_length is not None:
                    if isinstance(episode_length, (list, np.ndarray)):
                        episode_length = episode_length[0]
                    episode_metrics["train/episode_len"] = float(episode_length)
                self.log_metrics(episode_metrics, step=t)
                
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
            if self.best_adaptation_step is not None:
                f.write(
                    "Best adaptation checkpoint: "
                    f"step={self.best_adaptation_step}, "
                    f"success_rate={self.best_adaptation_success_rate:.4f}, "
                    f"path={self.best_adaptation_ckpt_path}\n"
                )

        # Save model, stats and data
        self.save_model_and_stats()

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stream AC(λ)')
    parser.add_argument('--env_name', type=str, default='UnitreeG1TransportBox-v1')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lamda', type=float, default=0.8)
    parser.add_argument('--total_steps', type=int, default=2_000_000)
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
    parser.add_argument('--cbp', action='store_true', default=False)
    parser.add_argument('--layernorm', action='store_true', default=False)
    parser.add_argument('--optimizer', type=str, default="AdaptiveObGD")
    parser.add_argument('--checkpoint', type=str, default="pretrained-models/transport-box/adam_ppo_pretrain.pt")
    parser.add_argument('--interpretability', action='store_true', default=False)
    parser.add_argument('--do_damage', action='store_true', default=False)
    parser.add_argument('--no_damage', action='store_false', dest='do_damage',
                        help='Disable perturbation/damage during training')
    parser.add_argument('--damage_start_step', type=int, default=500_000)
    parser.add_argument('--damage_steps', type=int, default=1_500_000, help='Steps between damage events')
    parser.add_argument('--damage_type', type=str, default='arm_hand_impairment', help='Type of damage to apply')
    parser.add_argument('--arm_shoulder_elbow_gain', type=float, default=0.14)
    parser.add_argument('--arm_finger_gain', type=float, default=0.07)
    parser.add_argument('--arm_stuck_joint_idxs', type=int, nargs='+', default=[14, 15, 21, 24])
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
        arm_shoulder_elbow_gain=args.arm_shoulder_elbow_gain,
        arm_finger_gain=args.arm_finger_gain,
        arm_stuck_joint_idxs=args.arm_stuck_joint_idxs,
        cbp=args.cbp,
        layernorm=args.layernorm,
        optimizer=args.optimizer,
        checkpoint=args.checkpoint,
        interpretability=args.interpretability
    )
    
    if args.mode == 'train':
        runner.train()
    else:
        runner.test()