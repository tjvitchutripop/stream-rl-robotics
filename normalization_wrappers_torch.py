import torch
import gymnasium as gym

class SampleMeanStd:
    def __init__(self, shape=(), device="cuda"):
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.p = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = 0
        self.device = device

    def update(self, x):
        x = x.to(dtype=torch.float32, device=self.device)
        if self.count == 0:
            self.mean = x.clone()
            self.p = torch.zeros_like(x)
        self.mean, self.var, self.p, self.count = self._update_mean_var_count_from_moments(
            self.mean, self.p, self.count, x
        )

    def _update_mean_var_count_from_moments(self, mean, p, count, sample):
        new_count = count + 1
        delta = sample - mean
        new_mean = mean + delta / new_count
        p = p + delta * (sample - new_mean)
        new_var = torch.ones_like(p) if new_count < 2 else p / (new_count - 1)
        return new_mean, new_var, p, new_count


class NormalizeObservation(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env: gym.Env, epsilon: float = 1e-8, device="cuda"):
        gym.utils.RecordConstructorArgs.__init__(self, epsilon=epsilon)
        gym.Wrapper.__init__(self, env)
        self.num_envs = 10_000
        self.is_vector_env = True

        shape = (
            self.single_observation_space.shape
            if self.is_vector_env
            else self.observation_space.shape
        )
        self.obs_stats = SampleMeanStd(shape=shape, device=device)
        self.epsilon = epsilon
        self.device = device

    def step(self, action):
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        obs = self._to_tensor(obs)
        norm_obs = self.normalize(obs)
        return norm_obs.float(), rews, terminateds, truncateds, infos

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._to_tensor(obs)
        norm_obs = self.normalize(obs)
        return norm_obs.float(), info

    def _to_tensor(self, obs):
        if isinstance(obs, torch.Tensor):
            return obs.to(self.device)
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if not self.is_vector_env:
            obs = obs.unsqueeze(0)
        return obs

    def normalize(self, obs):
        self.obs_stats.update(obs)
        norm = (obs - self.obs_stats.mean) / torch.sqrt(self.obs_stats.var + self.epsilon)
        return norm if self.is_vector_env else norm[0]


class ScaleReward(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env: gym.Env, gamma: float = 0.99, epsilon: float = 1e-8, device="cuda"):
        gym.utils.RecordConstructorArgs.__init__(self, gamma=gamma, epsilon=epsilon)
        gym.Wrapper.__init__(self, env)
        self.num_envs = 10_000
        self.is_vector_env = True

        self.reward_stats = SampleMeanStd(shape=(), device=device)
        self.reward_trace = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = device

    def step(self, action):
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        rews = self._to_tensor(rews)
        term = torch.tensor(terminateds, dtype=torch.bool, device=self.device) | \
               torch.tensor(truncateds, dtype=torch.bool, device=self.device)
        self.reward_trace = self.reward_trace * self.gamma * (~term).float() + rews
        norm_rews = self.normalize(rews)
        return obs.float(), norm_rews.item() if not self.is_vector_env else norm_rews, terminateds, truncateds, infos

    def _to_tensor(self, rews):
        if isinstance(rews, torch.Tensor):
            return rews.to(dtype=torch.float32, device=self.device)
        rews = torch.tensor(rews, dtype=torch.float32, device=self.device)
        if not self.is_vector_env:
            rews = rews.unsqueeze(0)
        return rews

    def normalize(self, rews):
        self.reward_stats.update(self.reward_trace)
        norm = rews / torch.sqrt(self.reward_stats.var + self.epsilon)
        return norm if self.is_vector_env else norm[0]  