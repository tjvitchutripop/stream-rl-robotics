import torch
import wandb
import numpy as np

class MLPWandBLogger:
    """
    Logs MLP weight and activation statistics to Weights & Biases during training.

    Features:
      - Weight histograms and norms
      - Activation histograms, mean, std
      - Dormant neuron ratio (low variance or tanh saturation)
      - Neuron drift (weight cosine similarity over time)
    """

    def __init__(self, model, log_interval=1000, activation_fn='tanh'):
        self.model = model
        self.log_interval = log_interval
        self.activation_fn = activation_fn.lower()
        self.activations = {}
        self.prev_weights = {}
        self._register_hooks()

    def _register_hooks(self):
        """Attach hooks to collect layer activations."""
        for name, layer in self.model.named_modules():
            if isinstance(layer, (torch.nn.Linear, torch.nn.Tanh, torch.nn.ReLU)):
                layer.register_forward_hook(self._make_activation_hook(name))

    def _make_activation_hook(self, name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self.activations[name] = output.detach().cpu()
        return hook

    def _log_weight_stats(self, step):
        """Log weight norms and histograms."""
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                wandb.log({
                    f"{name}/l2_norm": param.norm().item(),
                    f"{name}/hist": wandb.Histogram(param.detach().cpu().numpy())
                })

                # Track drift (cosine similarity with previous weights)
                prev = self.prev_weights.get(name)
                if prev is not None:
                    sim = torch.nn.functional.cosine_similarity(
                        param.flatten(), prev.flatten(), dim=0
                    ).item()
                    wandb.log({f"{name}/cosine_similarity": sim})
                self.prev_weights[name] = param.detach().clone()

    def _log_activation_stats(self, step):
        """Log activation statistics and dormant ratio."""
        for name, act in self.activations.items():
            data = act.numpy()
            wandb.log({
                f"{name}/act_mean": data.mean(),
                f"{name}/act_std": data.std(),
                f"{name}/act_hist": wandb.Histogram(data)
            })

            # Dormant unit detection
            if data.ndim == 2:  # [batch, neurons]
                if self.activation_fn == 'tanh':
                    saturated = np.mean((np.abs(data) > 0.97).astype(float), axis=0)
                    dormant_ratio = (saturated > 0.95).mean()
                else:  # for ReLU, "dormant" = always near zero
                    low_var = data.var(axis=0) < 1e-4
                    dormant_ratio = low_var.mean()
                wandb.log({f"{name}/dormant_ratio": dormant_ratio})

    def log(self, step):
        """Call this every `log_interval` steps inside your RL loop."""
        if step % self.log_interval != 0:
            return
        self._log_weight_stats(step)
        self._log_activation_stats(step)