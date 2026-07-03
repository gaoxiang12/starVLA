"""Flow-matching scheduler ported from zs-va (wan_va/utils/scheduler.py).

Self-contained (no external deps beyond torch). Provides the linear
interpolation noise schedule used to train / sample the Wan latent world
model in this repo.

Convention (same as source):
  sigma = 0 -> clean sample, sigma = 1 -> pure noise.
  add_noise:        x_sigma = (1 - sigma) * x0 + sigma * noise
  training_target:  v* = noise - x0      (velocity from clean to noise)
  step (sampling):  x_{prev} = x + v * (sigma_next - sigma_cur)
"""

import math

import torch


class FlowMatchScheduler:
    def __init__(
        self,
        num_inference_steps=100,
        num_train_timesteps=1000,
        shift=3.0,
        sigma_max=1.0,
        sigma_min=0.003 / 1.002,
        inverse_timesteps=False,
        extra_one_step=False,
        reverse_sigmas=False,
        exponential_shift=False,
        exponential_shift_mu=None,
        shift_terminal=None,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.exponential_shift = exponential_shift
        self.exponential_shift_mu = exponential_shift_mu
        self.shift_terminal = shift_terminal
        self.training = False
        self.linear_timesteps_weights = None
        self.set_timesteps(num_inference_steps)

    def set_timesteps(
        self,
        num_inference_steps=100,
        denoising_strength=1.0,
        training=False,
        shift=None,
        dynamic_shift_len=None,
    ):
        if shift is not None:
            self.shift = shift
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength

        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])

        if self.exponential_shift:
            mu = (
                self.calculate_shift(dynamic_shift_len)
                if dynamic_shift_len is not None
                else self.exponential_shift_mu
            )
            self.sigmas = math.exp(mu) / (math.exp(mu) + (1 / self.sigmas - 1))
        else:
            self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)

        if self.shift_terminal is not None:
            one_minus_z = 1 - self.sigmas
            scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
            self.sigmas = 1 - (one_minus_z / scale_factor)

        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps

        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing
            self.training = True
        else:
            self.training = False

    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample

    def return_to_timestep(self, timestep, sample, sample_stablized):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        model_output = (sample - sample_stablized) / sigma
        return model_output

    def sigma_for(self, timestep):
        """Return sigma for each entry of `timestep`, preserving its shape."""
        ts = timestep.detach().cpu().reshape(-1)
        idx = torch.argmin((self.timesteps[:, None] - ts[None, :]).abs(), dim=0)
        return self.sigmas[idx].reshape(timestep.shape).to(timestep.device)

    def add_noise(self, original_samples, noise, timestep):
        """Interpolate between clean sample and noise.

        `timestep` is a per-frame tensor whose shape is a prefix of
        `original_samples` (e.g. (B, T) for samples (B, T, H, W, C)). sigma is
        looked up per element and broadcast over the trailing dims.
        """
        sigma = self.sigma_for(timestep).to(original_samples)
        while sigma.ndim < original_samples.ndim:
            sigma = sigma.unsqueeze(-1)
        return (1 - sigma) * original_samples + sigma * noise

    def training_target(self, sample, noise, timestep):
        return noise - sample

    def training_weight(self, timestep):
        if self.linear_timesteps_weights is None:
            return torch.ones_like(timestep)
        timestep_id = torch.argmin(
            (self.timesteps[:, None].to(timestep.device) - timestep[None]).abs(), dim=0
        )
        weights = self.linear_timesteps_weights.to(timestep.device)[timestep_id]
        return weights.to(timestep.device)

    def sample_timesteps(self, n, device):
        """Draw n training timesteps uniformly from the discrete schedule."""
        idx = torch.randint(0, len(self.timesteps), (n,), device="cpu")
        return self.timesteps[idx].to(device)

    def calculate_shift(
        self,
        image_seq_len,
        base_seq_len: int = 256,
        max_seq_len: int = 8192,
        base_shift: float = 0.5,
        max_shift: float = 0.9,
    ):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu
