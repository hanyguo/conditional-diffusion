#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model architectures for Conditional DDPM/DDIM

Contains:
    - Diffusion: DDPM/DDIM forward and reverse diffusion process
    - CondDenoiser: Conditional noise prediction network
    - SinusoidalTimeEmbedding: Sinusoidal timestep embedding
"""
import math
import torch
import torch.nn as nn


# =============================================================================
# Diffusion Components
# =============================================================================
def make_beta_schedule(T=200, beta_start=1e-4, beta_end=2e-2):
    """Create linear beta schedule with T steps."""
    return torch.linspace(beta_start, beta_end, T)


class Diffusion:
    """DDPM forward/reverse diffusion process for continuous values with conditional inference."""
    
    def __init__(self, T=200, beta_start=1e-4, beta_end=2e-2, device='cpu'):
        self.device = device
        self.T = T
        betas = make_beta_schedule(T, beta_start, beta_end).to(device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]], dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))

    def q_sample(self, x0, t, noise=None):
        """Forward diffusion: q(x_t | x_0) = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * eps"""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        return sqrt_ab * x0 + sqrt_om * noise

    def p_mean_variance(self, model, x_t, t, cond):
        """Compute mean from eps_theta: mu_t = 1/sqrt(alpha_t) * (x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * eps_theta)"""
        betas_t = self.betas[t].view(-1, 1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].view(-1, 1)
        sqrt_one_minus_ab_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)

        eps_theta = model(x_t, t, cond)
        model_mean = sqrt_recip_alphas_t * (x_t - betas_t * eps_theta / sqrt_one_minus_ab_t)
        posterior_var = self.posterior_variance[t].view(-1, 1)
        posterior_log_var = self.posterior_log_variance_clipped[t].view(-1, 1)
        return model_mean, posterior_var, posterior_log_var

    @torch.no_grad()
    def sample(self, model, cond, shape, deterministic=False, method='ddpm', ddim_steps=None, ddim_eta=0.0):
        """Sample x_0 (features) given condition, supports DDPM/DDIM."""
        if method == 'ddim':
            steps = self.T if ddim_steps is None else max(1, min(self.T, int(ddim_steps)))
            eta = 0.0 if deterministic else float(ddim_eta)
            return self.ddim_sample(model, cond, shape, steps=steps, eta=eta)
        return self.ddpm_sample(model, cond, shape, deterministic=deterministic)

    @torch.no_grad()
    def ddpm_sample(self, model, cond, shape, deterministic=False):
        """Standard DDPM sampling."""
        model.eval()
        N = shape[0]
        x_t = torch.randn(shape, device=self.device)
        for i in reversed(range(self.T)):
            t = torch.full((N,), i, device=self.device, dtype=torch.long)
            mean, var, logvar = self.p_mean_variance(model, x_t, t, cond)
            if i > 0:
                if deterministic:
                    x_t = mean
                else:
                    noise = torch.randn_like(x_t)
                    x_t = mean + torch.exp(0.5 * logvar) * noise
            else:
                x_t = mean
        return x_t

    @torch.no_grad()
    def ddim_sample(self, model, cond, shape, steps=50, eta=0.0):
        """DDIM accelerated sampling. steps: number of timesteps (<=T), eta=0 for deterministic."""
        model.eval()
        N = shape[0]
        steps = max(1, min(self.T, int(steps)))
        # Uniformly spaced timesteps
        timesteps = torch.linspace(0, self.T - 1, steps, device=self.device).long().flip(0)
        x = torch.randn(shape, device=self.device)
        one_tensor = torch.tensor(1.0, device=self.device)
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((N,), t.item(), device=self.device, dtype=torch.long)
            eps = model(x, t_batch, cond)

            alpha_bar = self.alphas_cumprod[t]
            if i + 1 < len(timesteps):
                prev_t = timesteps[i + 1]
                alpha_bar_prev = self.alphas_cumprod[prev_t]
            else:
                alpha_bar_prev = one_tensor

            sqrt_alpha_bar = torch.sqrt(alpha_bar)
            sqrt_one_minus = torch.sqrt(1 - alpha_bar)
            sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)

            pred_x0 = (x - sqrt_one_minus * eps) / sqrt_alpha_bar

            if torch.isclose(alpha_bar_prev, one_tensor):
                x = pred_x0
                continue

            sigma = eta * torch.sqrt(torch.clamp((1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev), min=0.0))
            noise_term = 0.0
            if eta > 0:
                noise_term = sigma * torch.randn_like(x)
            dir_xt = torch.sqrt(torch.clamp(1 - alpha_bar_prev - sigma**2, min=0.0)) * eps
            x = sqrt_alpha_bar_prev * pred_x0 + dir_xt + noise_term
        return x


# =============================================================================
# Model Components
# =============================================================================
class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


class CondDenoiser(nn.Module):
    """Conditional noise predictor eps_theta(x_t, t, cond)."""
    
    def __init__(self, feat_dim, cond_dim, hidden=256, time_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(feat_dim + time_dim + hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, feat_dim),
        )

    def forward(self, x_t, t, cond):
        t_embed = self.time_mlp(t)
        c_embed = self.cond_mlp(cond)
        h = torch.cat([x_t, t_embed, c_embed], dim=1)
        eps = self.net(h)
        return eps
