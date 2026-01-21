#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Conditional DDPM/DDIM for Tabular Features
Predicting astrophysical features and redshift from flux

Usage:
    python cdmddim_modify.py --data_dir ./data/splits --cond_dim 143
    python cdmddim_modify.py --mode inference --checkpoint ./checkpoints/model.pth
    python cdmddim_modify.py --help
"""
import os
import math
import random
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# Argument Parser
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Conditional DDPM/DDIM for predicting astrophysical features from flux",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="/data/aai/scratch_ssd/hguo/DESI/",
                        help="Directory containing train.parquet, val.parquet, test.parquet")
    parser.add_argument("--cond_names", type=str, nargs="+",
                        default=['FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2'],
                        help="Condition column names (flux columns)")
    parser.add_argument("--target_names", type=str, nargs="+", 
                        default=['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'z'],
                        help="Target feature names")
    
    # Model arguments
    parser.add_argument("--hidden_dim", type=int, default=1024,
                        help="Hidden dimension of the denoiser network")
    parser.add_argument("--time_dim", type=int, default=256,
                        help="Time embedding dimension")
    parser.add_argument("--diffusion_steps", type=int, default=200,
                        help="Number of diffusion steps (T)")
    parser.add_argument("--beta_start", type=float, default=1e-4,
                        help="Starting beta value for noise schedule")
    parser.add_argument("--beta_end", type=float, default=2e-2,
                        help="Ending beta value for noise schedule")
    
    # Training arguments
    parser.add_argument("--mode", type=str, default="train", choices=["train", "inference"],
                        help="Run mode: train or inference only")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Maximum number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay for AdamW optimizer")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--ema_decay", type=float, default=0.9999,
                        help="EMA decay rate")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Random seed for reproducibility")
    
    # Sampling arguments
    parser.add_argument("--sampling_method", type=str, default="ddim", choices=["ddpm", "ddim"],
                        help="Sampling method: ddpm or ddim")
    parser.add_argument("--ddim_steps", type=int, default=50,
                        help="Number of DDIM sampling steps")
    parser.add_argument("--ddim_eta", type=float, default=1.0,
                        help="DDIM eta parameter (0=deterministic, >0=stochastic)")
    parser.add_argument("--num_samples", type=int, default=200,
                        help="Number of prediction samples per test instance")
    parser.add_argument("--pred_chunk", type=int, default=100000,
                        help="Chunk size for inference to avoid OOM")
    
    # I/O arguments
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/cond_ddpm_ema_best.pth",
                        help="Path to save/load model checkpoint")
    parser.add_argument("--output_dir", type=str, default="/data/aai/scratch/hguo/DESI/",
                        help="Directory to save prediction results")

    return parser.parse_args()


# =============================================================================
# Data Loading
# =============================================================================
def load_split_data(data_dir, cond_names, target_names):
    """
    Load train/val/test data from preprocessed Parquet files by column names.
    
    Args:
        data_dir: Directory containing train.parquet, val.parquet, test.parquet
        cond_names: List of condition column names
        target_names: List of target column names
    
    Returns:
        Xtr, ytr, Xval, yval, Xte, yte: Feature and target arrays for each subset
        id_tr, id_val, id_te: TARGETID arrays
    """
    train_df = pd.read_parquet(os.path.join(data_dir, 'train.parquet'))
    val_df = pd.read_parquet(os.path.join(data_dir, 'val.parquet'))
    test_df = pd.read_parquet(os.path.join(data_dir, 'test.parquet'))
    
    # Extract X and y by column names
    Xtr = train_df[cond_names].to_numpy(np.float32)
    ytr = train_df[target_names].to_numpy(np.float32)
    id_tr = train_df['TARGETID'].to_numpy()
    
    Xval = val_df[cond_names].to_numpy(np.float32)
    yval = val_df[target_names].to_numpy(np.float32)
    id_val = val_df['TARGETID'].to_numpy()
    
    Xte = test_df[cond_names].to_numpy(np.float32)
    yte = test_df[target_names].to_numpy(np.float32)
    id_te = test_df['TARGETID'].to_numpy()
    
    return Xtr, ytr, Xval, yval, Xte, yte, id_tr, id_val, id_te


class CondTabDataset(Dataset):
    """Dataset wrapper for condition-target pairs."""
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, i):
        return self.X[i], self.Y[i]


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

            sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev))
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


# =============================================================================
# Checkpoint Utilities
# =============================================================================
def save_checkpoint(path, ema_model, model, metadata=None):
    """Save model checkpoint."""
    ckpt = {
        'ema_state_dict': ema_model.state_dict(),
        'model_state_dict': model.state_dict(),
        'metadata': metadata or {},
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)
    print(f"Model saved to: {path}")


def load_checkpoint(path, ema_model, model, device):
    """Load model checkpoint."""
    ckpt = torch.load(path, map_location=device)
    if 'ema_state_dict' in ckpt:
        ema_model.load_state_dict(ckpt['ema_state_dict'])
    if 'model_state_dict' in ckpt:
        try:
            model.load_state_dict(ckpt['model_state_dict'])
        except Exception as e:
            print(f"Failed to load model weights, using EMA weights only: {e}")
    return ckpt.get('metadata', {})


@torch.no_grad()
def update_ema(model, ema_model, decay):
    """Update EMA model parameters."""
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


# =============================================================================
# Training and Evaluation
# =============================================================================
def train_epoch(model, ema_model, diff, dl_tr, opt, device, ema_decay):
    """Train for one epoch."""
    model.train()
    total = 0.0
    n = 0
    for cond_b, x0_b in dl_tr:
        cond_b = cond_b.to(device)
        x0_b = x0_b.to(device)
        B = x0_b.size(0)
        t = torch.randint(0, diff.T, (B,), device=device).long()
        noise = torch.randn_like(x0_b)
        x_t = diff.q_sample(x0_b, t, noise)
        eps_theta = model(x_t, t, cond_b)
        loss = nn.functional.mse_loss(eps_theta, noise)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        update_ema(model, ema_model, ema_decay)
        total += loss.item() * B
        n += B
    return total / n


@torch.no_grad()
def eval_recon(model, diff, X_src, y_src, device, feat_dim, args):
    """Evaluate reconstruction on validation/test set."""
    model.eval()
    N = min(2000, X_src.shape[0])
    cond = torch.from_numpy(X_src[:N]).to(device)
    
    sample_kwargs = dict(method=args.sampling_method, ddim_steps=args.ddim_steps, ddim_eta=args.ddim_eta)
    y_hat = diff.sample(model, cond, shape=(N, feat_dim), deterministic=False, **sample_kwargs)
    y_true = torch.from_numpy(y_src[:N]).to(device)

    mse = nn.functional.mse_loss(y_hat, y_true).item()
    rmse = float(np.sqrt(mse))
    return y_hat.cpu().numpy(), y_true.cpu().numpy(), mse, rmse


@torch.no_grad()
def reconstruct_on_test_multi(ema_model, diff, Xte, yte, device, feat_dim, args):
    """
    Multiple sampling (without averaging), returns predictions of shape (num_samples, N, feat_dim).
    Uses chunked inference to avoid OOM.
    """
    ema_model.eval()
        N_use = Xte.shape[0]
    chunk_size = max(1, args.pred_chunk)
    S = max(1, args.num_samples)
    y_hats = np.empty((S, N_use, feat_dim), dtype=np.float32)

    sample_kwargs = dict(method=args.sampling_method, ddim_steps=args.ddim_steps, ddim_eta=args.ddim_eta)

    for s in range(S):
        for start in range(0, N_use, chunk_size):
            end = min(N_use, start + chunk_size)
            cond_chunk = torch.from_numpy(Xte[start:end]).to(device)
            n_chunk = end - start
            y_hat_t = diff.sample(
                ema_model, cond_chunk,
                shape=(n_chunk, feat_dim),
                deterministic=False,
                **sample_kwargs,
            )
            y_hats[s, start:end] = y_hat_t.detach().cpu().numpy()

    y_true = yte[:N_use]
    return y_hats, y_true


def compute_metrics(y_pred, y_true):
    """Compute evaluation metrics (MSE, RMSE, MAE)."""
    err = y_pred - y_true
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae}


def save_predictions(y_preds_multi, y_true, X_cond, target_ids, cond_names, feature_names, out_path):
    """
    Save multiple prediction results, ground truth, and condition columns to Parquet file.
    
    Args:
        y_preds_multi: Predictions array of shape (num_samples, N, feat_dim)
        y_true: Ground truth array of shape (N, feat_dim)
        X_cond: Condition array of shape (N, cond_dim)
        target_ids: TARGETID array
        cond_names: List of condition column names
        feature_names: List of target feature names
        out_path: Output file path
    """
    S = y_preds_multi.shape[0]
    K = len(feature_names)
    
    # Build all data as a dict first, then create DataFrame at once (avoid fragmentation)
    data_dict = {'TARGETID': target_ids}
    
    # Add condition columns (ground truth flux)
    for j, col_name in enumerate(cond_names):
        data_dict[col_name] = X_cond[:, j]
    
    # Add ground truth targets (use original column names without prefix)
    for j, col_name in enumerate(feature_names):
        data_dict[col_name] = y_true[:, j]
    
    # Add multiple predictions
    for s in range(S):
        y_pred_s = y_preds_multi[s]
        for j, col_name in enumerate(feature_names):
            data_dict[f"pred{s+1}_{col_name}"] = y_pred_s[:, j]
    
    # Add prediction mean
    y_pred_mean = y_preds_multi.mean(axis=0)
    for j, col_name in enumerate(feature_names):
        data_dict[f"pred_mean_{col_name}"] = y_pred_mean[:, j]
    
    # Create DataFrame at once
    result_df = pd.DataFrame(data_dict)
    
    # Save
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result_df.to_parquet(out_path, index=False)
    print(f"Predictions saved to: {out_path} shape={result_df.shape}")


# =============================================================================
# Main Function
# =============================================================================
def main():
    args = parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from: {args.data_dir}")
    Xtr, ytr, Xval, yval, Xte, yte, id_tr, id_val, id_te = load_split_data(args.data_dir, args.cond_names, args.target_names)
    print(f"Train: {Xtr.shape} -> {ytr.shape} | Val: {Xval.shape} -> {yval.shape} | Test: {Xte.shape} -> {yte.shape}")
    
    # Create datasets and dataloaders
    ds_tr = CondTabDataset(Xtr, ytr)
    ds_val = CondTabDataset(Xval, yval)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)
    
    # Model dimensions
    feat_dim = ytr.shape[1]
    cond_dim = Xtr.shape[1]
    
    # Initialize models
    model = CondDenoiser(feat_dim=feat_dim, cond_dim=cond_dim, hidden=args.hidden_dim, time_dim=args.time_dim).to(device)
    diff = Diffusion(T=args.diffusion_steps, beta_start=args.beta_start, beta_end=args.beta_end, device=device)
    
    # EMA model
    ema_model = CondDenoiser(feat_dim=feat_dim, cond_dim=cond_dim, hidden=args.hidden_dim, time_dim=args.time_dim).to(device)
    ema_model.load_state_dict(model.state_dict())
    for p in ema_model.parameters():
        p.requires_grad_(False)
    
    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Training or loading
    skipped_training = False
    if args.mode == "inference" or os.path.isfile(args.checkpoint):
        if os.path.isfile(args.checkpoint):
            try:
                meta = load_checkpoint(args.checkpoint, ema_model, model, device)
                skipped_training = True
                print(f"Loaded existing model from {args.checkpoint}, skipping training.")
            except Exception as e:
                if args.mode == "inference":
                    raise RuntimeError(f"Inference mode requires valid checkpoint: {e}")
                print(f"Failed to load model, will retrain: {e}")

    # Training loop
    if not skipped_training and args.mode == "train":
        best_rmse = float('inf')
        best_state = None
        epochs_no_improve = 0
        
        for ep in range(1, args.epochs + 1):
            tr_loss = train_epoch(model, ema_model, diff, dl_tr, opt, device, args.ema_decay)
            _, _, mse_val, rmse_val = eval_recon(ema_model, diff, Xval, yval, device, feat_dim, args)
            print(f"[Ep{ep:03d}] train_noiseMSE={tr_loss:.6f}  val_recon MSE={mse_val:.6f} RMSE={rmse_val:.6f}")

            # Early stopping (based on EMA model validation RMSE)
            if rmse_val < best_rmse - 1e-6:
                best_rmse = rmse_val
                best_state = {k: v.detach().cpu().clone() for k, v in ema_model.state_dict().items()}
                epochs_no_improve = 0
else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(f"Early stopping at epoch {ep} (patience={args.patience}).")
                    break

        # Restore best EMA model
        if best_state is not None:
            ema_model.load_state_dict(best_state)
        print(f"Best val RMSE (original scale): {best_rmse}")

        # Save checkpoint
        save_checkpoint(args.checkpoint, ema_model, model, metadata={
            'feat_dim': int(feat_dim),
            'cond_dim': int(cond_dim),
            'T': int(diff.T),
            'beta_start': float(diff.betas[0].detach().cpu().item()),
            'beta_end': float(diff.betas[-1].detach().cpu().item()),
            'seed': int(args.seed),
        })

    # Inference
    print("Running inference on test set...")
    y_preds_multi, y_true = reconstruct_on_test_multi(ema_model, diff, Xte, yte, device, feat_dim, args)
    
    # Compute metrics using mean of predictions
    y_pred_mean = y_preds_multi.mean(axis=0)
    n_eval = min(2000, y_pred_mean.shape[0])
    metrics = compute_metrics(y_pred_mean[:n_eval], y_true[:n_eval])
    print(f"Test metrics (original scale): {metrics}")
    
    # Save predictions (including condition columns)
    out_path = os.path.join(args.output_dir, f"predictions_S{args.num_samples}.parquet")
    save_predictions(y_preds_multi, y_true, Xte, id_te, args.cond_names, args.target_names, out_path)
    
    print("Done!")


if __name__ == "__main__":
    main()
