import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models import Diffusion, CondDenoiser


# =============================================================================
# Argument Parser
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Conditional DDPM/DDIM for predicting astrophysical features from flux",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="./data",
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
                        help="Chunk size for sampling to avoid OOM")
    parser.add_argument("--eval_samples", type=int, default=2000,
                        help="Number of samples for validation/test metrics (<=0 for all)")
    
    # I/O arguments
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/model.pth",
                        help="Path to save/load model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./outputs",
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
    """Evaluate reconstruction on validation/test set using all data (chunked)."""
    model.eval()
    if args.eval_samples and args.eval_samples > 0:
        N = min(int(args.eval_samples), X_src.shape[0])
    else:
        N = X_src.shape[0]
    chunk_size = max(1, args.pred_chunk)
    
    sample_kwargs = dict(method=args.sampling_method, ddim_steps=args.ddim_steps, ddim_eta=args.ddim_eta)
    
    total_mse = 0.0
    total_samples = 0
    
    # Process in chunks to avoid OOM
    for start in range(0, N, chunk_size):
        end = min(N, start + chunk_size)
        cond_chunk = torch.from_numpy(X_src[start:end]).to(device)
        y_true_chunk = torch.from_numpy(y_src[start:end]).to(device)
        n_chunk = end - start
        
        # Single sample (deterministic=False)
        y_hat_chunk = diff.sample(
            model, cond_chunk, 
            shape=(n_chunk, feat_dim), 
            deterministic=False, 
            **sample_kwargs
        )
        
        # Accumulate MSE
        mse_chunk = nn.functional.mse_loss(y_hat_chunk, y_true_chunk, reduction='sum').item()
        total_mse += mse_chunk
        total_samples += n_chunk * feat_dim

    mse = total_mse / total_samples
    rmse = float(np.sqrt(mse))
    return mse, rmse


@torch.no_grad()
def reconstruct_on_test_multi(ema_model, diff, Xte, yte, device, feat_dim, args):
    """
    Multiple sampling (without averaging), returns predictions of shape (num_samples, N, feat_dim).
    Uses chunked sampling to avoid OOM.
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
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
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
    if os.path.isfile(args.checkpoint):
        try:
            meta = load_checkpoint(args.checkpoint, ema_model, model, device)
            skipped_training = True
            print(f"Loaded existing model from {args.checkpoint}, skipping training.")
        except Exception as e:
            print(f"Failed to load model, will retrain: {e}")
    
    # Training loop
    if not skipped_training:
        best_rmse = float('inf')
        best_state = None
        epochs_no_improve = 0
        
        for ep in range(1, args.epochs + 1):
            tr_loss = train_epoch(model, ema_model, diff, dl_tr, opt, device, args.ema_decay)
            mse_val, rmse_val = eval_recon(ema_model, diff, Xval, yval, device, feat_dim, args)
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

    # Sampling
    print("Running sampling on test set...")
    y_preds_multi, y_true = reconstruct_on_test_multi(ema_model, diff, Xte, yte, device, feat_dim, args)
    
    # Compute metrics using single sample (first sample) for fair comparison with validation
    # Use configured evaluation subset (default 2000)
    if args.eval_samples and args.eval_samples > 0:
        n_eval = min(int(args.eval_samples), y_true.shape[0])
    else:
        n_eval = y_true.shape[0]
    y_pred_single = y_preds_multi[0][:n_eval]  # Take the first sample
    metrics = compute_metrics(y_pred_single, y_true[:n_eval])
    print(f"Test metrics (single sample, all data): {metrics}")
    
    # Also compute metrics using mean of predictions (optional, for reference)
    y_pred_mean = y_preds_multi.mean(axis=0)
    metrics_mean = compute_metrics(y_pred_mean[:n_eval], y_true[:n_eval])
    print(f"Test metrics (mean of {args.num_samples} samples, all data): {metrics_mean}")
    
    # Save predictions (including condition columns)
    out_path = os.path.join(args.output_dir, f"samples_{args.num_samples}.parquet")
    save_predictions(y_preds_multi, y_true, Xte, id_te, args.cond_names, args.target_names, out_path)
    
    print("Done!")


if __name__ == "__main__":
    main()
