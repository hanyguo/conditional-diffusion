Codes for “Joint probabilistic inference of galaxy redshifts and rest-frame spectra from photometric fluxes with latent diffusion”.

A PyTorch implementation of a Conditional Denoising Diffusion Probabilistic Model (DDPM/DDIM) for predicting astrophysical properties and redshift from photometric flux data.

This model learns the conditional distribution $p(\mathbf{y} | \mathbf{x})$, where $\mathbf{x}$ is the input photometry (fluxes) and $\mathbf{y}$ are the target astrophysical features. By sampling from this distribution, it provides principled uncertainty estimation and handles the multi-modal nature of the inverse problem.

## Features

- **Conditional Generation**: Conditioned on 5 photometric bands (G, R, Z, W1, W2).
- **Tabular Diffusion**: Adapted for continuous tabular data using MLP architectures.
- **Fast Sampling**: Supports DDIM sampling for accelerated inference (50 steps).
- **Ensemble Prediction**: Generates multiple samples per object to capture posterior uncertainty.
- **Robust Training**: Includes EMA (Exponential Moving Average) and early stopping.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cdm-galaxy.git
   cd cdm-galaxy
   ```

2. Install dependencies:
   ```bash
   # Install PyTorch (adjust for your CUDA version)
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   
   # Install other requirements
   pip install -r requirements.txt
   ```

## Data Format

The code expects data in Parquet format with the following columns:
- **Conditions**: `FLUX_G`, `FLUX_R`, `FLUX_Z`, `FLUX_W1`, `FLUX_W2`
- **Targets**: `f1`...`f10`, `z` (redshift)
- **ID**: `TARGETID`

Prepare your data splits in a directory (e.g., `./data/splits/`):
- `train.parquet`
- `val.parquet`
- `test.parquet`

## Usage

### Training

Train the model on your dataset:

```bash
python train_and_sampling.py --data_dir ./data/splits --epochs 50 --batch_size 1024
```

Key arguments:
- `--data_dir`: Path to directory containing parquet files.
- `--output_dir`: Directory to save checkpoints and predictions.
- `--cond_names`: List of condition column names.
- `--target_names`: List of target column names.

### Sampling (Inference)

To generate predictions using a trained model checkpoint:

```bash
python train_and_sampling.py \
    --checkpoint ./checkpoints/cdm_best.pth \
    --output_dir ./results \
    --num_samples 200 \
    --sampling_method ddim \
    --ddim_steps 50
```

This will:
1. Load the model from `--checkpoint`.
2. Skip training (since checkpoint exists).
3. Generate 200 samples for each test object.
4. Save results to `results/samples_200.parquet`.

## Model Architecture

The model uses a conditional noise prediction network $\epsilon_\theta(x_t, t, c)$:
- **Condition Encoder**: 4-layer MLP (1024 dim) processes flux.
- **Time Embedding**: Sinusoidal embedding (256 dim).
- **Denoising Trunk**: 5-layer ResNet-style MLP (1024 dim).

Training uses the standard DDPM objective with continuous time steps. Sampling uses DDIM ($\eta=1.0$) for efficiency.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{YourName2026,
  title={Conditional Diffusion for Galaxy Properties},
  author={Your Name and Collaborators},
  journal={arXiv preprint},
  year={2026}
}
```
