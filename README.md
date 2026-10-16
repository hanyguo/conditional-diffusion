# Conditional diffusion for joint redshift and spectral latent inference

This repository contains the implementation of the **conditional diffusion model** used in:

> Guo, H.-Y. & Eriksen, M.  
> *Joint probabilistic inference of galaxy redshifts and rest-frame spectra from photometric fluxes with latent diffusion*  
> [arXiv:2605.10753](https://arxiv.org/abs/2605.10753)


This repository trains and samples the conditional diffusion model. Given broadband photometric fluxes $\mathbf{x}$, the model learns the conditional distribution $p(\mathbf{t}_0 \mid \mathbf{x})$, where $\mathbf{t}_0 = [\mathbf{s}; z]$ concatenates a spectral latent representation $\mathbf{s}$ with the galaxy redshift $z$. Drawing multiple samples from this distribution provides probabilistic predictions of redshift and spectral latents. Sampled latents can be decoded into rest-frame spectra with a `spender` decoder (Melchior et al. 2023); the public [`spender`](https://github.com/pmelchior/spender) repository provides the autoencoder implementation used for that step.


## Model architecture

The diffusion process follows the standard DDPM formulation, with a linear noise schedule. At inference time, samples are drawn with the DDIM sampler.

The noise-prediction network (`CondDenoiser`) is an MLP. The diffusion timestep is encoded by a sinusoidal embedding followed by a linear layer and a SiLU activation, and the photometric fluxes are encoded by a dedicated MLP. In the default setting, these embeddings are **concatenated** with the noisy target vector and passed through a denoising MLP with SiLU activations, which predicts the noise.
An alternative conditioning scheme based on **adaptive layer normalization (AdaLN)** is also implemented (`--conditioning_method adaln`). 
## Repository contents

- `models.py` - diffusion process (DDPM/DDIM) and conditional denoiser
- `train_and_sampling.py` - training, checkpointing, and multi-sample inference

## Data format

Place three Parquet files in `--data_dir`:

```text
data/
  train.parquet
  val.parquet
  test.parquet
```

Each file should contain:

- condition columns (default): `FLUX_G`, `FLUX_R`, `FLUX_Z`, `FLUX_W1`, `FLUX_W2`
- target columns (default): `f1`, `f2`, ..., `f10`, and `z`
- `TARGETID` (used when saving predictions)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python train_and_sampling.py --data_dir ./data --output_dir ./outputs
```

If `--checkpoint` already exists and matches the requested configuration, training is skipped and the script proceeds to sampling.

### Key arguments

- `--data_dir`: directory with `train.parquet`, `val.parquet`, `test.parquet`
- `--output_dir`: directory for prediction files
- `--checkpoint`: path to save/load model weights
- `--cond_names`: photometric condition columns
- `--target_names`: target columns (default: `f1` ... `f10` and `z`)
- `--conditioning_method`: `concat` (default) or `adaln`
- `--num_samples`: number of posterior draws per test galaxy

See `python train_and_sampling.py --help` for the full list.

Predictions are saved under `--output_dir` as a Parquet file containing the input fluxes, ground-truth targets, individual draws, and the mean over draws.

## Related code

- **spender** spectral autoencoder: https://github.com/pmelchior/spender

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
