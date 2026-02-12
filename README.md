Codes for “Joint probabilistic inference of galaxy redshifts and rest-frame spectra from photometric fluxes with latent diffusion”.

A PyTorch implementation of a Conditional Denoising Diffusion Probabilistic Model for predicting redshift and other features from photometric fluxes.

This model learns the conditional distribution $p(\mathbf{y} | \mathbf{x})$, where $\mathbf{x}$ is the input photometry (fluxes) and $\mathbf{y}$ are the target features. By sampling from this distribution, it provides principled uncertainty estimation.

## Usage


```bash
python train_and_sampling.py --data_dir ./data/splits --epochs 50 --batch_size 1024
```

Key arguments:
- `--data_dir`: Path to directory containing parquet files.
- `--output_dir`: Directory to save checkpoints and predictions.
- `--cond_names`: List of condition column names.
- `--target_names`: List of target column names.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

