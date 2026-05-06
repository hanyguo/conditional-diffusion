This repository contains the implementation of the conditional diffusion model used in the paper “Joint probabilistic inference of galaxy redshifts and rest-frame spectra from photometric fluxes with latent diffusion”.

The model learns the conditional distribution p(t | X), where X denotes the input photometric fluxes and t = [s; z] concatenates a spectral latent representation s with the galaxy redshift z. By drawing multiple samples from this distribution, it provides probabilistic predictions.

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

