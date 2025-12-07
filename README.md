# PJ-5

# Evidential Multimodal Implicit Priors for Time-Structured Diffusion Models

## Training

Edit the `config` dictionary in `train.py` to set:
- `dataset`: "mnist" | "cifar10" | "celeba"
- `num_epochs`, `batch_size`, `learning_rate`
- Multimodal prior scaling: `c_bern`, `c_fourier`, `c_wavelet`
- Other hyperparameters as needed

Run training:
```bash
python train.py
```

Checkpoints and config are saved to `./results_multimodal_implicit/`

## Evaluation

After training, evaluate the model:
```bash
python eval.py
```

The script will:
- Load the trained checkpoint from the training output directory
- Generate samples and save them
- Compute FID score against the test set

Adjust `output_dir` in `eval.py` if your results are in a different location.