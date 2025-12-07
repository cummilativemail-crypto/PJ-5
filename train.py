# ============================================================
# train.py
# Multimodal Time-Dependent Implicit Priors for Diffusion
# - Datasets: MNIST, CIFAR-10, CelebA (torchvision)
# - UNet2DModel + DDPMScheduler (diffusers)
# - Implicit conditioning via multimodal, time-dependent priors
#   encoded in the input channel (no extra network params)
# - KL-style regularizer on predicted noise (no extra params)
# - Saves checkpoints and config for later evaluation
# ============================================================

import os
import math
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm

from diffusers import DDPMScheduler, UNet2DModel
from copy import deepcopy


# ======================
# 0. Config dict
# ======================

config = {
    # ======================
    # Dataset and paths
    # ======================
    "dataset": "cifar10",           # "mnist" | "cifar10" | "celeba"
    "celeba_root": "./data/celeba", # used only if dataset == "celeba"
    "output_dir": "./results_multimodal_implicit",

    # ======================
    # Optimization / training
    # ======================
    "num_epochs": 1,
    "batch_size": 64,
    "learning_rate": 2e-4,
    "weight_decay": 0.0,
    "betas": (0.9, 0.999),
    "seed": 42,

    # ======================
    # Diffusion schedule
    # ======================
    "num_train_timesteps": 1000,  # training T
    "beta_start": 0.0001,
    "beta_end": 0.02,
    "beta_schedule": "linear",

    # ======================
    # Multimodal prior geometry
    # ======================
    "c_bern": 1.0,    # scaling for Bernoulli component
    "c_fourier": 0.3, # scaling for Fourier component
    "c_wavelet": 0.3, # scaling for wavelet-like (blob) component

    # ======================
    # Logging / checkpoints
    # ======================
    "save_checkpoint_every": 2000, # global steps
    "log_every": 200,              # global steps

    # ======================
    # KL-style noise regularization
    # ======================
    "lambda_kl_noise": 1e-4,       # weight on ||eps_pred||^2 term

    # ======================
    # Evaluation / FID settings
    # ======================
    "fid_num_fake": 5000,          # total fake images for FID
    "fid_batch": 16,               # fake batch size for FID
    "fid_steps": 250,              # reverse steps for FID sampling (<= num_train_timesteps)
}


os.makedirs(config["output_dir"], exist_ok=True)
os.makedirs(os.path.join(config["output_dir"], "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(config["output_dir"], "samples"), exist_ok=True)

# Save config as JSON for eval script
with open(os.path.join(config["output_dir"], "config.json"), "w") as f:
    json.dump(config, f, indent=2)

torch.manual_seed(config["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


# ======================
# 1. Dataset-specific setup
# ======================

def get_dataset_and_model_cfg(cfg):
    ds = cfg["dataset"].lower()
    if ds == "mnist":
        img_size = 28
        in_channels = 1
        num_classes = 10
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test_set  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    elif ds == "cifar10":
        img_size = 32
        in_channels = 3
        num_classes = 10
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        test_set  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    elif ds == "celeba":
        img_size = 64
        in_channels = 3
        num_classes = 10  # 10 attribute-based groups
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        train_set = datasets.CelebA(
            root=cfg["celeba_root"], split="train",
            target_type="attr", download=True, transform=transform
        )
        test_set = datasets.CelebA(
            root=cfg["celeba_root"], split="valid",
            target_type="attr", download=True, transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {cfg['dataset']}")
    return train_set, test_set, img_size, in_channels, num_classes

train_set, test_set, img_size, in_channels, num_classes = get_dataset_and_model_cfg(config)

train_loader = DataLoader(
    train_set,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)

print("Train size:", len(train_set), "Test size:", len(test_set))

def get_labels(batch_targets, cfg):
    ds = cfg["dataset"].lower()
    if ds in ["mnist", "cifar10"]:
        return batch_targets
    elif ds == "celeba":
        attrs = batch_targets  # Bx40
        codes = (attrs[:, :4] > 0).int()
        ints = codes[:, 0] + 2 * codes[:, 1] + 4 * codes[:, 2] + 8 * codes[:, 3]
        return (ints % num_classes)
    else:
        raise ValueError


# ==================================
# 2. UNet + DDPM scheduler (Diffusers)
# ==================================

model = UNet2DModel(
    sample_size=img_size,
    in_channels=in_channels,
    out_channels=in_channels,
    layers_per_block=1,
    block_out_channels=(64, 128, 256),
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
    norm_num_groups=8,
).to(device)

noise_scheduler = DDPMScheduler(
    num_train_timesteps=config["num_train_timesteps"],
    beta_start=config["beta_start"],
    beta_end=config["beta_end"],
    beta_schedule=config["beta_schedule"],
)

for buffer_name in ["betas", "alphas", "alphas_cumprod", "final_alpha_cumprod", "one"]:
    if hasattr(noise_scheduler, buffer_name):
        cur = getattr(noise_scheduler, buffer_name)
        if isinstance(cur, torch.Tensor):
            setattr(noise_scheduler, buffer_name, cur.to(device))

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["learning_rate"],
    betas=config["betas"],
    weight_decay=config["weight_decay"],
)


# ==============================================
# 3. Multimodal, time-dependent implicit priors
# ==============================================

H, W = img_size, img_size

# 3.1 Bernoulli component
mu_bern = {}
for k in range(num_classes):
    z = torch.bernoulli(0.5 * torch.ones(1, H, W))
    z = (2 * z - 1) * config["c_bern"]
    mu_bern[k] = z.to(device)

# 3.2 Fourier component
def build_fourier_map(k, H, W, scale):
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, H),
        torch.linspace(0, 1, W),
        indexing="ij",
    )
    base_omega = 2 * math.pi
    phase = (k / max(1, num_classes - 1)) * 2 * math.pi
    pattern = torch.cos(base_omega * xx + phase) + torch.sin(base_omega * yy - phase)
    pattern = pattern - pattern.mean()
    pattern = pattern / (pattern.std() + 1e-6)
    pattern = pattern * scale
    return pattern.unsqueeze(0)

mu_fourier = {}
for k in range(num_classes):
    mu_fourier[k] = build_fourier_map(k, H, W, config["c_fourier"]).to(device)

# 3.3 Wavelet-like component (Gaussian blobs)
def build_wavelet_like_map(k, H, W, scale):
    pattern = torch.zeros(1, H, W)
    num_blobs = 3
    for b in range(num_blobs):
        cx = (H // (num_blobs + 1)) * (b + 1)
        cy = (W // (num_blobs + 1)) * ((k + b) % num_blobs + 1)
        sigma = H / 8.0
        yy, xx = torch.meshgrid(
            torch.arange(H), torch.arange(W), indexing="ij"
        )
        gauss = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        pattern[0] += gauss
    pattern = pattern - pattern.mean()
    pattern = pattern / (pattern.std() + 1e-6)
    pattern = pattern * scale
    return pattern

mu_wavelet = {}
for k in range(num_classes):
    mu_wavelet[k] = build_wavelet_like_map(k, H, W, config["c_wavelet"]).to(device)

# 3.4 Time-dependent combination
def build_mu_class_time_dependent():
    T = config["num_train_timesteps"]
    mu_t = {}
    timesteps = torch.linspace(0.0, 1.0, T)
    for k in range(num_classes):
        mu_kt = []
        for tau in timesteps:
            g = (1.0 - tau)  # Fourier weight
            h = tau          # wavelet weight
            m = mu_bern[k] + g * mu_fourier[k] + h * mu_wavelet[k]
            m_centered = m - m.mean()
            m_norm = m_centered / (m_centered.std() + 1e-6)
            target_std = mu_bern[k].std().item()
            m_final = m_norm * target_std
            mu_kt.append(m_final)
        mu_t[k] = torch.stack(mu_kt, dim=0).to(device)  # (T,1,H,W)
    return mu_t

mu_time = build_mu_class_time_dependent()

def sample_prior_ids(labels):
    return labels.clamp(0, num_classes - 1)

def sample_xT_from_prior(batch_size, prior_ids):
    T = config["num_train_timesteps"]
    C_img = in_channels
    z = torch.randn((batch_size, C_img, H, W), device=device)
    muT = torch.stack([mu_time[int(pid)][T - 1] for pid in prior_ids])  # (B,1,H,W)
    if C_img > 1:
        muT = muT.repeat(1, C_img, 1, 1)
    return z + muT


# ========================
# 4. Training loop (MSE + KL-style noise regularizer)
# ========================

global_step = 0
model.train()

lambda_kl = config["lambda_kl_noise"]
mse_loss_fn = nn.MSELoss()

for epoch in range(config["num_epochs"]):
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}", leave=False)
    for imgs, labels in pbar:
        imgs = imgs.to(device)
        labels = get_labels(labels.to(device), config)

        bsz = imgs.size(0)
        prior_ids = sample_prior_ids(labels)
        _ = sample_xT_from_prior(bsz, prior_ids)  # conceptual

        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (bsz,), device=device, dtype=torch.long
        )

        with torch.no_grad():
            alphas_cumprod = noise_scheduler.alphas_cumprod[timesteps]
            sqrt_alphas_cumprod = alphas_cumprod.sqrt().view(-1, 1, 1, 1)
            sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod).sqrt().view(-1, 1, 1, 1)

            eps0 = torch.randn_like(imgs)

            mu_t = torch.stack(
                [mu_time[int(pid)][int(t)] for pid, t in zip(prior_ids, timesteps)]
            ).to(device)  # (B,1,H,W)
            if in_channels > 1:
                mu_t = mu_t.repeat(1, in_channels, 1, 1)

            eps_t = eps0 + mu_t
            x_t = sqrt_alphas_cumprod * imgs + sqrt_one_minus_alphas_cumprod * eps_t

        # Predicted standard Gaussian noise
        model_pred = model(x_t, timesteps).sample

        # Standard DDPM noise MSE loss
        loss_mse = mse_loss_fn(model_pred, eps0)

        # KL-style regularizer on predicted noise: ||eps_pred||^2
        loss_kl = (model_pred ** 2).mean()

        # Total loss
        loss = loss_mse + lambda_kl * loss_kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_step += 1

        if global_step % config["log_every"] == 0:
            pbar.set_postfix(
                loss=loss.item(),
                loss_mse=loss_mse.item(),
                loss_kl=loss_kl.item(),
                step=global_step,
            )

        if global_step % config["save_checkpoint_every"] == 0:
            ckpt_path = os.path.join(
                config["output_dir"], "checkpoints",
                f"{config['dataset']}_ckpt_step_{global_step}.pt"
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_step": global_step,
                    "epoch": epoch,
                    "config": config,
                    "mu_time": {k: v.cpu() for k, v in mu_time.items()},
                },
                ckpt_path,
            )

    print(
        f"[{config['dataset']}][Epoch {epoch+1}] "
        f"last batch total={loss.item():.4f}, mse={loss_mse.item():.4f}, kl={loss_kl.item():.4f}"
    )

# Save final checkpoint
final_ckpt = os.path.join(
    config["output_dir"], "checkpoints",
    f"{config['dataset']}_final.pt"
)
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "epoch": config["num_epochs"] - 1,
        "config": config,
        "mu_time": {k: v.cpu() for k, v in mu_time.items()},
    },
    final_ckpt,
)
print("Saved final checkpoint to:", final_ckpt)