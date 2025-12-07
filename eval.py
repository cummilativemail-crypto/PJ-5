# ============================================================
# eval.py
# Evaluation script for Multimodal Time-Dependent Implicit Priors
# - Loads config.json and a checkpoint from output_dir
# - Uses saved time-dependent priors (mu_time)
# - Runs sampling and FID evaluation (torchmetrics)
# ============================================================

import os
import json
import math
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm

from diffusers import DDPMScheduler, UNet2DModel
from torchmetrics.image.fid import FrechetInceptionDistance
from copy import deepcopy


# ======================
# 1. Load config and checkpoint
# ======================

output_dir = "./results_multimodal_implicit"  # adjust if needed
config_path = os.path.join(output_dir, "config.json")
with open(config_path, "r") as f:
    config = json.load(f)

# Provide safe defaults for FID-related keys if missing
config.setdefault("fid_num_fake", 5000)
config.setdefault("fid_batch", 16)
config.setdefault("fid_steps", 250)

ckpt_name = f"{config['dataset']}_final.pt"
ckpt_path = os.path.join(output_dir, "checkpoints", ckpt_name)
checkpoint = torch.load(ckpt_path, map_location="cpu")

print("Loaded checkpoint:", ckpt_path)

torch.manual_seed(config["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


# ======================
# 2. Dataset-specific setup (same as training)
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
        num_classes = 10
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

test_loader = DataLoader(
    test_set,
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

print("Test size:", len(test_set))


# ======================
# 3. Rebuild model and scheduler, load weights
# ======================

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

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

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

# Load time-dependent priors
mu_time = {int(k): v.to(device) for k, v in checkpoint["mu_time"].items()}
H, W = img_size, img_size


# ======================
# 4. Helper functions
# ======================

def get_labels(batch_targets, cfg):
    ds = cfg["dataset"].lower()
    if ds in ["mnist", "cifar10"]:
        return batch_targets
    elif ds == "celeba":
        attrs = batch_targets
        codes = (attrs[:, :4] > 0).int()
        ints = codes[:, 0] + 2 * codes[:, 1] + 4 * codes[:, 2] + 8 * codes[:, 3]
        return (ints % num_classes)
    else:
        raise ValueError

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

@torch.no_grad()
def sample_from_prior(prior_id, num_samples=64, num_steps=None):
    model.eval()
    if num_steps is None:
        num_steps = noise_scheduler.config.num_train_timesteps

    prior_ids = torch.full((num_samples,), prior_id, device=device, dtype=torch.long)
    x = sample_xT_from_prior(num_samples, prior_ids)

    local_scheduler = deepcopy(noise_scheduler)
    local_scheduler.set_timesteps(num_steps, device=device)
    timesteps = local_scheduler.timesteps

    for t in tqdm(timesteps, total=len(timesteps), leave=False,
                  desc=f"Sampling (prior {prior_id})"):
        tt = torch.full((num_samples,), t, device=device, dtype=torch.long)
        eps0_pred = model(x, tt).sample
        x = local_scheduler.step(eps0_pred, t.item(), x).prev_sample

    x = x.clamp(-1, 1)
    x = (x + 1) / 2.0
    return x


# Save some grids for visual inspection
samples_dir = os.path.join(output_dir, "samples_eval")
os.makedirs(samples_dir, exist_ok=True)
for pid in range(min(num_classes, 10)):
    imgs = sample_from_prior(pid, num_samples=16, num_steps=config["num_train_timesteps"])
    save_image(imgs, os.path.join(samples_dir, f"{config['dataset']}_eval_prior{pid}.png"), nrow=4)
print("Saved eval sample grids to:", samples_dir)


# ======================
# 5. FID evaluation (batch size >= 2)
# ======================

def to_fid_tensor(x):
    if x.min() < 0:
        x = (x + 1) / 2.0
    x = x.clamp(0, 1)
    if x.size(1) == 1:
        x = x.repeat(1, 3, 1, 1)
    x = (x * 255.0).clamp(0, 255).byte()
    return x

def ensure_min_batch_two(x):
    # x: (B, C, H, W), B>=1
    if x.size(0) == 1:
        x = x.repeat(2, 1, 1, 1)
    return x

fid_metric = FrechetInceptionDistance(feature=64).to(device)

print("Collecting real features for FID...")
with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc="Real", leave=False):
        imgs = imgs.to(device)
        imgs_uint8 = to_fid_tensor(imgs)
        imgs_uint8 = ensure_min_batch_two(imgs_uint8)
        fid_metric.update(imgs_uint8, real=True)

print("Generating fake images for FID...")

fid_num_fake = config.get("fid_num_fake", 5000)
fid_batch = max(2, config.get("fid_batch", 16))
fid_steps = config.get("fid_steps", 250)

num_fake = min(fid_num_fake, len(test_set))
generated = 0
batch_fake = fid_batch

fid_scheduler = deepcopy(noise_scheduler)
fid_scheduler.set_timesteps(fid_steps, device=device)
fid_timesteps = fid_scheduler.timesteps

while generated < num_fake:
    curr_batch = min(batch_fake, num_fake - generated)
    if curr_batch == 1:
        curr_batch = 2  # ensure >=2 for FID

    prior_ids = torch.randint(0, num_classes, (curr_batch,), device=device)
    x = sample_xT_from_prior(curr_batch, prior_ids)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        for t in fid_timesteps:
            tt = torch.full((curr_batch,), t, device=device, dtype=torch.long)
            eps0_pred = model(x, tt).sample
            x = fid_scheduler.step(eps0_pred, t.item(), x).prev_sample

    x = x.clamp(-1, 1)
    x_uint8 = to_fid_tensor(x)
    x_uint8 = ensure_min_batch_two(x_uint8)
    fid_metric.update(x_uint8, real=False)

    generated += curr_batch
    if generated % (batch_fake * 10) == 0:
        torch.cuda.empty_cache()

fid_value = fid_metric.compute().item()
print(f"[{config['dataset']}] FID (multimodal time-dependent implicit priors, eval): {fid_value:.4f}")
