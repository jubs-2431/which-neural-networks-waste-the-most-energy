"""
Edge AI Energy GAN - Training Script
Trains a Conditional WGAN-GP to learn the distribution of
inference energy measurements across edge hardware platforms.

Target: produce a generator capable of synthesising
        10,000 high-quality samples for downstream analysis.

Usage:
    python train.py                        # default 2000 epochs
    python train.py --epochs 3000          # longer run
    python train.py --resume               # resume from last checkpoint
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from gan_model   import build_gan_with_dims, gradient_penalty, LATENT_DIM
from data_utils  import (
    build_training_dataframe,
    DataScaler,
    EnergyDataset,
    parse_requested_keys,
    resolve_feature_columns,
)


# ─── Hyper-parameters ─────────────────────────────────────────────────────────
# Chosen for tabular GANs on small datasets (< 5 k rows):
#   - WGAN-GP with n_critic=5 prevents discriminator overfitting
#   - lr=1e-4 / β=(0.0, 0.9) are standard WGAN-GP settings
#   - batch=64 keeps gradient estimates stable at 3,000-row dataset size

DEFAULTS = dict(
    epochs        = 2000,
    batch_size    = 64,
    lr_g          = 1e-4,
    lr_d          = 1e-4,
    n_critic      = 5,       # D steps per G step
    lambda_gp     = 10.0,
    latent_dim    = LATENT_DIM,
    save_every    = 200,     # checkpoint interval (epochs)
    log_every     = 50,
    checkpoint_dir= "checkpoints",
    seed          = 42,
    rows_per_combo= 64,
    data_csv      = "real_benchmark_data.csv",
    devices       = "",
    models        = "",
    observed_only = False,
    feature_mode  = "raw",
)


# ─── Training Loop ────────────────────────────────────────────────────────────

def train(cfg: dict):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    df     = build_training_dataframe(
        csv_path=cfg["data_csv"],
        rows_per_combo=cfg["rows_per_combo"],
        random_seed=cfg["seed"],
        include_devices=parse_requested_keys(cfg.get("devices")),
        include_models=parse_requested_keys(cfg.get("models")),
        observed_only=cfg.get("observed_only", False),
    )
    feature_names = resolve_feature_columns(cfg.get("feature_mode", "raw"))
    scaler = DataScaler(features=feature_names).fit(df)
    ds     = EnergyDataset(df, scaler)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)

    # ── Models & Optimisers ───────────────────────────────────────────────────
    G, D = build_gan_with_dims(device, data_dim=len(feature_names))
    opt_G = optim.Adam(G.parameters(), lr=cfg["lr_g"], betas=(0.0, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=cfg["lr_d"], betas=(0.0, 0.9))

    # ── Resume ────────────────────────────────────────────────────────────────
    ckpt_dir   = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(exist_ok=True)
    start_epoch = 1

    if cfg.get("resume"):
        ckpts = sorted(ckpt_dir.glob("ckpt_epoch_*.pt"))
        if ckpts:
            ckpt = torch.load(ckpts[-1], map_location=device)
            G.load_state_dict(ckpt["G"])
            D.load_state_dict(ckpt["D"])
            opt_G.load_state_dict(ckpt["opt_G"])
            opt_D.load_state_dict(ckpt["opt_D"])
            start_epoch = ckpt["epoch"] + 1
            print(f"[train] Resumed from epoch {ckpt['epoch']}")

    # ── History ───────────────────────────────────────────────────────────────
    history = {"epoch": [], "loss_D": [], "loss_G": [], "wdist": []}

    print(f"[train] Starting training: epochs={cfg['epochs']}, "
          f"batch={cfg['batch_size']}, n_critic={cfg['n_critic']}")
    t0 = time.time()

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        epoch_loss_D, epoch_loss_G, epoch_wd = [], [], []

        for x_real, cond in loader:
            x_real = x_real.to(device)
            cond   = cond.to(device)
            bs     = x_real.size(0)

            # ── Discriminator step ────────────────────────────────────────────
            for _ in range(cfg["n_critic"]):
                z      = torch.randn(bs, cfg["latent_dim"], device=device)
                x_fake = G(z, cond).detach()

                d_real = D(x_real, cond)
                d_fake = D(x_fake, cond)
                gp     = gradient_penalty(D, x_real, x_fake, cond,
                                          device, cfg["lambda_gp"])
                loss_D = d_fake.mean() - d_real.mean() + gp

                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()

                w_dist = (d_real.mean() - d_fake.mean()).item()
                epoch_loss_D.append(loss_D.item())
                epoch_wd.append(w_dist)

            # ── Generator step ────────────────────────────────────────────────
            z      = torch.randn(bs, cfg["latent_dim"], device=device)
            x_fake = G(z, cond)
            loss_G = -D(x_fake, cond).mean()

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            epoch_loss_G.append(loss_G.item())

        # ── Logging ───────────────────────────────────────────────────────────
        mean_D = float(np.mean(epoch_loss_D))
        mean_G = float(np.mean(epoch_loss_G))
        mean_W = float(np.mean(epoch_wd))

        history["epoch"].append(epoch)
        history["loss_D"].append(mean_D)
        history["loss_G"].append(mean_G)
        history["wdist"].append(mean_W)

        if epoch % cfg["log_every"] == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:>5}/{cfg['epochs']}  "
                  f"D={mean_D:+.4f}  G={mean_G:+.4f}  "
                  f"W-dist={mean_W:.4f}  [{elapsed:.0f}s]")

        # ── Checkpoint ────────────────────────────────────────────────────────
        if epoch % cfg["save_every"] == 0:
            ckpt_path = ckpt_dir / f"ckpt_epoch_{epoch:05d}.pt"
            torch.save({
                "epoch": epoch,
                "G":     G.state_dict(),
                "D":     D.state_dict(),
                "opt_G": opt_G.state_dict(),
                "opt_D": opt_D.state_dict(),
                "cfg":   cfg,
            }, ckpt_path)
            print(f"[train] Checkpoint saved → {ckpt_path}")

    # ── Final save ────────────────────────────────────────────────────────────
    final_path = ckpt_dir / "generator_final.pt"
    torch.save({"G": G.state_dict(), "cfg": cfg}, final_path)
    print(f"[train] Training complete. Generator saved → {final_path}")

    # Save training history
    import json
    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return G, scaler


# ─── Entry Point ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Edge AI Energy GAN")
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", action="store_true", default=v)
        else:
            p.add_argument(f"--{k}", type=type(v), default=v)
    p.add_argument("--resume", action="store_true", default=False)
    return vars(p.parse_args())


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
