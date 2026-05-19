import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from data_utils import (
    BASELINE_CSV,
    LatencyScaler,
    build_training_dataframe,
    load_baseline_data,
)
from gan_model import Discriminator, Generator, LATENT_DIM, gradient_penalty


def parse_args():
    p = argparse.ArgumentParser(description="Train weaker-engine latency GAN")
    p.add_argument("--data_csv", type=str, default=str(BASELINE_CSV))
    p.add_argument("--rows_per_combo", type=int, default=64)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--n_critic", type=int, default=5)
    p.add_argument("--lambda_gp", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    return p.parse_args()


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    baseline = load_baseline_data(args.data_csv)
    grounded = build_training_dataframe(args.data_csv, rows_per_combo=args.rows_per_combo, random_seed=args.seed)
    num_combos = int(baseline["combo_idx"].nunique())
    combo_order = (
        baseline[["combo_idx", "combo_key", "device_key", "model_key"]]
        .drop_duplicates()
        .sort_values("combo_idx")
    )

    scaler = LatencyScaler().fit(grounded)
    x = torch.from_numpy(scaler.transform(grounded))
    cond = torch.ones((len(x), 1), dtype=torch.float32)
    ds = TensorDataset(x, cond)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    G = Generator(cond_dim=1).to(device)
    D = Discriminator(cond_dim=1).to(device)
    opt_g = optim.Adam(G.parameters(), lr=args.lr, betas=(0.0, 0.9))
    opt_d = optim.Adam(D.parameters(), lr=args.lr, betas=(0.0, 0.9))

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history = {"epoch": [], "loss_D": [], "loss_G": [], "wdist": []}

    for epoch in range(1, args.epochs + 1):
        loss_d_epoch, loss_g_epoch, wdist_epoch = [], [], []
        for x_real, cond in loader:
            x_real = x_real.to(device)
            cond = cond.to(device)
            bs = x_real.size(0)

            for _ in range(args.n_critic):
                z = torch.randn(bs, LATENT_DIM, device=device)
                x_fake = G(z, cond).detach()
                d_real = D(x_real, cond)
                d_fake = D(x_fake, cond)
                gp = gradient_penalty(D, x_real, x_fake, cond, device, args.lambda_gp)
                loss_d = d_fake.mean() - d_real.mean() + gp

                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()

                loss_d_epoch.append(loss_d.item())
                wdist_epoch.append((d_real.mean() - d_fake.mean()).item())

            z = torch.randn(bs, LATENT_DIM, device=device)
            x_fake = G(z, cond)
            loss_g = -D(x_fake, cond).mean()
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()
            loss_g_epoch.append(loss_g.item())

        history["epoch"].append(epoch)
        history["loss_D"].append(float(np.mean(loss_d_epoch)))
        history["loss_G"].append(float(np.mean(loss_g_epoch)))
        history["wdist"].append(float(np.mean(wdist_epoch)))

        if epoch == 1 or epoch % 100 == 0:
            print(
                f"Epoch {epoch:>4}/{args.epochs} "
                f"D={history['loss_D'][-1]:+.4f} "
                f"G={history['loss_G'][-1]:+.4f} "
                f"W={history['wdist'][-1]:+.4f}"
            )

    payload = {
        "G": G.state_dict(),
        "cfg": vars(args),
        "num_combos": num_combos,
        "combo_order": combo_order.to_dict("records"),
    }
    torch.save(payload, ckpt_dir / "generator_final.pt")
    with open(ckpt_dir / "history.json", "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    print(f"[train] Saved generator to {ckpt_dir / 'generator_final.pt'}")


if __name__ == "__main__":
    train(parse_args())
