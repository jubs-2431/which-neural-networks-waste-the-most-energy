import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_utils import (
    BASELINE_CSV,
    LatencyScaler,
    build_training_dataframe,
    load_baseline_data,
)
from gan_model import Generator, LATENT_DIM


def parse_args():
    p = argparse.ArgumentParser(description="Generate weaker-engine latency samples")
    p.add_argument("--checkpoint", type=str, default="checkpoints/generator_final.pt")
    p.add_argument("--data_csv", type=str, default=str(BASELINE_CSV))
    p.add_argument("--rows_per_combo", type=int, default=64)
    p.add_argument("--n_samples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="weaker_engine_synth_10k.csv")
    p.add_argument("--match_seed_moments", action="store_true", default=True)
    return p.parse_args()


def balanced_combo_indices(n_samples: int, combo_indices: list[int]) -> np.ndarray:
    n = len(combo_indices)
    base = n_samples // n
    extra = n_samples % n
    out = []
    for i, combo_idx in enumerate(combo_indices):
        count = base + (1 if i < extra else 0)
        out.extend([combo_idx] * count)
    return np.array(out, dtype=int)


def match_seed_moments(df: pd.DataFrame, seed_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    df = df.copy()
    rng = np.random.default_rng(seed)
    target = seed_df.groupby("combo_idx")["latency_ms"].agg(["mean", "std"]).fillna(0.0)
    for combo_idx, idx in df.groupby("combo_idx").groups.items():
        vals = df.loc[idx, "latency_ms"].to_numpy(dtype=float)
        current_mean = vals.mean()
        current_std = vals.std(ddof=1)
        desired_mean = float(target.loc[combo_idx, "mean"])
        desired_std = float(target.loc[combo_idx, "std"])

        if current_std > 1e-9 and desired_std > 0:
            vals = desired_mean + (vals - current_mean) * (desired_std / current_std)
        else:
            vals = rng.normal(desired_mean, max(desired_std, desired_mean * 0.05, 0.05), size=len(vals))
        df.loc[idx, "latency_ms"] = np.maximum(vals, 1e-6)
    return df


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    baseline = load_baseline_data(args.data_csv)
    grounded = build_training_dataframe(args.data_csv, rows_per_combo=args.rows_per_combo, random_seed=args.seed)
    num_combos = int(baseline["combo_idx"].nunique())
    combo_meta = (
        baseline.groupby(["combo_idx", "combo_key", "device_key", "model_key"], as_index=False)
        .agg(hw_name=("hw_name", lambda s: " | ".join(sorted(set(map(str, s))))))
        .sort_values("combo_idx")
        .reset_index(drop=True)
    )
    scaler = LatencyScaler().fit(grounded)

    ckpt = torch.load(Path(args.checkpoint), map_location=device)
    G = Generator(cond_dim=1).to(device)
    G.load_state_dict(ckpt["G"])
    G.eval()

    combo_indices = combo_meta["combo_idx"].tolist()
    sampled_combo_idx = balanced_combo_indices(args.n_samples, combo_indices)
    cond = np.eye(num_combos, dtype=np.float32)[sampled_combo_idx]

    outputs = []
    with torch.no_grad():
        for start in range(0, args.n_samples, 512):
            end = min(start + 512, args.n_samples)
            bs = end - start
            z = torch.randn(bs, LATENT_DIM, device=device)
            c = torch.ones((bs, 1), dtype=torch.float32, device=device)
            outputs.append(G(z, c).cpu().numpy())

    raw = np.vstack(outputs).reshape(-1)
    latency = scaler.inverse_transform(raw, sampled_combo_idx)
    out = pd.DataFrame({
        "combo_idx": sampled_combo_idx,
        "latency_ms": latency,
    })
    out = out.merge(combo_meta, on="combo_idx", how="left")

    if args.match_seed_moments:
        out = match_seed_moments(out, grounded, seed=args.seed)

    out["latency_ms"] = out["latency_ms"].round(6)
    out.to_csv(args.output, index=False)
    print(f"[generate] Saved {len(out):,} rows to {args.output}")
