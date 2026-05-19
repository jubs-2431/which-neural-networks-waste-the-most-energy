"""
Edge AI Energy GAN - Sample Generation Script

Supports two generation modes:
  - `gan`: load a trained generator checkpoint and sample from the GAN
  - `bootstrap`: resample directly from the grounded per-combo dataset

The bootstrap mode is the more reliable option when the benchmark evidence is
still sparse, because it preserves the grounded combo distributions without
asking a GAN to extrapolate from mostly estimated rows.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from gan_model  import Generator, DEVICES, MODELS, LATENT_DIM
from data_utils import (
    build_training_dataframe, DataScaler,
    DEVICE_PROFILES, MODEL_BASELINES, parse_requested_keys,
    resolve_feature_columns,
)


# ─── Inverse look-ups ─────────────────────────────────────────────────────────

IDX_TO_DEVICE = {v: k for k, v in DEVICES.items()}
IDX_TO_MODEL  = {v: k for k, v in MODELS.items()}

DEVICE_META = {
    "apple_silicon": ("Mobile laptop",  "#"),
    "raspberry_pi4": ("SBC",            "##"),
    "jetson_nano":   ("Edge GPU",       "###"),
    "coral_tpu":     ("Accelerator",    "####"),
    "stm32":         ("Microcontroller","#####"),
    "snapdragon888": ("Mobile SoC",     "######"),
}
MODEL_META = {
    "mobilenetv3_small": ("CNN (depthwise)",  2.54,  0.22),
    "mobilenetv2":       ("CNN (depthwise)",  3.50,  0.30),
    "resnet18":          ("CNN (baseline)",  11.69,  1.82),
    "tiny_vit_5m":       ("Transformer",     12.08,  1.30),
    "efficientnet_b0":   ("CNN (scaled)",     5.29,  0.39),
}


# ─── Balanced Condition Generator ─────────────────────────────────────────────

def balanced_conditions(n: int, combos: list[tuple[int, int]]):
    """
    Returns condition tensors that are balanced across all 30 combos
    (6 devices × 5 models), guaranteeing each combo appears ≥ floor(n/30) times.

    For n=10,000: each combo gets 333 rows, remainder distributed round-robin.
    """
    n_combos = len(combos)
    base     = n // n_combos
    extra    = n % n_combos

    device_idxs, model_idxs = [], []
    for i, (d, m) in enumerate(combos):
        count = base + (1 if i < extra else 0)
        device_idxs.extend([d] * count)
        model_idxs.extend([m]  * count)

    device_idxs = np.array(device_idxs, dtype=int)
    model_idxs  = np.array(model_idxs,  dtype=int)

    # Build one-hot condition matrix
    from gan_model import NUM_DEVICES, NUM_MODELS
    n_total = len(device_idxs)
    d_oh = np.zeros((n_total, NUM_DEVICES), dtype=np.float32)
    m_oh = np.zeros((n_total, NUM_MODELS),  dtype=np.float32)
    d_oh[np.arange(n_total), device_idxs] = 1.0
    m_oh[np.arange(n_total), model_idxs]  = 1.0
    cond = np.concatenate([d_oh, m_oh], axis=1)

    return cond, device_idxs, model_idxs


# ─── Core Generation Function ─────────────────────────────────────────────────

def generate_samples(
    G: Generator,
    scaler: DataScaler,
    combos: list[tuple[int, int]],
    n_samples: int = 10_000,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 512,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate n_samples rows from the trained Generator.

    Returns a DataFrame with physically meaningful units:
        device, device_tier, model, arch, params_M, flops_G,
        energy_J, power_W, latency_ms, energy_std,
        energy_scale_vs_paper, device_idx, model_idx
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    G.eval()

    cond_np, d_idxs, m_idxs = balanced_conditions(n_samples, combos)
    all_outputs = []

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end   = min(start + batch_size, n_samples)
            bs    = end - start
            z     = torch.randn(bs, LATENT_DIM, device=device)
            cond  = torch.from_numpy(cond_np[start:end]).to(device)
            fake  = G(z, cond).cpu().numpy()
            all_outputs.append(fake)

    raw = np.vstack(all_outputs)          # (n_samples, DATA_DIM)
    raw = np.clip(raw, 0.0, 1.0)          # safety clip before inverse scaling

    # Inverse-transform to physical units
    phys = scaler.inverse_transform(raw, d_idxs, m_idxs)

    # Clamp negative values that can occur near distribution tails
    phys = np.abs(phys)
    phys_df = pd.DataFrame(phys, columns=scaler.feature_names)

    rows = []
    for i in range(n_samples):
        d_key = IDX_TO_DEVICE[d_idxs[i]]
        m_key = IDX_TO_MODEL[m_idxs[i]]
        tier, _ = DEVICE_META[d_key]
        arch, params, flops = MODEL_META[m_key]
        paper_e = MODEL_BASELINES[m_key]["e"] * DEVICE_PROFILES[d_key]["es"]
        energy_j = float(phys_df.iloc[i].get("energy_J", np.nan))
        power_w = float(phys_df.iloc[i].get("power_W", np.nan))
        latency_ms = float(phys_df.iloc[i].get("latency_ms", np.nan))
        energy_std = float(phys_df.iloc[i].get("energy_std", np.nan))
        e_scale = energy_j / paper_e if paper_e > 0 else 1.0

        rows.append({
            "device":                d_key,
            "device_tier":           tier,
            "model":                 m_key,
            "arch":                  arch,
            "params_M":              params,
            "flops_G":               flops,
            "energy_J":              round(energy_j, 6) if np.isfinite(energy_j) else np.nan,
            "power_W":               round(power_w, 6) if np.isfinite(power_w) else np.nan,
            "latency_ms":            round(latency_ms, 3) if np.isfinite(latency_ms) else np.nan,
            "energy_std":            round(energy_std, 6) if np.isfinite(energy_std) else np.nan,
            "energy_scale_vs_paper": round(float(e_scale),    4),
            "device_idx":            int(d_idxs[i]),
            "model_idx":             int(m_idxs[i]),
        })

    df = pd.DataFrame(rows)
    print(f"[generate] Generated {len(df):,} samples across "
          f"{df['device'].nunique()} devices × {df['model'].nunique()} models")
    print_summary(df)
    return df


def bootstrap_samples(
    seed_df: pd.DataFrame,
    n_samples: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate balanced synthetic output by resampling each device/model combo
    from the grounded dataset with replacement.

    This preserves the empirical grounded distribution far more faithfully than
    the GAN when the measured evidence is sparse.
    """
    rng = np.random.default_rng(seed)
    combos = sorted(
        {
            (int(row.device_idx), int(row.model_idx))
            for row in seed_df[["device_idx", "model_idx"]].itertuples(index=False)
        }
    )
    cond_np, d_idxs, m_idxs = balanced_conditions(n_samples, combos)
    del cond_np  # only needed for GAN mode

    combo_tables = {
        (device, model): seed_df[
            (seed_df["device"] == device) & (seed_df["model"] == model)
        ].reset_index(drop=True)
        for device, model in seed_df[["device", "model"]].drop_duplicates().itertuples(index=False)
    }

    rows = []
    for i in range(n_samples):
        d_key = IDX_TO_DEVICE[d_idxs[i]]
        m_key = IDX_TO_MODEL[m_idxs[i]]
        combo_df = combo_tables[(d_key, m_key)]
        if combo_df.empty:
            raise ValueError(f"No grounded rows available for combo {(d_key, m_key)}")

        sampled = combo_df.iloc[int(rng.integers(0, len(combo_df)))]
        tier, _ = DEVICE_META[d_key]
        arch, params, flops = MODEL_META[m_key]
        paper_e = MODEL_BASELINES[m_key]["e"] * DEVICE_PROFILES[d_key]["es"]
        e_scale = float(sampled["energy_J"]) / paper_e if paper_e > 0 else 1.0

        rows.append({
            "device":                d_key,
            "device_tier":           tier,
            "model":                 m_key,
            "arch":                  arch,
            "params_M":              params,
            "flops_G":               flops,
            "energy_J":              round(float(sampled["energy_J"]), 6),
            "power_W":               round(float(sampled["power_W"]), 4),
            "latency_ms":            round(float(sampled["latency_ms"]), 3),
            "energy_std":            round(float(sampled["energy_std"]), 6),
            "energy_scale_vs_paper": round(float(e_scale), 4),
            "device_idx":            int(sampled["device_idx"]),
            "model_idx":             int(sampled["model_idx"]),
            "generation_method":     "bootstrap",
            "combo_source":          str(sampled.get("combo_source", "")),
        })

    df = pd.DataFrame(rows)
    print(f"[generate] Bootstrapped {len(df):,} samples across "
          f"{df['device'].nunique()} devices × {df['model'].nunique()} models")
    print_summary(df)
    return df


# ─── Summary Statistics ───────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    print("\n── Per-device mean energy (J) ──")
    summary = (
        df.groupby(["device", "model"])["energy_J"]
        .agg(["mean", "std", "count"])
        .round(5)
    )
    print(summary.to_string())
    print()


def postprocess_generated_df(
    df: pd.DataFrame,
    derive_power: bool = False,
    recompute_energy_std: bool = False,
) -> pd.DataFrame:
    df = df.copy()
    if derive_power:
        df["power_W"] = (
            (df["energy_J"] * 1000.0 / df["latency_ms"].clip(lower=1e-6))
            .round(6)
        )
    if recompute_energy_std:
        combo_std = (
            df.groupby(["device", "model"])["energy_J"]
            .transform("std")
            .fillna(0.0)
            .round(6)
        )
        df["energy_std"] = combo_std
    return df


def match_seed_variance(
    df: pd.DataFrame,
    seed_df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """
    Rescale per-combo deviations so generated feature spread matches the
    grounded seed distribution. This is useful for paper-aligned repeated-trial
    synthesis because WGANs on small tabular datasets often under-estimate
    within-combo variance even when the means/ranking are correct.
    """
    df = df.copy()
    combo_keys = ["device", "model"]
    target_std = (
        seed_df.groupby(combo_keys)[features]
        .std()
        .fillna(0.0)
    )

    for combo, idx in df.groupby(combo_keys).groups.items():
        combo_index = list(idx)
        if combo not in target_std.index:
            continue
        for feat in features:
            if feat not in df.columns or feat not in target_std.columns:
                continue
            vals = df.loc[combo_index, feat].to_numpy(dtype=float)
            if not np.isfinite(vals).all():
                continue
            current_mean = vals.mean()
            current_std = vals.std(ddof=1)
            desired_std = float(target_std.loc[combo, feat])
            if current_std <= 1e-9 or desired_std <= 0:
                continue
            adjusted = current_mean + (vals - current_mean) * (desired_std / current_std)
            df.loc[combo_index, feat] = np.maximum(adjusted, 1e-6)

    return df


# ─── Entry Point ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate 10k synthetic edge AI energy samples")
    p.add_argument("--method", type=str, default="gan", choices=["gan", "bootstrap"],
                   help="Generation method: trained GAN or grounded bootstrap sampler")
    p.add_argument("--checkpoint", type=str,
                   default="checkpoints/generator_final.pt",
                   help="Path to saved generator checkpoint")
    p.add_argument("--n_samples", type=int, default=10_000)
    p.add_argument("--output",    type=str, default="edge_ai_synthetic_10k.csv")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--data_csv",  type=str, default="real_benchmark_data.csv")
    p.add_argument("--rows_per_combo", type=int, default=64)
    p.add_argument("--devices",   type=str, default="")
    p.add_argument("--models",    type=str, default="")
    p.add_argument("--observed_only", action="store_true", default=False)
    p.add_argument("--feature_mode", type=str, default="raw",
                   choices=["raw", "paper_aligned"])
    p.add_argument("--match_seed_variance", action="store_true", default=False)
    p.add_argument("--derive_power", action="store_true", default=False)
    p.add_argument("--recompute_energy_std", action="store_true", default=False)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fit scaler on seed data (reproducible — same as training)
    df_seed = build_training_dataframe(
        csv_path=args.data_csv,
        rows_per_combo=args.rows_per_combo,
        random_seed=args.seed,
        include_devices=parse_requested_keys(args.devices),
        include_models=parse_requested_keys(args.models),
        observed_only=args.observed_only,
    )
    feature_names = resolve_feature_columns(args.feature_mode)
    scaler  = DataScaler(features=feature_names).fit(df_seed)
    combos = sorted(
        {
            (int(row.device_idx), int(row.model_idx))
            for row in df_seed[["device_idx", "model_idx"]].itertuples(index=False)
        }
    )

    if args.method == "gan":
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                "Run `python train.py` first to train the GAN."
            )

        print(f"[generate] Loading generator from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        ckpt_cfg = ckpt.get("cfg", {})
        ckpt_feature_mode = ckpt_cfg.get("feature_mode")
        if ckpt_feature_mode and ckpt_feature_mode != args.feature_mode:
            raise ValueError(
                f"Checkpoint feature_mode={ckpt_feature_mode!r} does not match "
                f"requested feature_mode={args.feature_mode!r}."
            )
        G = Generator(data_dim=len(feature_names)).to(device)
        G.load_state_dict(ckpt["G"])
        df_out = generate_samples(
            G, scaler, combos,
            n_samples  = args.n_samples,
            device     = device,
            seed       = args.seed,
        )
    else:
        df_out = bootstrap_samples(
            df_seed,
            n_samples=args.n_samples,
            seed=args.seed,
        )

    if args.feature_mode == "paper_aligned":
        args.derive_power = True
        args.recompute_energy_std = True

    if args.match_seed_variance:
        df_out = match_seed_variance(
            df_out,
            df_seed,
            features=["energy_J", "latency_ms"],
        )

    df_out = postprocess_generated_df(
        df_out,
        derive_power=args.derive_power,
        recompute_energy_std=args.recompute_energy_std,
    )

    # Save
    out_path = Path(args.output)
    df_out.to_csv(out_path, index=False)
    print(f"[generate] Saved {len(df_out):,} rows → {out_path}")
    print(f"           Columns: {list(df_out.columns)}")
