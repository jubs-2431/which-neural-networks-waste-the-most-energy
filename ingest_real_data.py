"""
Edge AI Energy GAN - Real Data Ingestion
Takes the CSV produced by scraper.py and replaces the estimated
scaling factors in data_utils.py with real measured values.

This is the bridge between scraper.py → data_utils.py → train.py

Usage:
    python ingest_real_data.py --csv real_benchmark_data.csv
    python ingest_real_data.py --csv real_benchmark_data.csv --dry_run
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


APPLE_SILICON_BASELINES = {
    "mobilenetv3_small": {"energy_J": 0.080, "latency_ms": 15.14, "power_W": 5.3},
    "mobilenetv2":       {"energy_J": 0.106, "latency_ms": 20.07, "power_W": 5.3},
    "resnet18":          {"energy_J": 0.110, "latency_ms": 20.68, "power_W": 5.3},
    "tiny_vit_5m":       {"energy_J": 0.219, "latency_ms": 41.33, "power_W": 5.3},
    "efficientnet_b0":   {"energy_J": 0.292, "latency_ms": 55.08, "power_W": 5.3},
}


def compute_real_scales(df: pd.DataFrame) -> dict:
    """
    For each (device, model) pair in the scraped CSV, compute
    the real energy_scale and latency_scale relative to Apple Silicon.
    Returns a nested dict: scales[device][model] = {es, ls, tdp}
    """
    scales = {}

    devices = df["device"].unique()
    for device in devices:
        if device == "apple_silicon":
            continue

        scales[device] = {}
        dev_df = df[df["device"] == device]

        for model, baseline in APPLE_SILICON_BASELINES.items():
            model_rows = dev_df[dev_df["model"] == model]
            if model_rows.empty:
                continue

            real_energy  = model_rows["energy_J"].dropna()
            real_latency = model_rows["latency_ms"].dropna()
            real_power   = model_rows["power_W"].dropna()

            entry = {}
            if not real_energy.empty:
                entry["energy_scale"] = round(
                    float(real_energy.mean()) / baseline["energy_J"], 4
                )
                entry["energy_J_mean"] = round(float(real_energy.mean()), 6)
                entry["energy_J_std"]  = round(float(real_energy.std()) if len(real_energy) > 1 else 0.0, 6)
                entry["n_energy"]      = len(real_energy)

            if not real_latency.empty:
                entry["latency_scale"] = round(
                    float(real_latency.mean()) / baseline["latency_ms"], 4
                )
                entry["latency_ms_mean"] = round(float(real_latency.mean()), 4)
                entry["n_latency"]       = len(real_latency)

            if not real_power.empty:
                entry["tdp_W"] = round(float(real_power.mean()), 3)

            if entry:
                scales[device][model] = entry

    return scales


def print_comparison(scales: dict):
    """Print real vs estimated scales side-by-side."""
    from data_utils import DEVICE_PROFILES

    print("\n── Real vs Estimated Scaling Factors ──")
    print(f"{'Device':<20} {'Model':<20} {'Est E-scale':>12} {'Real E-scale':>13} {'Est L-scale':>12} {'Real L-scale':>13}")
    print("-" * 95)

    for device, model_scales in scales.items():
        est = DEVICE_PROFILES.get(device, {})
        for model, vals in model_scales.items():
            real_es = vals.get("energy_scale",  "N/A")
            real_ls = vals.get("latency_scale", "N/A")
            est_es  = est.get("es", "N/A")
            est_ls  = est.get("ls", "N/A")
            print(f"{device:<20} {model:<20} {str(est_es):>12} {str(real_es):>13} {str(est_ls):>12} {str(real_ls):>13}")


def patch_data_utils(scales: dict, dry_run: bool = False):
    """
    Write the real scaling factors to real_device_profiles.json
    which data_utils.py will load preferentially over estimates.
    """
    output = {
        "generated_from": "ingest_real_data.py",
        "source":         "scraped real benchmark data",
        "note":           "Loaded by data_utils.py — overrides estimated DEVICE_PROFILES",
        "profiles":       {}
    }

    # Per-device mean scales across all models (for devices with multiple models)
    for device, model_scales in scales.items():
        all_es = [v["energy_scale"]  for v in model_scales.values() if "energy_scale"  in v]
        all_ls = [v["latency_scale"] for v in model_scales.values() if "latency_scale" in v]
        all_tdp= [v["tdp_W"]         for v in model_scales.values() if "tdp_W"         in v]

        output["profiles"][device] = {
            "energy_scale_mean":  round(np.mean(all_es),  4) if all_es  else None,
            "latency_scale_mean": round(np.mean(all_ls),  4) if all_ls  else None,
            "tdp_W":              round(np.mean(all_tdp), 3) if all_tdp else None,
            "per_model":          model_scales,
            "n_models_with_data": len(model_scales),
        }

    path = Path("real_device_profiles.json")
    if not dry_run:
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n[ingest] Saved real profiles → {path}")
        print("         data_utils.py will automatically load this file on next run.")
    else:
        print(f"\n[ingest] DRY RUN — would write to {path}:")
        print(json.dumps(output, indent=2))


def coverage_report(df: pd.DataFrame, scales: dict):
    """Report which device/model combos have real data vs still estimated."""
    from data_utils import DEVICES, MODELS

    print("\n── Coverage: real data vs still estimated ──")
    print(f"{'Device':<22} {'Model':<22} {'Status':<12} {'Source'}")
    print("-" * 80)

    for device in DEVICES:
        if device == "apple_silicon":
            continue
        for model in MODELS:
            has_real = device in scales and model in scales[device]
            if has_real:
                vals    = scales[device][model]
                n       = vals.get("n_energy", vals.get("n_latency", 0))
                sources = df[(df["device"]==device) & (df["model"]==model)]["source"].unique()
                status  = f"REAL ({n} rows)"
                src     = ", ".join(sources)
            else:
                status  = "estimated"
                src     = "data_utils.py defaults"
            print(f"{device:<22} {model:<22} {status:<12} {src}")


# ─── Entry Point ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Ingest real benchmark data into GAN pipeline")
    p.add_argument("--csv",     required=True, help="Path to real_benchmark_data.csv from scraper.py")
    p.add_argument("--dry_run", action="store_true", help="Print results without writing files")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}\nRun scraper.py first.")

    df = pd.read_csv(csv_path)
    print(f"[ingest] Loaded {len(df)} rows from {csv_path}")
    print(f"         Devices: {sorted(df['device'].unique())}")
    print(f"         Models:  {sorted(df['model'].unique())}")

    scales = compute_real_scales(df)
    print_comparison(scales)
    coverage_report(df, scales)
    patch_data_utils(scales, dry_run=args.dry_run)

    if not args.dry_run:
        print("\n── Next steps ──")
        print("1. Retrain the GAN with real baseline data:")
        print("   python train.py --epochs 2000")
        print("2. Generate 10,000 samples:")
        print("   python generate.py")
        print("3. Evaluate fidelity:")
        print("   python evaluate.py")
