"""
Edge AI Energy GAN - Evaluation Script
Measures how faithfully the GAN reproduces the real distribution
using statistical tests and fidelity metrics.

Metrics:
  - Wasserstein distance (per feature, per combo)
  - KS test (Kolmogorov-Smirnov) for distribution match
  - Coverage: % of real data covered by generated distribution
  - Energy correlation with FLOPs / params (paper finding replication)

Usage:
    python evaluate.py                        # uses generator_final.pt
    python evaluate.py --csv edge_ai_synthetic_10k.csv  # evaluate saved CSV
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from data_utils import resolve_feature_columns


# ─── Wasserstein Distance ─────────────────────────────────────────────────────

def wasserstein_per_feature(real: pd.DataFrame, fake: pd.DataFrame,
                             features: list[str]) -> dict:
    results = {}
    for feat in features:
        w = stats.wasserstein_distance(real[feat].values, fake[feat].values)
        results[feat] = round(w, 6)
    return results


# ─── KS Test ──────────────────────────────────────────────────────────────────

def ks_test_per_feature(real: pd.DataFrame, fake: pd.DataFrame,
                         features: list[str]) -> dict:
    results = {}
    for feat in features:
        ks_stat, p_val = stats.ks_2samp(real[feat].values, fake[feat].values)
        results[feat] = {"ks_stat": round(ks_stat, 4), "p_value": round(p_val, 4)}
    return results


# ─── Coverage Score ───────────────────────────────────────────────────────────

def coverage_score(real: pd.DataFrame, fake: pd.DataFrame,
                   feature: str = "energy_J", n_bins: int = 50) -> float:
    """
    Fraction of real-data histogram bins covered (non-zero) by generated data.
    Score of 1.0 = full coverage, < 0.8 = mode collapse.
    """
    r = real[feature].values
    f = fake[feature].values
    bins = np.linspace(min(r.min(), f.min()), max(r.max(), f.max()), n_bins + 1)

    real_hist, _ = np.histogram(r, bins=bins)
    fake_hist, _ = np.histogram(f, bins=bins)

    covered = np.sum((real_hist > 0) & (fake_hist > 0))
    total   = np.sum(real_hist > 0)
    return round(covered / total, 4) if total > 0 else 0.0


# ─── Correlation Replication ──────────────────────────────────────────────────

def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replicates paper finding: FLOPs and param count are weak predictors
    of energy. Computes Pearson & Spearman correlations.
    """
    rows = []
    for feat in ["flops_G", "params_M", "latency_ms"]:
        pearson, p_p = stats.pearsonr(df[feat], df["energy_J"])
        spearman, p_s = stats.spearmanr(df[feat], df["energy_J"])
        rows.append({
            "predictor":       feat,
            "pearson_r":       round(pearson,  4),
            "pearson_p":       round(p_p,      4),
            "spearman_rho":    round(spearman, 4),
            "spearman_p":      round(p_s,      4),
        })
    return pd.DataFrame(rows)


# ─── Per-Combo Statistics ─────────────────────────────────────────────────────

def per_combo_stats(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["device", "model"])["energy_J"]
        .agg(count="count", mean="mean", std="std",
             p5=lambda x: np.percentile(x, 5),
             p95=lambda x: np.percentile(x, 95))
        .round(5)
        .reset_index()
    )


def derive_power(df: pd.DataFrame) -> pd.Series:
    return df["energy_J"] * 1000.0 / df["latency_ms"].clip(lower=1e-6)


def per_combo_energy_std(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["device", "model"])["energy_J"]
        .agg(energy_std="std")
        .fillna(0.0)
        .reset_index()
    )


def compare_combo_energy_std(real_df: pd.DataFrame, fake_df: pd.DataFrame) -> pd.DataFrame:
    merged = per_combo_energy_std(real_df).merge(
        per_combo_energy_std(fake_df),
        on=["device", "model"],
        suffixes=("_real", "_fake"),
    )
    merged["abs_error"] = (merged["energy_std_fake"] - merged["energy_std_real"]).abs()
    merged["pct_error"] = (
        merged["abs_error"] / merged["energy_std_real"].clip(lower=1e-6) * 100.0
    )
    return merged.sort_values(["device", "model"]).reset_index(drop=True)


# ─── Full Evaluation ──────────────────────────────────────────────────────────

def evaluate(real_df: pd.DataFrame, fake_df: pd.DataFrame, feature_mode: str = "raw"):
    modeled_features = resolve_feature_columns(feature_mode)

    print("=" * 60)
    print("  Edge AI Energy GAN — Evaluation Report")
    print("=" * 60)
    print(f"  Real samples : {len(real_df):,}")
    print(f"  Fake samples : {len(fake_df):,}")
    print(f"  Feature mode : {feature_mode}")
    print()

    # ── Wasserstein ───────────────────────────────────────────────────────────
    wd = wasserstein_per_feature(real_df, fake_df, modeled_features)
    print("── Wasserstein Distance (modeled features; lower = better) ──")
    for feat, val in wd.items():
        bar = "█" * int(val * 200)
        print(f"  {feat:<15} {val:.6f}  {bar}")
    print()

    # ── KS Test ───────────────────────────────────────────────────────────────
    ks = ks_test_per_feature(real_df, fake_df, modeled_features)
    print("── KS Test (modeled features; p > 0.05 = distributions match) ──")
    for feat, res in ks.items():
        flag = "✓" if res["p_value"] > 0.05 else "✗"
        print(f"  {feat:<15} KS={res['ks_stat']:.4f}  p={res['p_value']:.4f}  {flag}")
    print()

    derived = {}
    if feature_mode == "paper_aligned":
        real_power = derive_power(real_df)
        fake_power = derive_power(fake_df)
        power_wd = round(stats.wasserstein_distance(real_power.values, fake_power.values), 6)
        power_ks_stat, power_p = stats.ks_2samp(real_power.values, fake_power.values)
        power_ks = {"ks_stat": round(power_ks_stat, 4), "p_value": round(power_p, 4)}
        derived["power_W"] = {"wasserstein": power_wd, **power_ks}

        print("── Derived Power Fidelity (from energy/latency) ──")
        flag = "✓" if power_ks["p_value"] > 0.05 else "✗"
        print(
            f"  power_W         WD={power_wd:.6f}  "
            f"KS={power_ks['ks_stat']:.4f}  p={power_ks['p_value']:.4f}  {flag}"
        )
        print()

        energy_std_cmp = compare_combo_energy_std(real_df, fake_df)
        energy_std_wd = round(
            stats.wasserstein_distance(
                energy_std_cmp["energy_std_real"].values,
                energy_std_cmp["energy_std_fake"].values,
            ),
            6,
        )
        energy_std_mae = round(float(energy_std_cmp["abs_error"].mean()), 6)
        energy_std_mape = round(float(energy_std_cmp["pct_error"].mean()), 2)
        derived["energy_std_summary"] = {
            "wasserstein": energy_std_wd,
            "mae": energy_std_mae,
            "mean_pct_error": energy_std_mape,
        }

        print("── Repeated-Trial Energy Spread Fidelity (per combo summary) ──")
        print(
            f"  energy_std      WD={energy_std_wd:.6f}  "
            f"mean_abs_err={energy_std_mae:.6f}  "
            f"mean_pct_err={energy_std_mape:.2f}%"
        )
        print(energy_std_cmp[[
            "device", "model", "energy_std_real", "energy_std_fake", "pct_error"
        ]].round(6).to_string(index=False))
        print()

    # ── Coverage ──────────────────────────────────────────────────────────────
    cov = coverage_score(real_df, fake_df, "energy_J")
    print(f"── Coverage (energy_J) : {cov:.1%} {'✓' if cov >= 0.8 else '⚠ possible mode collapse'}")
    print()

    # ── Correlations ──────────────────────────────────────────────────────────
    print("── Correlation with energy_J (paper finding replication) ──")
    corr = correlation_analysis(fake_df)
    print(corr.to_string(index=False))
    print()

    # ── Sample Count per Combo ────────────────────────────────────────────────
    print("── Sample counts per device/model combo ──")
    counts = fake_df.groupby(["device", "model"]).size().unstack(fill_value=0)
    print(counts.to_string())
    print()

    # ── Top 5 most/least efficient ────────────────────────────────────────────
    combo_stats = per_combo_stats(fake_df)
    print("── 5 most energy-efficient combos ──")
    print(combo_stats.nsmallest(5, "mean")[["device","model","mean","std","count"]].to_string(index=False))
    print()
    print("── 5 least energy-efficient combos ──")
    print(combo_stats.nlargest(5, "mean")[["device","model","mean","std","count"]].to_string(index=False))
    print("=" * 60)

    return {
        "wasserstein":  wd,
        "ks_test":      ks,
        "derived":      derived,
        "coverage":     cov,
        "correlations": corr.to_dict("records"),
    }


# ─── Entry Point ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate GAN output quality")
    p.add_argument("--csv", type=str, default="edge_ai_synthetic_10k.csv",
                   help="Path to generated CSV from generate.py")
    p.add_argument("--data_csv", type=str, default="real_benchmark_data.csv",
                   help="Path to real benchmark CSV used to build the training data")
    p.add_argument("--rows_per_combo", type=int, default=64,
                   help="Rows per combo used when constructing the grounded training dataset")
    p.add_argument("--devices", type=str, default="")
    p.add_argument("--models", type=str, default="")
    p.add_argument("--observed_only", action="store_true", default=False)
    p.add_argument("--feature_mode", type=str, default="raw",
                   choices=["raw", "paper_aligned"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Real (seed) data
    from data_utils import build_training_dataframe, parse_requested_keys
    real_df = build_training_dataframe(
        csv_path=args.data_csv,
        rows_per_combo=args.rows_per_combo,
        include_devices=parse_requested_keys(args.devices),
        include_models=parse_requested_keys(args.models),
        observed_only=args.observed_only,
    )

    # Generated data
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Generated CSV not found: {csv_path}\n"
            "Run `python generate.py` first."
        )
    fake_df = pd.read_csv(csv_path)

    evaluate(real_df, fake_df, feature_mode=args.feature_mode)
