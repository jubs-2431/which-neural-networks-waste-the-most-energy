import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from data_utils import BASELINE_CSV, build_training_dataframe


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate weaker-engine latency GAN")
    p.add_argument("--csv", type=str, default="weaker_engine_synth_10k.csv")
    p.add_argument("--data_csv", type=str, default=str(BASELINE_CSV))
    p.add_argument("--rows_per_combo", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def coverage_score(real: pd.Series, fake: pd.Series, n_bins: int = 50) -> float:
    bins = np.linspace(min(real.min(), fake.min()), max(real.max(), fake.max()), n_bins + 1)
    real_hist, _ = np.histogram(real.values, bins=bins)
    fake_hist, _ = np.histogram(fake.values, bins=bins)
    total = np.sum(real_hist > 0)
    covered = np.sum((real_hist > 0) & (fake_hist > 0))
    return float(covered / total) if total else 0.0


def combo_metrics(real_df: pd.DataFrame, fake_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for combo_key, real_combo in real_df.groupby("combo_key", sort=True):
        fake_combo = fake_df[fake_df["combo_key"] == combo_key]
        ks_stat, p_val = stats.ks_2samp(real_combo["latency_ms"], fake_combo["latency_ms"])
        rows.append({
            "combo_key": combo_key,
            "real_mean": float(real_combo["latency_ms"].mean()),
            "fake_mean": float(fake_combo["latency_ms"].mean()),
            "pct_err": abs(float(fake_combo["latency_ms"].mean()) - float(real_combo["latency_ms"].mean()))
                      / max(float(real_combo["latency_ms"].mean()), 1e-6) * 100.0,
            "real_std": float(real_combo["latency_ms"].std(ddof=1)),
            "fake_std": float(fake_combo["latency_ms"].std(ddof=1)),
            "wasserstein": float(stats.wasserstein_distance(real_combo["latency_ms"], fake_combo["latency_ms"])),
            "ks_stat": float(ks_stat),
            "p_value": float(p_val),
            "pass": bool(p_val > 0.05),
        })
    return pd.DataFrame(rows).sort_values("combo_key").reset_index(drop=True)


if __name__ == "__main__":
    args = parse_args()
    real_df = build_training_dataframe(args.data_csv, rows_per_combo=args.rows_per_combo, random_seed=args.seed)
    fake_path = Path(args.csv)
    if not fake_path.exists():
        raise FileNotFoundError(fake_path)
    fake_df = pd.read_csv(fake_path)

    wd = stats.wasserstein_distance(real_df["latency_ms"], fake_df["latency_ms"])
    ks_stat, p_val = stats.ks_2samp(real_df["latency_ms"], fake_df["latency_ms"])
    coverage = coverage_score(real_df["latency_ms"], fake_df["latency_ms"])
    metrics = combo_metrics(real_df, fake_df)
    pass_rate = float(metrics["pass"].mean())
    mean_pct_err = float(metrics["pct_err"].mean())

    overall_pass = (p_val > 0.05) and (coverage >= 0.95) and (pass_rate >= 0.80) and (mean_pct_err <= 5.0)

    print("=" * 60)
    print("Weaker Engines Latency GAN — Evaluation Report")
    print("=" * 60)
    print(f"Real samples : {len(real_df):,}")
    print(f"Fake samples : {len(fake_df):,}")
    print(f"Global Wasserstein : {wd:.6f}")
    print(f"Global KS         : {ks_stat:.4f}")
    print(f"Global p-value    : {p_val:.4f}")
    print(f"Coverage          : {coverage:.1%}")
    print(f"Combo KS pass rate: {pass_rate:.1%}")
    print(f"Mean combo %% err : {mean_pct_err:.2f}%")
    print()
    print(metrics[[
        "combo_key", "real_mean", "fake_mean", "pct_err", "wasserstein", "ks_stat", "p_value", "pass"
    ]].round(6).to_string(index=False))
    print()
    print(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    print("=" * 60)

    report_path = fake_path.with_suffix(".eval.txt")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(
            f"global_wasserstein={wd:.6f}\n"
            f"global_ks={ks_stat:.6f}\n"
            f"global_p={p_val:.6f}\n"
            f"coverage={coverage:.6f}\n"
            f"combo_pass_rate={pass_rate:.6f}\n"
            f"mean_combo_pct_err={mean_pct_err:.6f}\n"
            f"overall={'PASS' if overall_pass else 'FAIL'}\n"
        )
    print(f"[evaluate] Saved summary to {report_path}")
