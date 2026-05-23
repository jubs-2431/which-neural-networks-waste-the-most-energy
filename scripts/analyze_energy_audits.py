#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def r2(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    mask = x.notna() & y.notna()
    corr = float(np.corrcoef(x[mask], y[mask])[0, 1])
    return corr * corr, corr


def load_summary(audit_dir: Path) -> pd.DataFrame:
    summary_path = audit_dir / "measured_energy_summary.csv"
    env_path = audit_dir / "measurement_environment_energy.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    df = pd.read_csv(summary_path)
    if env_path.exists():
        env = json.loads(env_path.read_text())
        df["audit_dir"] = audit_dir.name
        df["backend_env"] = env.get("backend", env.get("device", ""))
        df["batch_size_env"] = env.get("input_shape", ["", "", "", ""])[0]
        df["image_size_env"] = env.get("input_shape", ["", "", "", ""])[-1]
    return df


def load_trials(audit_dir: Path) -> pd.DataFrame:
    trials_path = audit_dir / "measured_energy_trials.csv"
    if not trials_path.exists():
        raise FileNotFoundError(trials_path)
    df = pd.read_csv(trials_path)
    df["audit_dir"] = audit_dir.name
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize direct powermetrics audit results.")
    parser.add_argument("audit_dirs", nargs="+", type=Path)
    args = parser.parse_args()

    arch = pd.read_csv(PROJECT_ROOT / "measured_architecture_benchmark.csv")[
        ["architecture", "flops_G", "params_M"]
    ].rename(columns={"architecture": "model"})

    for raw_dir in args.audit_dirs:
        audit_dir = raw_dir if raw_dir.is_absolute() else PROJECT_ROOT / raw_dir
        summary = load_summary(audit_dir).merge(arch, on="model", how="left")
        trials = load_trials(audit_dir)

        print(f"\n== {audit_dir.name} ==")
        print(f"models={summary['model'].nunique()} windows={len(trials)}")
        for predictor in ("latency_mean_ms", "flops_G", "params_M"):
            if predictor in summary:
                val, corr = r2(summary[predictor], summary["energy_mean_J"])
                print(f"{predictor}: R2={val:.4f} r={corr:.4f}")

        print("\nEnergy ranking:")
        cols = ["model", "family", "energy_mean_J", "latency_mean_ms", "flops_G"]
        print(summary.sort_values("energy_mean_J")[cols].to_string(index=False))

        print("\nLargest retained windows:")
        top = trials.nlargest(5, "energy_J_per_inference")[
            [
                "schedule_index",
                "model",
                "trial",
                "energy_J_per_inference",
                "latency_ms_per_inference",
                "mean_power_W",
                "raw_powermetrics_log",
            ]
        ]
        print(top.to_string(index=False))


if __name__ == "__main__":
    main()
