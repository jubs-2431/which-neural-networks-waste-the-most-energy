#!/usr/bin/env python3
"""Validate that HPEC paper tables match committed benchmark artifacts."""

from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "HPEC2026_Submission.tex"

MODEL_LABELS = {
    "MobileNetV3-S": "mobilenetv3_small",
    "MobileNetV2": "mobilenetv2",
    "ResNet-18": "resnet18",
    "Tiny-ViT-5M": "tiny_vit_5m",
    "EfficientNet-B0": "efficientnet_b0",
}

PRIMARY_FLOPS = {
    "mobilenetv3_small": 0.22,
    "mobilenetv2": 0.30,
    "resnet18": 1.82,
    "tiny_vit_5m": 1.30,
    "efficientnet_b0": 0.39,
}

ARCH_LABELS = {
    "Global avg. MLP tiny": "global_avg_mlp_tiny",
    "Patch mixer small": "patch_mixer_small",
    "Patch mixer medium": "patch_mixer_medium",
    "CNN tiny 2conv": "cnn_tiny_2conv",
    "CNN small 4conv": "cnn_small_4conv",
    "Depthwise small": "depthwise_small",
    "Grouped conv CNN": "grouped_conv_cnn",
    "Depthwise medium": "depthwise_medium",
    "Attention gate CNN": "attention_gate_cnn",
    "CNN medium 6conv": "cnn_medium_6conv",
    "Squeeze-expand CNN": "squeeze_expand_cnn",
    "Residual CNN small": "residual_cnn_small",
    "Inverted residual small": "inverted_residual_small",
    "Bottleneck residual": "bottleneck_residual",
    "Inverted residual wide": "inverted_residual_wide",
    "Residual CNN medium": "residual_cnn_medium",
    "ConvNeXt micro": "convnext_micro",
}


def expect_close(label: str, got: float, expected: float, decimals: int, issues: list[str]) -> None:
    tolerance = 0.5 * (10 ** -decimals) + 1e-12
    if abs(got - expected) > tolerance:
        issues.append(
            f"{label}: paper has {got}, expected {expected} at {decimals} decimals"
        )


def table_body(tex: str, label: str) -> str:
    match = re.search(
        rf"\\label\{{{re.escape(label)}\}}(?P<body>.*?)\\end\{{tabular\}}",
        tex,
        re.DOTALL,
    )
    if match is None:
        raise ValueError(f"could not find table body for {label}")
    return match.group("body")


def validate_hpec_tables(issues: list[str]) -> None:
    tex = TEX.read_text()

    primary = pd.read_csv(ROOT / "paper_supplemental_metrics.csv").set_index("model")
    for label, flops, latency, energy, edp in re.findall(
        r"(MobileNetV3-S|MobileNetV2|ResNet-18|Tiny-ViT-5M|EfficientNet-B0)"
        r" & ([0-9.]+) & ([0-9.]+) & ([0-9.]+) & ([0-9.]+)\\\\",
        table_body(tex, "tab:primary"),
    ):
        model = MODEL_LABELS[label]
        expect_close(f"{label} FLOPs", float(flops), PRIMARY_FLOPS[model], 2, issues)
        expect_close(f"{label} latency", float(latency), primary.loc[model, "paper_latency_ms"], 2, issues)
        expect_close(f"{label} energy", float(energy), primary.loc[model, "paper_energy_J"], 3, issues)
        expect_close(
            f"{label} EDP",
            float(edp),
            primary.loc[model, "paper_energy_J"] * primary.loc[model, "paper_latency_ms"],
            3,
            issues,
        )

    audit_specs = [
        ("M1", ROOT / "measured_energy_powermetrics_m1/measured_energy_summary.csv"),
        ("M4 Pro", ROOT / "measured_energy_powermetrics/measured_energy_summary.csv"),
        ("M5 Pro", ROOT / "measured_energy_powermetrics_m5pro/measured_energy_summary.csv"),
    ]
    figure_body = re.search(
        r"\\begin\{figure\*\}\[t\](?P<body>.*?)\\end\{figure\*\}",
        tex,
        re.DOTALL,
    )
    if figure_body is None:
        issues.append("could not find audit energy figure")
    else:
        figure_text = figure_body.group("body")
        for description, csv_path in audit_specs:
            audit = pd.read_csv(csv_path)
            for _, row in audit.iterrows():
                rounded = f"{row.energy_mean_J:.3f}"
                if rounded not in figure_text:
                    issues.append(
                        f"{description} {row.model} energy {rounded} not found in audit energy figure"
                    )

    arch = pd.read_csv(ROOT / "measured_architecture_benchmark.csv").set_index("architecture")
    for label, _family, params, flops, latency, p95, cv in re.findall(
        r"([^&\n]+) & ([^&\n]+) & ([0-9.]+) & ([0-9.]+) & ([0-9.]+) & ([0-9.]+) & ([0-9.]+)\\\\",
        tex,
    ):
        label = label.strip()
        if label not in ARCH_LABELS:
            continue
        row = arch.loc[ARCH_LABELS[label]]
        expect_close(f"{label} params", float(params), row.params_M, 3, issues)
        expect_close(f"{label} FLOPs", float(flops), row.flops_G, 4, issues)
        expect_close(f"{label} latency", float(latency), row.latency_mean_ms, 3, issues)
        expect_close(f"{label} p95", float(p95), row.latency_p95_ms, 3, issues)
        expect_close(f"{label} CV", float(cv), row.latency_cv, 3, issues)


def validate_energy_audit_internal_consistency(audit_dir: Path, description: str, issues: list[str]) -> None:
    energy_trials = pd.read_csv(audit_dir / "measured_energy_trials.csv")
    energy_summary = pd.read_csv(audit_dir / "measured_energy_summary.csv")
    raw_logs = list((audit_dir / "raw_powermetrics").glob("*.txt"))

    if len(energy_trials) != 50:
        issues.append(f"expected 50 {description} energy trial rows, found {len(energy_trials)}")
    if len(raw_logs) != 50:
        issues.append(f"expected 50 {description} raw powermetrics logs, found {len(raw_logs)}")

    expected_windows = dict(zip(energy_summary.model, energy_summary.n_windows))
    actual_windows = energy_trials.groupby("model").size().to_dict()
    if actual_windows != expected_windows:
        issues.append(f"{description} energy window counts mismatch: {actual_windows} != {expected_windows}")

    grouped = energy_trials.groupby("model").agg(
        energy_mean=("energy_J_per_inference", "mean"),
        latency_mean=("latency_ms_per_inference", "mean"),
        power_mean=("mean_power_W", "mean"),
    )
    for _, row in energy_summary.iterrows():
        calc = grouped.loc[row.model]
        expect_close(f"{description} {row.model} summary energy", row.energy_mean_J, calc.energy_mean, 6, issues)
        expect_close(f"{description} {row.model} summary latency", row.latency_mean_ms, calc.latency_mean, 6, issues)
        expect_close(f"{description} {row.model} summary power", row.mean_power_W, calc.power_mean, 6, issues)


def validate_artifact_internal_consistency(issues: list[str]) -> None:
    validate_energy_audit_internal_consistency(ROOT / "measured_energy_powermetrics_m1", "M1", issues)
    validate_energy_audit_internal_consistency(ROOT / "measured_energy_powermetrics", "M4", issues)
    validate_energy_audit_internal_consistency(ROOT / "measured_energy_powermetrics_m5pro", "M5", issues)

    arch_trials = pd.read_csv(ROOT / "measured_architecture_trials.csv")
    arch_summary = pd.read_csv(ROOT / "measured_architecture_benchmark.csv")
    if len(arch_summary) != 17:
        issues.append(f"expected 17 architecture rows, found {len(arch_summary)}")
    if len(arch_trials) != 510:
        issues.append(f"expected 510 architecture trial rows, found {len(arch_trials)}")
    if not all(arch_trials.groupby("architecture").size() == 30):
        issues.append("not every architecture has exactly 30 latency trials")

    trial_means = arch_trials.groupby("architecture").latency_ms.mean()
    for _, row in arch_summary.iterrows():
        expect_close(
            f"{row.architecture} latency summary",
            row.latency_mean_ms,
            trial_means[row.architecture],
            6,
            issues,
        )

    baseline = pd.read_csv(ROOT / "paper_baseline_comparison.csv")
    expected_order = list(arch_summary.sort_values("latency_mean_ms").architecture)
    if list(baseline.architecture) != expected_order:
        issues.append("paper_baseline_comparison.csv is not sorted by latency_mean_ms")
    if list(baseline.latency_rank) != list(range(1, 18)):
        issues.append("paper_baseline_comparison.csv latency_rank is not 1..17")


def main() -> None:
    issues: list[str] = []
    validate_hpec_tables(issues)
    validate_artifact_internal_consistency(issues)
    if issues:
        for issue in issues:
            print(f"FAIL: {issue}")
        raise SystemExit(1)
    print("OK: HPEC tables and benchmark artifacts are consistent.")


if __name__ == "__main__":
    main()
