#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "measured_architecture_benchmark.csv"
DEFAULT_TRIALS = PROJECT_ROOT / "measured_architecture_trials.csv"
DEFAULT_ENV = PROJECT_ROOT / "measurement_environment.json"
DEFAULT_RELEASE_COMPARISON = PROJECT_ROOT / "paper_baseline_comparison.csv"


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        stride: int = 1,
        groups: int = 1,
        act: bool = True,
    ) -> None:
        pad = kernel // 2
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=pad, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if act:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class Head(nn.Sequential):
    def __init__(self, channels: int, classes: int = 1000) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, classes),
        )


class BasicBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(channels, channels, 3),
            ConvBNAct(channels, channels, 3, act=False),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class BottleneckBlock(nn.Module):
    def __init__(self, channels: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(channels, hidden, 1),
            ConvBNAct(hidden, hidden, 3),
            ConvBNAct(hidden, channels, 1, act=False),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class InvertedResidual(nn.Module):
    def __init__(self, channels: int, expand_ratio: int = 4) -> None:
        super().__init__()
        hidden = channels * expand_ratio
        self.net = nn.Sequential(
            ConvBNAct(channels, hidden, 1),
            ConvBNAct(hidden, hidden, 3, groups=hidden),
            ConvBNAct(hidden, channels, 1, act=False),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class FireBlock(nn.Module):
    def __init__(self, in_ch: int, squeeze: int, expand: int) -> None:
        super().__init__()
        self.squeeze = ConvBNAct(in_ch, squeeze, 1)
        self.expand1 = ConvBNAct(squeeze, expand, 1)
        self.expand3 = ConvBNAct(squeeze, expand, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze(x)
        return torch.cat([self.expand1(x), self.expand3(x)], dim=1)


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 4) -> None:
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.BatchNorm2d(channels)
        self.pw = nn.Sequential(
            nn.Conv2d(channels, channels * expansion, 1),
            nn.GELU(),
            nn.Conv2d(channels * expansion, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pw(self.norm(self.dw(x)))


class SpatialGate(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class MixerBlock(nn.Module):
    def __init__(self, tokens: int, channels: int, token_hidden: int, channel_hidden: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.token_mlp = nn.Sequential(
            nn.Linear(tokens, token_hidden),
            nn.GELU(),
            nn.Linear(token_hidden, tokens),
        )
        self.norm2 = nn.LayerNorm(channels)
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, channel_hidden),
            nn.GELU(),
            nn.Linear(channel_hidden, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x).transpose(1, 2)
        x = x + self.token_mlp(y).transpose(1, 2)
        x = x + self.channel_mlp(self.norm2(x))
        return x


class PatchMixer(nn.Module):
    def __init__(self, channels: int, patch: int, depth: int, token_hidden: int, channel_hidden: int) -> None:
        super().__init__()
        tokens = (224 // patch) * (224 // patch)
        self.patch = nn.Conv2d(3, channels, patch, stride=patch)
        self.blocks = nn.Sequential(
            *[MixerBlock(tokens, channels, token_hidden, channel_hidden) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(channels)
        self.head = nn.Linear(channels, 1000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch(x).flatten(2).transpose(1, 2)
        x = self.blocks(x)
        x = self.norm(x).mean(dim=1)
        return self.head(x)


def simple_cnn(widths: list[int], name: str) -> tuple[str, nn.Module, str]:
    layers: list[nn.Module] = []
    in_ch = 3
    for i, out_ch in enumerate(widths):
        stride = 2 if i in {0, 1, 3} else 1
        layers.append(ConvBNAct(in_ch, out_ch, 3, stride=stride))
        in_ch = out_ch
    layers.append(Head(in_ch))
    return name, nn.Sequential(*layers), "plain_cnn"


def depthwise_cnn(widths: list[int], name: str) -> tuple[str, nn.Module, str]:
    layers: list[nn.Module] = [ConvBNAct(3, widths[0], 3, stride=2)]
    in_ch = widths[0]
    for i, out_ch in enumerate(widths[1:]):
        stride = 2 if i in {0, 2} else 1
        layers.extend(
            [
                ConvBNAct(in_ch, in_ch, 3, stride=stride, groups=in_ch),
                ConvBNAct(in_ch, out_ch, 1),
            ]
        )
        in_ch = out_ch
    layers.append(Head(in_ch))
    return name, nn.Sequential(*layers), "depthwise_cnn"


def architecture_registry() -> OrderedDict[str, tuple[Callable[[], nn.Module], str]]:
    return OrderedDict(
        [
            ("global_avg_mlp_tiny", (lambda: nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 1000)), "mlp")),
            ("cnn_tiny_2conv", (lambda: simple_cnn([16, 32], "cnn_tiny_2conv")[1], "plain_cnn")),
            ("cnn_small_4conv", (lambda: simple_cnn([16, 32, 48, 64], "cnn_small_4conv")[1], "plain_cnn")),
            ("cnn_medium_6conv", (lambda: simple_cnn([24, 48, 64, 96, 128, 128], "cnn_medium_6conv")[1], "plain_cnn")),
            ("depthwise_small", (lambda: depthwise_cnn([16, 24, 32, 48], "depthwise_small")[1], "depthwise_cnn")),
            ("depthwise_medium", (lambda: depthwise_cnn([24, 32, 48, 64, 96], "depthwise_medium")[1], "depthwise_cnn")),
            ("inverted_residual_small", (lambda: nn.Sequential(ConvBNAct(3, 24, 3, stride=2), InvertedResidual(24, 3), ConvBNAct(24, 48, 3, stride=2), InvertedResidual(48, 3), Head(48)), "inverted_residual")),
            ("inverted_residual_wide", (lambda: nn.Sequential(ConvBNAct(3, 32, 3, stride=2), InvertedResidual(32, 4), ConvBNAct(32, 64, 3, stride=2), InvertedResidual(64, 4), InvertedResidual(64, 4), Head(64)), "inverted_residual")),
            ("residual_cnn_small", (lambda: nn.Sequential(ConvBNAct(3, 32, 3, stride=2), BasicBlock(32), ConvBNAct(32, 64, 3, stride=2), BasicBlock(64), Head(64)), "residual_cnn")),
            ("residual_cnn_medium", (lambda: nn.Sequential(ConvBNAct(3, 48, 3, stride=2), BasicBlock(48), ConvBNAct(48, 96, 3, stride=2), BasicBlock(96), BasicBlock(96), Head(96)), "residual_cnn")),
            ("bottleneck_residual", (lambda: nn.Sequential(ConvBNAct(3, 64, 3, stride=2), BottleneckBlock(64, 16), ConvBNAct(64, 128, 3, stride=2), BottleneckBlock(128, 32), Head(128)), "bottleneck_cnn")),
            ("grouped_conv_cnn", (lambda: nn.Sequential(ConvBNAct(3, 32, 3, stride=2), ConvBNAct(32, 64, 3, stride=2, groups=4), ConvBNAct(64, 96, 3, groups=8), ConvBNAct(96, 128, 1), Head(128)), "grouped_cnn")),
            ("squeeze_expand_cnn", (lambda: nn.Sequential(ConvBNAct(3, 32, 3, stride=2), FireBlock(32, 16, 32), ConvBNAct(64, 96, 3, stride=2), FireBlock(96, 24, 48), Head(96)), "squeeze_cnn")),
            ("convnext_micro", (lambda: nn.Sequential(ConvBNAct(3, 48, 4, stride=4), ConvNeXtBlock(48), ConvNeXtBlock(48), ConvBNAct(48, 96, 2, stride=2), ConvNeXtBlock(96), Head(96)), "convnext_like")),
            ("attention_gate_cnn", (lambda: nn.Sequential(ConvBNAct(3, 32, 3, stride=2), ConvBNAct(32, 64, 3, stride=2), SpatialGate(64), ConvBNAct(64, 96, 3), SpatialGate(96), Head(96)), "attention_cnn")),
            ("patch_mixer_small", (lambda: PatchMixer(channels=64, patch=16, depth=2, token_hidden=96, channel_hidden=128), "mixer")),
            ("patch_mixer_medium", (lambda: PatchMixer(channels=96, patch=16, depth=3, token_hidden=128, channel_hidden=192), "mixer")),
        ]
    )


def run_cmd(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def hardware_profile() -> dict[str, str]:
    raw = run_cmd(["system_profiler", "SPHardwareDataType"]) or ""
    keep_prefixes = {
        "Model Name",
        "Model Identifier",
        "Model Number",
        "Chip",
        "Total Number of Cores",
        "Memory",
        "System Firmware Version",
        "OS Loader Version",
    }
    profile: dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        if key in keep_prefixes:
            profile[key] = value
    return profile


def os_profile() -> dict[str, str]:
    raw = run_cmd(["sw_vers"]) or ""
    profile: dict[str, str] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        profile[key.strip()] = value.strip()
    return profile


def environment_metadata(args: argparse.Namespace) -> dict[str, Any]:
    sysctl_keys = [
        "hw.model",
        "hw.machine",
        "hw.memsize",
        "hw.ncpu",
        "hw.physicalcpu",
        "hw.logicalcpu",
        "machdep.cpu.brand_string",
    ]
    raw_sysctl = {key: run_cmd(["sysctl", "-n", key]) for key in sysctl_keys}
    sysctl = {key: value for key, value in raw_sysctl.items() if value is not None}
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version.replace("\n", " "),
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "cuda_available": torch.cuda.is_available(),
        "device": "cpu",
        "dtype": "float32",
        "input_shape": [args.batch_size, 3, args.image_size, args.image_size],
        "warmup_trials": args.warmup,
        "measured_trials": args.trials,
        "threads_requested": args.threads,
        "constant_power_proxy_W": args.constant_power_w,
        "git_commit": run_cmd(["git", "rev-parse", "HEAD"]),
        "git_branch": run_cmd(["git", "branch", "--show-current"]),
        "git_dirty": bool(run_cmd(["git", "status", "--porcelain"])),
        "hardware_profile": hardware_profile(),
        "os_profile": os_profile(),
        "uname": run_cmd(["uname", "-a"]),
        "sysctl": sysctl,
        "sysctl_note": "Empty values mean macOS denied sysctl reads in the current sandbox.",
        "note": "Latency is directly measured. Energy and EDP proxy columns use a clearly labeled constant-power assumption, not powermetrics.",
    }


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def profile_macs(model: nn.Module, sample: torch.Tensor) -> int:
    macs = 0
    hooks = []

    def conv_hook(module: nn.Conv2d, _inp: tuple[torch.Tensor], out: torch.Tensor) -> None:
        nonlocal macs
        out_elements = out.numel()
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
        macs += int(out_elements * kernel_ops)

    def linear_hook(module: nn.Linear, inp: tuple[torch.Tensor], out: torch.Tensor) -> None:
        nonlocal macs
        in_features = module.in_features
        out_features = module.out_features
        batch_ops = out.numel() // out_features
        macs += int(batch_ops * in_features * out_features)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    with torch.inference_mode():
        model(sample)

    for hook in hooks:
        hook.remove()
    return macs


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct
    low = math.floor(idx)
    high = math.ceil(idx)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - idx) + ordered[high] * (idx - low)


def measure_latency(model: nn.Module, sample: torch.Tensor, warmup: int, trials: int) -> list[float]:
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)

        latencies: list[float] = []
        for _ in range(trials):
            start = time.perf_counter()
            model(sample)
            end = time.perf_counter()
            latencies.append((end - start) * 1000.0)
    return latencies


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def benchmark(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    sample = torch.randn(args.batch_size, 3, args.image_size, args.image_size)
    rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []

    for idx, (name, (factory, family)) in enumerate(architecture_registry().items(), start=1):
        model = factory().eval()
        params = count_params(model)
        model_size_mb = params * 4 / (1024 * 1024)
        macs = profile_macs(model, sample)
        flops = 2 * macs
        latencies = measure_latency(model, sample, args.warmup, args.trials)
        mean_ms = statistics.fmean(latencies)
        std_ms = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
        latency_s = mean_ms / 1000.0
        energy_proxy_j = args.constant_power_w * latency_s
        edp_proxy_j_s = energy_proxy_j * latency_s
        row = {
            "architecture": name,
            "family": family,
            "batch_size": args.batch_size,
            "input_shape": f"{args.batch_size}x3x{args.image_size}x{args.image_size}",
            "trials": args.trials,
            "warmup_trials": args.warmup,
            "params": params,
            "params_M": round(params / 1_000_000, 6),
            "model_size_MB_fp32": round(model_size_mb, 6),
            "macs": macs,
            "macs_G": round(macs / 1_000_000_000, 6),
            "flops": flops,
            "flops_G": round(flops / 1_000_000_000, 6),
            "latency_mean_ms": round(mean_ms, 6),
            "latency_std_ms": round(std_ms, 6),
            "latency_p50_ms": round(percentile(latencies, 0.50), 6),
            "latency_p95_ms": round(percentile(latencies, 0.95), 6),
            "latency_cv": round(std_ms / mean_ms if mean_ms > 0 else 0.0, 6),
            "throughput_fps": round(1000.0 / mean_ms if mean_ms > 0 else 0.0, 6),
            "macs_per_ms": round(macs / mean_ms if mean_ms > 0 else 0.0, 3),
            "constant_power_proxy_W": args.constant_power_w,
            "energy_proxy_J_constant_power": round(energy_proxy_j, 9),
            "edp_proxy_J_s_constant_power": round(edp_proxy_j_s, 12),
            "measurement_type": "direct_latency_repeated_trials",
            "energy_note": "Energy/EDP proxy uses constant-power assumption; not direct powermetrics measurement.",
        }
        rows.append(row)
        for trial_idx, latency_ms in enumerate(latencies, start=1):
            trial_rows.append(
                {
                    "architecture": name,
                    "family": family,
                    "trial": trial_idx,
                    "latency_ms": round(latency_ms, 6),
                    "batch_size": args.batch_size,
                    "input_shape": f"{args.batch_size}x3x{args.image_size}x{args.image_size}",
                }
            )
        print(f"[{idx:02d}/{len(architecture_registry())}] {name}: {mean_ms:.3f} +/- {std_ms:.3f} ms")

    return rows, trial_rows


def write_baseline_comparison(path: Path, benchmark_rows: list[dict[str, Any]]) -> None:
    rows = sorted(benchmark_rows, key=lambda r: float(r["latency_mean_ms"]))
    comparison = []
    for rank, row in enumerate(rows, start=1):
        comparison.append(
            {
                "latency_rank": rank,
                "architecture": row["architecture"],
                "family": row["family"],
                "params_M": row["params_M"],
                "model_size_MB_fp32": row["model_size_MB_fp32"],
                "macs_G": row["macs_G"],
                "flops_G": row["flops_G"],
                "latency_mean_ms": row["latency_mean_ms"],
                "latency_std_ms": row["latency_std_ms"],
                "latency_p95_ms": row["latency_p95_ms"],
                "throughput_fps": row["throughput_fps"],
                "energy_proxy_J_constant_power": row["energy_proxy_J_constant_power"],
                "edp_proxy_J_s_constant_power": row["edp_proxy_J_s_constant_power"],
                "measurement_type": row["measurement_type"],
            }
        )
    write_csv(path, comparison)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure local CPU latency for self-contained architecture variants.")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--constant-power-w", type=float, default=5.3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trials-output", type=Path, default=DEFAULT_TRIALS)
    parser.add_argument("--env-output", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--comparison-output", type=Path, default=DEFAULT_RELEASE_COMPARISON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    env = environment_metadata(args)
    rows, trial_rows = benchmark(args)
    write_csv(args.output, rows)
    write_csv(args.trials_output, trial_rows)
    write_baseline_comparison(args.comparison_output, rows)
    args.env_output.write_text(json.dumps(env, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[write] {args.output}")
    print(f"[write] {args.trials_output}")
    print(f"[write] {args.comparison_output}")
    print(f"[write] {args.env_output}")


if __name__ == "__main__":
    main()
