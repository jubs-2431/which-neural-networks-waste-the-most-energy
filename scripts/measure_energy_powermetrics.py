#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import timm
import torch
import torchvision
import torchvision.models as tv_models


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from benchmark_architectures import architecture_registry

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "measured_energy_powermetrics"
DEFAULT_TRIALS = DEFAULT_OUTPUT_DIR / "measured_energy_trials.csv"
DEFAULT_SUMMARY = DEFAULT_OUTPUT_DIR / "measured_energy_summary.csv"
DEFAULT_ENV = DEFAULT_OUTPUT_DIR / "measurement_environment_energy.json"
DEFAULT_RAW_DIR = DEFAULT_OUTPUT_DIR / "raw_powermetrics"

PAPER5_MODEL_ORDER = [
    "mobilenetv3_small",
    "mobilenetv2",
    "resnet18",
    "tiny_vit_5m",
    "efficientnet_b0",
]
ARCHITECTURE17_MODEL_ORDER = list(architecture_registry().keys())
MODEL_FAMILIES = {name: family for name, (_, family) in architecture_registry().items()}
MODEL_FAMILIES.update(
    {
        "mobilenetv3_small": "mobilenet",
        "mobilenetv2": "mobilenet",
        "resnet18": "residual_cnn",
        "tiny_vit_5m": "vision_transformer",
        "efficientnet_b0": "efficientnet",
    }
)

POWER_RE = re.compile(
    r"^\s*(?P<label>[A-Za-z0-9_ /()+-]*Power(?:\s*\([^)]*\))?)\s*:\s*"
    r"(?P<value>[0-9]+(?:\.[0-9]+)?)\s*(?P<unit>mW|W)\b",
    re.IGNORECASE,
)

T_CRIT_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def run_cmd(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def hardware_profile() -> dict[str, str]:
    raw = run_cmd(["system_profiler", "SPHardwareDataType"]) or ""
    keep = {
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
        if key in keep:
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


def build_model(name: str) -> torch.nn.Module:
    registry = architecture_registry()
    if name in registry:
        factory, _family = registry[name]
        return factory()
    if name == "mobilenetv3_small":
        return tv_models.mobilenet_v3_small(weights=None)
    if name == "mobilenetv2":
        return tv_models.mobilenet_v2(weights=None)
    if name == "resnet18":
        return tv_models.resnet18(weights=None)
    if name == "tiny_vit_5m":
        return timm.create_model("tiny_vit_5m_224", pretrained=False)
    if name == "efficientnet_b0":
        return tv_models.efficientnet_b0(weights=None)
    raise ValueError(f"unknown model: {name}")


def selected_models(args: argparse.Namespace) -> list[str]:
    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    elif args.model_set == "paper5":
        models = PAPER5_MODEL_ORDER[:]
    elif args.model_set == "architecture17":
        models = ARCHITECTURE17_MODEL_ORDER[:]
    elif args.model_set == "paper5_plus_architecture17":
        models = PAPER5_MODEL_ORDER + [m for m in ARCHITECTURE17_MODEL_ORDER if m not in PAPER5_MODEL_ORDER]
    else:
        raise ValueError(f"unknown model set: {args.model_set}")

    known = set(PAPER5_MODEL_ORDER) | set(ARCHITECTURE17_MODEL_ORDER)
    unknown = [m for m in models if m not in known]
    if unknown:
        raise ValueError(f"unknown model(s): {', '.join(unknown)}")
    return models


def normalize_power_label(label: str) -> str | None:
    key = " ".join(label.lower().replace("_", " ").split())
    if key.startswith("combined power") or key.startswith("processor power") or key.startswith("package power"):
        return "combined"
    if key.startswith("cpu power"):
        return "cpu"
    if key.startswith("gpu power"):
        return "gpu"
    if key.startswith("ane power"):
        return "ane"
    return None


def parse_power_w(raw_text: str) -> dict[str, Any]:
    values: dict[str, list[float]] = {"combined": [], "cpu": [], "gpu": [], "ane": []}
    for line in raw_text.splitlines():
        match = POWER_RE.match(line)
        if not match:
            continue
        key = normalize_power_label(match.group("label"))
        if key is None:
            continue
        value = float(match.group("value"))
        if match.group("unit").lower() == "mw":
            value /= 1000.0
        values[key].append(value)

    means = {
        f"{key}_power_W_mean": statistics.fmean(vals) if vals else float("nan")
        for key, vals in values.items()
    }
    counts = {f"{key}_sample_count": len(vals) for key, vals in values.items()}
    if values["combined"]:
        mean_power_w = means["combined_power_W_mean"]
        power_source = "combined"
        sample_count = len(values["combined"])
    else:
        available = [k for k in ("cpu", "gpu", "ane") if values[k]]
        mean_power_w = sum(means[f"{k}_power_W_mean"] for k in available) if available else float("nan")
        power_source = "+".join(available) if available else "unparsed"
        sample_count = max((len(values[k]) for k in values), default=0)

    return {
        **means,
        **counts,
        "mean_power_W": mean_power_w,
        "power_source": power_source,
        "parsed_samples": sample_count,
    }


def mps_available() -> bool:
    return bool(torch.backends.mps.is_built() and torch.backends.mps.is_available())


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "mps":
        if not mps_available():
            raise RuntimeError("MPS backend was requested, but torch.backends.mps.is_available() is false")
        return torch.device("mps")
    raise ValueError(f"unknown backend: {name}")


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def warmup(model: torch.nn.Module, sample: torch.Tensor, warmups: int, device: torch.device) -> None:
    with torch.inference_mode():
        for _ in range(warmups):
            model(sample)
            synchronize_if_needed(device)


def run_inference_window(
    model: torch.nn.Module,
    sample: torch.Tensor,
    seconds: float,
    device: torch.device,
) -> tuple[int, float]:
    synchronize_if_needed(device)
    deadline = time.perf_counter() + seconds
    count = 0
    start = time.perf_counter()
    with torch.inference_mode():
        while time.perf_counter() < deadline:
            model(sample)
            synchronize_if_needed(device)
            count += 1
    synchronize_if_needed(device)
    end = time.perf_counter()
    return count, end - start


def run_powermetrics_window(
    args: argparse.Namespace,
    model_name: str,
    trial: int,
    raw_path: Path,
    model: torch.nn.Module,
    sample: torch.Tensor,
) -> dict[str, Any]:
    sample_interval_s = args.sample_interval_ms / 1000.0
    sample_count = max(1, math.ceil(args.window_seconds / sample_interval_s))
    window_seconds = sample_count * sample_interval_s
    command = [
        "sudo",
        "powermetrics",
        "--samplers",
        args.samplers,
        "-i",
        str(args.sample_interval_ms),
        "-n",
        str(sample_count),
        "-f",
        "text",
        "-o",
        str(raw_path),
    ]

    start_utc = datetime.now(timezone.utc).isoformat()
    proc = subprocess.Popen(command)
    time.sleep(args.powermetrics_start_delay_s)
    inference_count, elapsed_s = run_inference_window(model, sample, window_seconds, args.torch_device)
    return_code = proc.wait()
    end_utc = datetime.now(timezone.utc).isoformat()
    if return_code != 0:
        raise RuntimeError(f"powermetrics exited with status {return_code} for {model_name} trial {trial}")

    raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
    power = parse_power_w(raw_text)
    mean_power_w = float(power["mean_power_W"])
    total_window_energy_j = mean_power_w * elapsed_s if math.isfinite(mean_power_w) else float("nan")
    energy_j = total_window_energy_j / inference_count if inference_count else float("nan")
    latency_ms = (elapsed_s * 1000.0 / inference_count) if inference_count else float("nan")

    return {
        "model": model_name,
        "family": MODEL_FAMILIES.get(model_name, ""),
        "backend": args.backend,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "trial": trial,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "window_seconds_requested": args.window_seconds,
        "window_seconds_actual": round(elapsed_s, 6),
        "powermetrics_sample_interval_ms": args.sample_interval_ms,
        "powermetrics_requested_samples": sample_count,
        "parsed_samples": power["parsed_samples"],
        "power_source": power["power_source"],
        "inferences": inference_count,
        "latency_ms_per_inference": round(latency_ms, 6),
        "mean_power_W": round(mean_power_w, 6) if math.isfinite(mean_power_w) else "",
        "combined_power_W_mean": round(float(power["combined_power_W_mean"]), 6)
        if math.isfinite(float(power["combined_power_W_mean"]))
        else "",
        "cpu_power_W_mean": round(float(power["cpu_power_W_mean"]), 6)
        if math.isfinite(float(power["cpu_power_W_mean"]))
        else "",
        "gpu_power_W_mean": round(float(power["gpu_power_W_mean"]), 6)
        if math.isfinite(float(power["gpu_power_W_mean"]))
        else "",
        "ane_power_W_mean": round(float(power["ane_power_W_mean"]), 6)
        if math.isfinite(float(power["ane_power_W_mean"]))
        else "",
        "total_window_energy_J": round(total_window_energy_j, 9)
        if math.isfinite(total_window_energy_j)
        else "",
        "energy_J_per_inference": round(energy_j, 9) if math.isfinite(energy_j) else "",
        "warmup_trials": args.warmups,
        "cooldown_seconds": args.cooldown_seconds,
        "raw_powermetrics_log": str(raw_path.relative_to(PROJECT_ROOT)),
    }


def finite_floats(rows: list[dict[str, Any]], key: str) -> list[float]:
    vals: list[float] = []
    for row in rows:
        value = row.get(key, "")
        if value == "":
            continue
        f = float(value)
        if math.isfinite(f):
            vals.append(f)
    return vals


def ci95(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, mean, float("nan")
    sd = statistics.stdev(values)
    sem = sd / math.sqrt(len(values))
    tcrit = T_CRIT_95.get(len(values) - 1, 1.96)
    return mean - tcrit * sem, mean + tcrit * sem, sem


def summarize(rows: list[dict[str, Any]], model_order: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for model in model_order:
        group = [r for r in rows if r["model"] == model]
        energies = finite_floats(group, "energy_J_per_inference")
        latencies = finite_floats(group, "latency_ms_per_inference")
        powers = finite_floats(group, "mean_power_W")
        infs = [int(r["inferences"]) for r in group]
        ci_low, ci_high, sem = ci95(energies)
        out.append(
            {
                "model": model,
                "family": MODEL_FAMILIES.get(model, ""),
                "backend": group[0].get("backend", "") if group else "",
                "batch_size": group[0].get("batch_size", "") if group else "",
                "image_size": group[0].get("image_size", "") if group else "",
                "n_windows": len(group),
                "parsed_windows": len(energies),
                "total_inferences": sum(infs),
                "inferences_per_window_mean": round(statistics.fmean(infs), 3) if infs else "",
                "inferences_per_window_std": round(statistics.stdev(infs), 3) if len(infs) > 1 else 0.0,
                "energy_mean_J": round(statistics.fmean(energies), 9) if energies else "",
                "energy_std_J": round(statistics.stdev(energies), 9) if len(energies) > 1 else 0.0,
                "energy_sem_J": round(sem, 9) if math.isfinite(sem) else "",
                "energy_ci95_low_J": round(ci_low, 9) if math.isfinite(ci_low) else "",
                "energy_ci95_high_J": round(ci_high, 9) if math.isfinite(ci_high) else "",
                "latency_mean_ms": round(statistics.fmean(latencies), 6) if latencies else "",
                "latency_std_ms": round(statistics.stdev(latencies), 6) if len(latencies) > 1 else 0.0,
                "mean_power_W": round(statistics.fmean(powers), 6) if powers else "",
                "mean_power_std_W": round(statistics.stdev(powers), 6) if len(powers) > 1 else 0.0,
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def environment_metadata(args: argparse.Namespace, powermetrics_template: list[str], model_order: list[str]) -> dict[str, Any]:
    git_status = run_cmd(["git", "status", "--porcelain"])
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version.replace("\n", " "),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "timm_version": timm.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "backend": args.backend,
        "device": str(args.torch_device),
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "dtype": "float32",
        "input_shape": [args.batch_size, 3, args.image_size, args.image_size],
        "model_set": args.model_set,
        "models": model_order,
        "model_families": {name: MODEL_FAMILIES.get(name, "") for name in model_order},
        "windows_per_model": args.windows,
        "window_seconds": args.window_seconds,
        "sample_interval_ms": args.sample_interval_ms,
        "warmups": args.warmups,
        "cooldown_seconds": args.cooldown_seconds,
        "randomized_order": True,
        "random_seed": args.seed,
        "powermetrics_command_template": powermetrics_template,
        "powermetrics_note": "Run by subprocess as sudo; raw text logs are stored under raw_powermetrics.",
        "evidence_policy": {
            "raw_logs_retained": True,
            "per_window_inference_counts_retained": True,
            "environment_metadata_retained": True,
            "direct_energy_source": "powermetrics",
            "backend": f"{args.backend} PyTorch",
            "backend_scope_note": "Each backend/device condition is reported separately; CPU, MPS/Metal, ANE, GPU, quantized, batch-size, and input-size runs are not merged.",
            "outlier_policy": "No measured windows are filtered or winsorized. High-power windows remain in trials, summaries, confidence intervals, and paper analysis.",
            "sampling_note": "1 Hz powermetrics is coarse for sub-100 ms inference, so each row averages many inferences inside a measurement window.",
        },
        "hardware_profile": hardware_profile(),
        "os_profile": os_profile(),
        "uname": run_cmd(["uname", "-a"]),
        "git_commit": run_cmd(["git", "rev-parse", "HEAD"]),
        "git_branch": run_cmd(["git", "branch", "--show-current"]),
        "git_dirty": None if git_status is None else bool(git_status),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect repeated Apple-Silicon inference-energy windows with powermetrics."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--model-set",
        choices=["paper5", "architecture17", "paper5_plus_architecture17"],
        default="paper5",
        help="which benchmark model list to measure",
    )
    parser.add_argument("--models", default="", help="optional comma-separated explicit model list")
    parser.add_argument("--windows", type=int, default=10, help="measurement windows per model")
    parser.add_argument("--window-seconds", type=float, default=20.0)
    parser.add_argument("--sample-interval-ms", type=int, default=1000)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--cooldown-seconds", type=float, default=10.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--backend", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--samplers", default="cpu_power,gpu_power,ane_power")
    parser.add_argument("--powermetrics-start-delay-s", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser()
    if not args.output_dir.is_absolute():
        args.output_dir = PROJECT_ROOT / args.output_dir
    args.output_dir = args.output_dir.resolve()
    model_order = selected_models(args)
    args.torch_device = resolve_device(args.backend)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw_powermetrics"
    raw_dir.mkdir(parents=True, exist_ok=True)
    trials_path = args.output_dir / "measured_energy_trials.csv"
    summary_path = args.output_dir / "measured_energy_summary.csv"
    env_path = args.output_dir / "measurement_environment_energy.json"

    subprocess.run(["sudo", "-v"], check=True)

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(max(1, args.threads))
    except RuntimeError:
        pass
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    sample = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=args.torch_device)

    print(f"[load] building {len(model_order)} models from model_set={args.model_set}")
    models = {name: build_model(name).eval().to(args.torch_device) for name in model_order}

    schedule: list[tuple[int, str]] = []
    for repeat in range(1, args.windows + 1):
        order = model_order[:]
        rng.shuffle(order)
        schedule.extend((repeat, name) for name in order)

    powermetrics_template = [
        "sudo",
        "powermetrics",
        "--samplers",
        args.samplers,
        "-i",
        str(args.sample_interval_ms),
        "-n",
        str(math.ceil(args.window_seconds / (args.sample_interval_ms / 1000.0))),
        "-f",
        "text",
        "-o",
        "<raw_log_path>",
    ]
    env_path.write_text(
        json.dumps(environment_metadata(args, powermetrics_template, model_order), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rows: list[dict[str, Any]] = []
    total = len(schedule)
    for idx, (repeat, model_name) in enumerate(schedule, start=1):
        print(f"[{idx:02d}/{total}] {model_name} repeat={repeat}: warmup {args.warmups}")
        warmup(models[model_name], sample, args.warmups, args.torch_device)
        raw_path = raw_dir / f"{idx:03d}_{model_name}_repeat{repeat}.txt"
        row = run_powermetrics_window(args, model_name, repeat, raw_path, models[model_name], sample)
        row["schedule_index"] = idx
        rows.append(row)
        write_csv(trials_path, rows)
        write_csv(summary_path, summarize(rows, model_order))
        print(
            "    energy={energy_J_per_inference} J, latency={latency_ms_per_inference} ms, "
            "n={inferences}, samples={parsed_samples}, raw={raw_powermetrics_log}".format(**row)
        )
        if idx < total and args.cooldown_seconds > 0:
            print(f"    cooldown {args.cooldown_seconds:.1f}s")
            time.sleep(args.cooldown_seconds)

    write_csv(trials_path, rows)
    write_csv(summary_path, summarize(rows, model_order))
    print(f"[write] {trials_path}")
    print(f"[write] {summary_path}")
    print(f"[write] {env_path}")
    print(f"[write] {raw_dir}")


if __name__ == "__main__":
    main()
