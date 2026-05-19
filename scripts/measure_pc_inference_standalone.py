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
import shutil
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import timm
import torch
import torchvision
import torchvision.models as tv_models

try:
    from benchmark_architectures import architecture_registry, count_params, profile_macs
    HAVE_BENCHMARK_HELPERS = True
except ImportError:
    HAVE_BENCHMARK_HELPERS = False

    def architecture_registry() -> dict[str, tuple[Any, str]]:
        return {}

    def count_params(model: torch.nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())

    def profile_macs(model: torch.nn.Module, sample: torch.Tensor) -> int:
        raise RuntimeError("benchmark_architectures.py is unavailable")


PROJECT_ROOT = Path.cwd()
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "pc_measurements"

PAPER_MODEL_ORDER = [
    "mobilenetv3_small",
    "mobilenetv2",
    "resnet18",
    "tiny_vit_5m",
    "efficientnet_b0",
]

PAPER_MODEL_MACS = {
    "mobilenetv3_small": 56_510_400,
    "mobilenetv2": 300_774_272,
    "resnet18": 1_814_073_344,
    "tiny_vit_5m": 1_165_342_848,
    "efficientnet_b0": 385_814_752,
}

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


def build_paper_model(name: str) -> torch.nn.Module:
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
    raise ValueError(f"unknown paper model: {name}")


def available_models(suite: str) -> dict[str, tuple[str, torch.nn.Module]]:
    models: dict[str, tuple[str, torch.nn.Module]] = {}
    if suite in {"paper", "all"}:
        for name in PAPER_MODEL_ORDER:
            models[name] = ("paper_model", build_paper_model(name))
    if suite in {"synthetic", "all"}:
        for name, (factory, family) in architecture_registry().items():
            models[name] = (family, factory())
    return models


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class PowerMeter:
    name = "none"
    measurement_type = "latency_only_no_power_backend"

    def start(self, raw_path: Path) -> Any:
        return None

    def stop(self, state: Any, elapsed_s: float) -> dict[str, Any]:
        return {
            "mean_power_W": "",
            "total_window_energy_J": "",
            "parsed_power_samples": 0,
            "power_note": "No supported direct power backend was selected or detected.",
            "raw_power_text": "",
        }


class NvidiaSmiPowerMeter(PowerMeter):
    name = "nvidia-smi"
    measurement_type = "direct_gpu_power_nvidia_smi"

    def __init__(self, sample_interval_ms: int) -> None:
        self.sample_interval_ms = sample_interval_ms

    @staticmethod
    def available() -> bool:
        return shutil.which("nvidia-smi") is not None

    @staticmethod
    def parse_power_w(line: str, *, allow_plain: bool = False) -> float | None:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) > 1:
            candidates = [parts[-1]]
        elif allow_plain:
            candidates = [line]
        else:
            return None
        for candidate in candidates:
            if not candidate or candidate.upper() in {"N/A", "[N/A]"}:
                continue
            match = re.search(r"([-+]?\d+(?:\.\d+)?)\s*(?:W)?\s*$", candidate)
            if match:
                return float(match.group(1))
        return None

    @staticmethod
    def parse_power_q_avg_w(text: str) -> float | None:
        in_power_samples = False
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if line == "Power Samples":
                in_power_samples = True
                continue
            if in_power_samples and line and not line.startswith(("Duration", "Number of Samples", "Max", "Min", "Avg")):
                in_power_samples = False
            if in_power_samples and line.startswith("Avg"):
                _, _, value = line.partition(":")
                return NvidiaSmiPowerMeter.parse_power_w(value, allow_plain=True)
        return None

    def start(self, raw_path: Path) -> dict[str, Any]:
        stop_event = threading.Event()
        samples: list[tuple[float, str]] = []

        def poll() -> None:
            cmd = [
                "nvidia-smi",
                "--query-gpu=timestamp,power.draw",
                "--format=csv,noheader,nounits",
            ]
            power_q_cmd = ["nvidia-smi", "-q", "-d", "POWER"]
            interval_s = max(0.05, self.sample_interval_ms / 1000.0)
            while not stop_event.is_set():
                try:
                    out = subprocess.check_output(
                        cmd,
                        text=True,
                        stderr=subprocess.DEVNULL,
                        encoding="utf-8",
                        errors="replace",
                    ).strip()
                    if out:
                        parsed_any = False
                        for line in out.splitlines():
                            samples.append((time.perf_counter(), line.strip()))
                            parsed_any = parsed_any or self.parse_power_w(line) is not None
                        if not parsed_any:
                            power_q = subprocess.check_output(
                                power_q_cmd,
                                text=True,
                                stderr=subprocess.DEVNULL,
                                encoding="utf-8",
                                errors="replace",
                            )
                            fallback_power = self.parse_power_q_avg_w(power_q)
                            if fallback_power is not None:
                                samples.append((time.perf_counter(), f"nvidia-smi -q Power Samples Avg, {fallback_power}"))
                            samples.append((time.perf_counter(), power_q.strip()))
                except Exception:
                    pass
                stop_event.wait(interval_s)

        thread = threading.Thread(target=poll, daemon=True)
        thread.start()
        return {"stop_event": stop_event, "thread": thread, "samples": samples}

    def stop(self, state: dict[str, Any], elapsed_s: float) -> dict[str, Any]:
        state["stop_event"].set()
        state["thread"].join(timeout=5)
        raw_lines = [line for _timestamp, line in state["samples"]]
        stdout = "\n".join(raw_lines)
        powers: list[float] = []
        for line in stdout.splitlines():
            power_w = self.parse_power_w(line)
            if power_w is not None:
                powers.append(power_w)
        mean_power = statistics.fmean(powers) if powers else float("nan")
        energy = mean_power * elapsed_s if math.isfinite(mean_power) else float("nan")
        return {
            "mean_power_W": round(mean_power, 6) if math.isfinite(mean_power) else "",
            "total_window_energy_J": round(energy, 9) if math.isfinite(energy) else "",
            "parsed_power_samples": len(powers),
            "power_note": "GPU power sampled with nvidia-smi; energy is GPU-only, not whole-system energy.",
            "raw_power_text": stdout,
        }


class RaplPowerMeter(PowerMeter):
    name = "rapl"
    measurement_type = "direct_cpu_package_energy_rapl"

    def __init__(self, energy_path: Path, max_range_path: Path | None) -> None:
        self.energy_path = energy_path
        self.max_range_path = max_range_path

    @staticmethod
    def detect() -> "RaplPowerMeter | None":
        candidates = sorted(Path("/sys/class/powercap").glob("intel-rapl:*/energy_uj"))
        if not candidates:
            return None
        energy_path = candidates[0]
        max_path = energy_path.with_name("max_energy_range_uj")
        return RaplPowerMeter(energy_path, max_path if max_path.exists() else None)

    def read_uj(self) -> int:
        return int(self.energy_path.read_text().strip())

    def start(self, raw_path: Path) -> int:
        return self.read_uj()

    def stop(self, state: int, elapsed_s: float) -> dict[str, Any]:
        end = self.read_uj()
        delta = end - state
        if delta < 0 and self.max_range_path is not None:
            delta += int(self.max_range_path.read_text().strip())
        energy = delta / 1_000_000.0
        mean_power = energy / elapsed_s if elapsed_s > 0 else float("nan")
        return {
            "mean_power_W": round(mean_power, 6) if math.isfinite(mean_power) else "",
            "total_window_energy_J": round(energy, 9),
            "parsed_power_samples": 2,
            "power_note": f"CPU package energy read from {self.energy_path}.",
            "raw_power_text": "",
        }


def choose_power_meter(args: argparse.Namespace, device: torch.device) -> PowerMeter:
    backend = args.power_backend
    if backend == "auto":
        if device.type == "cuda" and NvidiaSmiPowerMeter.available():
            backend = "nvidia-smi"
        elif platform.system().lower() == "linux" and RaplPowerMeter.detect() is not None:
            backend = "rapl"
        else:
            backend = "none"

    if backend == "nvidia-smi":
        if not NvidiaSmiPowerMeter.available():
            raise RuntimeError("nvidia-smi backend requested, but nvidia-smi was not found on PATH.")
        return NvidiaSmiPowerMeter(args.power_sample_interval_ms)
    if backend == "rapl":
        meter = RaplPowerMeter.detect()
        if meter is None:
            raise RuntimeError("RAPL backend requested, but /sys/class/powercap/intel-rapl was not found.")
        return meter
    if backend == "none":
        return PowerMeter()
    raise ValueError(f"unknown power backend: {backend}")


def warmup(model: torch.nn.Module, sample: torch.Tensor, device: torch.device, warmups: int) -> None:
    with torch.inference_mode():
        for _ in range(warmups):
            model(sample)
        synchronize(device)


def run_window(model: torch.nn.Module, sample: torch.Tensor, device: torch.device, seconds: float) -> tuple[int, float]:
    deadline = time.perf_counter() + seconds
    count = 0
    synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        while time.perf_counter() < deadline:
            model(sample)
            count += 1
    synchronize(device)
    end = time.perf_counter()
    return count, end - start


def finite_floats(rows: list[dict[str, Any]], key: str) -> list[float]:
    vals: list[float] = []
    for row in rows:
        value = row.get(key, "")
        if value == "":
            continue
        value_f = float(value)
        if math.isfinite(value_f):
            vals.append(value_f)
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def summarize(rows: list[dict[str, Any]], model_order: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for model_name in model_order:
        group = [r for r in rows if r["model"] == model_name]
        if not group:
            continue
        energies = finite_floats(group, "energy_J_per_inference")
        latencies = finite_floats(group, "latency_ms_per_inference")
        powers = finite_floats(group, "mean_power_W")
        infs = [int(r["inferences"]) for r in group]
        ci_low, ci_high, sem = ci95(energies)
        mean_energy = statistics.fmean(energies) if energies else float("nan")
        mean_latency = statistics.fmean(latencies) if latencies else float("nan")
        out.append(
            {
                "model": model_name,
                "family": group[0]["family"],
                "n_windows": len(group),
                "parsed_energy_windows": len(energies),
                "total_inferences": sum(infs),
                "inferences_per_window_mean": round(statistics.fmean(infs), 3) if infs else "",
                "params": group[0]["params"],
                "params_M": group[0]["params_M"],
                "macs": group[0]["macs"],
                "macs_G": group[0]["macs_G"],
                "flops": group[0]["flops"],
                "flops_G": group[0]["flops_G"],
                "latency_mean_ms": round(mean_latency, 6) if math.isfinite(mean_latency) else "",
                "latency_std_ms": round(statistics.stdev(latencies), 6) if len(latencies) > 1 else 0.0,
                "mean_power_W": round(statistics.fmean(powers), 6) if powers else "",
                "mean_power_std_W": round(statistics.stdev(powers), 6) if len(powers) > 1 else 0.0,
                "energy_mean_J": round(mean_energy, 9) if math.isfinite(mean_energy) else "",
                "energy_std_J": round(statistics.stdev(energies), 9) if len(energies) > 1 else 0.0,
                "energy_sem_J": round(sem, 9) if math.isfinite(sem) else "",
                "energy_ci95_low_J": round(ci_low, 9) if math.isfinite(ci_low) else "",
                "energy_ci95_high_J": round(ci_high, 9) if math.isfinite(ci_high) else "",
                "edp_J_ms": round(mean_energy * mean_latency, 9)
                if math.isfinite(mean_energy) and math.isfinite(mean_latency)
                else "",
                "measurement_type": group[0]["measurement_type"],
                "power_note": group[0]["power_note"],
            }
        )
    return out


def environment_metadata(args: argparse.Namespace, device: torch.device, meter: PowerMeter) -> dict[str, Any]:
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
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "device": str(device),
        "dtype": "float32",
        "input_shape": [args.batch_size, 3, args.image_size, args.image_size],
        "suite": args.suite,
        "windows_per_model": args.windows,
        "window_seconds": args.window_seconds,
        "warmups": args.warmups,
        "threads": args.threads,
        "random_seed": args.seed,
        "power_backend": meter.name,
        "measurement_type": meter.measurement_type,
        "nvidia_smi_path": shutil.which("nvidia-smi") or "",
        "git_commit": run_cmd(["git", "rev-parse", "HEAD"]),
        "git_branch": run_cmd(["git", "branch", "--show-current"]),
        "git_dirty": bool(run_cmd(["git", "status", "--porcelain"])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure inference latency and, when available, direct energy on a PC. "
            "Supports NVIDIA GPU power through nvidia-smi and Linux Intel RAPL CPU package energy."
        )
    )
    parser.add_argument("--suite", choices=["paper", "synthetic", "all"], default="paper")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--power-backend", choices=["auto", "none", "nvidia-smi", "rapl"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--windows", type=int, default=10)
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--cooldown-seconds", type=float, default=2.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--power-sample-interval-ms", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw_power"
    raw_dir.mkdir(parents=True, exist_ok=True)
    trials_path = args.output_dir / "pc_inference_trials.csv"
    summary_path = args.output_dir / "pc_inference_summary.csv"
    env_path = args.output_dir / "pc_measurement_environment.json"

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is false.")

    meter = choose_power_meter(args, device)
    models = available_models(args.suite)
    model_order = list(models.keys())
    sample = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)

    env_path.write_text(
        json.dumps(environment_metadata(args, device, meter), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    loaded: dict[str, tuple[str, torch.nn.Module, dict[str, Any]]] = {}
    print(f"[setup] device={device}, suite={args.suite}, power_backend={meter.name}")
    for name, (family, model) in models.items():
        model = model.eval().to(device)
        params = count_params(model)
        if HAVE_BENCHMARK_HELPERS:
            try:
                macs = profile_macs(model.cpu(), torch.randn(args.batch_size, 3, args.image_size, args.image_size))
            finally:
                model = model.to(device)
        else:
            macs = PAPER_MODEL_MACS.get(name, 0) * args.batch_size
        flops = 2 * macs
        loaded[name] = (
            family,
            model,
            {
                "params": params,
                "params_M": round(params / 1_000_000, 6),
                "macs": macs,
                "macs_G": round(macs / 1_000_000_000, 6),
                "flops": flops,
                "flops_G": round(flops / 1_000_000_000, 6),
            },
        )

    schedule: list[tuple[int, str]] = []
    for repeat in range(1, args.windows + 1):
        order = model_order[:]
        rng.shuffle(order)
        schedule.extend((repeat, name) for name in order)

    rows: list[dict[str, Any]] = []
    total = len(schedule)
    for idx, (repeat, name) in enumerate(schedule, start=1):
        family, model, static = loaded[name]
        print(f"[{idx:03d}/{total}] {name} repeat={repeat}")
        warmup(model, sample, device, args.warmups)
        raw_path = raw_dir / f"{idx:03d}_{name}_repeat{repeat}_{meter.name}.txt"
        start_utc = datetime.now(timezone.utc).isoformat()
        power_state = meter.start(raw_path)
        inference_count, elapsed_s = run_window(model, sample, device, args.window_seconds)
        power = meter.stop(power_state, elapsed_s)
        end_utc = datetime.now(timezone.utc).isoformat()
        raw_power_text = str(power.pop("raw_power_text", "") or "")
        raw_path.write_text(
            raw_power_text
            if raw_power_text
            else json.dumps(power, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        total_energy = power["total_window_energy_J"]
        energy_per_inf = ""
        if total_energy != "" and inference_count:
            energy_per_inf = round(float(total_energy) / inference_count, 9)
        latency_ms = elapsed_s * 1000.0 / inference_count if inference_count else float("nan")
        images_processed = inference_count * args.batch_size
        latency_ms_per_image = elapsed_s * 1000.0 / images_processed if images_processed else float("nan")
        energy_per_image = ""
        if total_energy != "" and images_processed:
            energy_per_image = round(float(total_energy) / images_processed, 9)
        row = {
            "model": name,
            "family": family,
            "repeat": repeat,
            "start_utc": start_utc,
            "end_utc": end_utc,
            "device": str(device),
            "measurement_type": meter.measurement_type,
            "window_seconds_requested": args.window_seconds,
            "window_seconds_actual": round(elapsed_s, 6),
            "inferences": inference_count,
            "batch_size": args.batch_size,
            "images_processed": images_processed,
            "latency_ms_per_inference": round(latency_ms, 6),
            "latency_ms_per_image": round(latency_ms_per_image, 6),
            "mean_power_W": power["mean_power_W"],
            "total_window_energy_J": total_energy,
            "energy_J_per_inference": energy_per_inf,
            "energy_J_per_image": energy_per_image,
            "parsed_power_samples": power["parsed_power_samples"],
            "power_note": power["power_note"],
            "raw_power_log": display_path(raw_path),
            "warmup_trials": args.warmups,
            "cooldown_seconds": args.cooldown_seconds,
            **static,
        }
        rows.append(row)
        write_csv(trials_path, rows)
        write_csv(summary_path, summarize(rows, model_order))
        print(
            "    latency={latency_ms_per_inference} ms, energy={energy_J_per_inference}, "
            "power={mean_power_W}, n={inferences}".format(**row)
        )
        if idx < total and args.cooldown_seconds > 0:
            time.sleep(args.cooldown_seconds)

    write_csv(trials_path, rows)
    write_csv(summary_path, summarize(rows, model_order))
    print(f"[write] {trials_path}")
    print(f"[write] {summary_path}")
    print(f"[write] {env_path}")


if __name__ == "__main__":
    main()
