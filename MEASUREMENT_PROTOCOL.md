# Measurement Protocol

This repository now separates three evidence types:

- Direct Apple-Silicon energy measurements from the original five-model paper table.
- Direct local repeated-trial latency measurements from the expanded 17-architecture PyTorch sweep.
- Synthetic or proxy quantities, which are clearly labeled and should not be described as direct measurements.

## Expanded Architecture Sweep

Run:

```bash
.venv/bin/python scripts/benchmark_architectures.py --trials 30 --warmup 10 --threads 1
```

Outputs:

- `measured_architecture_benchmark.csv`: one row per architecture with params, model size, MACs, FLOPs, latency mean/std/p50/p95, throughput, and clearly labeled proxy energy/EDP columns.
- `measured_architecture_trials.csv`: all 510 raw latency trials, 30 per architecture.
- `measurement_environment.json`: exact measurement-machine metadata, PyTorch version, benchmark settings, input shape, and Git revision.
- `paper_baseline_comparison.csv`: latency-ranked comparison table for paper/review use.

## What Is Directly Measured

- Latency is directly measured with `time.perf_counter()` under `torch.inference_mode()`.
- Each architecture receives 10 warm-up runs and 30 measured runs.
- Input shape is `1 x 3 x 224 x 224`.
- Execution is CPU-only float32 PyTorch.
- Thread count is fixed to one with `torch.set_num_threads(1)` for lower scheduling variance.

## What Is Not Directly Measured

- The expanded architecture sweep does not directly measure wall-power, CPU package power, or energy.
- `energy_proxy_J_constant_power` and `edp_proxy_J_s_constant_power` assume a constant `5.3 W` reference power only to make latency-normalized comparisons possible.
- These proxy columns must not be described as powermetrics measurements.

## Direct Energy Extension

This release includes a direct repeated `powermetrics` audit for the five paper
models under `measured_energy_powermetrics/`. To reproduce or extend it, run:

```bash
python3 scripts/measure_energy_powermetrics.py \
  --windows 10 \
  --window-seconds 20 \
  --sample-interval-ms 1000 \
  --warmups 10 \
  --cooldown-seconds 10 \
  --threads 1
```

This script prompts for `sudo` because macOS requires superuser privileges for
`powermetrics`. It writes:

- `measured_energy_powermetrics/measured_energy_trials.csv`: one row per
  measured model/window with measured power, latency, inference count, and
  per-inference energy.
- `measured_energy_powermetrics/measured_energy_summary.csv`: per-model
  measured mean, standard deviation, SEM, 95% CI, and window count.
- `measured_energy_powermetrics/measurement_environment_energy.json`: exact
  chip, macOS, Python, PyTorch, torchvision, timm, thread count, warmups,
  cooldown, randomized order seed, and `powermetrics` command template.
- `measured_energy_powermetrics/raw_powermetrics/`: raw text logs, one file per
  model/window.

The script uses CPU-only float32 inference, randomized model order across
repeats, a fixed input shape of `1 x 3 x 224 x 224`, and
`powermetrics --samplers cpu_power,gpu_power,ane_power`.

The included audit used 10 windows per model, 20 seconds per window, 20
`powermetrics` samples per window at 1 Hz, 10 warm-up inferences before each
window, 10 seconds of cooldown between windows, and one PyTorch CPU thread.
It was collected on the local Apple M1 MacBook Pro recorded in
`measured_energy_powermetrics/measurement_environment_energy.json`; it is a
new reproducibility audit and should not be presented as the missing original
raw windows behind `paper_apple_silicon_benchmark.csv`.
