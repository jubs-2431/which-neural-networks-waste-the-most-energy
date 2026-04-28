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

To turn the expanded sweep into direct energy evidence, run the benchmark while collecting synchronized `powermetrics` traces for each architecture. Add the resulting per-trial power traces and integrated energy values as new measured columns rather than replacing the current latency-only sweep.
