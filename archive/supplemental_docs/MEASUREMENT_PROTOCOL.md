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
models under `measured_energy_powermetrics/`. To reproduce that smaller audit,
run:

```bash
.venv/bin/python scripts/measure_energy_powermetrics.py \
  --model-set paper5 \
  --windows 10 \
  --window-seconds 20 \
  --sample-interval-ms 1000 \
  --warmups 10 \
  --cooldown-seconds 10 \
  --threads 1
```

To apply the stronger follow-up protocol and turn the 17-architecture latency
sweep into direct energy evidence, run one matched CPU-only audit per machine:

```bash
.venv/bin/python scripts/measure_energy_powermetrics.py \
  --model-set architecture17 \
  --output-dir measured_energy_powermetrics_m4pro_arch17_cpu \
  --windows 10 \
  --window-seconds 30 \
  --sample-interval-ms 1000 \
  --warmups 20 \
  --cooldown-seconds 10 \
  --threads 1 \
  --batch-size 1 \
  --image-size 224 \
  --seed 20260523
```

For friend machines, keep every argument the same and change only the output
directory so the artifacts remain machine-specific:

```bash
# Apple M1
.venv/bin/python scripts/measure_energy_powermetrics.py \
  --model-set architecture17 \
  --output-dir measured_energy_powermetrics_m1_arch17_cpu \
  --windows 10 \
  --window-seconds 30 \
  --sample-interval-ms 1000 \
  --warmups 20 \
  --cooldown-seconds 10 \
  --threads 1 \
  --batch-size 1 \
  --image-size 224 \
  --seed 20260523

# Apple M5 Pro
.venv/bin/python scripts/measure_energy_powermetrics.py \
  --model-set architecture17 \
  --output-dir measured_energy_powermetrics_m5pro_arch17_cpu \
  --windows 10 \
  --window-seconds 30 \
  --sample-interval-ms 1000 \
  --warmups 20 \
  --cooldown-seconds 10 \
  --threads 1 \
  --batch-size 1 \
  --image-size 224 \
  --seed 20260523
```

For the current limitation checks, keep these audits separate from the main
CPU batch-1 224x224 table:

```bash
# MPS/Metal backend comparison.
.venv/bin/python scripts/measure_energy_powermetrics.py \
  --model-set architecture17 \
  --backend mps \
  --output-dir measured_energy_powermetrics_m4pro_arch17_mps_b1_224 \
  --windows 5 \
  --window-seconds 20 \
  --sample-interval-ms 1000 \
  --warmups 20 \
  --cooldown-seconds 5 \
  --threads 1 \
  --batch-size 1 \
  --image-size 224 \
  --seed 20260524

# Batch-size sensitivity.
.venv/bin/python scripts/measure_energy_powermetrics.py \
  --model-set architecture17 \
  --backend cpu \
  --output-dir measured_energy_powermetrics_m4pro_arch17_cpu_b4_224 \
  --windows 5 \
  --window-seconds 20 \
  --sample-interval-ms 1000 \
  --warmups 20 \
  --cooldown-seconds 5 \
  --threads 1 \
  --batch-size 4 \
  --image-size 224 \
  --seed 20260524

# Input-size sensitivity. Patch mixers are excluded because their token MLP
# is shape-fixed to 224x224 in this architecture registry.
.venv/bin/python scripts/measure_energy_powermetrics.py \
  --backend cpu \
  --models global_avg_mlp_tiny,cnn_tiny_2conv,cnn_small_4conv,cnn_medium_6conv,depthwise_small,depthwise_medium,inverted_residual_small,inverted_residual_wide,residual_cnn_small,residual_cnn_medium,bottleneck_residual,grouped_conv_cnn,squeeze_expand_cnn,convnext_micro,attention_gate_cnn \
  --output-dir measured_energy_powermetrics_m4pro_arch15_cpu_b1_128 \
  --windows 5 \
  --window-seconds 20 \
  --sample-interval-ms 1000 \
  --warmups 20 \
  --cooldown-seconds 5 \
  --threads 1 \
  --batch-size 1 \
  --image-size 128 \
  --seed 20260524
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
It was collected on the local Apple M4 Pro MacBook Pro recorded in
`measured_energy_powermetrics/measurement_environment_energy.json`; it is a
new reproducibility audit and should not be presented as the missing original
raw windows behind `paper_apple_silicon_benchmark.csv`.

Outlier policy: no measured windows are filtered or winsorized. High-power
windows remain in raw logs, trial CSVs, summary statistics, confidence
intervals, and paper analysis.
