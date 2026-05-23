# Mac Apple Silicon Inference Energy Paper

This is the Apple-Silicon version of the neural-network inference-energy
project. It studies CPU-only PyTorch inference on Apple Silicon and supports
the IEEE HPEC 2026 draft:

**When FLOPs Mislead: Benchmarking Neural Network Inference Energy on Apple Silicon**

## Start Here

- Current easy-to-find paper copy:
  - `current_paper/Mac_Apple_Silicon_HPEC2026_Submission.pdf`
  - `current_paper/Mac_Apple_Silicon_HPEC2026_Submission.tex`
- Submission-compatible paper paths still used by scripts and prior links:
  - `HPEC2026_Submission.pdf`
  - `HPEC2026_Submission.tex`
- Collected older/submission versions:
  - `papers/`

## Main Data

- `paper_apple_silicon_benchmark.csv`: original five-model paper table.
- `measured_energy_powermetrics/`: direct M4 Pro five-model audit.
- `measured_energy_powermetrics_friend_m4/`: second direct M4 Pro five-model audit.
- `measured_energy_powermetrics_m5pro/`: direct M5 Pro audit.
- `measured_architecture_benchmark.csv`: 17-architecture latency sweep.
- `measured_architecture_trials.csv`: raw latency sweep trials.
- `measured_energy_powermetrics_m4pro_arch17_cpu/`: direct M4 Pro 17-architecture CPU energy audit.
- `paper_supplemental_metrics.csv`: paper support table.

The paper now uses the M4 Pro 17-architecture `powermetrics` audit as the main
expanded direct-energy result. The older five-model M4/M5 audits remain
supporting evidence and are not merged into one undifferentiated device claim.

## Active Scripts

- `scripts/measure_energy_powermetrics.py`: run a direct macOS `powermetrics`
  energy audit.
- `scripts/benchmark_architectures.py`: run the 17-architecture latency sweep.
- `scripts/validate_hpec_consistency.py`: check that the paper tables match
  committed CSV artifacts.
- `scripts/measure_pc_inference.py`: PC/NVIDIA measurement helper kept here so
  the same script can be shared with the Windows project.

## Validate

Run this before submitting or pushing paper edits:

```bash
python3 scripts/validate_hpec_consistency.py
```

Build the HPEC PDF with Tectonic:

```bash
tectonic -X compile HPEC2026_Submission.tex
```

## Reproduce The Mac Energy Audit

This requires macOS and `sudo` access because `powermetrics` needs elevated
privileges.

Five-model audit:

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

Full 17-architecture M4 Pro CPU audit:

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

Short follow-up audits for limitations checks:

```bash
# MPS/Metal backend comparison, batch 1, 224x224.
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

# Batch-size sensitivity on CPU, batch 4, 224x224.
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

# Input-size sensitivity on CPU, 128x128. Patch mixers are excluded because
# their token MLP is shape-fixed to 224x224 in this architecture registry.
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

For another machine, keep the protocol fixed and change only `--output-dir`,
for example `measured_energy_powermetrics_m1_arch17_cpu` or
`measured_energy_powermetrics_m5pro_arch17_cpu`.

## Current 17-Architecture Energy Result

The expanded M4 Pro audit has 170 measured windows: 10 windows per
architecture across 17 architecture variants, 30 seconds per window, 20 warmups
per window, 1 Hz `powermetrics` sampling, raw logs, per-window inference
counts, confidence intervals, and environment metadata.

Across the 17 direct-energy rows:

- Latency predicts measured energy with `R^2 = 0.9925`.
- FLOPs reaches only `R^2 = 0.4008`.
- Parameter count reaches only `R^2 = 0.0977`.

The `energy_proxy_J_constant_power` and `edp_proxy_J_s_constant_power` columns
in `measured_architecture_benchmark.csv` remain constant-power proxy fields.
Direct energy for those architectures is stored separately in
`measured_energy_powermetrics_m4pro_arch17_cpu/`.

Outlier policy: no measured windows are filtered or winsorized. High-power
windows remain in raw logs, trial CSVs, summary statistics, confidence
intervals, and paper analysis.

## Archive

Old drafts, GAN/synthetic experiments, external benchmark notes, and legacy
LaTeX support files were moved into `archive/` so the root folder stays focused:

- `archive/older_papers/`
- `archive/supplemental_docs/`
- `archive/gan_workflow/`
- `archive/legacy_data/`
- `archive/legacy_latex_support/`
- `archive/generated_builds/`
- `archive/generated_cache/`

These files were preserved for reference; they are not the current paper.

## Related Project

The separate Windows/NVIDIA RTX paper is in:

```text
/Users/romikadiam/windows-rtx-gpu-inference-energy-paper
```
