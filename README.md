# Which Neural Networks Waste the Most Energy?

Code and paper artifacts for the study:

*Which Neural Networks Waste the Most Energy? A Study of Edge AI Inference with Paper-Aligned Synthetic Trial Modeling*

This repository contains:
- the tabular WGAN-GP code used for synthetic inference-trial generation
- the Apple-Silicon paper-aligned benchmark inputs
- evaluation scripts for fidelity and ranking preservation
- the revised LaTeX manuscript and supporting appendix/CSV artifacts

## Repo Layout

```text
GAN/
├── train.py
├── generate.py
├── evaluate.py
├── gan_model.py
├── data_utils.py
├── paper_revised_latex.tex
├── benchmark_metadata.json
├── paper_apple_silicon_benchmark.csv
├── paper_alignment_comparison.csv
├── paper_alignment_power_std_comparison.csv
├── paper_supplemental_metrics.csv
├── PAPER_DATA_APPENDIX.md
└── README.md
```

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Two Workflows

### 1. General Residual GAN

This is the broader multi-device tabular GAN workflow.

```bash
python train.py --epochs 2000
python generate.py --n_samples 10000 --output edge_ai_synthetic_10k.csv
python evaluate.py --csv edge_ai_synthetic_10k.csv
```

### 2. Paper-Aligned Apple-Silicon Workflow

This is the workflow that matches the revised manuscript.

Key points:
- train only on the Apple-Silicon benchmark rows used by the paper
- treat `energy_J` and `latency_ms` as the modeled quantities
- derive `power_W` from energy and latency
- report repeated-trial `energy_std` only after within-model variance calibration

Recommended commands:

```bash
python train.py \
  --data_csv paper_apple_silicon_benchmark.csv \
  --devices apple_silicon \
  --observed_only

python generate.py \
  --checkpoint checkpoints_paper_apple_20260410/generator_final.pt \
  --data_csv paper_apple_silicon_benchmark.csv \
  --devices apple_silicon \
  --observed_only \
  --match_seed_variance \
  --derive_power \
  --recompute_energy_std \
  --n_samples 10000 \
  --output paper_apple_silicon_synth_10k_fixed.csv

python evaluate.py \
  --csv paper_apple_silicon_synth_10k_fixed.csv \
  --data_csv paper_apple_silicon_benchmark.csv \
  --devices apple_silicon \
  --observed_only \
  --feature_mode paper_aligned
```

## Current Paper-Aligned Results

These are the current calibrated paper-aligned evaluation results from
`paper_apple_silicon_synth_10k_fixed.csv`.

- `energy_J`: Wasserstein `0.003939`, KS `0.0636`, `p=0.1561`
- `latency_ms`: Wasserstein `0.693447`, KS `0.0669`, `p=0.1195`
- derived `power_W`: Wasserstein `0.045739`, KS `0.0465`, `p=0.4984`
- repeated-trial `energy_std`: matched after calibration
- energy coverage: `100%`
- ranking preserved: `mobilenetv3_small < mobilenetv2 < resnet18 < tiny_vit_5m < efficientnet_b0`

## Main Files

- `train.py`: WGAN-GP training loop with conditional generation by device/model
- `generate.py`: GAN sampling, postprocessing, and variance calibration
- `evaluate.py`: fidelity metrics, KS tests, coverage, and paper-aligned derived-metric evaluation
- `data_utils.py`: grounded seed construction, combo-aware scaling, and feature-mode support
- `paper_revised_latex.tex`: canonical IEEE-style revised manuscript
- `benchmark_metadata.json`: machine-readable benchmark scope, workflow defaults, and robustness checks

## Paper-Specific Artifacts

- `paper_apple_silicon_benchmark.csv`: Apple-Silicon benchmark rows aligned to the paper
- `benchmark_metadata.json`: reproducibility metadata and predictor sensitivity values
- `paper_apple_silicon_synth_10k_fixed.csv`: calibrated synthetic dataset used by the revised manuscript
- `paper_alignment_comparison.csv`: measured vs synthetic means
- `paper_alignment_power_std_comparison.csv`: derived power and calibrated spread comparison
- `paper_supplemental_metrics.csv`: paper-safe derived/support values
- `PAPER_DATA_APPENDIX.md`: explanation of what is measured, derived, and synthetic

## Important Interpretation Notes

- Mean energy and latency come from the paper's Apple-Silicon benchmark.
- Model-specific power values are derived from `energy_J * 1000 / latency_ms`.
- Trial-spread values are synthetic support estimates, not direct measurements.
- The paper-aligned evaluation should use `--feature_mode paper_aligned` in `evaluate.py`.

## Status

The repository is set up as the companion codebase for the paper and is pushed to:

`https://github.com/jubs-2431/which-neural-networks-waste-the-most-energy`
