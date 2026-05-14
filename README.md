# Which Neural Networks Waste the Most Energy?

Companion repository for a small edge-AI inference-energy study. The current
main paper draft is an IEEE HPEC 2026 submission titled:

**When FLOPs Mislead: Benchmarking Neural Network Inference Energy on Apple Silicon**

The central result is narrow but reproducible: for five image-classification
architectures under CPU-only PyTorch inference on Apple Silicon, measured
per-inference energy tracks latency much more closely than FLOPs.

## Current Paper Files

All available paper versions are collected in [`papers/`](papers/):

- [`papers/HPEC2026_Submission.pdf`](papers/HPEC2026_Submission.pdf): current IEEE HPEC 2026 draft.
- [`papers/HPEC2026_Upload_Copy.pdf`](papers/HPEC2026_Upload_Copy.pdf): upload-safe copy of the HPEC PDF.
- [`papers/FTC2026_Anonymous_Submission.pdf`](papers/FTC2026_Anonymous_Submission.pdf): anonymized FTC 2026 version.
- [`papers/Revised_IEEE_Style_Manuscript.pdf`](papers/Revised_IEEE_Style_Manuscript.pdf): earlier revised IEEE-style manuscript.

Some root-level paper `.tex` and `.pdf` files are also kept for compatibility
with existing links and scripts, but `papers/` is the canonical place to find
all collected versions.

## What Is Measured

This repository separates direct measurements from derived or synthetic support
data.

- `paper_apple_silicon_benchmark.csv`: original five-model Apple-Silicon
  benchmark table used by the paper.
- `measured_energy_powermetrics/`: direct repeated `powermetrics` audit for the
  five paper models, including raw logs, trial rows, summary rows, and machine
  metadata.
- `measured_architecture_benchmark.csv`: direct latency-only benchmark for 17
  self-contained PyTorch architecture variants.
- `measured_architecture_trials.csv`: raw latency trials for the 17-architecture
  sweep.
- `paper_alignment_comparison.csv`, `paper_alignment_power_std_comparison.csv`,
  and `paper_supplemental_metrics.csv`: paper-aligned synthetic/support
  summaries.

Important interpretation:

- The five-model energy audit is direct measured energy.
- The 17-architecture sweep is direct latency only.
- Energy/EDP columns in the 17-architecture sweep are constant-power proxies,
  not `powermetrics` measurements.
- Synthetic spread estimates are support data, not replacement measurements.

## Repository Layout

```text
.
├── papers/                                  # collected paper versions
├── measured_energy_powermetrics/            # direct M4 Pro energy audit
├── scripts/
│   ├── benchmark_architectures.py           # 17-architecture latency sweep
│   ├── measure_energy_powermetrics.py       # five-model direct energy audit
│   └── validate_hpec_consistency.py         # paper/table consistency check
├── train.py                                 # residual WGAN-GP training
├── generate.py                              # synthetic sample generation
├── evaluate.py                              # synthetic fidelity evaluation
├── data_utils.py
├── gan_model.py
├── HPEC2026_Submission.tex/.pdf
├── paper_revised_latex_all_fixes.tex/.pdf
└── README.md
```

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Validate The Paper Numbers

Run this before submitting or pushing paper edits:

```bash
python3 scripts/validate_hpec_consistency.py
```

This checks that the HPEC tables match the committed CSV artifacts and that the
energy/latency summaries recompute from the raw trial rows.

## Reproduce The Latency Sweep

```bash
python3 scripts/benchmark_architectures.py --trials 30 --warmup 10 --threads 1
```

Outputs:

- `measured_architecture_benchmark.csv`
- `measured_architecture_trials.csv`
- `measurement_environment.json`
- `paper_baseline_comparison.csv`

## Reproduce The Direct Energy Audit

This requires macOS and `sudo` access because `powermetrics` needs elevated
privileges:

```bash
python3 scripts/measure_energy_powermetrics.py \
  --windows 10 \
  --window-seconds 20 \
  --sample-interval-ms 1000 \
  --warmups 10 \
  --cooldown-seconds 10 \
  --threads 1
```

Outputs are written under `measured_energy_powermetrics/`.

## Paper-Aligned Synthetic Workflow

The GAN workflow is supplemental. It models the Apple-Silicon rows using
`energy_J` and `latency_ms`, then derives power and spread values for support
tables.

```bash
python3 train.py \
  --data_csv paper_apple_silicon_benchmark.csv \
  --devices apple_silicon \
  --observed_only \
  --feature_mode paper_aligned

python3 generate.py \
  --checkpoint checkpoints/generator_final.pt \
  --data_csv paper_apple_silicon_benchmark.csv \
  --devices apple_silicon \
  --observed_only \
  --feature_mode paper_aligned \
  --match_seed_variance \
  --n_samples 10000 \
  --output paper_apple_silicon_synth_10k_fixed.csv

python3 evaluate.py \
  --csv paper_apple_silicon_synth_10k_fixed.csv \
  --data_csv paper_apple_silicon_benchmark.csv \
  --devices apple_silicon \
  --observed_only \
  --feature_mode paper_aligned
```

## Supporting Documentation

- [`MEASUREMENT_PROTOCOL.md`](MEASUREMENT_PROTOCOL.md): what is measured versus proxy/synthetic.
- [`PAPER_DATA_APPENDIX.md`](PAPER_DATA_APPENDIX.md): paper data provenance and limitations.
- [`ONLINE_BENCHMARK_AUDIT.md`](ONLINE_BENCHMARK_AUDIT.md): external benchmark source audit.
- [`RELEASE_MANIFEST.md`](RELEASE_MANIFEST.md): release file checklist.

## Status

The repository is intended to be the companion artifact for the HPEC 2026
submission. The public remote is:

```text
https://github.com/jubs-2431/which-neural-networks-waste-the-most-energy
```
