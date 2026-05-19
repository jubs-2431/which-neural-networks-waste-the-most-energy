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
- `measured_energy_powermetrics/`: direct M4 Pro audit.
- `measured_energy_powermetrics_friend_m4/`: second direct M4 Pro audit.
- `measured_energy_powermetrics_m5pro/`: direct M5 Pro audit.
- `measured_architecture_benchmark.csv`: 17-architecture latency sweep.
- `measured_architecture_trials.csv`: raw latency sweep trials.
- `paper_supplemental_metrics.csv`: paper support table.

The paper now compares the M4 audit, second M4 audit, and M5 audit without
deleting the original M4 data.

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
mkdir -p build
/Users/romikadiam/.local/bin/tectonic --outdir build --keep-logs HPEC2026_Submission.tex
```

## Reproduce The Mac Energy Audit

This requires macOS and `sudo` access because `powermetrics` needs elevated
privileges.

```bash
python3 scripts/measure_energy_powermetrics.py \
  --windows 10 \
  --window-seconds 20 \
  --sample-interval-ms 1000 \
  --warmups 10 \
  --cooldown-seconds 10 \
  --threads 1
```

To keep another machine separate, pass a custom output directory:

```bash
python3 scripts/measure_energy_powermetrics.py \
  --output-dir measured_energy_powermetrics_m5pro \
  --windows 10 \
  --window-seconds 20 \
  --sample-interval-ms 1000 \
  --warmups 10 \
  --cooldown-seconds 10 \
  --threads 1
```

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
