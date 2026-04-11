# Edge AI Energy GAN

Measured-aware synthetic data pipeline for **10,000 inference energy
measurements** across 6 edge hardware platforms and 5 neural network
architectures.

The current GAN trains on combo-normalized residuals instead of raw physical
values. That change prevents the mode collapse seen in the earlier raw-scale
WGAN-GP setup.

Based on: *"Which Neural Networks Waste the Most Energy? A Study of Edge AI Inference"*
(Aryan Shah, Texas Academy of Mathematics and Science / UNT)

---

## Design Decisions

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Noise σ | 5% of mean | Matches powermetrics measurement variance from paper |
| Trials per combo | 100 | 30 combos × 100 = 3,000 seed rows — enough for stable GAN training |
| Target samples | 10,000 | ~333 per combo — statistically sufficient for correlation analysis |
| Architecture | WGAN-GP + spectral norm | Prevents mode collapse on small tabular datasets |
| n_critic | 5 | Standard WGAN-GP; prevents discriminator overfitting |
| Latent dim | 64 | Sufficient capacity for 4-dimensional output |

---

## Files

```
edge_ai_gan/
├── gan_model.py      # Generator, Discriminator, gradient penalty
├── data_utils.py     # Seed data generation, DataScaler, EnergyDataset
├── train.py          # WGAN-GP training loop with checkpointing
├── generate.py       # Load trained G → produce 10,000 samples → CSV
├── evaluate.py       # Wasserstein, KS test, coverage, correlation metrics
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the GAN  (~5 min CPU / ~1 min GPU)
python train.py --epochs 2000

# 3. Generate 10,000 samples with the residualized GAN
python generate.py --n_samples 10000 --output edge_ai_synthetic_10k.csv

# 4. Optional: compare against the grounded bootstrap baseline
python generate.py --method bootstrap \
  --output edge_ai_synthetic_10k_bootstrap.csv

# 5. Evaluate fidelity
python evaluate.py --csv edge_ai_synthetic_10k.csv
```

## Recommended Use

- Use `python generate.py` for the current residualized GAN.
- Use `python generate.py --method bootstrap ...` as a grounded baseline.
- Keep checking the GAN against `evaluate.py` whenever the benchmark file
  changes materially.

## Paper-Aligned Workflow

To match `main.pdf` as closely as possible, use the Apple-Silicon-only benchmark
file and restrict generation to observed combos:

```bash
python train.py \
  --data_csv paper_apple_silicon_benchmark.csv \
  --devices apple_silicon \
  --observed_only

python generate.py \
  --data_csv paper_apple_silicon_benchmark.csv \
  --devices apple_silicon \
  --observed_only \
  --output paper_apple_silicon_synth_10k.csv
```

---

## Device Profiles

| Device | TDP (W) | Energy scale | Latency scale | Tier |
|--------|---------|-------------|---------------|------|
| Apple Silicon (paper baseline) | 5.3 | 1.0× | 1.0× | Mobile laptop |
| Raspberry Pi 4 | 3.4 | 2.1× | 3.8× | SBC |
| Jetson Nano | 5.0 | 1.6× | 2.2× | Edge GPU |
| Coral Edge TPU | 2.0 | 0.3× | 0.4× | Accelerator |
| STM32 MCU | 0.05 | 0.8× | 45× | Microcontroller |
| Snapdragon 888 | 4.5 | 0.85× | 0.7× | Mobile SoC |

Scales are relative to the Apple Silicon baseline from the paper.

---

## Output CSV Schema

| Column | Type | Description |
|--------|------|-------------|
| `device` | str | Device key |
| `device_tier` | str | Hardware category |
| `model` | str | Model key |
| `arch` | str | Architecture type |
| `params_M` | float | Parameter count (millions) |
| `flops_G` | float | GFLOPs |
| `energy_J` | float | Energy per inference (Joules) |
| `power_W` | float | Average power draw (Watts) |
| `latency_ms` | float | Inference latency (ms) |
| `energy_std` | float | Std dev of energy across trials |
| `energy_scale_vs_paper` | float | Ratio to paper's Apple Silicon measurement |
| `device_idx` | int | Integer device encoding |
| `model_idx` | int | Integer model encoding |

---

## Next Steps (Future Work from Paper)

1. **Real validation** — run powermetrics / powertop on each physical device and
   compare against generated distributions using evaluate.py
2. **Quantization study** — add int8/fp16 variants as separate model entries
3. **Expanded architectures** — MobileViT, EfficientFormer, MCUNet
4. **Regression model** — train a lightweight predictor from architecture
   descriptors (FLOPs, params, arch type) to energy using this synthetic dataset
