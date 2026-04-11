# GAN for Weaker Engines

This folder contains a separate workflow for non-Apple hardware baselines.

Design choices:
- uses a real-world latency baseline CSV built from official docs, benchmark
  providers, and published benchmark tables
- models `latency_ms` only, because that is the most consistently available
  cross-device metric for weaker engines
- treats power and energy as optional support metadata rather than forcing
  fabricated labels where the source data does not provide them
- trains a combo-conditioned latency GAN with post-generation moment calibration

## Files

- `weaker_engine_latency_baseline.csv`
- `data_utils.py`
- `gan_model.py`
- `train.py`
- `generate.py`
- `evaluate.py`

## Run

```bash
python train.py --epochs 400 --checkpoint_dir checkpoints
python generate.py --checkpoint checkpoints/generator_final.pt --output weaker_engine_synth_10k.csv
python evaluate.py --csv weaker_engine_synth_10k.csv
```

## Passing Criteria

`evaluate.py` reports `OVERALL: PASS` when all of these hold:

- global KS `p > 0.05`
- coverage `>= 95%`
- per-combo KS pass rate `>= 80%`
- mean per-combo latency error `<= 5%`
