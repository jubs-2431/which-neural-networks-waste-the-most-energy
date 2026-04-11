# Paper Data Appendix

This note documents the strongest paper-aligned benchmark data derived from
`/Users/aryanshah/Downloads/main.pdf` and the limits of the available evidence.

## Directly Reported by the Paper

Source:
- `/Users/aryanshah/Downloads/main.pdf`

Extracted from Table 3:
- Device: Apple Silicon
- Runtime setting: CPU-only inference in PyTorch
- Models:
  - `mobilenetv3_small`
  - `mobilenetv2`
  - `resnet18`
  - `tiny_vit_5m`
  - `efficientnet_b0`
- Reported metrics:
  - `energy_J`
  - `latency_ms`
  - rounded `power_W` shown as `5.3 W`

## Supplemental Values Added

File:
- `/Users/aryanshah/Downloads/GAN/paper_supplemental_metrics.csv`

These supplemental values are documented so they can be described honestly in
the paper.

### 1. Derived Power

For each model:

`power_W = energy_J * 1000 / latency_ms`

This was used because Table 3 rounds every power value to `5.3 W`, while the
reported energy and latency imply slightly different model-specific average
power values.

Derived powers:
- `mobilenetv3_small`: `5.284016 W`
- `mobilenetv2`: `5.281515 W`
- `resnet18`: `5.319149 W`
- `tiny_vit_5m`: `5.298814 W`
- `efficientnet_b0`: `5.301380 W`

These are derived from the paper's own Table 3 values, not imported from a
separate source.

### 2. Synthetic Repeated-Trial Energy Spread

The paper text states that repeated trials were run, but Table 3 does not
publish per-model standard deviations. To support repeated-trial style analysis,
the paper-aligned GAN was trained on the Apple-Silicon-only benchmark and used
to generate repeated synthetic samples.

The resulting per-model synthetic `energy_std` values are:
- `mobilenetv3_small`: `0.003867 J`
- `mobilenetv2`: `0.007057 J`
- `resnet18`: `0.006286 J`
- `tiny_vit_5m`: `0.014270 J`
- `efficientnet_b0`: `0.016949 J`

These should be described as synthetic repeated-trial spread estimates, not as
directly measured statistics from the paper.

## Online Source Check

I also checked for public Apple-Silicon model-specific inference-power sources
that were apples-to-apples with the paper setup. I did not find a strong public
source that simultaneously matched:
- Apple Silicon
- CPU-only PyTorch inference
- the same five models
- model-specific average power or trial-level variance

Because of that, the supplemental values above stay anchored to the paper's own
reported energy and latency instead of importing mismatched online numbers.

## Recommended Wording

If you add this to the paper, the safest framing is:

- Mean energy and latency are directly reported from the Apple-Silicon
  experiments in Table 3.
- Average power can be derived from the reported energy and latency values.
- Trial-to-trial spread estimates were generated using a paper-aligned synthetic
  model and should be treated as synthetic support data rather than direct
  measurements.
