# Weaker Engines Pause Notes

Saved on: 2026-04-11

## Current Goal

Create a separate weaker-engine GAN workflow alongside the main Apple-Silicon
paper pipeline:

- collect stronger baseline data for non-Apple devices
- build a separate folder/workflow for weaker engines
- train/generate/evaluate a separate GAN that passes reliability checks

## Current Non-Apple Coverage Already in Repo

Source file:
- `real_benchmark_data.csv`

Current device coverage:
- `coral_tpu`
- `raspberry_pi4`
- `stm32`

Current counts by device/model:

```text
model          mobilenetv2  resnet18
device
coral_tpu                4         0
raspberry_pi4            1         1
stm32                    1         0
```

## Existing Rows Already Curated

### Coral TPU
- 4 rows for `mobilenetv2`
- latency around `2.6-3.0 ms`
- power recorded as `2.0 W`
- energy inferred from latency and official Coral power

### Raspberry Pi 4
- `mobilenetv2`: `26.4 ms`
- `resnet18`: `100.3 ms`
- latency only in current CSV

### STM32
- `mobilenetv2`: `1118.27 ms`
- latency only in current CSV

## Existing Strong Source Leads Already Recorded

See:
- `ONLINE_BENCHMARK_AUDIT.md`

Best currently identified source families:

- Coral official benchmarks and model catalog
- PyTorch Raspberry Pi 4 official tutorial benchmark table
- STMicroelectronics STM32 AI Model Zoo benchmark tables
- JCSE 2023 edge-device inference paper for Jetson/Coral latency candidates
- MLCommons Tiny source family for STM32-class evidence

## Important Constraint

The repo currently has much stronger Apple-Silicon evidence than non-Apple
evidence. For weaker engines, most safe claims will need to be phrased as:

- latency-grounded
- partially inferred for power/energy
- exploratory synthetic extension rather than strong measured validation

## Planned Next Steps

1. Collect stronger primary-source rows for:
   - `jetson_nano`
   - `snapdragon888`
   - more `coral_tpu`
   - more `raspberry_pi4`
   - more `stm32`
2. Build a separate curated benchmark CSV for weaker engines only.
3. Create a separate folder/workflow, likely `gan_for_weaker_engines/`.
4. Reuse the current GAN pipeline with isolated configs/scripts.
5. Run reliability tests and iterate until the weaker-engine workflow passes.

## Suggested Resume Point

When resuming, start with source collection for:

- Jetson Nano official or peer-reviewed latency tables
- Snapdragon 888 primary-source benchmark tables
- additional STM32 official image-classification rows
- additional Raspberry Pi official benchmark rows

Then create the separate weaker-engine folder and benchmark CSV.
