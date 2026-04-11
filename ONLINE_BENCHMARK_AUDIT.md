# Online Benchmark Audit

This file records online sources that are strong enough to use when curating
`real_benchmark_data.csv`.

## Strong Sources Found

### Coral Edge TPU

Source:
https://www.coral.ai/docs/edgetpu/benchmarks/

What it supports:
- Official Coral latency benchmarks.
- Explicit device context for the Coral Dev Board and USB Accelerator.
- Official device power statement: 2 W / 4 TOPS from Coral docs.

Usable examples from the official page:
- MobileNet v2 (224x224), Dev Board with Edge TPU: 2.6 ms
- MobileNet v2 (224x224), USB Accelerator on desktop CPU: 2.6 ms
- EfficientNet-EdgeTpu-S (224x224), Dev Board with Edge TPU: 5.5 ms

Notes:
- This page does not publish `energy_J` directly.
- `energy_J` could be inferred from latency and power, but that should be
  marked as inferred, not measured.
- These are not the exact same model names as `efficientnet_b0`.

### Coral Model Catalog

Source:
https://www.coral.ai/models/all/

What it supports:
- Official Coral model names and single-inference latency figures for some
  Edge TPU models.

Usable examples:
- EfficientNet-EdgeTpu (S), 224x224: 5.0 ms
- SSD MobileNet V2, 300x300: 7.3 ms

Notes:
- Good for validating model naming and latency ranges.
- Still does not provide measured `energy_J`.

### MLPerf Tiny / MLCommons

Source:
https://mlcommons.org/benchmarks/inference-tiny/

Supporting repositories:
- https://github.com/mlcommons/tiny_results_v1.2
- https://github.com/mlcommons/tiny_results_v1.1

What it supports:
- Official benchmark framework and official submission/result repositories.
- Valid source family for STM32-class measurements.

Notes:
- Strong source for TinyML devices.
- The benchmark tasks are MLPerf Tiny tasks, not your exact ImageNet
  classification lineup, so these rows are not directly comparable to
  MobileNetV2 / ResNet18 ImageNet-style classification unless carefully mapped.

### JCSE 2023: Edge Devices Inference Performance Comparison

Source PDF:
https://jcse.kiise.org/files/V17N2-02.pdf

What it supports:
- Exact latency numbers for Jetson Nano FP16 and Coral for selected models.
- Peer-reviewed paper with extracted table values.

Directly usable values from Table 1:
- Coral USB, MobileNetV2, 224: mean 2.73 ms
- Coral PCIe, MobileNetV2, 224: mean 3.46 ms
- Jetson FP16, MobileNetV2, 224: mean 12.08 ms
- Jetson FP16, EfficientNetV2B0, 224: mean 19.64 ms
- Jetson FP16, ResNet-50, 224: mean 28.47 ms

Notes:
- This paper gives latency only, not power and energy.
- Table 1 reports runs without classification heads, so these are
  feature-extractor timings unless another configuration is explicitly cited
  from the downloadable CSV.
- It is useful for candidate rows and for studying relative latency scaling,
  but it should not be mixed with classifier rows without labeling.

### PyTorch Raspberry Pi 4 Tutorial

Source:
https://docs.pytorch.org/tutorials/intermediate/realtime_rpi.html?highlight=mobilenet

What it supports:
- Official PyTorch Raspberry Pi 4 benchmark table.
- Exact model-time-per-frame numbers for common torchvision models.

Directly usable values:
- Raspberry Pi 4, MobileNetV2: 26.4 ms model time
- Raspberry Pi 4, ResNet18: 100.3 ms model time

Candidate-only values:
- Raspberry Pi 4, MobileNetV3-Large: 30.7 ms

Notes:
- This is latency only, not power or direct energy.
- It is a strong official source for Raspberry Pi latency.

### STMicroelectronics STM32 AI Model Zoo

Source:
https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/main/image_classification/mobilenetv2/README.md

What it supports:
- Official STMicroelectronics benchmark tables for exact `MobileNet v2`
  ImageNet variants on named STM32 boards.
- Explicit board, runtime, quantization, frequency, and inference time
  metadata.

Directly usable values:
- STM32H747I-DISCO, MobileNet v2 1.0, ImageNet 224x224, INT8, CPU: 1118.27 ms

Candidate-only values:
- STM32MP257F-DK2, MobileNet v2 1.0 per-tensor, ImageNet 224x224, NPU/GPU:
  12.15 ms
- STM32MP257F-DK2, MobileNet v2 1.0 per-channel, ImageNet 224x224, NPU/GPU:
  75.91 ms

Notes:
- This is one of the strongest exact `stm32` sources found so far.
- The MPU/NPU rows are valid, but they represent a different STM32 hardware
  class than MCU-only boards, so merging them into a single `stm32` bucket
  would blur device meaning unless the taxonomy is refined.

### OpenReview BiBench Hardware Inference Tables

Source PDF:
https://openreview.net/pdf?id=e1u9PVnwNr

What it supports:
- Exact hardware inference timing tables for Apple M1 and Apple M1 Max.
- Explicit thread counts and runtime-library context.

Candidate-only values:
- Apple M1, ResNet18, Larq FP32, 1 thread: 44.334 ms

Notes:
- This is useful evidence that `apple_silicon` rows can be sourced online.
- The benchmark is runtime-specific and thread-specific, so it should stay in
  the candidate set unless you split the taxonomy by runtime or normalize
  methodology.

## Important Limits

- Public sources for exact `energy_J` and `power_W` on your target
  device/model pairs are sparse.
- Many sources are not apples-to-apples:
  - different runtimes
  - different precisions
  - different input sizes
  - classification vs feature-extractor mode
  - ImageNet classifier vs Edge TPU-specific variants
- Some high-quality rows are still not exact matches for your current model set:
  - `mobilenet_v3_large` vs `mobilenetv3_small`
  - `efficientnet_edgetpu_s` / `efficientnetv2b0` vs `efficientnet_b0`
  - feature-extractor timing vs classifier timing
- Some sources are exact on model and metric but still risky to merge into the
  current taxonomy:
  - broad device labels like `stm32` that currently mix MCU and MPU/NPU classes
- The current CSV should not treat inferred energy as if it were directly
  measured.

## Recommended Curation Policy

- Keep a row only if the source explicitly states the device, model, and metric.
- Add a `source_url` field for traceability.
- Add a `measurement_type` field with one of:
  - `measured`
  - `reported`
  - `inferred`
- Do not mix inferred energy with measured energy without labeling it.
- Prefer latency-only rows over fabricated full rows.

## Next Safe Step

Build a new curated CSV with:
- validated measured rows
- source URLs
- explicit flags for inferred values
- separate handling for latency-only evidence
