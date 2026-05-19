# Windows RTX GPU Inference Energy Paper

This is the Windows/NVIDIA version of the neural-network inference-energy
project. It uses GPU-only power samples from `nvidia-smi` and is kept separate
from the Apple-Silicon paper.

Current title:

**When FLOPs Mislead on PC GPUs: Comparing RTX 3050 and RTX 4060 Inference Energy**

## Start Here

- Current paper PDF:
  - `paper/Windows_RTX_GPU_Inference_Energy.pdf`
- Current paper LaTeX:
  - `paper/Windows_RTX_GPU_Inference_Energy.tex`

## Data

- `data/rtx3050_batch1_main/`: batch-size-1 RTX 3050 Laptop GPU measurement
  run from `aryan.zip`.
  - `pc_inference_summary.csv`
  - `pc_inference_trials.csv`
  - `pc_measurement_environment.json`
  - `raw_power/`
- `data/rtx3050_batch_sweep/`: RTX 3050 Laptop GPU batch-size sweep from
  `aryan.zip`.
  - `batch_1/`
  - `batch_4/`
  - `batch_8/`
  - `batch_16/`
  - `batch_32/`
- `data/rtx4060_batch1_main/`: batch-size-1 RTX 4060 measurement run.
  - `pc_inference_summary.csv`
  - `pc_inference_trials.csv`
  - `pc_measurement_environment.json`
  - `raw_power/`
- `data/rtx4060_batch_sweep/`: RTX 4060 batch-size sweep.
  - `batch_1/`
  - `batch_4/`
  - `batch_8/`
  - `batch_16/`
  - `batch_32/`

Important interpretation: these are GPU-only energy measurements, not whole-PC
wall power and not CPU package energy.

## Scripts

- `scripts/measure_pc_inference.py`: primary PC/NVIDIA measurement script.
- `scripts/measure_pc_inference_standalone.py`: standalone friend-run version.
- `scripts/benchmark_architectures.py`: shared model definitions.

Mac-only scripts from the original Apple-Silicon project were moved to
`archive/unused_original_mac_scripts/` so they do not look like part of the
Windows workflow.

## Re-run RTX 4060 Measurements

From this folder on the Windows PC:

```bash
python -m pip install -r requirements.txt
python scripts/measure_pc_inference.py --suite paper --device cuda --power-backend nvidia-smi --output-dir pc_measurements_rtx4060 --windows 10 --window-seconds 10 --warmups 10
```

For a batch sweep:

```bash
python scripts/measure_pc_inference.py --suite paper --device cuda --power-backend nvidia-smi --batch-size 1 --windows 5 --window-seconds 5 --warmups 10 --output-dir pc_measurements_rtx4060_batch/batch_1
python scripts/measure_pc_inference.py --suite paper --device cuda --power-backend nvidia-smi --batch-size 4 --windows 5 --window-seconds 5 --warmups 10 --output-dir pc_measurements_rtx4060_batch/batch_4
python scripts/measure_pc_inference.py --suite paper --device cuda --power-backend nvidia-smi --batch-size 8 --windows 5 --window-seconds 5 --warmups 10 --output-dir pc_measurements_rtx4060_batch/batch_8
python scripts/measure_pc_inference.py --suite paper --device cuda --power-backend nvidia-smi --batch-size 16 --windows 5 --window-seconds 5 --warmups 10 --output-dir pc_measurements_rtx4060_batch/batch_16
python scripts/measure_pc_inference.py --suite paper --device cuda --power-backend nvidia-smi --batch-size 32 --windows 5 --window-seconds 5 --warmups 10 --output-dir pc_measurements_rtx4060_batch/batch_32
```

Use a run as energy data only if every model has parsed power windows and
nonempty `mean_power_W` and `energy_mean_J` values in the summary CSV.

## Build The Paper

On the Mac, from this folder:

```bash
mkdir -p build
/Users/romikadiam/.local/bin/tectonic --outdir build --keep-logs paper/Windows_RTX_GPU_Inference_Energy.tex
```

## Related Project

The Apple-Silicon paper is in:

```text
/Users/romikadiam/mac-apple-silicon-inference-energy-paper
```
