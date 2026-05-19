# Friend Windows GPU Measurement Instructions

Use `measure_pc_inference_standalone.py` if you are sending only one script
instead of the full repository.

## Install dependencies

Install a CUDA-enabled PyTorch build for the friend's GPU, then install timm:

```powershell
pip install timm
```

If `torch`, `torchvision`, or CUDA support are missing, install PyTorch from
the command shown at https://pytorch.org/get-started/locally/.

## Check the machine

```powershell
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

`torch.cuda.is_available()` must print `True` for the CUDA GPU measurement.

## Main batch-1 run

From the folder containing `measure_pc_inference_standalone.py`:

```powershell
python measure_pc_inference_standalone.py --suite paper --device cuda --power-backend nvidia-smi --windows 10 --window-seconds 10 --warmups 10 --output-dir pc_measurements
```

## Batch sweep

Run each command:

```powershell
python measure_pc_inference_standalone.py --suite paper --device cuda --power-backend nvidia-smi --windows 5 --window-seconds 5 --warmups 10 --batch-size 1 --output-dir pc_measurements_batch\batch_1
python measure_pc_inference_standalone.py --suite paper --device cuda --power-backend nvidia-smi --windows 5 --window-seconds 5 --warmups 10 --batch-size 4 --output-dir pc_measurements_batch\batch_4
python measure_pc_inference_standalone.py --suite paper --device cuda --power-backend nvidia-smi --windows 5 --window-seconds 5 --warmups 10 --batch-size 8 --output-dir pc_measurements_batch\batch_8
python measure_pc_inference_standalone.py --suite paper --device cuda --power-backend nvidia-smi --windows 5 --window-seconds 5 --warmups 10 --batch-size 16 --output-dir pc_measurements_batch\batch_16
python measure_pc_inference_standalone.py --suite paper --device cuda --power-backend nvidia-smi --windows 5 --window-seconds 5 --warmups 10 --batch-size 32 --output-dir pc_measurements_batch\batch_32
```

If batch 32 runs out of GPU memory, keep the smaller completed batch folders.

## Zip outputs

```powershell
Compress-Archive -Path pc_measurements,pc_measurements_batch -DestinationPath friend_pc_measurements.zip -Force
```

The zip must include `raw_power/` folders. If any rows have
`parsed_power_samples` equal to 0, the run is latency-only until the raw power
logs are inspected.
