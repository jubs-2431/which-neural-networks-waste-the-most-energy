# Release Manifest

Target repository:

- `https://github.com/jubs-2431/which-neural-networks-waste-the-most-energy`

Final paper artifacts:

- `paper_apple_silicon_only.pdf`: compiled paper PDF.
- `paper_apple_silicon_only.tex`: final LaTeX source.
- `tables/expanded_architecture_latency.tex`: expanded measured-latency architecture table included by the paper source.

Measured-data artifacts:

- `paper_apple_silicon_benchmark.csv`: original five-model Apple-Silicon energy/latency benchmark rows.
- `measured_architecture_benchmark.csv`: 17 measured architecture variants with repeated-trial latency statistics and baseline complexity metrics.
- `measured_architecture_trials.csv`: raw per-trial latency data for the expanded architecture sweep.
- `measurement_environment.json`: exact local environment metadata for the expanded measurement run.
- `paper_baseline_comparison.csv`: ranked baseline comparison with params, model size, MACs, FLOPs, latency, and proxy EDP.

Synthetic/support artifacts:

- `paper_alignment_comparison.csv`
- `paper_alignment_power_std_comparison.csv`
- `paper_supplemental_metrics.csv`
- `PAPER_DATA_APPENDIX.md`

Scripts:

- `scripts/benchmark_architectures.py`: repeatable local architecture latency benchmark.
- `train.py`
- `generate.py`
- `evaluate.py`
- `data_utils.py`
- `gan_model.py`

Release rule:

- Main branch should contain the final PDF, source, metadata, measured CSVs, and scripts listed above.
- Do not describe proxy energy/EDP columns as direct measurements.
