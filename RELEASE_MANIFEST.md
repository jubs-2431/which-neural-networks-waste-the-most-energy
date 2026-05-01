# Release Manifest

Target repository:

- `https://github.com/jubs-2431/which-neural-networks-waste-the-most-energy`

Final paper artifacts:

- `paper_revised_latex_all_fixes.pdf`: compiled IEEE-style paper PDF.
- `paper_revised_latex_all_fixes.tex`: final LaTeX source.
- `FTC2026_Anonymous_Submission.pdf`: anonymized FTC 2026 submission PDF using the Springer-style conference template.
- `FTC2026_Anonymous_Submission.tex`: source for the anonymized FTC 2026 submission PDF.
- `benchmark_metadata_all_fixes.json`: metadata for the final paper's benchmark and predictor checks.

Measured-data artifacts:

- `paper_apple_silicon_benchmark.csv`: original five-model Apple-Silicon energy/latency benchmark rows.
- `measured_architecture_benchmark.csv`: 17 measured architecture variants with repeated-trial latency statistics and baseline complexity metrics.
- `measured_architecture_trials.csv`: raw per-trial latency data for the expanded architecture sweep.
- `measurement_environment.json`: exact local environment metadata for the expanded measurement run.
- `measured_energy_powermetrics/measured_energy_trials.csv`: 50 direct repeated `powermetrics` energy windows for the five paper models.
- `measured_energy_powermetrics/measured_energy_summary.csv`: per-model measured energy mean, standard deviation, SEM, 95% CI, window count, latency, and power summaries.
- `measured_energy_powermetrics/measurement_environment_energy.json`: exact hardware, software, thread, warmup, cooldown, randomized-order, and `powermetrics` command metadata for the energy audit.
- `measured_energy_powermetrics/raw_powermetrics/`: raw `powermetrics` text logs, one per measured energy window.
- `paper_baseline_comparison.csv`: ranked baseline comparison with params, model size, MACs, FLOPs, latency, and proxy EDP.
- `benchmark_metadata_all_fixes.json`: metadata referenced by the final paper.

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
- Do not describe the new Apple M4 Pro repeated-energy audit as the missing original raw windows for `paper_apple_silicon_benchmark.csv`; it is a follow-up direct measurement artifact.
