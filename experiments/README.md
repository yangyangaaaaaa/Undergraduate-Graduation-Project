# Experiments

This directory keeps the scripts and manifests needed to reproduce the thesis experiments.

## Layout

- `manifests/`: run specifications and checkpoint/result mapping.
- `scripts/anchor0624_ablation/`: 16-cell G/P/E/V mechanism ablation.
- `scripts/paper_generalization/`: paper-style benchmark and generalization evaluation.
- `scripts/appendix_dataset_param/`: training-dataset, parameter, and seed-stability experiments.
- `scripts/appendix_gate_valdist/`: gate-floor, validation-distance, reward-control, and gate-function experiments.
- `scripts/ultra_long_eval/`: long-distance grid stress tests.

The scripts are intentionally kept as experiment assets. They assume the same dataset and checkpoint layout used in the original GPU environment. See `docs/reproducibility_zh.md` for the required data and checkpoint notes.
