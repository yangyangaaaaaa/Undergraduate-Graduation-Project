# Results Inventory (2026-05-22)

This note collects the main local result directories and summary files for the recent experiment rounds.

## 1. Factorial ablation and paper-style benchmark comparison

- Factorial generalization / 16-branch ablation:
  - Directory: `F:\bishe\GeoExplorer\analysis\pipeline_20260516_anchor0624_factorial_generalization`
  - Main files:
    - `anchor0624_generalization_table.csv`
    - `anchor0624_generalization_summary.md`
    - `anchor0624_generalization_aggregate.json`

- Paper-style benchmark comparison:
  - Directory: `F:\bishe\GeoExplorer\analysis\pipeline_20260516_paper_baseline_compare`
  - Main files:
    - `paper_baseline_compare_table.csv`
    - `paper_baseline_compare_summary.md`
    - `paper_baseline_compare_aggregate.json`

- AirLoc MM-GAG aerial supplemental row:
  - Directory: `F:\bishe\GeoExplorer\analysis\pipeline_20260516_paper_baseline_compare\airloc_mmgag_aerial_20260521`
  - Main file:
    - `airloc_mmgag_aerial_eval_with_sg.json`

## 2. Dataset / parameter appendix experiments

- Dataset comparison and first-round parameter appendix:
  - Directory: `F:\bishe\GeoExplorer\analysis\pipeline_20260519_appendix_dataset_param_compare`
  - Main files:
    - `appendix_dataset_sr_table.csv`
    - `appendix_dataset_sg_table.csv`
    - `appendix_param_sr_table.csv`
    - `appendix_param_sg_table.csv`
    - `appendix_seed_stability_table.csv`
    - `appendix_summary_zh.md`
    - `appendix_all_results.json`
    - `appendix_long_table.csv`

- Gate-floor / validation-distance full-range follow-up:
  - Directory: `F:\bishe\GeoExplorer\analysis\pipeline_20260519_appendix_gate_valdist_dense_followup`
  - Main files:
    - `appendix_gate_valdist_sr_table.csv`
    - `appendix_gate_valdist_sg_table.csv`
    - `appendix_gate_valdist_per_distance.csv`
    - `appendix_summary_zh.md`
    - `appendix_all_results.json`
    - `appendix_long_table.csv`

- Reward-control strict algorithm ablation download:
  - Directory: `F:\bishe\GeoExplorer\analysis\pipeline_20260519_appendix_gate_valdist_dense_followup_reward_control_download_20260520_220705`

## 3. Reward-gate supplementary ablations

- Reward-gate same-protocol linear fill-in:
  - Directory: `F:\bishe\GeoExplorer\analysis\pipeline_20260520_reward_gate_linear_main_primary_eval`

- Reward-gate tables stored in:
  - `F:\bishe\GeoExplorer\analysis\pipeline_20260519_appendix_gate_valdist_dense_followup`
  - Main files:
    - `appendix_reward_gate_type_main_primary_table.csv`
    - `appendix_reward_gate_type_main_primary_table_with_linear.csv`
    - `appendix_reward_gate_type_mmgag_only_table_with_linear.csv`
    - `appendix_reward_gate_pb_two_factor_table_zh.md`

## 4. Ultra-long stress tests

- Revised formal ultra-long stress test (recommended for writing):
  - Directory: `F:\bishe\GeoExplorer\analysis\pipeline_20260521_ultra_long_grid_stress_v2`
  - Main files:
    - `ultra_long_full_summary.csv`
    - `ultra_long_full_per_distance.csv`
    - `ultra_long_full_summary_zh.md`
    - `ultra_long_full_aggregate.json`

- Exploratory 25x25 stress test:
  - Directory: `F:\bishe\GeoExplorer\analysis\pipeline_20260521_ultra_long_grid_stress_v3_grid25`
  - Main files:
    - `ultra_long_full_summary.csv`
    - `ultra_long_full_per_distance.csv`
    - `ultra_long_full_summary_zh.md`
    - `ultra_long_full_aggregate.json`

- Older coarse ultra-long protocol (do not use for final writing):
  - Directory: `F:\bishe\GeoExplorer\analysis\pipeline_20260521_ultra_long_grid_stress`

## 5. Writing-support material

- Chapter 4 material pack:
  - File: `F:\bishe\GeoExplorer\analysis\chapter4_airloc_ultralong_split_material_20260521_zh.md`

- Reward-formula ablation write-up:
  - File: `F:\bishe\GeoExplorer\analysis\ablation_reward_formula_report_20260520_zh.md`

- Reward-gate evaluation protocol audit:
  - File: `F:\bishe\GeoExplorer\analysis\reward_gate_eval_protocol_audit_20260520_zh.md`
