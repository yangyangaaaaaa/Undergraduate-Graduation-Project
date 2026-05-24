# 结果文件索引

本文件说明仓库中各结果文件的用途，以及它们对应论文中的实验部分。

## 主结果表

| 文件 | 用途 |
| --- | --- |
| `results/tables/main_benchmark/paper_baseline_compare_table.csv` | MASA、MM-GAG、SwissView、xBD 上的主对比结果。 |
| `results/tables/main_benchmark/paper_baseline_compare_aggregate.json` | 主对比结果聚合数据。 |
| `results/tables/main_benchmark/airloc_mmgag_aerial_eval_with_sg.json` | AirLoc 在 MM-GAG aerial 目标上的补充结果。 |
| `results/tables/main_benchmark/airloc_mmgag_aerial_table_row.csv` | 可直接并入 MM-GAG A 模态表的 AirLoc 行。 |
| `results/tables/main_benchmark/official_retrain_compare_table.csv` | GOMAA-Geo 与原始 GeoExplorer 风格重训结果对照。 |

## 消融实验

| 文件 | 用途 |
| --- | --- |
| `results/tables/ablation/anchor0624_generalization_table.csv` | 16 个 G/P/E/V 分支消融主表。 |
| `results/tables/ablation/anchor0624_generalization_aggregate.json` | 16 分支消融聚合结果。 |
| `results/tables/ablation/reward_gate_type_mmgag_only_table_with_linear.csv` | 不同门控函数与 PBRS 的 MM-GAG-only 补充消融。 |
| `results/tables/ablation/reward_control_long_table.csv` | `external_only`、`intrinsic_only`、无衰减混合奖励等严格奖励控制结果。 |
| `results/tables/ablation/gp_factorial_summary.csv` | G/P 奖励机制可视化辅助统计。 |
| `results/tables/ablation/gp_reward_case_summary.csv` | G/P 典型案例辅助统计。 |

## 数据集、参数和稳定性实验

| 文件 | 用途 |
| --- | --- |
| `results/tables/appendix/appendix_dataset_sr_table.csv` | 不同训练数据组合的 SR 对比。 |
| `results/tables/appendix/appendix_dataset_sg_table.csv` | 不同训练数据组合的 SG 对比。 |
| `results/tables/appendix/appendix_param_sr_table.csv` | 第一轮参数实验 SR。 |
| `results/tables/appendix/appendix_param_sg_table.csv` | 第一轮参数实验 SG。 |
| `results/tables/appendix/appendix_seed_stability_table.csv` | 随机种子稳定性结果。 |
| `results/tables/appendix/appendix_gate_valdist_sr_table.csv` | 门控下界与验证距离全范围 SR。 |
| `results/tables/appendix/appendix_gate_valdist_sg_table.csv` | 门控下界与验证距离全范围 SG。 |
| `results/tables/appendix/appendix_gate_valdist_per_distance.csv` | 门控下界与验证距离的分距离结果。 |

## 长距离扩展测试

| 文件 | 用途 |
| --- | --- |
| `results/tables/ultra_long/ultra_long_v2_summary.csv` | `8 x 8` 与 `10 x 10` 正式长距离测试汇总。 |
| `results/tables/ultra_long/ultra_long_v2_per_distance.csv` | 正式长距离测试分距离结果。 |
| `results/tables/ultra_long/ultra_long_v3_grid25_summary.csv` | `25 x 25` 探索性压力测试汇总。 |
| `results/tables/ultra_long/ultra_long_v3_grid25_per_distance.csv` | `25 x 25` 探索性压力测试分距离结果。 |

## P0/P1 补充评测

| 文件 | 用途 |
| --- | --- |
| `results/tables/supplement_eval/budget_sensitivity_summary.csv` | P0 8x8/10x10 预算敏感性汇总。 |
| `results/tables/supplement_eval/task_seed_summary.csv` | P0 MM-GAG 与 ultra-long task-bank seed 稳定性汇总。 |
| `results/tables/supplement_eval/p1_grid25_budget_summary.csv` | P1 25x25 超大网格预算敏感性汇总。 |
| `results/tables/supplement_eval/p1_grid25_seed_summary.csv` | P1 25x25 超大网格 task-bank seed 稳定性汇总。 |
| `results/tables/supplement_eval/p0_supplement_aggregate.json` | P0 补充评测完整聚合数据。 |
| `results/tables/supplement_eval/p1_grid25_aggregate.json` | P1 25x25 补充评测完整聚合数据。 |

## 补充统计与行为分析

| 文件 | 用途 |
| --- | --- |
| `results/tables/statistical_analysis/main_benchmark_ci_table.csv` | 主表各方法 SR 的成功次数、任务数与 95% 置信区间。 |
| `results/tables/statistical_analysis/main_benchmark_diff_ci_table.csv` | 本文方法相对 GOMAA-Geo、Random policy、DiT-AGL 的 SR 差值置信区间。 |
| `results/tables/statistical_analysis/ultra_long_ci_table.csv` | 长距离扩展测试各方法 SR 置信区间。 |
| `results/tables/statistical_analysis/ultra_long_diff_ci_table.csv` | 长距离扩展测试中本文方法相对基线的 SR 差值置信区间。 |
| `results/tables/trajectory_analysis/trajectory_behavior_summary.csv` | 轨迹案例库的总体行为指标，包括目标距离缩短、单调接近率和重复访问率。 |
| `results/tables/trajectory_analysis/trajectory_behavior_by_distance.csv` | 按距离桶统计的轨迹行为指标。 |
| `results/tables/trajectory_analysis/c4_failure_profile.csv` | `C=4` 短距离失败样例画像，用于解释短距离弱项。 |
| `results/tables/reward_process/reward_process_summary.csv` | 奖励分量过程统计，包括外部奖励、门控内在奖励、PBRS 与总奖励。 |
| `results/tables/experiment_plan/supplement_experiment_plan.csv` | 后续建议补跑的预算敏感性、任务种子稳定性、目标线索鲁棒性等实验方案。 |

## 图件

| 文件夹 | 内容 |
| --- | --- |
| `results/figures/chapter2_dataset/` | 数据集示意图、数据集总览图和手动画图素材包。 |
| `results/figures/chapter3_method/` | 方法总体结构图。 |
| `results/figures/chapter4_trajectories/` | 典型成功轨迹和三方法困难样例对比。 |
| `results/figures/reward_story/` | G/P 机制辅助可视化图。 |
| `results/figures/supplement/` | 置信区间、轨迹行为、C=4 失败画像和奖励过程曲线等补充分析图。 |

## 中文报告

| 文件 | 用途 |
| --- | --- |
| `results/reports/ablation_reward_formula_report_20260520_zh.md` | 以奖励公式组织的消融实验报告。 |
| `results/reports/chapter4_airloc_ultralong_split_material_20260521_zh.md` | AirLoc、长距离实验和数据划分材料。 |
| `results/reports/reward_gate_eval_protocol_audit_20260520_zh.md` | 消融测试阶段是否调用奖励模块的协议审查。 |
| `results/reports/supplement_experiment_analysis_zh.md` | 补充统计、轨迹行为、奖励过程分析与后续实验方案。 |
| `results/reports/p0_supplement_eval_inventory_zh.md` | P0 预算敏感性与 task-bank seed 复评整理说明。 |
| `results/reports/p1_grid25_analysis_zh.md` | P1 25x25 超大网格压力测试整理与分析。 |
| `results/reports/supplement_eval_overview_zh.md` | P0/P1 补充实验总报告，可作为第 4 章补充材料来源。 |
| `results/reports/xbd_protocol_correction_20260519_zh.md` | xBD 评测口径说明。 |
| `results/reports/source_result_inventory_20260522.md` | 原始本地结果目录索引。 |

## 代码与实验设置对应

| 文件夹 | 内容 |
| --- | --- |
| `experiments/manifests/` | 每组实验的 run 列表、参数设置和 checkpoint 映射。 |
| `experiments/scripts/paper_generalization/` | 主表和泛化评测脚本。 |
| `experiments/scripts/anchor0624_ablation/` | 16 分支消融训练/监控脚本。 |
| `experiments/scripts/appendix_dataset_param/` | 数据集、参数和种子实验脚本。 |
| `experiments/scripts/appendix_gate_valdist/` | 门控、验证距离和奖励控制实验脚本。 |
| `experiments/scripts/ultra_long_eval/` | 长距离扩展测试脚本。 |
| `code/tools/build_supplement_experiment_analysis.py` | 从已有结果生成补充统计表、轨迹行为表、奖励过程表、图件和后续实验方案。 |
