# 结果文件索引

本文件说明仓库中各结果文件的用途，以及它们对应论文中的实验部分。正式数值以 `results/tables/` 中的 CSV/JSON 为准，图件用于展示和解释。

## 主结果表

| 文件 | 用途 |
| --- | --- |
| `results/tables/main_benchmark/paper_baseline_compare_table.csv` | MASA、MM-GAG、SwissView、xBD 上的主对比结果。 |
| `results/tables/main_benchmark/airloc_mmgag_aerial_table_row.csv` | 可并入 MM-GAG A 模态表的 AirLoc 行。 |
| `results/tables/main_benchmark/official_retrain_compare_table.csv` | 统一协议下的基线重训结果对照。 |

## 消融实验

| 文件 | 用途 |
| --- | --- |
| `results/tables/ablation/anchor0624_generalization_table.csv` | 16 个 G/P/E/V 分支消融主表。 |
| `results/tables/ablation/reward_gate_type_mmgag_only_table_with_linear.csv` | 不同门控函数与 PBRS 的补充消融。 |
| `results/tables/ablation/reward_control_long_table.csv` | `external_only`、`intrinsic_only`、无衰减混合奖励等严格奖励控制结果。 |
| `results/tables/ablation/gp_factorial_summary.csv` | G/P 奖励机制可视化辅助统计。 |

## 数据集、参数和稳定性实验

| 文件 | 用途 |
| --- | --- |
| `results/tables/appendix/appendix_dataset_sr_table.csv` | 不同训练数据组合的 SR 对比。 |
| `results/tables/appendix/appendix_dataset_sg_table.csv` | 不同训练数据组合的 SG 对比。 |
| `results/tables/appendix/appendix_param_sr_table.csv` | 第一轮参数实验 SR。 |
| `results/tables/appendix/appendix_param_sg_table.csv` | 第一轮参数实验 SG。 |
| `results/tables/appendix/appendix_seed_stability_table.csv` | 随机种子稳定性结果。 |
| `results/tables/appendix/appendix_gate_valdist_per_distance.csv` | 门控下界与验证距离的分距离结果。 |

## 长距离扩展测试

| 文件 | 用途 |
| --- | --- |
| `results/tables/ultra_long/ultra_long_v2_summary.csv` | `8 x 8` 与 `10 x 10` 正式长距离测试汇总。 |
| `results/tables/ultra_long/ultra_long_v2_per_distance.csv` | 正式长距离测试分距离结果。 |
| `results/tables/ultra_long/ultra_long_v3_grid25_summary.csv` | `25 x 25` 探索性压力测试汇总。 |
| `results/tables/ultra_long/ultra_long_v3_grid25_per_distance.csv` | `25 x 25` 探索性压力测试分距离结果。 |

## 补充统计与行为分析

| 文件 | 用途 |
| --- | --- |
| `results/tables/statistical_analysis/main_benchmark_ci_table.csv` | 主表各方法 SR 的成功次数、任务数与 95% 置信区间。 |
| `results/tables/statistical_analysis/main_benchmark_diff_ci_table.csv` | 本文方法相对基线的 SR 差值置信区间。 |
| `results/tables/statistical_analysis/ultra_long_ci_table.csv` | 长距离扩展测试各方法 SR 置信区间。 |
| `results/tables/trajectory_analysis/trajectory_behavior_summary.csv` | 轨迹案例库的总体行为指标。 |
| `results/tables/trajectory_analysis/trajectory_behavior_by_distance.csv` | 按距离桶统计的轨迹行为指标。 |
| `results/tables/reward_process/reward_process_summary.csv` | 奖励分量过程统计。 |

## 图件目录

| 文件夹 | 内容 |
| --- | --- |
| `results/figures/showcase/` | 首页展示图、证据墙、polished 图卡和同步轨迹 GIF。 |
| `results/figures/defense_showcase/` | 从最终答辩 PPT 抽取并整理的核心展示媒体。 |
| `results/figures/acceptance_demo/` | 验收演示阶段精选动图和索引图。 |
| `results/figures/reward_cases/` | 奖励机制、动作归因、同任务演化和失败边界案例图。 |
| `results/figures/presentation_assets/` | PPT 候选素材和说明。 |
| `results/figures/chapter2_dataset/` | 数据集示意图、数据集总览图和手动画图素材包。 |
| `results/figures/chapter3_method/` | 方法总体结构图。 |
| `results/figures/chapter4_trajectories/` | 典型成功轨迹和三方法困难样例对比。 |
| `results/figures/reward_story/` | G/P 机制辅助可视化图。 |
| `results/figures/supplement/` | 置信区间、轨迹行为、短距离失败画像和奖励过程曲线等补充分析图。 |

## 中文报告

| 文件 | 用途 |
| --- | --- |
| `results/reports/supplement_eval_overview_zh.md` | P0/P1 补充实验总报告。 |
| `results/reports/reward_stepwise_global_analysis_zh.md` | 奖励分量与动作关系分析。 |
| `results/reports/defense_visualization_strategy_review_zh.md` | 答辩可视化材料选择说明。 |
| `results/reports/short_distance_c123_summary_20260609_zh.md` | 短距离边界分析。 |
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
| `code/tools/` | 从已有结果生成补充统计表、轨迹行为表、奖励过程表和展示图的离线工具。 |
