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

## 图件

| 文件夹 | 内容 |
| --- | --- |
| `results/figures/chapter2_dataset/` | 数据集示意图、数据集总览图和手动画图素材包。 |
| `results/figures/chapter3_method/` | 方法总体结构图。 |
| `results/figures/chapter4_trajectories/` | 典型成功轨迹和三方法困难样例对比。 |
| `results/figures/reward_story/` | G/P 机制辅助可视化图。 |

## 中文报告

| 文件 | 用途 |
| --- | --- |
| `results/reports/ablation_reward_formula_report_20260520_zh.md` | 以奖励公式组织的消融实验报告。 |
| `results/reports/chapter4_airloc_ultralong_split_material_20260521_zh.md` | AirLoc、长距离实验和数据划分材料。 |
| `results/reports/reward_gate_eval_protocol_audit_20260520_zh.md` | 消融测试阶段是否调用奖励模块的协议审查。 |
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
