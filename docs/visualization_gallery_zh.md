# 可视化结果画廊

本文实验材料不仅保留数值表格，也整理了可直接放入项目展示页、论文补充材料或答辩展示的可视化结果。所有展示图默认位于 `results/figures/showcase/`，完整文件清单见 `results/figures/showcase/showcase_manifest.json`。

## 生成方式

```bash
python code/tools/build_visual_showcase.py
```

该脚本读取 `results/tables/` 中的真实实验结果表，批量生成 PNG/SVG 图表，并复制已生成的轨迹 GIF、静态轨迹图和奖励机制图。图表可复现，数据来源可追溯到对应 CSV 或 JSON 表。

## 总览图

| 图 | 用途 | 数据来源 |
|---|---|---|
| ![benchmark](../results/figures/showcase/benchmark_overview_dashboard.png) | 项目首页主图，概括主基准平均 SR、GOMAA 对比和长距离提升。 | `results/tables/main_benchmark/paper_baseline_compare_table.csv` 与长距离表 |
| ![rank](../results/figures/showcase/method_rank_lollipop.png) | 展示不同方法在可用主表行上的整体排序。 | `paper_baseline_compare_table.csv` |
| ![dataset](../results/figures/showcase/dataset/figure_dataset_overview.png) | 展示训练与评测涉及的数据集类型。 | 已整理数据集图素材 |

## MM-GAG 与距离桶

| 图 | 用途 | 说明 |
|---|---|---|
| ![mmgag](../results/figures/showcase/mmgag_modality_sr.png) | 展示 MM-GAG 航拍图像目标、地面图像目标和文本目标的 SR。 | AiRLoc 只补充 A 行，不填 G/T。 |
| ![distance](../results/figures/showcase/distance_bucket_curves.png) | 展示 C=4 到 C=8 距离桶下的成功率曲线。 | 可用于说明中远距离优势更明显。 |
| ![trajectory stats](../results/figures/showcase/trajectory_statistics.png) | 从 2205 条轨迹记录中汇总成功率和绕行步数。 | 这是可视化案例库统计，不替代主表评测。 |
| ![trajectory heatmap](../results/figures/showcase/trajectory_success_heatmap.png) | 轨迹案例库按方法和距离的成功率热力图。 | 用于快速解释轨迹材料覆盖情况。 |

## 消融与奖励机制

| 图 | 用途 | 数据来源 |
|---|---|---|
| ![16-cell](../results/figures/showcase/generalization_heatmap_16cell.png) | 16 组 G/P/E/V 机制消融热力图。 | `results/tables/ablation/anchor0624_generalization_table.csv` |
| ![factor](../results/figures/showcase/ablation_factor_effects.png) | 展示 G、P、E、V 四个因子的平均主效应。 | 16 组机制消融表 |
| ![gate](../results/figures/showcase/reward_gate_pb_comparison.png) | 比较不同 `lambda_t` 门控函数和 PBRS 开关。 | `reward_gate_type_mmgag_only_table_with_linear.csv` |
| ![control](../results/figures/showcase/reward_control_endpoints.png) | 外部奖励、内在奖励和完整混合奖励的严格控制对比。 | `reward_control_long_table.csv` |
| ![reward schematic](../results/figures/showcase/reward_story/figure_g_reward_design_schematic.png) | 奖励机制示意图，可用于方法解释。 | 奖励故事图包 |

## 参数、数据集与稳定性

| 图 | 用途 | 数据来源 |
|---|---|---|
| ![param](../results/figures/showcase/parameter_sensitivity_curves.png) | 覆盖门控下限、PBRS 系数、熵系数、训练预算和验证距离集合。 | `appendix_param_sr_table.csv` 与 `appendix_gate_valdist_sr_table.csv` |
| ![trainset](../results/figures/showcase/dataset_training_transfer.png) | 比较不同训练数据组合的整体与迁移 SR。 | `appendix_dataset_sr_table.csv` |
| ![seed](../results/figures/showcase/seed_stability.png) | 展示 seed 42、123、321 的结果波动。 | `appendix_seed_stability_table.csv` |

## 长距离压力测试

| 图 | 用途 | 说明 |
|---|---|---|
| ![ultra sr](../results/figures/showcase/ultra_long_sr_curves.png) | 8x8 与 10x10 扩展网格下的 SR 曲线。 | 预算分别为 B=24 和 B=32。 |
| ![ultra sg](../results/figures/showcase/ultra_long_sg_curves.png) | 8x8 与 10x10 扩展网格下的 SG 曲线。 | SG 越低表示到达后剩余预算越多或路径更高效。 |
| ![grid25](../results/figures/showcase/grid25_stress_curves.png) | 25x25 探索性压力测试。 | 该设置难度显著更高，主要作为补充观察。 |

## 动态与静态轨迹

轨迹媒体位于以下目录：

- `results/figures/showcase/trajectories/gifs/`：21 个动态轨迹 GIF，覆盖 C=4、C=6、C=8 的成功、失败和三方法困难案例。
- `results/figures/showcase/trajectories/comparison_png/`：7 张三方法静态对比图，适合论文或答辩页。
- `results/figures/showcase/trajectories/static_png/`：21 张单方法静态轨迹图，适合补充材料。

典型困难案例：

| Ours | GOMAA-Geo | GeoExplorer |
|---|---|---|
| ![ours](../results/figures/showcase/trajectories/gifs/three_method_hardcase__img189_d6_s20_g14_r0__anchor0624.gif) | ![gomaa](../results/figures/showcase/trajectories/gifs/three_method_hardcase__img189_d6_s20_g14_r0__gomaa.gif) | ![geoexplorer](../results/figures/showcase/trajectories/gifs/three_method_hardcase__img189_d6_s20_g14_r0__pristine.gif) |

静态对比图示例：

![hardcase static](../results/figures/showcase/trajectories/comparison_png/three_method_hardcase__img189_d6_s20_g14_r0__comparison.png)

## 奖励机制故事图

奖励机制图位于 `results/figures/showcase/reward_story/`，包括机制示意、G/P 2x2 路径对比、奖励矩阵、全局证据、结果矩阵和 GIF 回放。

| 图 | 用途 |
|---|---|
| ![anchor mechanism](../results/figures/showcase/reward_story/figure_a_anchor_mechanism_img189.png) | 展示完整方法在代表性样例中的奖励引导。 |
| ![gp paths](../results/figures/showcase/reward_story/figure_b_gp_2x2_paths_img189.png) | 展示 G 与 P 四种组合对路径的影响。 |
| ![reward replay](../results/figures/showcase/reward_story/gifs/gif_anchor_reward_replay_img189.gif) | 动态回放奖励机制如何影响搜索过程。 |

## 补充统计与行为分析图

这些图位于 `results/figures/supplement/`，由 `code/tools/build_supplement_experiment_analysis.py` 从已有结果表和任务级记录生成。它们更适合放在论文补充分析或答辩问答页，用来回答“结果是否稳定”“为什么短距离不一定最好”“轨迹行为是否真的更稳”等问题。

| 图 | 用途 | 数据来源 |
|---|---|---|
| ![mmgag ci](../results/figures/supplement/mmgag_diff_ci.png) | 展示 MM-GAG A/G/T 上本文方法相对 GOMAA-Geo 的 SR 差值及近似 95% CI。 | `results/tables/statistical_analysis/main_benchmark_diff_ci_table.csv` |
| ![ultra ci](../results/figures/supplement/ultra_long_diff_ci.png) | 展示长距离扩展测试中本文方法相对基线的 SR 差值及近似 95% CI。 | `results/tables/statistical_analysis/ultra_long_diff_ci_table.csv` |
| ![behavior](../results/figures/supplement/trajectory_behavior_metrics.png) | 用成功率、距离缩短率、单调接近率和重复访问率解释轨迹行为差异。 | `results/tables/trajectory_analysis/trajectory_behavior_by_distance.csv` |
| ![c4 failure](../results/figures/supplement/c4_failure_profile.png) | 分析 `C=4` 短距离失败样例，支撑“方法优势集中在中远距离”的论文表述。 | `results/tables/trajectory_analysis/c4_failure_profile.csv` |
| ![reward traces](../results/figures/supplement/reward_component_traces.png) | 展示成功/失败样例中奖励分量、门控值和总奖励随步数变化的趋势。 | `results/tables/reward_process/reward_process_summary.csv` |

## 使用建议

- 项目首页优先使用 `benchmark_overview_dashboard.png`、`mmgag_modality_sr.png`、`ultra_long_sr_curves.png` 和一组困难案例 GIF。
- 论文正文优先使用 SVG 或 PDF 版本，答辩展示优先使用 PNG 和 GIF。
- 主表结论以 `results/tables/` 中的 CSV/JSON 为准，展示图用于帮助读者快速理解趋势。
