# Undergraduate Graduation Project

本仓库用于整理本科毕业设计《好奇心驱动的无人机主动定位目标导航方法》的代码、实验结果、可视化材料、论文文档和复现说明。项目面向离散网格下的无人机主动目标定位导航任务：给定航拍搜索区域和目标线索后，智能体从起始网格出发，根据当前位置观测、目标表示和历史搜索序列选择移动动作，并在有限预算内尽可能到达目标网格。

![Benchmark overview](results/figures/showcase/benchmark_overview_dashboard.png)

## 一眼看结果

- 主结果：在 9 个共享基准评测上，本文方法平均 SR 为 `0.580`，GOMAA-Geo 为 `0.519`，平均提升约 `+0.062`。
- 跨模态：在 MM-GAG 的航拍图像、地面图像和文本目标上，本文方法 SR 分别为 `0.617`、`0.639`、`0.625`。
- 机制消融：16 组 G/P/E/V 消融中，完整分支 `g1_p1_e1_v1` 的主泛化均值为 `0.6211`，高于同数据无新增机制控制组 `0.5545`。
- 长距离：8x8 与 10x10 扩展网格中，本文方法在更长起终点距离下保持更稳定的成功率；25x25 压力测试也保留了对比材料。
- 可视化：仓库整理了 `103` 个展示文件，包括 `22` 个 GIF、`57` 张 PNG、`24` 个 SVG，以及轨迹、奖励机制、参数敏感和长距离测试图。
- 补充分析：新增置信区间、轨迹行为、C=4 失败画像和奖励过程曲线，用于支撑结果稳定性与中远距离优势解释。

## 方法概览

本文方法保留 GeoExplorer 的目标编码、历史动作-观测序列建模和 Actor-Critic 策略推理框架，重点改进训练阶段的混合奖励：

```text
r_t = r_ex,t + lambda_t r_in,t + r_p,t
```

其中 `r_ex,t` 为外在目标奖励，`r_in,t` 为下一步特征预测误差构造的好奇心内在奖励，`lambda_t` 为随距离变化的门控权重，`r_p,t` 为势函数奖励塑形项。推理阶段不再计算混合奖励或更新 PPO 参数，只使用训练后的策略网络进行动作选择。

![Method overview](results/figures/showcase/dataset/figure3_1_method_overview_revised.png)

## 结果展示

| 主题 | 图示 | 说明 |
|---|---|---|
| MM-GAG 跨模态 | ![MM-GAG modality](results/figures/showcase/mmgag_modality_sr.png) | A/G/T 三类目标形式下，本文方法均高于 GOMAA-Geo；AiRLoc 仅补充航拍目标 A 行。 |
| 距离桶曲线 | ![Distance buckets](results/figures/showcase/distance_bucket_curves.png) | C=4 到 C=8 上展示不同距离条件的成功率变化。 |
| 机制消融 | ![Ablation heatmap](results/figures/showcase/generalization_heatmap_16cell.png) | 16 组 G/P/E/V 组合的主泛化均值，完整方法位于最佳单元。 |
| 奖励设计 | ![Reward gate](results/figures/showcase/reward_gate_pb_comparison.png) | 对比常数、线性、正弦、二次幂和混合门控，以及 PBRS 开关。 |
| 参数敏感 | ![Parameter sensitivity](results/figures/showcase/parameter_sensitivity_curves.png) | 覆盖门控下限、PBRS 系数、熵系数、训练预算和验证距离集合。 |
| 长距离压力测试 | ![Ultra long](results/figures/showcase/ultra_long_sr_curves.png) | 8x8 与 10x10 扩展网格下的长距离成功率曲线。 |
| 补充行为分析 | ![Trajectory behavior](results/figures/supplement/trajectory_behavior_metrics.png) | 用轨迹单调接近率、重复访问率和距离缩短率解释中远距离搜索行为。 |

## 动态轨迹

下面的 GIF 展示同一个困难样例中三种方法的搜索过程。更多 C=4、C=6、C=8 成功与失败案例见 [可视化画廊](docs/visualization_gallery_zh.md)。

| Ours | GOMAA-Geo | GeoExplorer |
|---|---|---|
| ![Ours hard case](results/figures/showcase/trajectories/gifs/three_method_hardcase__img189_d6_s20_g14_r0__anchor0624.gif) | ![GOMAA hard case](results/figures/showcase/trajectories/gifs/three_method_hardcase__img189_d6_s20_g14_r0__gomaa.gif) | ![GeoExplorer hard case](results/figures/showcase/trajectories/gifs/three_method_hardcase__img189_d6_s20_g14_r0__pristine.gif) |

## 仓库结构

- `code/geoexplorer_active/`：本文方法的干净代码入口，包括训练、评测、数据预处理和模型定义。
- `code/tools/build_visual_showcase.py`：从 `results/tables/` 重新生成 GitHub 展示图和媒体清单。
- `code/tools/build_supplement_experiment_analysis.py`：生成置信区间、轨迹行为、奖励过程和后续补跑实验方案。
- `experiments/`：主表、消融实验、参数实验和长距离测试对应的脚本与 manifest。
- `results/tables/`：已整理的主实验、消融、附录参数、长距离和轨迹记录表。
- `results/figures/`：论文图、数据集图、轨迹图、奖励机制图和 GitHub 展示图。
- `docs/`：代码结构、实验总结、数据划分、结果索引、复现说明和可视化画廊。
- `thesis/`：论文正文与 Markdown 草稿。
- `materials/`：任务书、开题报告、中期报告、外文翻译等毕设过程材料。

## 复现入口

环境配置：

```bash
cd code/geoexplorer_active
conda env create -f environment.yml
conda activate geoexplorer
```

基础训练与评测入口：

```bash
python pretrain.py
python train.py
python validate.py
```

重新生成展示图：

```bash
python code/tools/build_visual_showcase.py
python code/tools/build_supplement_experiment_analysis.py
```

仓库不包含训练 checkpoint、原始大规模数据包和本地临时缓存。结果表保留 checkpoint 路径或 run 名称，用于和原始实验设置对应；大文件权重需要按复现说明另行准备。

## 文档导航

- [可视化画廊](docs/visualization_gallery_zh.md)
- [实验结果总览](docs/experiment_summary_zh.md)
- [复现说明](docs/reproducibility_zh.md)
- [结果文件索引](docs/result_inventory_zh.md)
- [代码结构说明](docs/code_structure_zh.md)
