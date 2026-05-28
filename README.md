# Undergraduate Graduation Project

<p align="center">
  <b>好奇心驱动的无人机主动目标定位导航方法</b><br>
  Active target geo-localization with distance-aware curiosity reward shaping
</p>

<p align="center">
  <img src="results/figures/showcase/experience/hero_experience.png" width="100%" alt="好奇心驱动的无人机主动目标定位展示图">
</p>

## 项目一眼看懂

本仓库整理本科毕业设计《好奇心驱动的无人机主动目标定位导航方法》的代码、实验结果、可视化材料、论文文档和复现说明。任务面向离散网格下的无人机主动目标定位导航：给定航拍搜索区域和目标线索后，智能体从起始网格出发，根据当前位置观测、目标表示和历史搜索序列选择移动动作，并在有限搜索预算内尽可能到达目标网格。

<table>
  <tr>
    <td width="25%" align="center"><b>主基准平均 SR</b><br><code>0.580</code></td>
    <td width="25%" align="center"><b>相对 GOMAA 提升</b><br><code>+0.062</code></td>
    <td width="25%" align="center"><b>MM-GAG 平均提升</b><br><code>+0.083</code></td>
    <td width="25%" align="center"><b>长距离平均提升</b><br><code>+0.093</code></td>
  </tr>
</table>

## 方法框架

方法由三部分组成：首先将航拍图像、地面图像或文本描述编码为统一的目标表示；随后用 Transformer 建模历史动作与观测特征序列；最后由 Actor-Critic 策略网络根据当前状态和历史信息选择下一步动作。混合奖励机制只用于训练阶段优化策略，推理阶段直接使用训练好的策略网络。

<p align="center">
  <img src="results/figures/showcase/dataset/figure3_1_method_overview_revised.png" width="100%" alt="方法框架图">
</p>

## 数据集与任务场景

实验材料覆盖航拍目标、地面目标、文本目标和灾前灾后场景。下面的图用于快速说明不同数据设置对应的目标线索和搜索区域形式。

<p align="center">
  <img src="results/figures/showcase/dataset/chapter2_typical_dataset_scene_examples.png" width="100%" alt="数据集与任务场景示例">
</p>

## 轨迹对比

下面的 GIF 把同一个困难样例下的三种方法同步放在一张图中。新版在三个轨迹面板之间保留细间隔，避免画面直接挤在一起，同时保持统一裁剪、统一步数和统一进度条，便于直接观察搜索行为差异。

<p align="center">
  <img src="results/figures/showcase/experience/trajectory_theater_gifs/three_method_hardcase__img189_d6_s20_g14_r0__theater.gif" width="100%" alt="同步轨迹对比动图">
</p>

<p align="center">
  <img src="results/figures/showcase/experience/trajectory_storyboard_experience.png" width="100%" alt="轨迹对比结果图">
</p>

## 核心结果证据墙

证据墙把总体性能、跨模态目标、长距离搜索和轨迹行为放在同一页中，用来快速说明本文方法的提升来自哪些实验现象。完整数值仍以 `results/tables/` 中的表格为准。

<p align="center">
  <img src="results/figures/showcase/experience/evidence_wall_experience.png" width="100%" alt="核心结果证据墙">
</p>

## 主要实验结果

以下结果图按单列展示，避免 GitHub 页面压缩后看不清坐标、数值和图例。每张图对应一组核心实验结论。

### 结果总览

汇总主基准平均成功率、相对提升、跨模态任务和长距离任务表现，用于快速把握整体实验结论。

<p align="center">
  <img src="results/figures/showcase/polished/hero_dashboard.png" width="100%" alt="实验结果总览">
</p>

### 跨模态目标定位

比较航拍图像、地面图像和文本描述三类目标线索，展示目标表示在不同输入形式下的适应能力。

<p align="center">
  <img src="results/figures/showcase/polished/mmgag_modality_panel.png" width="100%" alt="跨模态目标定位表现">
</p>

### 模块消融实验

对门控、势函数塑形、内在奖励和价值分支进行组合消融，说明完整机制相对各个删减版本的收益。

<p align="center">
  <img src="results/figures/showcase/polished/ablation_story_panel.png" width="100%" alt="模块消融实验">
</p>

### 奖励设计分析

比较不同距离门控函数和 PBRS 开关，解释训练阶段混合奖励如何改善策略学习。

<p align="center">
  <img src="results/figures/showcase/polished/reward_design_panel.png" width="100%" alt="奖励设计对比">
</p>

### 长距离搜索与压力测试

展示 8x8、10x10 和 25x25 网格下的预算敏感性与随机种子稳定性，评估搜索范围扩大后的表现。

<p align="center">
  <img src="results/figures/showcase/polished/long_range_panel.png" width="100%" alt="长距离搜索与压力测试">
</p>

### 轨迹行为统计

从成功率、接近目标比例、单调接近率和重复访问率解释本文方法为什么在中远距离任务中更稳定。

<p align="center">
  <img src="results/figures/showcase/polished/trajectory_behavior_panel.png" width="100%" alt="轨迹行为统计">
</p>

## 结果与材料组织

- `code/geoexplorer_active/`：本文方法的干净代码入口，包括训练、评测、数据预处理和模型定义。
- `code/tools/build_visual_showcase.py`：从 `results/tables/` 重新生成 GitHub 展示图、experience 图、polished 图卡、同步 GIF 和媒体清单。
- `code/tools/build_showcase_experience.py`：生成首页首屏海报、轨迹剧场、证据墙和首页叙事资产。
- `code/tools/build_supplement_experiment_analysis.py`：生成置信区间、轨迹行为、奖励过程和补充实验分析图。
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
