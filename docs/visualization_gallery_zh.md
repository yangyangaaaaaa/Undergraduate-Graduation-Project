# 可视化结果画廊

本画廊整理项目展示页、论文补充材料和答辩展示可直接使用的图件。展示素材分为两类：一类用于首页开场、方法说明和轨迹对比，另一类用于展示主要实验结果。这样既保留数据表驱动的严谨性，也避免首页只像普通统计图堆叠。

## 生成方式

```bash
python code/tools/build_visual_showcase.py
```

该脚本会读取 `results/tables/` 中的真实实验结果表，重新生成基础展示图、首页叙事素材、核心结果图、同步三联 GIF，并复制轨迹与奖励机制媒体。新版核心文件位于 `results/figures/showcase/experience/`、`results/figures/showcase/polished/` 和 `results/figures/showcase/trajectories/triptych_gifs/`。

## 首页展示素材

这些图用于项目首页和答辩开场页，目标是先讲清楚“任务是什么、方法强在哪、轨迹为什么更稳”。

<p align="center">
  <img src="../results/figures/showcase/experience/hero_experience.png" width="100%" alt="首页展示海报">
</p>

### 方法框架

<p align="center">
  <img src="../results/figures/showcase/dataset/figure3_1_method_overview_revised.png" width="100%" alt="方法框架图">
</p>

### 数据集与任务场景

<p align="center">
  <img src="../results/figures/showcase/dataset/chapter2_typical_dataset_scene_examples.png" width="100%" alt="数据集与任务场景示例">
</p>

### 核心结果证据墙

<p align="center">
  <img src="../results/figures/showcase/experience/evidence_wall_experience.png" width="100%" alt="核心结果证据墙">
</p>

<p align="center">
  <img src="../results/figures/showcase/experience/trajectory_theater_gifs/three_method_hardcase__img189_d6_s20_g14_r0__theater.gif" width="100%" alt="同步轨迹对比动图">
</p>

<p align="center">
  <img src="../results/figures/showcase/experience/trajectory_storyboard_experience.png" width="100%" alt="轨迹对比结果图">
</p>

## 主要实验结果

主要实验结果按单列展示，避免 GitHub 页面把坐标和数值压得过小。每张图底部都给出简短结论，方便答辩或 README 快速阅读。

### 结果总览

汇总主实验、跨模态任务和长距离压力测试，先给出整体提升幅度。

<p align="center">
  <img src="../results/figures/showcase/polished/hero_dashboard.png" width="100%" alt="实验结果总览">
</p>

### 跨模态目标定位

比较航拍图像、地面图像和文本描述三类目标线索下的成功率。

<p align="center">
  <img src="../results/figures/showcase/polished/mmgag_modality_panel.png" width="100%" alt="跨模态目标定位表现">
</p>

### 模块消融实验

展示 G/P/E/V 四个机制开关的组合消融和平均边际效应。

<p align="center">
  <img src="../results/figures/showcase/polished/ablation_story_panel.png" width="100%" alt="模块消融实验">
</p>

### 奖励设计分析

比较不同距离门控函数、PBRS 开关和奖励端点对照。

<p align="center">
  <img src="../results/figures/showcase/polished/reward_design_panel.png" width="100%" alt="奖励设计分析">
</p>

### 长距离搜索与压力测试

观察扩大网格和搜索预算后，本文方法相对基线是否仍然保持优势。

<p align="center">
  <img src="../results/figures/showcase/polished/long_range_panel.png" width="100%" alt="长距离搜索与压力测试">
</p>

### 轨迹行为统计

用接近目标、单调接近和重复访问等指标解释轨迹为什么更稳定。

<p align="center">
  <img src="../results/figures/showcase/polished/trajectory_behavior_panel.png" width="100%" alt="轨迹行为统计">
</p>

### 奖励过程分解

按成功/失败轨迹分解外在奖励、门控内在奖励、PBRS 和总奖励。

<p align="center">
  <img src="../results/figures/showcase/polished/reward_process_panel.png" width="100%" alt="奖励过程分解">
</p>

## 同步三联动图

三联动图将同一案例下的本文方法、GOMAA-Geo 和 GeoExplorer 轨迹同步放在一张 GIF 中。新版采用无白底、无卡片、无外边框的三联拼接，并在面板之间保留细间隔，只保留图内方法名、步数和极细进度线，适合项目首页、答辩页和补充材料快速展示。

<p align="center">
  <img src="../results/figures/showcase/trajectories/triptych_gifs/three_method_hardcase__img189_d6_s20_g14_r0__triptych.gif" width="100%" alt="困难样例三联轨迹动图">
</p>

可用三联动图清单：

| 案例 | 文件 |
| --- | --- |
| C=4 失败/绕行样例 | `results/figures/showcase/trajectories/triptych_gifs/c4_anchor_failure_or_detour__img025_d4_s12_g00_r0__triptych.gif` |
| C=4 成功样例 | `results/figures/showcase/trajectories/triptych_gifs/c4_anchor_success__img011_d4_s03_g11_r0__triptych.gif` |
| C=6 失败/绕行样例 | `results/figures/showcase/trajectories/triptych_gifs/c6_anchor_failure_or_detour__img050_d6_s14_g00_r0__triptych.gif` |
| C=6 成功样例 | `results/figures/showcase/trajectories/triptych_gifs/c6_anchor_success__img006_d6_s24_g06_r0__triptych.gif` |
| C=8 失败/绕行样例 | `results/figures/showcase/trajectories/triptych_gifs/c8_anchor_failure_or_detour__img054_d8_s20_g04_r0__triptych.gif` |
| C=8 成功样例 | `results/figures/showcase/trajectories/triptych_gifs/c8_anchor_success__img000_d8_s24_g00_r0__triptych.gif` |
| 三方法困难样例 | `results/figures/showcase/trajectories/triptych_gifs/three_method_hardcase__img189_d6_s20_g14_r0__triptych.gif` |

## 论文补充分析图

<table>
  <tr>
    <td width="50%" valign="top">
      <img src="../results/figures/supplement/mmgag_diff_ci.png" width="100%" alt="mmgag ci"><br>
      <b>MM-GAG 差值置信区间</b><br>
      展示 A/G/T 三种目标模态下本文方法相对 GOMAA-Geo 的 SR 差值及近似 95% CI。
    </td>
    <td width="50%" valign="top">
      <img src="../results/figures/supplement/ultra_long_diff_ci.png" width="100%" alt="ultra ci"><br>
      <b>长距离差值置信区间</b><br>
      展示 8x8/10x10 长距离扩展测试中的相对提升。
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <img src="../results/figures/supplement/p0_budget_sensitivity.png" width="100%" alt="p0 budget"><br>
      <b>P0 预算敏感性</b><br>
      展示 8x8/10x10 下预算变化对 SR 的影响。
    </td>
    <td width="50%" valign="top">
      <img src="../results/figures/supplement/p1_grid25_budget_sensitivity.png" width="100%" alt="p1 budget"><br>
      <b>25x25 压力测试</b><br>
      展示极端大网格下预算变化、相对领先和绝对困难性。
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <img src="../results/figures/supplement/c4_failure_profile.png" width="100%" alt="c4 failure"><br>
      <b>C=4 失败画像</b><br>
      支撑“方法优势集中在中远距离，短距离不一定占优”的解释。
    </td>
    <td width="50%" valign="top">
      <img src="../results/figures/supplement/reward_component_traces.png" width="100%" alt="reward traces"><br>
      <b>奖励过程曲线</b><br>
      展示外在奖励、门控内在奖励、PBRS 和总奖励随步数变化的趋势。
    </td>
  </tr>
</table>

## 原始展示图与素材索引

旧版基础图仍保留在 `results/figures/showcase/`，包括 `benchmark_overview_dashboard.png`、`mmgag_modality_sr.png`、`distance_bucket_curves.png`、`generalization_heatmap_16cell.png`、`reward_gate_pb_comparison.png`、`parameter_sensitivity_curves.png`、`ultra_long_sr_curves.png`、`grid25_stress_curves.png` 等。这些图继续作为可复现中间结果保存，但项目首页优先使用 polished 图卡。

轨迹媒体目录：

- `results/figures/showcase/trajectories/triptych_gifs/`：7 个新版同步三联动图。
- `results/figures/showcase/trajectories/gifs/`：21 个原始单方法轨迹 GIF。
- `results/figures/showcase/trajectories/comparison_png/`：7 张三方法静态对比图。
- `results/figures/showcase/trajectories/static_png/`：21 张单方法静态轨迹图。

奖励机制图目录：

- `results/figures/showcase/reward_story/`：机制示意、G/P 2x2 路径对比、奖励矩阵、结果矩阵和奖励回放 GIF。

## 使用建议

- 项目首页优先使用 `polished/hero_dashboard.png`、`polished/mmgag_modality_panel.png`、`polished/ablation_story_panel.png`、`polished/reward_design_panel.png`、`polished/long_range_panel.png` 和一张 triptych GIF。
- 答辩展示优先使用 PNG 和 GIF；论文正文可继续使用 SVG/PDF 或从源表重新导出。
- 主表结论以 `results/tables/` 中的 CSV/JSON 为准，展示图用于帮助读者快速理解趋势。
