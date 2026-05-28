# Undergraduate Graduation Project

<p align="center">
  <b>好奇心驱动的无人机主动定位目标导航方法</b><br>
  Active target geo-localization with distance-aware curiosity reward shaping
</p>

<p align="center">
  <img src="results/figures/showcase/polished/hero_dashboard.png" width="100%" alt="Polished benchmark dashboard">
</p>

## 项目概览

本仓库整理本科毕业设计《好奇心驱动的无人机主动定位目标导航方法》的代码、实验结果、可视化材料、论文文档和复现说明。任务面向离散网格下的无人机主动目标定位导航：给定航拍搜索区域和目标线索后，智能体从起始网格出发，根据当前位置观测、目标表示和历史搜索序列选择移动动作，并在有限预算内尽可能到达目标网格。

<table>
  <tr>
    <td width="25%" align="center"><b>主基准平均 SR</b><br><code>0.580</code></td>
    <td width="25%" align="center"><b>相对 GOMAA 提升</b><br><code>+0.062</code></td>
    <td width="25%" align="center"><b>MM-GAG 平均提升</b><br><code>+0.083</code></td>
    <td width="25%" align="center"><b>展示文件</b><br><code>128</code></td>
  </tr>
</table>

## 方法概览

本文方法保留 GeoExplorer 的目标编码、历史动作-观测序列建模和 Actor-Critic 策略推理框架，重点改进训练阶段的混合奖励：

```text
r_t = r_ex,t + lambda_t r_in,t + r_p,t
```

其中 `r_ex,t` 为外在目标奖励，`r_in,t` 为下一步特征预测误差构造的好奇心内在奖励，`lambda_t` 为随距离变化的门控权重，`r_p,t` 为势函数奖励塑形项。推理阶段不再计算混合奖励或更新 PPO 参数，只使用训练后的策略网络进行动作选择。

<p align="center">
  <img src="results/figures/showcase/dataset/figure3_1_method_overview_revised.png" width="88%" alt="Method overview">
</p>

## 科研风格结果展示

<table>
  <tr>
    <td width="50%" valign="top">
      <img src="results/figures/showcase/polished/mmgag_modality_panel.png" width="100%" alt="MM-GAG modality panel"><br>
      <b>MM-GAG 跨模态目标</b><br>
      A/G/T 三类目标形式下，本文方法均保持高于 GOMAA-Geo 的成功率；AiRLoc 仅作为航拍目标补充基线。
    </td>
    <td width="50%" valign="top">
      <img src="results/figures/showcase/polished/ablation_story_panel.png" width="100%" alt="Ablation story panel"><br>
      <b>机制消融</b><br>
      16 组 G/P/E/V 消融中，完整分支 <code>g1_p1_e1_v1</code> 取得最高主泛化均值。
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <img src="results/figures/showcase/polished/reward_design_panel.png" width="100%" alt="Reward design panel"><br>
      <b>奖励机制</b><br>
      线性距离门控与 PBRS 组合在 MM-GAG 平均 SR 上表现最好，纯内在奖励无法单独支撑目标定位。
    </td>
    <td width="50%" valign="top">
      <img src="results/figures/showcase/polished/long_range_panel.png" width="100%" alt="Long-range stress panel"><br>
      <b>超长距离与压力测试</b><br>
      8x8、10x10 和 25x25 网格测试展示扩大搜索范围后的相对优势与极端难度。
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <img src="results/figures/showcase/polished/trajectory_behavior_panel.png" width="100%" alt="Trajectory behavior panel"><br>
      <b>轨迹行为分析</b><br>
      用成功率、距离缩短率、单调接近率和重复访问率解释中远距离搜索行为差异。
    </td>
    <td width="50%" valign="top">
      <img src="results/figures/showcase/polished/reward_process_panel.png" width="100%" alt="Reward process panel"><br>
      <b>奖励过程补充</b><br>
      展示外在奖励、门控内在奖励、PBRS 和总奖励在成功/失败轨迹中的变化。
    </td>
  </tr>
</table>

## 同步动态轨迹

下面的 GIF 将同一个困难样例中的三种方法同步放在一张图中，避免原先三张 GIF 分列时高度不统一、播放不同步和换行混乱的问题。更多成功、失败和绕行案例见 [可视化画廊](docs/visualization_gallery_zh.md)。

<p align="center">
  <img src="results/figures/showcase/trajectories/triptych_gifs/three_method_hardcase__img189_d6_s20_g14_r0__triptych.gif" width="100%" alt="Synchronized trajectory replay">
</p>

## 结果与材料组织

- `code/geoexplorer_active/`：本文方法的干净代码入口，包括训练、评测、数据预处理和模型定义。
- `code/tools/build_visual_showcase.py`：从 `results/tables/` 重新生成 GitHub 展示图、同步三联 GIF 和媒体清单。
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
