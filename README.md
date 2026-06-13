# 好奇心驱动的无人机主动定位目标导航方法

本科毕业设计项目归档仓库。当前首页按 2026 年 6 月结题答辩 PPT 重新整理，展示内容依据最终 PPT 中实际嵌入的媒体与论文定稿结论整理，动图已做 GitHub 展示压缩。

<p align="center">
  <img src="results/figures/defense_readme_20260613/method_framework.png" width="100%" alt="研究框架">
</p>

## 项目概览

本项目研究低空场景下的无人机主动目标定位导航。给定一个航拍搜索区域和目标线索后，无人机从起点出发，在每一步根据当前位置观测、历史动作和目标表示选择移动方向，并在有限搜索预算内尽量到达目标网格。

任务并不是一次性图像识别，也不是已知终点路径规划。它更接近“边移动、边观察、边修正”的连续搜索问题。本文方法把航拍图像、地面图像和文本描述等目标线索统一编码，并在 PPO 策略学习中加入好奇心驱动的混合奖励机制。

## 核心思想

混合奖励只用于训练阶段，正式测试阶段只加载训练好的 checkpoint 输出动作，不再调用奖励函数。

训练信号由三部分组成：

- 外在奖励：保证目标导向，到达或靠近目标时提供直接反馈。
- 好奇心内在奖励：由状态预测误差产生，用于补充探索动力。
- 距离调节与势函数塑形：在远距离阶段鼓励探索，在接近目标时抑制偏离并转向收敛。

## 答辩主线展示

下面几组图依据最终结题答辩 PPT 嵌入媒体抽取与压缩，保留 PPT 的讲述顺序：任务设置、连续搜索、跨模态目标、灾后场景、长距离搜索和实验分析。

### 连续搜索轨迹

<p align="center">
  <img src="results/figures/defense_readme_20260613/c8_route_demo.gif" width="100%" alt="C8 连续搜索轨迹">
</p>

### 跨模态文本目标

<p align="center">
  <img src="results/figures/defense_readme_20260613/multimodal_text_target.gif" width="100%" alt="文本目标搜索轨迹">
</p>

### 灾后场景搜索

<p align="center">
  <img src="results/figures/defense_readme_20260613/xbd_disaster_route.gif" width="100%" alt="xBD 灾后搜索轨迹">
</p>

### 长距离扩展任务

<p align="center">
  <img src="results/figures/defense_readme_20260613/long_distance_grid10.gif" width="100%" alt="10x10 长距离搜索轨迹">
</p>

### 短距离失败边界

<p align="center">
  <img src="results/figures/defense_readme_20260613/short_distance_failure.png" width="100%" alt="短距离失败案例">
</p>

## 实验结论

论文定稿与答辩 PPT 的主要结论是一致的：

- 在 MM-GAG 上，航拍图像、地面图像、文本目标的平均成功率分别为 `0.6170`、`0.6391`、`0.6247`。
- 在 `10 x 10` 长距离扩展测试中，平均成功率为 `0.7480`，平均最终距离为 `0.7360`。
- 方法优势主要体现在中远距离、多步搜索和跨模态目标线索场景。
- C1-C3 短距离补充评测显示，短距离不是本文方法的优势区间；局部相似模块容易导致徘徊和末端定位波动。

<p align="center">
  <img src="results/figures/defense_readme_20260613/performance_analysis.png" width="100%" alt="实验结果分析">
</p>

<p align="center">
  <img src="results/figures/defense_readme_20260613/reward_distance_analysis.png" width="100%" alt="奖励分距离分析">
</p>

<p align="center">
  <img src="results/figures/defense_readme_20260613/ablation_analysis.png" width="100%" alt="消融分析">
</p>

## 仓库内容

- `code/geoexplorer_active/`：整理后的核心代码入口。
- `code/tools/`：结果整理、可视化生成、服务器验收与短距离评估脚本。
- `docs/`：代码结构、实验复现、结题收尾、服务器验收、PPT 汇报脉络等说明文档。
- `results/tables/`：主实验、消融实验、补充实验和短距离评估表格。
- `results/reports/`：实验总结、奖励分析、续训诊断、短距离 C1-C3 评估报告。
- `results/figures/defense_readme_20260613/`：从最终结题答辩 PPT 抽取并压缩的首页展示素材。
- `thesis/`：论文草稿与 Word 版本归档。
- `materials/project_admin/`：任务书、开题、中期、外文翻译等过程材料。

## 复现与验收

基础训练与评估入口：

```bash
cd code/geoexplorer_active
conda env create -f environment.yml
conda activate geoexplorer
python pretrain.py
python train.py
python validate.py
```

服务器验收脚本已整理到 `code/tools/`。远端部署时使用环境变量或安全凭据传入登录信息，不要把密码、token 或 cookie 写入仓库。

常用验收命令：

```bash
/root/geoexplorer/run_acceptance_demo
/root/geoexplorer/run_acceptance_demo --visual-only
/root/geoexplorer/run_acceptance_case_pack
/root/geoexplorer/run_acceptance_train --status
/root/geoexplorer/run_c123_eval
```

## 进一步阅读

- [结题收尾总览](docs/project_closing_summary_zh.md)
- [PPT 汇报脉络](docs/defense_ppt_storyline_zh.md)
- [服务器验收指南](docs/server_acceptance_guide_zh.md)
- [实验结果总览](docs/experiment_summary_zh.md)
- [可视化结果画廊](docs/visualization_gallery_zh.md)
- [复现说明](docs/reproducibility_zh.md)
