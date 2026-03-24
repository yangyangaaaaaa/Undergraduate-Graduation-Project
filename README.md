# GeoExplorer 本科毕业设计仓库

本仓库用于本人本科毕业设计的资料整理、代码复现与实验记录。课题题目为“好奇心驱动的无人机主动定位目标导航方法”，核心工作是在 GeoExplorer 论文的基础上完成复现、代码理解、局部改进和实验验证。

## 课题目标

根据任务书、开题答辩材料和论文原文，本课题围绕低空无人机在预定义搜索区域内的主动目标定位与导航问题展开，重点解决以下问题：

- 在复杂、未知或部分可观测环境中，传统仅依赖目标导向奖励的导航方法探索效率低。
- 在长程搜索任务中，外部奖励稀疏，智能体容易陷入局部最优或盲目搜索。
- 需要结合多模态目标信息与环境状态建模，提升策略的鲁棒性与泛化能力。

本仓库当前的主线工作是：

1. 复现 GeoExplorer 论文方法，并完成论文到代码的一一对应理解。
2. 在原始方法基础上，对内在奖励与外在奖励的平衡机制进行改进。
3. 补齐命令行环境下的推理结果保存与可视化功能，支持路径图、热力图和奖励分布图导出。
4. 在 SwissView 系列数据集上完成实验与分析，形成毕业论文结果。

## 研究基础

本课题直接参考论文：

- `GeoExplorer: Active Geo-localization with Curiosity-Driven Exploration`

任务书和开题材料中明确了以下研究方向：

- 多模态目标引导：目标可以由俯视图像、地面图像或文本描述给出。
- 动作-状态联合建模：不仅预测下一步动作，还建模动作如何改变状态。
- 好奇心驱动探索：在外在目标奖励之外引入内在奖励，缓解奖励稀疏问题。
- 实验验证与消融分析：在仿真或离线数据场景下验证方法有效性。

## 本仓库的改进重点

在前期讨论基础上，本仓库将论文中的“固定外在奖励 + 内在奖励”思路进一步细化为“距离感知的自适应奖励平衡”：

- 当智能体距离目标较远、外部奖励稀疏时，提高内在好奇心奖励权重，鼓励探索未知区域。
- 当智能体逐渐接近目标时，提高外在奖励权重，减少无效探索，增强目标收敛能力。
- 该改进不改变 GeoExplorer 的主体网络结构，主要修改奖励融合逻辑、日志记录与实验分析流程。

当前代码侧的主要改动包括：

- 新增 `reward_mode = ex_only | static_mix | adaptive_mix`
- 支持 `d_near`、`d_far`、`lambda_in` 等自适应奖励参数
- 训练与验证阶段记录 `r_ex / r_in / w_ex / w_in / r_total`
- 新增命令行可视化导出，支持：
  - 搜索路径叠加图
  - visited / end patch 热力图
  - intrinsic reward 热力图
  - 轨迹与指标文件保存

## 仓库结构

仓库当前按“原始代码 + 修改版本 + 资料”组织：

- `GeoExplorer/`
  - 原始 GeoExplorer 代码与网页材料
- `geoexploxer_edit/`
  - 本课题实际使用的修改版代码与 SwissView 数据
  - 其中 `geoexploxer_edit/GeoExplorer/` 是主要代码目录
- `相关文献/`
  - 与课题相关的论文资料，包括 GeoExplorer 原文及扩展阅读
- `GeoExploer.pdf`
  - 本地保留的论文 PDF
- `08122215_张洋_任务书.docx`
  - 毕设任务书
- `08122215_张洋_开题报告.doc`
  - 开题报告
- `08122215_张洋_开题答辩.pptx`
  - 开题答辩材料
- `08122215_张洋_外文文献翻译.docx`
  - 论文翻译材料

## 代码与论文对应关系

修改版代码中已经补充了论文到代码的映射文档：

- `geoexploxer_edit/GeoExplorer/PAPER_CODE_MAPPING.md`

当前理解主线如下：

1. Feature Representation
   - `data_utils/get_patches.py`
   - `data_utils/get_sat_embeddings.py`
   - `data_utils/get_grd_embeddings.py`
2. Action-State Dynamics Modeling
   - `pretrain.py`
   - `models/pretrain_model.py`
   - `models/model_falcon.py`
3. Curiosity-Driven Exploration
   - `train.py`
   - `models/ppo.py`
4. Validation And Visualization
   - `validate.py`
   - `utils/visualization.py`

## 数据与实验范围

结合当前设备条件和毕设范围，实验主线主要聚焦在 SwissView 数据集：

- `SwissView100`
  - 用于基础验证和主结果测试
- `SwissViewMonuments`
  - 用于未见目标泛化实验和论文风格可视化案例

实验对比建议固定为三组：

- `ex_only`
- `static_mix`
- `adaptive_mix`

核心评价指标包括：

- Success Ratio
- 平均成功步数
- 偏离最优步数
- 最终到目标的网格距离
- 唯一访问 patch 数

## 运行说明

### 1. 环境准备

原始仓库环境文件位于：

- `GeoExplorer/environment.yml`
- `geoexploxer_edit/GeoExplorer/environment.yml`

建议在 AutoDL 平台或具备高显存 GPU 的 Linux 环境下运行训练。

### 2. 数据预处理

SwissView 数据处理入口包括：

- `geoexploxer_edit/GeoExplorer/data_utils/get_patches.py`
- `geoexploxer_edit/GeoExplorer/data_utils/get_sat_embeddings.py`
- `geoexploxer_edit/GeoExplorer/data_utils/get_grd_embeddings.py`

### 3. 训练流程

动作-状态预训练：

```bash
python pretrain.py
```

PPO 探索训练：

```bash
python train.py
```

### 4. 推理与可视化

推理入口：

```bash
python validate.py
```

当前修改版支持在命令行环境下直接保存可视化结果，适合无图形界面的服务器环境。典型输出包括：

- `metrics.csv`
- `trajectory.json`
- `path_overlay.png`
- `reward_heatmap.png`
- `composite.png`
- `visited_heatmap.png`
- `end_heatmap.png`

## 当前阶段结论

就本科毕设范围而言，本仓库已经形成了比较清晰的三段式路线：

1. 读懂论文并完成代码映射。
2. 实现“靠近目标时外在奖励为主、远离目标时内在奖励为主”的奖励平衡改进。
3. 通过 SwissView 系列数据集完成复现、对比实验和可视化分析。

这一路线与任务书中的“多模态信息融合、动态路径生成、算法实验与性能分析”保持一致，也与开题答辩中提出的“混合奖励函数 + 动作-状态建模”创新点一致。

## 说明

- 本仓库同时保留了原始 GeoExplorer 代码和修改版代码，便于对照分析。
- 修改版以 `geoexploxer_edit/GeoExplorer/` 为主。
- 若后续继续扩展，可将更多实验日志、结果图和论文写作材料补充到仓库中。

