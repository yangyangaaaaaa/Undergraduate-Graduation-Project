# Undergraduate Graduation Project

本仓库用于整理本科毕业设计《好奇心驱动的无人机主动定位目标导航方法》的代码、实验结果、论文材料与复现说明。

项目面向离散网格下的无人机主动目标定位导航任务。给定航拍搜索区域和目标线索后，智能体从起始网格出发，根据当前位置观测、目标表示和历史搜索序列选择移动动作，并在有限搜索预算内尽可能到达目标网格。

## 项目内容

- `code/geoexplorer_active/`：本文方法的干净代码入口，包括训练、评测、数据预处理和模型定义。
- `experiments/`：论文主表、消融实验、参数实验和长距离测试对应的实验脚本与 manifest。
- `results/`：已整理的结果表、图件和中文分析报告。
- `docs/`：代码结构、实验结论、数据划分和复现说明。
- `thesis/`：论文正文与 Markdown 草稿。
- `materials/`：任务书、开题报告、中期报告、外文翻译等毕设过程材料。

仓库中不包含训练 checkpoint、原始大规模数据包和本地临时缓存。结果表保留了 checkpoint 路径或 run 名称，用于和原始实验设置对应。

## 方法概述

本文方法以 GeoExplorer 主体网络为基础，保留目标编码、历史动作-观测序列建模和 Actor-Critic 策略推理框架，重点改进训练阶段的奖励构造：

```text
r_t = r_ex,t + lambda_t r_in,t + r_p,t
```

其中 `r_ex,t` 是外在目标奖励，`r_in,t` 是由下一步特征预测误差构造的好奇心内在奖励，`lambda_t` 是随距离变化的门控权重，`r_p,t` 是势函数奖励塑形项。推理阶段不再计算混合奖励或更新 PPO 参数，只使用训练后的策略网络选择动作。

## 主要结果

在 MM-GAG 航拍图像、地面图像和文本目标三类任务上，本文方法的总体 SR 分别为 `0.6170`、`0.6391` 和 `0.6247`，优势主要集中在 `C=6` 到 `C=8` 的中远距离条件。

在 16 组 G/P/E/V 机制消融中，完整方法 `g1_p1_e1_v1` 的主泛化均值为 `0.6211`，高于同数据无新增机制控制组 `g0_p0_e0_v0` 的 `0.5545`。

在长距离扩展测试中，本文方法在 `8 x 8` 网格、预算 `B=24` 时 SR 为 `0.7460`，在 `10 x 10` 网格、预算 `B=32` 时 SR 为 `0.7480`，均高于 GOMAA-Geo 和 GeoExplorer 原始风格基线。

详细表格见 [实验结果总览](docs/experiment_summary_zh.md) 和 `results/tables/`。

## 复现入口

环境配置：

```bash
cd code/geoexplorer_active
conda env create -f environment.yml
conda activate geoexplorer
```

预训练、训练和评测入口：

```bash
python pretrain.py
python train.py
python validate.py
```

批量实验脚本位于 `experiments/scripts/`。复现实验前需要准备对应数据 embedding 和 checkpoint；仓库仅保存代码、manifest、结果表和说明文档，不保存大型权重文件。

## 文档导航

- [实验结果总览](docs/experiment_summary_zh.md)
- [复现说明](docs/reproducibility_zh.md)
- [结果文件索引](docs/result_inventory_zh.md)
- [代码结构说明](docs/code_structure_zh.md)
