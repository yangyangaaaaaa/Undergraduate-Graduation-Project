# 复现说明

本说明记录论文实验的复现入口。仓库保存代码、实验脚本、manifest、结果表和图件；训练 checkpoint、原始大规模数据包和生成的 embedding 数组不进入仓库。

## 环境

建议使用 Linux GPU 环境。基础环境文件位于：

```bash
code/main/environment.yml
```

创建环境：

```bash
cd code/main
conda env create -f environment.yml
conda activate code
```

## 数据划分

| 用途 | 口径 |
| --- | --- |
| 训练集 | 默认使用 MASA + MM-GAG，MASA 提供主要航拍目标样本，MM-GAG 补充航拍、地面和文本目标样本。 |
| 验证集 | 训练过程中用于模型选择，不参与最终论文表格统计。 |
| 测试集 | 使用固定任务库，按数据集、目标模态和初始距离条件统计 SR/SG。 |
| 迁移测试 | MASA、MM-GAG、SwissView100、SwissViewMonuments、xBD-pre、xBD-disaster。 |

标准评测使用 `5 x 5` 网格、搜索预算 `B=10`、初始距离 `C={4,5,6,7,8}`。动作空间为上、右、下、左四个合法移动动作。

## 训练入口

预训练动作-特征预测模块：

```bash
python pretrain.py
```

训练 PPO 策略：

```bash
python train.py
```

常用环境变量：

```bash
GEOEXPLORER_REWARD=in
GEOEXPLORER_GATE_MODE=linear
GEOEXPLORER_GATE_FLOOR=0.405
GEOEXPLORER_PBRS_COEF=1.0
GEOEXPLORER_VAL_DISTS=7,8
```

这些变量沿用实验阶段脚本命名，用于保证结果表与原始训练记录可以对应。

## 评测入口

单模型评测：

```bash
python validate.py
```

论文表格使用的批量评测脚本位于：

```text
experiments/scripts/paper_generalization/
experiments/scripts/appendix_dataset_param/
experiments/scripts/appendix_gate_valdist/
experiments/scripts/ultra_long_eval/
```

## Checkpoint 对应关系

结果表中的 `checkpoint` 列保留原始 GPU 环境中的 checkpoint 路径。仓库不上传权重文件，但 run 名称、manifest 和结果表可以对应到具体实验设置。

| 论文称呼 | 结果表 run | 说明 |
| --- | --- | --- |
| 本文方法 | `g1_p1_e1_v1_seed321_t480k` | 线性距离门控 + PBRS + 低熵设置 + `VAL_DISTS=7,8`。 |
| 同数据无新增机制控制组 | `g0_p0_e0_v0_seed321_t480k` | 与本文方法同数据、同训练步数，但不启用 G/P/E/V。 |
| 好奇心探索基线 | `pristine_seed321_t480k` | 统一协议下的基线重评估。 |
| GOMAA-Geo | `formal_ppo_seed42_t480k` | 外部对比方法 checkpoint。 |

## 长距离扩展测试

长距离测试用于验证中远距离搜索优势，不替代标准 `5 x 5` 主评测。

| 设置 | 网格 | 距离 | 预算 | 任务数 |
| --- | --- | --- | ---: | ---: |
| 正式扩展 1 | `8 x 8` | `D={10,11,12,13,14}` | `24` | `1000` |
| 正式扩展 2 | `10 x 10` | `D={14,15,16,17,18}` | `32` | `1000` |
| 探索性压力测试 | `25 x 25` | `D={12,16,20,24,28,32,36,40,44,48}` | `60` | `500` |

论文正文建议使用 `8 x 8` 和 `10 x 10` 的正式扩展结果。`25 x 25` 结果可作为探索性补充，不作为主结论。
