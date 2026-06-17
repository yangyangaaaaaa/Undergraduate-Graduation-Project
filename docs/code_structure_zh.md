# 代码结构说明

本文方法的主代码入口位于 `code/main/`。该目录保留复现论文实验所需的核心训练、评测、数据预处理和模型定义文件；生成图表、整理结果和验收材料的离线脚本放在 `code/tools/`。

## 主入口

| 文件 | 作用 |
| --- | --- |
| `config.py` | 统一配置入口，支持通过环境变量切换奖励、门控函数、PBRS 系数、训练数据和验证距离。 |
| `pretrain.py` | 训练动作-特征预测模块，为好奇心内在奖励提供预测误差来源。 |
| `train.py` | PPO 训练入口，实现外在奖励、内在奖励、距离门控和势函数奖励塑形。 |
| `validate.py` | 固定 checkpoint 的推理评测入口，使用合法动作约束下的策略输出完成评测。 |
| `environment.yml` | Linux GPU 环境依赖。 |

## 模型模块

| 目录/文件 | 作用 |
| --- | --- |
| `models/model_falcon.py` | 历史动作-观测序列建模与状态特征输出。 |
| `models/ppo.py` | Actor-Critic 策略网络、PPO 更新和多模态评测函数。 |
| `models/pretrain_model.py` | 下一步特征预测模型。 |
| `models/decision_transformer.py` | 序列建模组件。 |

## 数据与工具

| 目录/文件 | 作用 |
| --- | --- |
| `data_utils/get_patches.py` | 将航拍图像划分为规则网格 patch。 |
| `data_utils/get_sat_embeddings.py` | 航拍图像 embedding 生成。 |
| `data_utils/get_grd_embeddings.py` | 地面图像 embedding 生成。 |
| `data_utils/prepare_masa_embeddings.py` | MASA 数据预处理入口。 |
| `data_utils/prepare_mmgag_embeddings.py` | MM-GAG 航拍、地面和文本目标预处理入口。 |
| `data_utils/prepare_xbd_embeddings.py` | xBD pre/post-disaster 评测数据预处理入口。 |
| `data_utils/sequence.py` | 历史动作-观测序列构造。 |
| `utils/get_test_config.py` | 固定距离任务配置生成。 |
| `utils/random_seed.py` | 随机种子控制。 |
| `utils/swissviewmonuments_metadata.py` | SwissViewMonuments 目标元数据。 |

## 奖励实现位置

训练阶段的奖励组合写在 `train.py`：

```text
r_t = r_ex,t + lambda_t r_in,t + r_p,t
```

- `r_ex,t` 由 `ppo_agent.get_reward()` 给出。
- `r_in,t` 来自预测特征与真实观测特征之间的均方误差。
- `lambda_t` 由 `gate_weight()` 根据当前距离计算。
- `r_p,t` 由 `pbrs_bonus()` 按势函数奖励塑形计算。

推理阶段只使用训练后的策略网络。奖励函数不参与动作选择，也不再更新 PPO 参数。
