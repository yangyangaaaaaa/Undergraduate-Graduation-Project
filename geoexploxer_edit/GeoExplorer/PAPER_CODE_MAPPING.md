# GeoExplorer 论文-代码对应表

## 1. 总体流程

| 论文阶段 | 论文作用 | 代码入口 | 关键输入 | 关键输出 |
| --- | --- | --- | --- | --- |
| Feature Representation | 把搜索区域和目标模态映射到共享特征空间 | `data_utils/get_patches.py`, `data_utils/get_sat_embeddings.py`, `data_utils/get_grd_embeddings.py` | 原始航拍图、地面图 | patch 图像、卫星特征 `npy`、地面特征 `npy` |
| Action-State Dynamics Modeling (DM) | 监督式学习动作-状态联合建模 | `pretrain.py`, `models/pretrain_model.py`, `models/model_falcon.py` | 轨迹序列、动作序列、patch 序列 | 预训练好的 action-state model checkpoint |
| Curiosity-Driven Exploration (CE) | 用 PPO 在 DM 输出上训练主动探索策略 | `train.py`, `models/ppo.py` | DM checkpoint、训练特征、奖励配置 | 探索策略 checkpoint、训练日志 |
| Validation / Analysis | 在 SwissView 上做验证、泛化与可视化分析 | `validate.py`, `models/ppo.py`, `utils/visualization.py` | checkpoint、测试特征、可视化参数 | `metrics.csv`、轨迹图、visited/end 热力图、reward 热力图 |

## 2. Feature Representation

### 论文描述
- 搜索区域的当前状态和目标模态先经过编码器变成对齐特征。
- 航拍目标、地面目标和文本目标理论上都可以进入统一流程。

### 代码对应
- `data_utils/get_patches.py`
  - 作用：把 1500x1500 的搜索区域切成 `5x5` patch。
  - 输出：`patch_0.jpg ... patch_24.jpg`
- `data_utils/get_sat_embeddings.py`
  - 作用：对每个航拍 patch 提取卫星特征。
  - 模型：`MVRL/Sat2Cap`
- `data_utils/get_grd_embeddings.py`
  - 作用：对地面图像提取 CLIP 特征。
  - 模型：`openai/clip-vit-base-patch32`

### 当前实现差异
- 文本模态在验证代码里保留了接口，但仓库里没有与论文一致的完整文本预处理脚本。
- 训练和验证真正跑通的主链是 aerial / ground 两个模态。

## 3. Action-State Dynamics Modeling (DM)

### 论文描述
- 论文用 causal Transformer 同时预测动作和状态。
- 训练监督来自随机生成的动作-状态轨迹。

### 代码对应
- `utils/get_test_config.py`
  - `get_random_sequence`
  - `PretrainRandomSequences`
- `models/pretrain_model.py`
  - `MaskedActionModeling`
  - 联合优化动作预测损失和状态预测损失
- `models/model_falcon.py`
  - 自定义 Falcon 风格 backbone
  - 通过 `state_preds` 与 `state_gt` 提供状态预测监督
- `pretrain.py`
  - Lightning 训练入口

### 当前实现差异
- 代码里虽然写了 `tiiuae/falcon-7b`，但实际是读取 tokenizer/config 思路后重新实例化了一个自定义尺寸的 Falcon 风格模型，不是直接加载原始 7B 预训练权重。
- 代码实现的状态建模损失是手工加到动作预测头上的工程化落地，不是论文公式逐行翻译版。

## 4. Curiosity-Driven Exploration (CE)

### 论文描述
- 论文把目标导向外在奖励和好奇心内在奖励结合起来，引导 agent 在搜索区域里探索。
- 本次改进版进一步把两种奖励改为距离感知的动态平衡。

### 代码对应
- `models/ppo.py`
  - `ActorCritic`
  - `PPO`
  - `compute_reward_components`
  - `rollout_episode`
- `train.py`
  - PPO 训练主循环
  - 每步和每轮训练日志导出

### 当前改进实现
- `reward_mode`
  - `ex_only`
  - `static_mix`
  - `adaptive_mix`
- `adaptive_mix`
  - 定义 `d_t` 为当前 patch 到目标 patch 的 Manhattan 距离
  - 用 `w_in(d_t)=clip((d_t-d_near)/(d_far-d_near),0,1)` 控制内在奖励占比
  - 用 `w_ex(d_t)=1-w_in(d_t)` 控制外在奖励占比
  - 总奖励：`r_total = w_ex * r_ex + w_in * lambda_in * r_in`

## 5. Validation And Visualization

### 论文描述
- 论文网站展示了三类分析图：
  - 路径可视化
  - 路径 visited / end 统计热力图
  - intrinsic reward 可视化

### 代码对应
- `validate.py`
  - 命令行验证入口
  - 支持 `--save_vis`, `--output_dir`, `--num_trials`, `--policy_mode`, `--selected_ids`, `--save_reward_map`
- `utils/visualization.py`
  - 保存 `trajectory.json`
  - 保存 `path_overlay.png`
  - 保存 `reward_heatmap.png`
  - 保存 `composite.png`
  - 保存汇总 `visited_heatmap.png` 和 `end_heatmap.png`

## 6. 本仓库当前主实验边界

- 主实验建议只放在 `SwissView100` 与 `SwissViewMonuments`。
- `SwissViewMonuments` 最适合做论文风格案例图，因为它和论文的 unseen target 可视化最接近。
- Masa/xBD/MM-GAG 更适合作为后续扩展，不作为本次毕设的必要主线。
