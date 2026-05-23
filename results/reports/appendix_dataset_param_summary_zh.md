# 附录补充实验汇总（自动生成）

- 生成时间：`2026-05-21T06:45:39+00:00`
- 训练 run 数：`24`
- 评测协议：`5x5`, `B=10`, `C={4,5,6,7,8}`, greedy，任务种子 `20260519`。
- `mean_transfer` 默认取 SwissViewMonuments aerial/ground 与 xBD-pre/xBD-disaster，用于减少训练域同域测试造成的偏差。

## 训练数据集比较（按 mean_transfer_sr 排序）

| 训练数据 | Run | Transfer SR | All SR | Transfer SG | All SG |
| --- | --- | --- | --- | --- | --- |
| MASA+MM-GAG | dataset_masa_plus_mmgag_seed321_t480k | 0.5422 | 0.5765 | 1.4070 | 1.2885 |
| MASA+SwissView100 | dataset_masa_plus_swissview_seed321_t480k | 0.5196 | 0.5389 | 1.5214 | 1.4590 |
| MASA+MM-GAG+SwissView100 | dataset_all_three_seed321_t480k | 0.4879 | 0.5060 | 1.8591 | 1.7877 |
| MM-GAG+SwissView100 | dataset_mmgag_plus_swissview_seed321_t480k | 0.4846 | 0.5060 | 1.9128 | 1.8599 |
| SwissView100 | dataset_swissview_only_seed321_t480k | 0.4340 | 0.4594 | 2.2050 | 2.1371 |
| MASA | dataset_masa_only_seed321_t480k | 0.4283 | 0.4230 | 2.0537 | 2.1114 |
| MM-GAG | dataset_mmgag_only_seed321_t480k | 0.4213 | 0.4329 | 2.0628 | 2.0248 |

- transfer 最优训练数据：`MASA+MM-GAG`。
- all-benchmark 最优训练数据：`MASA+MM-GAG`。

## 参数敏感性（按参数族分组）

### ent_coef

| Value | Run | Transfer SR | All SR | Transfer SG | All SG |
| --- | --- | --- | --- | --- | --- |
| 0.005 | dataset_masa_plus_mmgag_seed321_t480k | 0.5422 | 0.5765 | 1.4070 | 1.2885 |
| 0.0025 | param_ent_0p0025_seed321_t480k | 0.5376 | 0.5497 | 1.4675 | 1.4166 |
| 0.0100 | param_ent_0p01_seed321_t480k | 0.4715 | 0.4729 | 1.7639 | 1.7864 |
| 0.0075 | param_ent_0p0075_seed321_t480k | 0.4409 | 0.4344 | 1.9910 | 2.0582 |

### gate_floor

| Value | Run | Transfer SR | All SR | Transfer SG | All SG |
| --- | --- | --- | --- | --- | --- |
| 0.405 | dataset_masa_plus_mmgag_seed321_t480k | 0.5422 | 0.5765 | 1.4070 | 1.2885 |
| 0.350 | param_gate_floor_0p35_seed321_t480k | 0.4908 | 0.4861 | 1.6459 | 1.6788 |
| 0.500 | param_gate_floor_0p5_seed321_t480k | 0.4868 | 0.4987 | 1.6851 | 1.6200 |
| 0.650 | param_gate_floor_0p65_seed321_t480k | 0.4863 | 0.5089 | 1.7515 | 1.6526 |
| 0.250 | param_gate_floor_0p25_seed321_t480k | 0.4833 | 0.5009 | 1.7484 | 1.7052 |

### pbrs_coef

| Value | Run | Transfer SR | All SR | Transfer SG | All SG |
| --- | --- | --- | --- | --- | --- |
| 0.10 | dataset_masa_plus_mmgag_seed321_t480k | 0.5422 | 0.5765 | 1.4070 | 1.2885 |
| 0.00 | param_pbrs_0_seed321_t480k | 0.5300 | 0.5535 | 1.4173 | 1.3290 |
| 0.20 | param_pbrs_0p2_seed321_t480k | 0.5139 | 0.5307 | 1.5382 | 1.4880 |
| 0.05 | param_pbrs_0p05_seed321_t480k | 0.4951 | 0.5134 | 1.6142 | 1.5409 |
| 0.15 | param_pbrs_0p15_seed321_t480k | 0.4795 | 0.4933 | 1.7808 | 1.7120 |

### train_budget

| Value | Run | Transfer SR | All SR | Transfer SG | All SG |
| --- | --- | --- | --- | --- | --- |
| 480k | dataset_masa_plus_mmgag_seed321_t480k | 0.5422 | 0.5765 | 1.4070 | 1.2885 |
| 720k | param_budget_720k_seed321_t720k | 0.5384 | 0.5673 | 1.3903 | 1.2851 |
| 240k | param_budget_240k_seed321_t240k | 0.4918 | 0.4923 | 1.6329 | 1.6221 |

### val_dists

| Value | Run | Transfer SR | All SR | Transfer SG | All SG |
| --- | --- | --- | --- | --- | --- |
| 7,8 | dataset_masa_plus_mmgag_seed321_t480k | 0.5422 | 0.5765 | 1.4070 | 1.2885 |
| 6,7,8 | param_val_dists_678_seed321_t480k | 0.5108 | 0.5334 | 1.5902 | 1.4900 |
| 4,5,6,7,8 | param_val_dists_45678_seed321_t480k | 0.4354 | 0.4282 | 2.0864 | 2.1324 |

## 随机种子稳定性

- `mean_transfer_sr` 平均：`0.5179`
- `mean_transfer_sr` 标准差（population）：`0.0386`

## 写作提醒

- 如果某个训练数据在同域 benchmark 很高，但 transfer mean 不高，正文不要用它证明泛化。
- 如果当前参数不是每一列都第一，可以表述为“综合性能和远距离/跨域稳定性最优”。
- `xBD` 行使用当前构造的 deterministic paper-test800 子集，应保持该 caveat。
