# 门控与验证距离补充实验汇总（自动生成）

- 生成时间：`2026-05-22T08:01:18+00:00`
- 训练 run 数：`15`
- 训练数据固定：`MASA+MM-GAG`；默认配置为 `gate_floor=0.405`, `PBRS=0.10`, `entropy=0.005`, `VAL_DISTS=7,8`。
- 评测协议：`5x5`, `B=10`, `C={4,5,6,7,8}`, greedy，任务种子 `20260519`。
- `mean_transfer` 默认取 SwissViewMonuments aerial/ground 与 xBD-pre/xBD-disaster，用于减少训练域同域测试造成的偏差。
- `gate_floor=0` 和 `gate_floor=1` 是门控端点，不是纯外部/纯内部奖励；纯奖励端点见 `reward_control`。

- transfer 最优补充行：`gate_floor_dense=0.400`。
- all-benchmark 最优补充行：`gate_floor_dense=0.800`。

## 参数敏感性（按参数族分组）

### gate_floor_dense

| Value | Run | Transfer SR | All SR | Transfer SG | All SG |
| --- | --- | --- | --- | --- | --- |
| 0.400 | gate_floor_dense_0p4_seed321_t480k | 0.5449 | 0.5550 | 1.4416 | 1.3641 |
| 0.800 | gate_floor_dense_0p8_seed321_t480k | 0.5370 | 0.5608 | 1.4192 | 1.3277 |
| 0.600 | gate_floor_dense_0p6_seed321_t480k | 0.5285 | 0.5497 | 1.4490 | 1.3567 |
| 0.200 | gate_floor_dense_0p2_seed321_t480k | 0.4950 | 0.5130 | 1.6351 | 1.5669 |
| 0.700 | gate_floor_dense_0p7_seed321_t480k | 0.4930 | 0.5084 | 1.5060 | 1.4460 |
| 0.300 | gate_floor_dense_0p3_seed321_t480k | 0.4755 | 0.4686 | 1.8765 | 1.9608 |
| 0.000 | gate_floor_dense_0_seed321_t480k | 0.4755 | 0.4746 | 1.8948 | 1.9141 |
| 0.100 | gate_floor_dense_0p1_seed321_t480k | 0.4729 | 0.4813 | 1.9390 | 1.9150 |
| 0.900 | gate_floor_dense_0p9_seed321_t480k | 0.4678 | 0.4582 | 1.8286 | 1.9135 |
| 1.000 | gate_floor_dense_1_seed321_t480k | 0.3623 | 0.3564 | 2.1532 | 2.1581 |

### val_dists_bias

| Value | Run | Transfer SR | All SR | Transfer SG | All SG |
| --- | --- | --- | --- | --- | --- |
| 8 | val_dists_8_seed321_t480k | 0.4963 | 0.4941 | 1.6939 | 1.7396 |
| 4,5,6 | val_dists_456_seed321_t480k | 0.4959 | 0.4819 | 1.8742 | 1.9600 |
| 5,6,7 | val_dists_567_seed321_t480k | 0.4903 | 0.5137 | 1.6847 | 1.5803 |
| 4,5 | val_dists_45_seed321_t480k | 0.4540 | 0.4311 | 1.9699 | 2.0906 |
| 4 | val_dists_4_seed321_t480k | 0.4468 | 0.4441 | 1.9671 | 2.0184 |

## 结果解读提醒

- `gate_floor_dense` 用于回答 intrinsic 距离衰减强弱是否必要，尤其看 `0.0` 与 `1.0` 两个端点。
- `val_dists_bias` 用于回答 checkpoint 选择偏远距离是否必要，应结合 `appendix_gate_valdist_per_distance.csv` 看 d4-d8 形态。
- `reward_control` 中 `external_only` 和 `intrinsic_only` 才能对应纯外部/纯内部奖励；不要把 `gate_floor=0/1` 写成纯奖励端点。

- `xBD` 行使用当前构造的 deterministic paper-test800 子集，应保持该 caveat。
