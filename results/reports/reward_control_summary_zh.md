# 门控与验证距离补充实验汇总（自动生成）

- 生成时间：`2026-05-20T13:52:59+00:00`
- 训练 run 数：`3`
- 训练数据固定：`MASA+MM-GAG`；默认配置为 `gate_floor=0.405`, `PBRS=0.10`, `entropy=0.005`, `VAL_DISTS=7,8`。
- 评测协议：`5x5`, `B=10`, `C={4,5,6,7,8}`, greedy，任务种子 `20260519`。
- `mean_transfer` 默认取 SwissViewMonuments aerial/ground 与 xBD-pre/xBD-disaster，用于减少训练域同域测试造成的偏差。
- `gate_floor=0` 和 `gate_floor=1` 是门控端点，不是纯外部/纯内部奖励；纯奖励端点见 `reward_control`。

- transfer 最优补充行：`reward_control=external_only`。
- all-benchmark 最优补充行：`reward_control=external_only`。

## 参数敏感性（按参数族分组）

### reward_control

| Value | Run | Transfer SR | All SR | Transfer SG | All SG |
| --- | --- | --- | --- | --- | --- |
| external_only | reward_external_only_seed321_t480k | 0.5441 | 0.5654 | 1.5402 | 1.4933 |
| intrinsic_plus_external_no_decay_no_pbrs | reward_intrinsic_no_decay_seed321_t480k | 0.5281 | 0.5477 | 1.4061 | 1.3258 |
| intrinsic_only | reward_intrinsic_only_seed321_t480k | 0.1306 | 0.1205 | 3.7342 | 3.8553 |

## 结果解读提醒

- `gate_floor_dense` 用于回答 intrinsic 距离衰减强弱是否必要，尤其看 `0.0` 与 `1.0` 两个端点。
- `val_dists_bias` 用于回答 checkpoint 选择偏远距离是否必要，应结合 `appendix_gate_valdist_per_distance.csv` 看 d4-d8 形态。
- `reward_control` 中 `external_only` 和 `intrinsic_only` 才能对应纯外部/纯内部奖励；不要把 `gate_floor=0/1` 写成纯奖励端点。

- `xBD` 行使用当前构造的 deterministic paper-test800 子集，应保持该 caveat。
