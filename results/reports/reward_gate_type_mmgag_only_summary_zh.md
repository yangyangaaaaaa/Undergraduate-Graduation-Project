# reward_gate_type 按 MM-GAG 三模态口径汇总

本表把 9 个 `reward_gate_type` run 与补评估的 `linear_0.405` 两个格子合并。指标只看 MM-GAG：
`MM-GAG 平均 SR = mean(mmgag_aerial, mmgag_ground, mmgag_text)`。

| Rank | Row | MM-GAG A | MM-GAG G | MM-GAG T | 平均 SR |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `linear_0.405_pb` | 0.6128 | 0.6102 | 0.6051 | 0.6094 |
| 2 | `external_pbrs` | 0.6094 | 0.5949 | 0.5787 | 0.5943 |
| 3 | `constant_0.405_pb` | 0.5949 | 0.5949 | 0.5728 | 0.5875 |
| 4 | `linear_0.405_no_pb` | 0.5830 | 0.5949 | 0.5685 | 0.5821 |
| 5 | `blend_lp_pb` | 0.5779 | 0.5736 | 0.5345 | 0.5620 |
| 6 | `blend_lp_no_pb` | 0.5336 | 0.5404 | 0.5183 | 0.5308 |
| 7 | `power2_pb` | 0.5157 | 0.5234 | 0.4894 | 0.5095 |
| 8 | `constant_0.405_no_pb` | 0.4970 | 0.5021 | 0.4826 | 0.4939 |
| 9 | `power2_no_pb` | 0.4826 | 0.4638 | 0.4519 | 0.4661 |
| 10 | `sine_pb` | 0.4323 | 0.4315 | 0.4034 | 0.4224 |
| 11 | `sine_no_pb` | 0.3770 | 0.3991 | 0.3830 | 0.3864 |

## 与论文大表不完全一致的原因

论文大表中 `本文方法` 的 MM-GAG aerial 结果 `0.6170` 来自 20260516 paper-style evaluation，任务种子为 `20260516`。

本表中的 `linear_0.405_pb` 是为了补齐 reward-gate 表，在 20260519 appendix 协议下重新评估同一个 `g1_p1_e1_v1` checkpoint，任务种子为 `20260519`，因此 MM-GAG aerial 为 `0.6128`。

两者 checkpoint 和算法身份一致，但固定生成的 task bank 不同；整体均值差异很小，分距离桶如 D=6 会更敏感。因此论文里不要把 20260516 的分距离数和 20260519 的补充消融数放在同一张表里横比。

## 测试阶段说明

本表使用独立的 `paper_geo_evaluator.py` 进行 evaluation-only 测试。测试阶段只加载 policy checkpoint 和 LLM checkpoint，并使用 `select_greedy_action` 选择动作；训练期的 `gate_weight`、`pbrs_bonus`、`reward_ex/reward_in`、entropy loss 和 validation-distance checkpoint-selection 逻辑不参与测试。

因此，门控函数和 PBRS 的影响在测试时只通过已训练好的 checkpoint 权重体现。本表不应解释为“测试时打开或关闭某个奖励模块”，而应解释为“不同训练奖励设计得到的策略 checkpoint 在同一测试协议下的表现”。
