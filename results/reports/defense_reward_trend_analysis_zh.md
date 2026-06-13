# 混合奖励训练趋势图说明

本版图按“训练阶段趋势”组织，不使用最终结果柱状图作为主视觉。主图 A/B 优先使用补跑的 MM-GAG 三模态固定模型检查点评估，展示策略参数随训练进度的变化；正式 MM-GAG 表格用于确认最终方法排名。PCHIP 平滑线或历史最优包络只用于视觉连接，不当作新增实验数据。

## 关键结论
- 本文方法在正式 MM-GAG 评估中的平均 SR 为 60.94%，高于 `external_pbrs` 的 59.43%。
- 在 C=6-8 中长距离正式评估区间，本文方法平均成功率为 79.86%；其中 C=8 真实评估点为 93.19%。
- MM-GAG 三模态固定模型检查点趋势中，本文方法在 C=8 上达到最高真实观测 92.62%（进度 99.6%）；同一趋势表中最高行是 本文方法 92.62%。
- 训练预算趋势显示，本文方法在 480k 附近达到正式 MM-GAG 平均 SR 最优：60.94%。720k 未继续提升，答辩时应强调“最优模型检查点/平台期”，不要用最后一步替代最优点。
- 训练验证 best-so-far 中，本文方法进入 80% 区间的平均训练进度约为 39.1%；最优后回落约 2.2 个百分点。
- 后续 `480k -> 960k` 续训固定评估已完成，但没有支持“本文方法持续领先”的假设。续训历史最佳 C8 mean SR 为 `linear_gate_no_pbrs=0.9596`，本文方法为 `0.9113`；最终 checkpoint 为 `linear_gate_no_pbrs=0.9326`，本文方法为 `0.8326`；后期平均为 `constant_gate_pbrs=0.9208`，本文方法为 `0.8631`。因此续训曲线只作负向诊断/备查，不应放在主结果页。

## 真实值说明
- `mmgag_checkpoint_eval_real_points.csv`：补跑的 MM-GAG 三模态固定模型检查点评估真实观测点。
- `mmgag_checkpoint_c8_training_trend_real_points.csv`：主图 A/B 使用的 MM-GAG C=8 三模态趋势点、均值、模态分项和历史最优包络。
- `mmgag_checkpoint_c8_method_summary.csv`：各 reward-gate 方法 C=8 最优成功率、首次达到 90% 的进度、最低剩余距离。
- `fixed_checkpoint_eval_real_points.csv`：固定模型检查点评估的所有真实观测点。
- `fixed_checkpoint_c8_training_trend_real_points.csv`：MASA 固定样本 C=8 趋势聚合点、均值、最优观测和历史最优包络，作为备查。
- `fixed_checkpoint_c8_method_summary.csv`：各方法 C=8 最优成功率、首次达到 90% 的进度、最低剩余距离。
- `formal_mmgag_distance_real_points.csv`：正式 MM-GAG C=4..8 距离桶真实点，用于答辩补充和表格。
- `formal_mmgag_budget_real_points.csv`：240k/480k/720k 三个真实预算评估点，用于说明最优模型检查点。
- `training_convergence_real_summary_by_seed.csv`：训练日志验证成功率、收敛进度与回落统计。
- `training_route_shared_case_real_records.csv`：路线图中每条轨迹的真实训练采样记录。
- `proposed_reward_process_real_curves.csv` 与 `proposed_reward_distance_real_points.csv`：本文方法训练奖励分量与距离统计真实值。
- 已检测到 MM-GAG 检查点评估：4 个方法，3 个模态，120 条真实 checkpoint 评估记录。
- `dense_mmgag_continuation_c8_method_summary.csv`、`dense_mmgag_continuation_c8_final_checkpoint.csv`、`dense_mmgag_continuation_c8_late_stage_summary.csv`：`480k -> 960k` 续训后的诊断表。它们用于说明末端优势未能延续，不用于替换主趋势图。

## 答辩表述
“这里展示的是训练阶段的奖励机制，而不是测试时额外使用奖励函数。混合奖励通过距离门控调节内在探索奖励，再用 PBRS 提供朝目标推进的形状约束，因此训练过程中更容易形成中长距离连续行动。图中的模型检查点趋势说明策略参数在训练过程中逐步获得远距离到达能力；最终评估时只加载模型检查点并执行策略，奖励机制的作用体现在已经学到的策略参数中。”
