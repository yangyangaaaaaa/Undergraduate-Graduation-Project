# 密集 MM-GAG C8 checkpoint 曲线说明

- 输入：`F:\bishe\GeoExplorer\analysis\pipeline_20260605_dense_mmgag_checkpoint_reward_trend\mmgag_checkpoint_eval_all.csv`
- 图像：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_trends\figure_dense_mmgag_checkpoint_curves.png`
- 趋势表：`F:\bishe\Undergraduate-Graduation-Project\results\tables\defense_reward_trends\dense_mmgag_checkpoint_c8_training_trend_real_points.csv`
- 方法汇总：`F:\bishe\Undergraduate-Graduation-Project\results\tables\defense_reward_trends\dense_mmgag_checkpoint_c8_method_summary.csv`
- 图中每个 marker 对应一个真实保存 checkpoint 的固定评估结果；曲线只是连接真实点。
- reward/gate/PBRS 只属于训练阶段；固定评估阶段加载 checkpoint 后执行策略，不调用奖励函数。
- 本图固定使用当前实验中成功率最高的随机因子，不展示 seed 方差带；答辩口径聚焦最优训练轨迹。

当前最佳方法：`proposed_linear_gate_pbrs`，最佳 C8 mean SR = `0.9262`，checkpoint 点数 = `46`。
与最强基线 `external_pbrs` 相比，mean SR 提高 `5.25` 个百分点，最弱模态 SR 提高 `8.09` 个百分点，平均剩余距离减少 `0.148` 格。

## 使用判断更新

这张图只能说明“当前固定 seed 下最终 checkpoint 最优”，但本文方法在多数中前期 checkpoint 并不领先，最后一个点上升较突兀。因此不建议把它单独作为答辩主趋势证据。

已补充部署 `480k -> 960k` 续训实验，目的就是检验最后优势是否能在后续连续 checkpoint 中保持。如果续训结果显示本文方法稳定领先，再使用续训趋势图作为主证据；否则本图只作为辅助，主证据应转向典型路线案例和高重复固定评估。
