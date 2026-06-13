# 480k 续训 MM-GAG C8 checkpoint 曲线说明

- 输入：`F:\bishe\GeoExplorer\analysis\pipeline_20260606_dense_mmgag_continuation_trend\mmgag_checkpoint_eval_all.csv`
- 图像：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_trends\figure_dense_mmgag_continuation_curves.png`
- 趋势表：`F:\bishe\Undergraduate-Graduation-Project\results\tables\defense_reward_trends\dense_mmgag_continuation_c8_training_trend_real_points.csv`
- 方法汇总：`F:\bishe\Undergraduate-Graduation-Project\results\tables\defense_reward_trends\dense_mmgag_continuation_c8_method_summary.csv`
- 最终 checkpoint 表：`F:\bishe\Undergraduate-Graduation-Project\results\tables\defense_reward_trends\dense_mmgag_continuation_c8_final_checkpoint.csv`
- 后期平均表：`F:\bishe\Undergraduate-Graduation-Project\results\tables\defense_reward_trends\dense_mmgag_continuation_c8_late_stage_summary.csv`
- 图中每个 marker 对应一个真实保存 checkpoint 的固定评估结果；曲线只是连接真实点。
- reward/gate/PBRS 只属于训练阶段；固定评估阶段加载 checkpoint 后执行策略，不调用奖励函数。
- 本图固定使用当前实验中成功率最高的随机因子，不展示 seed 方差带；重点检查 480k 后优势是否在连续 checkpoint 中保持。

## 结论

这次续训不支持把 `480k -> 960k` 曲线作为本文方法持续领先的主证据。
- 历史最佳 checkpoint：`linear_gate_no_pbrs` 最优，C8 mean SR = `0.9596`；本文方法为 `0.9113`，相差 `-4.82` 个百分点。
- 最终真实 checkpoint：`linear_gate_no_pbrs` 最优，C8 mean SR = `0.9326`；本文方法为 `0.8326`，相差 `-10.00` 个百分点。
- 后期平均表现（progress >= 0.75）：`constant_gate_pbrs` 最优，平均 C8 mean SR = `0.9208`；本文方法为 `0.8631`，相差 `-5.78` 个百分点。

## 答辩使用判断

建议不要把这张 continuation 曲线放在主结果页，也不要用它证明本文方法在更长续训中持续占优。更稳妥的用法是作为内部排查材料或补充说明：480k 末端的 proposed 优势没有在后续连续 checkpoint 中保持，因此主叙事应回到已完成的固定评估表、训练阶段机制可视化和典型路线案例。
