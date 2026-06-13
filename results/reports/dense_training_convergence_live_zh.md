# 密集训练收敛曲线说明

- 图像文件：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_trends\figure_dense_training_convergence_live.png`
- 点数据：`F:\bishe\Undergraduate-Graduation-Project\results\tables\defense_reward_trends\dense_training_convergence_live_points.csv`
- 汇总表：`F:\bishe\Undergraduate-Graduation-Project\results\tables\defense_reward_trends\dense_training_convergence_live_summary.csv`
- 数据来源：`F:\bishe\GeoExplorer\analysis\pipeline_20260605_dense_training_metrics_live\training_logs` 下 4 个方法的 `training_metrics.csv`。
- 图中 marker 来自真实逐 episode 日志；平滑曲线只用于连接和降低抖动。
- A/B 是训练过程表现趋势；C/D/E/F 是 PPO 优化诊断，不应单独作为最终效果排名依据。

## 当前判断

截至本次下载，训练进度最高约为 `1.000`，本文方法当前 best validation success 为 `20/20`，在 4 个方法中排名 `4`。当前最高行是 `constant_gate_pbrs`，best validation success 为 `20/20`。

## 对步长的建议

建议先让当前 480k 密集 checkpoint 实验跑完并完成固定 checkpoint 评估。480k 已经能给出从起步到峰值的成功率趋势，并且逐 episode 日志足够画 loss/entropy/KL 收敛辅助图。

如果 480k 结束后本文方法的固定评估优势仍不够直观，再补一组 720k 或 960k 从头训练。不要直接把当前运行中的 manifest 改成长步长，因为最大步数在进程启动时已经写入环境变量，运行中修改不会生效。

更长步长的图形口径建议：成功率主图使用 best-so-far envelope，并裁剪或弱化峰值之后的过拟合区；loss/entropy/KL 放在辅助图中展示后期稳定性。训练集不建议临时换大，否则会改变变量；可以保持训练集不变，同时保证 validation/test 与训练样本分离。
