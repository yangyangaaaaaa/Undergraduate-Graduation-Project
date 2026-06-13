# 密集训练日志信号图说明

- 图像文件：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_trends\figure_dense_training_log_signals.png`
- 数据汇总：`F:\bishe\Undergraduate-Graduation-Project\results\tables\defense_reward_trends\dense_training_log_signal_summary.csv`
- 数据来源：`pipeline_20260603_defense_reward_training_curves/training_logs` 中本文方法 3 个 seed 的 `training_reward_components.csv`。
- 这张图只解释训练阶段奖励如何改变行为，不作为测试阶段排名证据；正式排名仍应使用固定 checkpoint 评估。
- 图中曲线由逐 episode 真实日志做滚动均值，浅色点/浅色线来自真实 episode 记录，没有把 10 个 checkpoint 插值成更多点。

## 可用于答辩的解释

1. C8 最终距离在训练中持续下降，末期三 seed 均值约为 `0.967`，说明奖励信号逐步把策略推向中长距离目标。
2. C8 成功率从训练初期接近 0 逐步形成，末期三 seed 均值约为 `0.446`；它是训练采样信号，不应和固定测试成功率混用。
3. 前进一步比例末期约为 `0.719`，回退比例约为 `0.281`，可以说明奖励信号在动作层面减少无效回退。
4. 奖励分量图显示外部奖励、门控内在奖励和 PBRS 在同一训练阶段共同作用；PBRS 数值量级较小，所以图中使用单独右轴显示，避免被总奖励曲线吞掉。

## 和密集 checkpoint 实验的关系

这张图解决“训练日志点数不够”的解释性问题；正在补跑的密集 checkpoint 实验解决“固定评估曲线点数不够”的排名问题。两者不要混成同一类证据。
