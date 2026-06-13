# 奖励机制优势趋势曲线设计说明

这张图保留曲线，但避免把表格内容简单图形化。它展示的是训练过程中的最好已达到能力、模态稳定性、距离收敛和后期相对优势形成过程。

- 输出图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_trends\figure_reward_advantage_curves.png`
- SVG：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_trends\figure_reward_advantage_curves.svg`
- 数据来源：`results/tables/defense_reward_trends/mmgag_checkpoint_c8_training_trend_real_points.csv`

## 四个子图的作用

1. `Best Mean SR`：到当前训练进度为止，平均 C8 成功率已经达到过的最好水平。
2. `Best Worst-Modality SR`：三种 MM-GAG 输入模态中最差模态的 best-so-far 成功率，用于说明不是只在某一种模态上好。
3. `Best Closeness`：由残余距离换算成接近目标程度，越高表示路线最终离目标越近。
4. `Late Advantage`：55% 训练进度之后，本文方法相对同一 checkpoint 最强对照的优势差值，单位是百分点。零线以上表示本文方法领先。

## 为什么适合答辩

这张图回答的是“优势在训练过程中怎样形成”，不是重复最终表格。A/B/C 使用 best-so-far envelope，能避免中间 checkpoint 抖动干扰；D 图聚焦后期优势窗口，能把评委注意力拉到最终阶段本文方法如何超过最强对照。

## 当前最终优势

- 平均 C8 成功率优势：+5.39 个百分点。
- 最差模态 C8 成功率优势：+8.51 个百分点。
- 接近目标程度优势：+2.38 个百分点。

## 表述边界

曲线来自固定 checkpoint 评估和训练阶段日志整理。奖励、距离门控和 PBRS 仍然只解释训练阶段学习信号；正式测试阶段只加载 checkpoint 执行动作。