# 同一任务训练演化图说明

这张图不展示表格数值，而是展示同一任务在训练过程中的路线行为变化。

- 输出图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\evolution_cases\same_task_evolution_c8_seed123_img20_s0_g24.png`
- SVG：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\evolution_cases\same_task_evolution_c8_seed123_img20_s0_g24.svg`
- 清单：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\evolution_cases\same_task_evolution_manifest.json`

## 图面含义

- 列表示奖励设置：`Ext`、`Ext+Int`、`Gate`、`Ours`。
- 上排是同一任务的训练早期路线，下排是训练后期路线。
- 绿色圆点为起点，黄色星标为目标；绿色终点表示到达，红色叉表示未到达。

## 为什么这类图有价值

表格只能说最后是否成功、最终距离是多少；这张图展示的是策略如何在同一空间任务上改变行动路径。为了减少空白，主图只保留早期和后期两个阶段。这个案例中，三个对照在后期仍停在目标附近但未到达，而本文方法在后期形成连续接近目标的路线并成功到达。

## 建议使用

这页适合作为动作归因图之后的补充页：先用动作归因说明奖励如何给动作反馈，再用这张同一任务演化图说明反馈最终怎样改变路线行为。

## 表述边界

该图来自训练日志中的真实采样片段，解释训练阶段学习过程；正式测试仍以固定 checkpoint 评估和论文结果表格为准。