# 非表格式训练动态可视化说明

本页只用于展示表格难以表达的信息，不替代正式结果表格。

- 输出图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_trends\figure_reward_dynamic_phase.png`
- SVG：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_trends\figure_reward_dynamic_phase.svg`
- 数据来源：`results/tables/defense_reward_trends/mmgag_checkpoint_c8_training_trend_real_points.csv`

## 为什么这不是表格重复

1. A 图是训练相图：横轴是 C8 成功率，纵轴是残余距离，箭头表示 checkpoint 的时间顺序。它展示方法在训练中如何移动，而不是只给最后一行数值。
2. B 图展示三种 MM-GAG 模态的训练期展开范围。浅色带越窄，说明同一方法在 aerial/ground/text 三种输入上越稳定；这类波动结构用表格不直观。
3. C 图只看后半程收敛：它用曲线形状展示后期是否继续向低残余距离区域移动，避免只盯最终 checkpoint。

## 建议讲法

这页可以放在主趋势图之后。主趋势图说明本文方法最后达到更高 C8 成功率；这页进一步说明优势不是一个孤立终点，而是训练轨迹逐步进入“高成功率、低残余距离”的区域。图中蓝色轨迹代表本文方法，最终进入左图右上方高成功率且低残余距离的浅蓝目标区域；右侧两图说明这种优势在三种 MM-GAG 模态上也保持一致，并在训练后期继续收敛。

## 表述边界

奖励、距离门控和 PBRS 仍然只解释训练阶段的学习信号；正式测试阶段只加载 checkpoint 执行动作。

## 当前关键事实

- 本文方法末端 C8 平均成功率为 92.62%，平均残余距离为 0.227。
- 这些数值可放进表格或口头说明，不建议直接堆到图面上。