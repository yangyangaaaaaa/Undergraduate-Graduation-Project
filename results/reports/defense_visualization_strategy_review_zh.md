# 结题汇报可视化展示策略评估

生成时间：2026-06-05

## 已检查范围

- 论文最新版：`D:\桌面\literature\张洋_本科毕业设计论文 - 副本 (2).docx`
- 仓库首页与画廊：`README.md`、`docs/visualization_gallery_zh.md`
- 结果目录：`results/figures/`、`results/tables/`、`results/reports/`
- 训练阶段日志：`F:\bishe\GeoExplorer\analysis\pipeline_20260603_defense_reward_training_curves\training_logs`
- 已有重点图：训练趋势图、PPT 聚焦路线图、奖励组成曲线、24 个完整案例图、正式实验表格和补充实验图。

## 当前判断

现有材料已经能证明“结果更好”，但单看趋势曲线和路线图，评委仍可能追问：奖励机制到底怎样影响训练中的动作选择。新增的动作归因图补上了这个缺口：它把真实俯视图路线、距离曲线、每步动作、距离变化、外部奖励、门控内在奖励、PBRS 和总奖励放在同一页。

不建议为了这一轮可视化再重写训练代码或重跑大规模训练。当前训练日志已经包含足够的真实每步奖励分解，且能选出 C7/C8 中对照失败、本文方法到达的典型案例。继续重跑的收益主要是统计量更稳，但对答辩展示的边际收益不如把现有证据组织清楚。

## 推荐 PPT 主线

1. 方法框架图：使用用户原图，先说明混合奖励是训练阶段机制。
2. 训练趋势图：`results\figures\defense_reward_trends\figure_reward_trend_main.png`，说明本文方法在 C8 中长距离检查点趋势中达到最高成功率。
3. 动作归因图：优先使用 `action_attribution_01_case_04_C8_seed123_ep120_img5_s4_g20.png` 或 `action_attribution_02_case_08_C8_seed123_ep55_img6_s4_g20.png`，说明每一步奖励如何把动作推向目标。
4. 真实俯视图路线：使用 `ppt_focus_cases` 中同一案例的 `focus_case_*.png`，只看路线，不放过多文字。
5. 奖励组成曲线：使用同一案例的 `reward_case_*.png`，作为机制补充页；若 PPT 时间紧，可放到备份页。
6. 正式结果表格：用论文第四章表格收束，强调正式测试只加载策略 checkpoint，不调用奖励函数。

## 新增输出

- 动作归因图目录：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\action_attribution`
- 总览图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\action_attribution\action_attribution_contact_sheet.png`
- 说明文件：`F:\bishe\Undergraduate-Graduation-Project\results\reports\reward_action_attribution_cases_zh.md`
- 生成脚本：`F:\bishe\Undergraduate-Graduation-Project\code\tools\build_reward_action_attribution_figures.py`

## 新增非表格式补充图

后续可视化不应再把表格内容画成柱状图或表格式热图。优先使用表格无法表达的结果：训练轨迹、同一任务路线演化、模态波动范围、动作归因和失败/恢复过程。

- 训练动态相图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_trends\figure_reward_dynamic_phase.png`
  - 用途：展示方法在训练 checkpoint 上如何移动到“高 C8 成功率、低残余距离”的区域。
  - 价值：表格只能给最终值；相图能展示训练路径、模态波动和后期收敛形态。
  - 说明文件：`F:\bishe\Undergraduate-Graduation-Project\results\reports\defense_non_table_visuals_zh.md`
- 同一任务训练演化图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\evolution_cases\same_task_evolution_c8_seed123_img20_s0_g24.png`
  - 用途：同一 C8 任务、同一起点和目标，用紧凑 2×4 版式比较 Ext、Ext+Int、Gate、Ours 在训练早期和后期的实际路线。
  - 价值：直接展示“行为如何变好”，不是复述成功率表格；该例中三个对照后期仍未到达，本文方法后期到达。
  - 说明文件：`F:\bishe\Undergraduate-Graduation-Project\results\reports\same_task_evolution_visual_zh.md`

## 优先使用案例

- `case_04`：C8，训练进度 26.9%，五个对照未到达，本文方法到达。适合讲远距离训练指导优势。
- `case_08`：C8，训练进度 12.5%，五个对照未到达，本文方法到达。适合强调训练早期已经形成方向差异。
- `case_12`：C7，训练进度 16.9%，五个对照未到达，本文方法有一次回退但恢复并到达。适合讲“走出回退/循环”。

## 备用图建议

- 若评委更关注整体统计：使用 `figure_reward_trend_main.png` 和正式表格，不要用柱状图替代趋势图。
- 若评委要求“不要把表格画成图”：使用 `figure_reward_dynamic_phase.png` 或 `same_task_evolution_c8_seed123_img20_s0_g24.png`，它们展示训练轨迹和路线演化。
- 若评委更关注机制公式：使用 `reward_case_*.png`，说明总奖励由外部项、门控内在项和 PBRS 相加。
- 若评委更关注直观行为：使用 `focus_case_contact_sheet.png` 或 1-2 张单案例俯视图路线。
- 若评委追问泛化：引用 MM-GAG 三模态表、MASA/SwissView/xBD 表，而不是训练阶段奖励图。

## 表述边界

必须保持以下表述：

> 奖励、距离门控和 PBRS 只在训练阶段提供学习信号；测试阶段只加载训练好的策略 checkpoint 执行动作。

避免说法：

- “测试时 PBRS 引导路线”
- “推理时根据奖励函数选择动作”
- “动作归因图代表正式测试过程”

更准确的说法：

- “动作归因图展示训练日志中的真实片段，用于解释奖励如何改变策略学习信号。”
- “正式测试结果仍以第四章表格和固定 checkpoint 评估为准。”
