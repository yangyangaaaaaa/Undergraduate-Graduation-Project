# 奖励机制与答辩证据精选图

本目录汇总训练阶段奖励机制相关的核心可视化，主要用于解释“奖励如何影响策略学习”，不表示测试阶段仍调用奖励函数。正式测试阶段只加载训练好的策略 checkpoint，根据当前状态输出动作。

## 内容

| 子目录 | 说明 |
| --- | --- |
| `01_action_attribution/` | 训练日志中的动作奖励归因图，把路线、动作、距离变化、外在奖励、门控内在奖励、PBRS 和总奖励放到同一页。 |
| `02_route_reward_cases/` | PPT 聚焦路线与奖励曲线案例，适合一页讲路线、一页讲奖励分解。 |
| `03_evolution_cases/` | 同一 C8 任务在训练早期和后期的路线演化。 |
| `04_c4_wandering_failures/` | C4 短距离徘徊失败边界，解释局部相似模块导致的末端波动。 |
| `05_reward_mechanism_axis/` | 2026-06-08 历史会话最终确认的奖励机制分阶段直线轴图。 |

配套中文报告包括：

- `results/reports/reward_action_attribution_cases_zh.md`
- `results/reports/reward_guided_ppt_focus_cases_zh.md`
- `results/reports/same_task_evolution_visual_zh.md`
- `results/reports/c4_all_failed_wandering_routes_zh.md`
- `results/reports/reward_stepwise_global_analysis_zh.md`

文件级信息见 `manifest.json`。
