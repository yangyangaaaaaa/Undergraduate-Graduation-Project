# P0 补充评测整理说明

- 整理时间：2026-05-24T14:48:01+08:00
- 原始来源：`F:\bishe\GeoExplorer\analysis\pipeline_20260524_p0_supplement_eval`
- 实验性质：仅评测 existing checkpoints，不重新训练。
- 主要用途：补强预算敏感性、任务库 seed 稳定性和超长距离结论的可信度。

## 关键观察

- 预算敏感性中，当前最大优势出现在 `10x10`、budget `40`：本文方法相对 GOMAA-Geo 的 SR 差值为 `+0.1250`。
- Ultra-long seed 复评中，本文方法在可达高距离端的平均 SR 约为 `0.7265`。
- MM-GAG seed 复评用于检查任务库随机性的影响，本文方法三种目标模态平均 SR 约为 `0.6156`。

## 已整理文件

- `results/tables/supplement_eval/budget_sensitivity_summary.csv`
- `results/tables/supplement_eval/budget_sensitivity_table.csv`
- `results/tables/supplement_eval/p0_supplement_aggregate.json`
- `results/tables/supplement_eval/p0_supplement_long_table.csv`
- `results/tables/supplement_eval/p0_supplement_per_distance.csv`
- `results/reports/p0_supplement_eval_summary_zh.md`
- `results/tables/supplement_eval/task_seed_mmgag_table.csv`
- `results/tables/supplement_eval/task_seed_summary.csv`
- `results/tables/supplement_eval/task_seed_ultra_table.csv`
- `results/figures/supplement/p0_budget_sensitivity.png`
- `results/figures/supplement/p0_budget_advantage.png`
- `results/figures/supplement/p0_task_seed_stability.png`
