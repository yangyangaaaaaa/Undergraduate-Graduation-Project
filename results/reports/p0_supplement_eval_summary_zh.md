# P0 补充评测结果

- 生成时间：2026-05-24T05:14:21+00:00
- 类型：evaluation-only，不重新训练模型。
- 目的：补充预算敏感性和 task-bank seed 稳定性证据。
- 距离桶说明：8x8 与 10x10 使用当前网格内可达的高距离端；更大距离范围另由 25x25 探索性压力测试承担。

## 预算敏感性

| Grid | Budget | Ours SR | GOMAA SR | GeoExplorer SR | Ours-GOMAA |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10x10 | 20 | 0.6840 | 0.5890 | 0.2660 | +0.0950 |
| 10x10 | 24 | 0.7210 | 0.6150 | 0.2800 | +0.1060 |
| 10x10 | 28 | 0.7400 | 0.6210 | 0.2900 | +0.1190 |
| 10x10 | 32 | 0.7480 | 0.6290 | 0.3000 | +0.1190 |
| 10x10 | 36 | 0.7590 | 0.6370 | 0.3000 | +0.1220 |
| 10x10 | 40 | 0.7660 | 0.6410 | 0.3030 | +0.1250 |
| 8x8 | 16 | 0.6970 | 0.6220 | 0.3460 | +0.0750 |
| 8x8 | 20 | 0.7300 | 0.6570 | 0.3600 | +0.0730 |
| 8x8 | 24 | 0.7460 | 0.6790 | 0.3680 | +0.0670 |
| 8x8 | 28 | 0.7530 | 0.6850 | 0.3770 | +0.0680 |
| 8x8 | 32 | 0.7620 | 0.6930 | 0.3800 | +0.0690 |

## Task-bank Seed 稳定性

| Family | Benchmark | Method | Seeds | Mean SR | Std | Min | Max |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| mmgag | mmgag_aerial | GeoExplorer-anchor0624 | 20260521,20260522,20260523 | 0.6139 | 0.0162 | 0.5983 | 0.6306 |
| mmgag | mmgag_aerial | GOMAA-Geo | 20260521,20260522,20260523 | 0.5217 | 0.0145 | 0.5055 | 0.5336 |
| mmgag | mmgag_ground | GeoExplorer-anchor0624 | 20260521,20260522,20260523 | 0.6196 | 0.0266 | 0.5983 | 0.6494 |
| mmgag | mmgag_ground | GOMAA-Geo | 20260521,20260522,20260523 | 0.5379 | 0.0098 | 0.5302 | 0.5489 |
| mmgag | mmgag_text | GeoExplorer-anchor0624 | 20260521,20260522,20260523 | 0.6133 | 0.0128 | 0.6051 | 0.6281 |
| mmgag | mmgag_text | GOMAA-Geo | 20260521,20260522,20260523 | 0.5339 | 0.0118 | 0.5209 | 0.5438 |
| ultra_long | masa_aerial_10x10 | GeoExplorer-anchor0624 | 20260521,20260522,20260523 | 0.7330 | 0.0130 | 0.7250 | 0.7480 |
| ultra_long | masa_aerial_10x10 | GOMAA-Geo | 20260521,20260522,20260523 | 0.6127 | 0.0144 | 0.6020 | 0.6290 |
| ultra_long | masa_aerial_10x10 | GeoExplorer-pristine | 20260521,20260522,20260523 | 0.2790 | 0.0184 | 0.2660 | 0.3000 |
| ultra_long | masa_aerial_8x8 | GeoExplorer-anchor0624 | 20260521,20260522,20260523 | 0.7200 | 0.0225 | 0.7060 | 0.7460 |
| ultra_long | masa_aerial_8x8 | GOMAA-Geo | 20260521,20260522,20260523 | 0.6700 | 0.0139 | 0.6540 | 0.6790 |
| ultra_long | masa_aerial_8x8 | GeoExplorer-pristine | 20260521,20260522,20260523 | 0.3567 | 0.0147 | 0.3400 | 0.3680 |

## 输出文件

- `p0_supplement_long_table.csv`：所有 job 汇总。
- `budget_sensitivity_table.csv` 与 `budget_sensitivity_summary.csv`：预算敏感性结果。
- `task_seed_mmgag_table.csv`、`task_seed_ultra_table.csv` 与 `task_seed_summary.csv`：任务库 seed 复评结果。
- `p0_supplement_per_distance.csv`：分距离结果。
