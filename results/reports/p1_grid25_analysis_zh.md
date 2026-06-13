# P1 25x25 超大网格补充实验整理与分析

- 整理时间：2026-05-24T16:47:12+08:00
- 原始来源：`F:\bishe\GeoExplorer\analysis\pipeline_20260524_p1_grid25_eval`
- 实验性质：仅评测 existing checkpoints，不重新训练。
- 设置：MASA aerial 25x25 网格，距离桶为 `12,16,20,24,28,32,36,40,44,48`，每个距离桶 5 次重复。
- 方法：本文方法、GOMAA-Geo、GeoExplorer-pristine。

## 预算敏感性结果

| Budget | 本文方法 | GOMAA-Geo | GeoExplorer | 本文-GOMAA |
| ---: | ---: | ---: | ---: | ---: |
| 40 | 0.0500 | 0.0380 | 0.0340 | +0.0120 |
| 50 | 0.1720 | 0.1500 | 0.0720 | +0.0220 |
| 60 | 0.1820 | 0.1580 | 0.0840 | +0.0240 |
| 70 | 0.1840 | 0.1580 | 0.0900 | +0.0260 |

## Seed 稳定性结果

| 方法 | Seeds | Mean SR | Std | Min | Max |
| --- | --- | ---: | ---: | ---: | ---: |
| This work | 20260524,20260525,20260526 | 0.1853 | 0.0172 | 0.1700 | 0.2040 |
| GOMAA-Geo | 20260524,20260525,20260526 | 0.1680 | 0.0111 | 0.1580 | 0.1800 |
| GeoExplorer | 20260524,20260525,20260526 | 0.0820 | 0.0072 | 0.0740 | 0.0880 |

## 分析结论

- 25x25 是明显更困难的压力测试，所有方法的 SR 都显著低于 8x8/10x10；这说明该设置更适合放在补充实验或鲁棒性分析中，而不适合作为主表。
- 在预算敏感性中，本文方法在所有预算下均高于 GOMAA-Geo。最大优势出现在 budget `70`，差值为 `+0.0260`。
- 在与既有 25x25 formal 设置一致的 budget 60 下，本文方法 SR 为 `0.1820`，GOMAA-Geo 为 `0.1580`，差值 `+0.0240`。
- 三个 task-bank seed 的均值显示，本文方法 Mean SR 为 `0.1853`，GOMAA-Geo 为 `0.1680`，平均优势 `+0.0173`；相对 GeoExplorer-pristine 的平均优势为 `+0.1033`。
- 从趋势上看，25x25 下本文方法优势仍存在，但幅度小于 10x10；这提示模型在极大网格中仍受训练分布和搜索预算限制，论文中应表述为“极端压力测试下仍保持相对领先”，不要夸大为“极端长距离完全解决”。

## 已整理文件

- `results/tables/supplement_eval/p1_grid25_aggregate.json`
- `results/tables/supplement_eval/p1_grid25_budget_summary.csv`
- `results/tables/supplement_eval/p1_grid25_budget_table.csv`
- `results/tables/supplement_eval/p1_grid25_long_table.csv`
- `results/tables/supplement_eval/p1_grid25_per_distance.csv`
- `results/tables/supplement_eval/p1_grid25_seed_summary.csv`
- `results/tables/supplement_eval/p1_grid25_seed_table.csv`
- `results/figures/supplement/p1_grid25_budget_sensitivity.png`
- `results/figures/supplement/p1_grid25_budget_advantage.png`
- `results/figures/supplement/p1_grid25_per_distance_seed_mean.png`
- `results/figures/supplement/p1_grid25_seed_stability.png`
