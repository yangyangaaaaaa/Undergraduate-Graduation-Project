# C=1,2,3 短距离完整评测汇总

- 评测口径：只评测，不重训；`5x5` 网格，`B=10`，距离桶 `C=1,2,3`，任务种子 `20260516`。
- 对比方法：Random、GOMAA-Geo、DiT-AGL、GeoExplorer-anchor0624，并额外保留验收常用的 GeoExplorer-pristine。
- 消融实验：anchor0624 的 16 个 G/P/E/V 因子组合，全用现有 seed321 480k checkpoint。
- xBD-disaster 口径：灾后图像中搜索，目标仍使用灾前图像嵌入。

## 对比方法均值

| 方法 | 覆盖 benchmark 数 | 支持范围内平均 SR | 支持范围内平均 SG |
| --- | ---: | ---: | ---: |
| Random policy | 4 | 0.3653 | 2.0558 |
| GeoExplorer-pristine | 9 | 0.2870 | 2.8837 |
| GOMAA-Geo | 9 | 0.2737 | 3.0870 |
| GeoExplorer-anchor0624 | 9 | 0.2535 | 3.3600 |
| DiT-AGL | 4 | 0.2465 | 1.9714 |

## 消融 Top 分支

| 排名 | 分支 | G | P | E | V | transfer mean | all mean |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | g0_p1_e1_v1 | 0 | 1 | 1 | 1 | 0.3294 | 0.3345 |
| 2 | g1_p1_e1_v0 | 1 | 1 | 1 | 0 | 0.3263 | 0.3186 |
| 3 | g0_p0_e1_v0 | 0 | 0 | 1 | 0 | 0.3227 | 0.3189 |
| 4 | g0_p1_e0_v0 | 0 | 1 | 0 | 0 | 0.3070 | 0.3047 |
| 5 | g0_p1_e0_v1 | 0 | 1 | 0 | 1 | 0.3019 | 0.3061 |
| 6 | g1_p0_e0_v0 | 1 | 0 | 0 | 0 | 0.3009 | 0.2874 |
| 7 | g1_p0_e1_v0 | 1 | 0 | 1 | 0 | 0.2864 | 0.2776 |
| 8 | g1_p1_e0_v1 | 1 | 1 | 0 | 1 | 0.2801 | 0.2756 |

## 关键对照

- 完整方法 `g1_p1_e1_v1` transfer mean：`0.2454`。
- 同数据无新增机制控制 `g0_p0_e0_v0` transfer mean：`0.2466`。
- 完整方法 - 控制组：`-0.0012`。

## 主效应

- `G_gate`：on `0.2755`，off `0.2851`，差值 `-0.0095`。
- `P_pbrs`：on `0.2892`，off `0.2715`，差值 `0.0177`。
- `E_low_entropy`：on `0.2845`，off `0.2761`，差值 `0.0084`。
- `V_val78`：on `0.2715`，off `0.2891`，差值 `-0.0177`。

## 文件

- 总表：`/root/geoexplorer/analysis/short_distance_c123_eval_20260609_161734/tables/all_run_metrics.csv`
- 对比方法逐 benchmark 表：`/root/geoexplorer/analysis/short_distance_c123_eval_20260609_161734/tables/comparison_method_metrics.csv`
- 对比方法均值表：`/root/geoexplorer/analysis/short_distance_c123_eval_20260609_161734/tables/comparison_method_summary.csv`
- 消融分支汇总表：`/root/geoexplorer/analysis/short_distance_c123_eval_20260609_161734/tables/ablation_branch_summary.csv`
- JSON 汇总：`/root/geoexplorer/analysis/short_distance_c123_eval_20260609_161734/short_distance_c123_aggregate.json`
