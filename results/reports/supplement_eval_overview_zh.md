# 补充实验总报告：预算敏感性、任务库稳定性与超大网格压力测试

- 生成时间：2026-05-24T16:48:28+08:00
- 实验性质：全部为 evaluation-only，均复用已训练 checkpoint，不包含新增训练。
- 实验范围：P0 包含 8x8/10x10 预算敏感性与 MM-GAG/ultra-long task-bank seed 复评；P1 包含 25x25 超大网格预算敏感性与 seed 复评。
- 对比方法：本文方法、GOMAA-Geo、GeoExplorer-pristine；MM-GAG seed 复评只比较本文方法与 GOMAA-Geo。

## 1. 预算敏感性

| Grid | Budget range | 本文方法 SR 范围 | GOMAA SR 范围 | 本文-GOMAA 差值范围 | 主要结论 |
| --- | --- | ---: | ---: | ---: | --- |
| 8x8 | 16-32 | 0.6970-0.7620 | 0.6220-0.6930 | 0.0670-0.0750 | 优势稳定，预算增加后双方都提升。 |
| 10x10 | 20-40 | 0.6840-0.7660 | 0.5890-0.6410 | 0.0950-0.1250 | 优势随预算扩大更明显，支撑中长距离论点。 |
| 25x25 | 40-70 | 0.0500-0.1840 | 0.0380-0.1580 | 0.0120-0.0260 | 极端网格下绝对 SR 较低，但相对领先仍存在。 |

## 2. Task-bank Seed 稳定性

| Setting | 方法 | Mean SR | Std | Min-Max |
| --- | --- | ---: | ---: | ---: |
| MM-GAG aerial | GeoExplorer-anchor0624 | 0.6139 | 0.0162 | 0.5983-0.6306 |
| MM-GAG ground | GeoExplorer-anchor0624 | 0.6196 | 0.0266 | 0.5983-0.6494 |
| MM-GAG text | GeoExplorer-anchor0624 | 0.6133 | 0.0128 | 0.6051-0.6281 |
| MM-GAG aerial | GOMAA-Geo | 0.5217 | 0.0145 | 0.5055-0.5336 |
| MM-GAG ground | GOMAA-Geo | 0.5379 | 0.0098 | 0.5302-0.5489 |
| MM-GAG text | GOMAA-Geo | 0.5339 | 0.0118 | 0.5209-0.5438 |
| Ultra-long 10x10 | GeoExplorer-anchor0624 | 0.7330 | 0.0130 | 0.7250-0.7480 |
| Ultra-long 8x8 | GeoExplorer-anchor0624 | 0.7200 | 0.0225 | 0.7060-0.7460 |
| Ultra-long 10x10 | GOMAA-Geo | 0.6127 | 0.0144 | 0.6020-0.6290 |
| Ultra-long 8x8 | GOMAA-Geo | 0.6700 | 0.0139 | 0.6540-0.6790 |
| 25x25 | GeoExplorer-anchor0624 | 0.1853 | 0.0172 | 0.1700-0.2040 |
| 25x25 | GOMAA-Geo | 0.1680 | 0.0111 | 0.1580-0.1800 |

## 3. 综合分析

- MM-GAG 三种目标模态的 seed 复评中，本文方法平均 SR 为 `0.6156`，GOMAA-Geo 为 `0.5312`，平均优势 `0.0844`。这说明主表结论不是单一任务库随机 seed 造成的偶然现象。
- 8x8/10x10 ultra-long seed 复评中，本文方法平均 SR 为 `0.7265`，GOMAA-Geo 为 `0.6413`，平均优势 `0.0852`。其中 10x10 预算敏感性优势最高达到 `0.1250`，更符合“中长距离优势更突出”的论文叙事。
- 25x25 压力测试中，本文方法 seed 平均 SR 为 `0.1853`，GOMAA-Geo 为 `0.1680`，平均优势 `0.0173`。但绝对 SR 明显偏低，说明该设置超过了当前训练分布和预算能力，应作为探索性鲁棒性补充，而不是主结果表。
- 论文写法建议：主文强调 8x8/10x10 的稳定提升；25x25 用作补充压力测试，措辞应保守，例如“在极端大网格下仍保持相对领先，但任务难度显著上升”。

## 4. 对应文件

- `results/tables/supplement_eval/`：P0/P1 所有长表、分距离表、预算表和 seed 汇总表。
- `results/figures/supplement/p0_*.png`：8x8/10x10 预算与 seed 稳定性图。
- `results/figures/supplement/p1_grid25_*.png`：25x25 预算、优势、分距离和 seed 稳定性图。
- `results/reports/p0_supplement_eval_inventory_zh.md`、`results/reports/p1_grid25_analysis_zh.md`：单项实验整理说明。
