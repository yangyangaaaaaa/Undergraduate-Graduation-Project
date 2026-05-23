# 补充实验分析与后续实验方案

生成时间：2026-05-23T23:55:41+08:00

本文件汇总基于现有结果可直接完成的后处理分析，并列出下一批建议补跑的 evaluation-only 实验。这里没有重新训练模型，也没有改变服务器或系统配置。

## 1. 统计置信度补充

主表和长距离表已经包含任务级成功次数，因此可以补充成功率置信区间和方法差值的近似 95% CI。差值 CI 使用两个独立比例的正态近似，适合作为论文中的稳健性说明；如果后面要更严格，可以在保留逐任务结果后做 paired bootstrap。

| Benchmark | Ours SR | GOMAA SR | Diff | CI Low | CI High |
| --- | ---: | ---: | --- | --- | --- |
| mmgag_aerial | 0.6170 | 0.5336 | 0.0834 | 0.0436 | 0.1232 |
| mmgag_ground | 0.6391 | 0.5523 | 0.0868 | 0.0473 | 0.1263 |
| mmgag_text | 0.6247 | 0.5472 | 0.0774 | 0.0377 | 0.1172 |

长距离扩展实验中，本文方法相对 GOMAA-Geo 的差值如下。8x8 和 10x10 的差值为正，且 10x10 的优势更明显；25x25 仍建议定位为探索性压力测试。

| Grid | Budget | Ours SR | GOMAA SR | Diff | CI Low | CI High |
| --- | --- | ---: | ---: | --- | --- | --- |
| 10x10 | 32 | 0.7480 | 0.6290 | 0.1190 | 0.0787 | 0.1593 |
| 25x25 | 60 | 0.2040 | 0.1800 | 0.0240 | -0.0248 | 0.0728 |
| 8x8 | 24 | 0.7460 | 0.6790 | 0.0670 | 0.0274 | 0.1066 |

## 2. 轨迹行为补充

轨迹行为分析不只看是否成功，还看搜索过程是否更接近目标方向。建议在论文中把这些指标作为定性可视化的量化支撑：成功率说明结果，单调接近比例和重复访问率说明过程。

| Method | SR | Progress | Monotonic | Revisit | Detour |
| --- | ---: | --- | --- | --- | --- |
| GOMAA-Geo | 0.4245 | 0.5905 | 0.7380 | 0.1409 | 3.3633 |
| GeoExplorer-anchor0624 | 0.4449 | 0.5921 | 0.7458 | 0.1259 | 3.2354 |
| GeoExplorer-pristine | 0.4122 | 0.5973 | 0.7384 | 0.1355 | 3.2952 |

本文方法在 C=6 和 C=8 的成功率、目标距离缩短和单调接近趋势更有解释价值；C=4 不一定占优，应该在正文中诚实说明这是中远距离优化带来的取舍。

| C | N | SR | Progress | Monotonic | Revisit |
| --- | --- | ---: | --- | --- | --- |
| 4 | 375 | 0.2507 | 0.3533 | 0.6346 | 0.1730 |
| 6 | 300 | 0.6067 | 0.8189 | 0.8420 | 0.0832 |
| 8 | 60 | 0.8500 | 0.9500 | 0.9600 | 0.0455 |

## 3. C=4 弱项分析

C=4 是一个必须主动解释的点。短距离下最优路径很短，任何探索性绕行都会快速拉低成功率或 SG；而本文方法的奖励设计更偏向在中远距离维持探索方向和目标收敛。

| Method | C4 SR | Fail Rate | Fail FinalDist | Fail Revisit | Fail Backtrack |
| --- | ---: | --- | --- | --- | --- |
| GOMAA-Geo | 0.2507 | 0.7493 | 3.4235 | 0.2245 | 0.2511 |
| GeoExplorer-anchor0624 | 0.2507 | 0.7493 | 3.4520 | 0.2281 | 0.2361 |
| GeoExplorer-pristine | 0.3333 | 0.6667 | 3.3680 | 0.2167 | 0.2289 |

## 4. 奖励过程补充

奖励过程记录显示，混合奖励只用于训练阶段的行为塑形分析；推理阶段仍然是策略网络根据状态特征选动作。这里的曲线和表格用于解释为什么门控内在奖励和 PBRS 能改善训练出的策略，而不是说测试时还在计算奖励后选动作。

| C | Success | N | Ext | Gated In | PBRS | Total | Mean Lambda |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | False | 1 | -8.0000 | -0.1611 | 0.0051 | -8.1560 | 0.9851 |
| 4 | True | 1 | 5.0000 | 0.4141 | 0.0552 | 5.4693 | 0.6281 |
| 6 | False | 1 | -4.0000 | 2.3872 | 0.0196 | -1.5932 | 0.8810 |
| 6 | True | 2 | 6.0000 | 1.6360 | 0.0925 | 7.7285 | 0.6529 |
| 8 | False | 1 | 0.0000 | 1.8638 | 0.0894 | 1.9532 | 0.7397 |
| 8 | True | 1 | 9.0000 | 2.4306 | 0.1245 | 11.5551 | 0.6653 |

## 5. 下一批建议补跑实验

优先级从 P0 到 P2。P0 是最推荐先跑的，因为它们不需要重新训练，只复用现有 checkpoint 做评测，能最快增强论文可信度。

| Priority | Experiment | Purpose | Cost | Use |
| --- | --- | --- | --- | --- |
| P0 | budget_sensitivity | Verify whether the long-distance advantage remains under tighter and looser search budgets. | Evaluation only; no training. | Chapter 4 long-distance robustness supplement. |
| P0 | task_bank_seed_rerun | Check whether main conclusions are stable to evaluation task sampling. | Evaluation only; moderate GPU time. | Reliability note or appendix table. |
| P1 | target_cue_robustness | Probe cross-modal robustness when target cues are degraded. | Requires writing cue-perturbation evaluator; no retraining. | Optional new angle for cross-modal target adaptation. |
| P1 | reward_trace_expansion | Make the reward mechanism explanation less dependent on a few showcase cases. | Inference/logging only; light. | Mechanism visualization supplement. |
| P2 | larger_grid_middle_scale | Bridge formal 10x10 and noisy 25x25 observations. | Evaluation plus embedding generation if grid not cached. | Appendix only unless results are very clean. |

## 6. 文件索引

- `results/tables/statistical_analysis/main_benchmark_ci_table.csv`：主表 SR 置信区间。
- `results/tables/statistical_analysis/main_benchmark_diff_ci_table.csv`：本文方法与基线的 SR 差值 CI。
- `results/tables/statistical_analysis/ultra_long_diff_ci_table.csv`：长距离实验差值 CI。
- `results/tables/trajectory_analysis/trajectory_behavior_summary.csv`：轨迹行为总体指标。
- `results/tables/trajectory_analysis/c4_failure_profile.csv`：C=4 失败画像。
- `results/tables/reward_process/reward_process_summary.csv`：奖励分量过程汇总。
- `results/tables/experiment_plan/supplement_experiment_plan.csv`：后续补跑实验清单。
- `results/figures/supplement/`：对应图件。
