# 结题汇报材料交接清单

生成时间：2026-06-06 13:45（Asia/Shanghai）；续训结论更新：2026-06-06 19:50（Asia/Shanghai）

本文档给论文 agent 使用，目标是把当前 `F:\bishe` 文件夹中对结题汇报有价值的材料整理成可直接写 PPT、讲稿和论文补充说明的证据包。本文档不记录任何远程密码、密钥或私有凭据。

## 结论先行

当前最适合答辩展示的主线不是单纯把最终表格画成图，而是用“训练阶段奖励如何指导动作”作为可视化核心，再用正式评估表格兜底结果。推荐主叙事是：混合奖励机制只在训练阶段提供学习信号，通过外部奖励、门控内在奖励和 PBRS 共同改变每一步动作反馈，使策略在中长距离任务中更容易形成连续接近目标的路线；正式测试阶段只加载训练好的 checkpoint 执行动作，不调用奖励函数。

最推荐作为主视觉证据的是 `defense_reward_training_stage/ppt_focus_cases` 与 `action_attribution` 两组图。它们基于真实训练日志和真实俯视图，能让评委直接看到同一起点和目标下，本文方法到达目标，而对照方法停在目标外、回退或重复访问。趋势曲线可以保留，但需要分级使用：旧版 `figure_reward_trend_main.png` 可说明训练检查点趋势；新版 480k 密集 checkpoint 图只能作为辅助，因为本文方法主要在最后 checkpoint 超过基线，不能单独作为主证据。`480k -> 960k` 续训评估已经完成，结果没有支持“后续 checkpoint 持续领先”，因此不能把续训曲线升级为主趋势证据。

续训实验最终状态截至 2026-06-06 19:50：四个方法的 `480k -> 960k` 续训训练和固定 checkpoint 评估均已完成，远程评估 `12/12` 完成，本地合并表和曲线已生成。关键结论是负向/诊断性的：历史最佳 C8 mean SR 为 `linear_gate_no_pbrs=0.9596`，本文方法为 `0.9113`；最终 checkpoint 为 `linear_gate_no_pbrs=0.9326`，本文方法为 `0.8326`；后期平均（progress >= 0.75）为 `constant_gate_pbrs=0.9208`，本文方法为 `0.8631`。论文 agent 不应使用续训曲线证明本文方法持续占优。

## 推荐汇报顺序

1. 方法与任务：先放用户提供的架构图和数据集图，说明任务是地理定位/目标导航，混合奖励属于训练阶段机制。
2. 训练阶段机制：放 1 张动作归因图，说明每步动作如何收到外部项、内在项和 PBRS 的组合反馈。
3. 真实路线行为：放同一案例的俯视路线图，突出本文方法蓝色实线到达目标，对照方法虚线未到达。
4. 奖励组成曲线：放同一案例的奖励曲线图，展示距离曲线与奖励组成对齐。
5. 训练趋势：放训练 checkpoint 趋势图，重点说“固定 checkpoint 的训练过程观察”，不要说测试阶段使用奖励。
6. 正式结果：回到论文表格或精简数值，说明正式评估中本文方法在 MM-GAG 三模态和外部基线上更好。
7. 备份页：放同一任务早晚期路线演化、相图、GIF、补充实验曲线，用于回答追问。

## 主证据候选

### 1. 用户原图：方法架构与数据集

用途：答辩开场，先讲清楚研究对象和方法框架。用户已明确要求使用原图。

推荐文件：

- `F:\bishe\架构图.png`
- `F:\bishe\数据集.png`
- `F:\bishe\Undergraduate-Graduation-Project\results\figures\chapter3_method\figure3_1_method_overview_revised.png`
- `F:\bishe\Undergraduate-Graduation-Project\results\figures\chapter2_dataset\chapter2_typical_dataset_scene_examples.png`
- `F:\bishe\Undergraduate-Graduation-Project\results\figures\chapter2_dataset\figure_dataset_overview.png`

使用建议：PPT 中优先使用 `chapter3_method` 和 `chapter2_dataset` 下已归档版本，因为它们已同步进毕业设计仓库；如果需要追溯原始素材，再引用根目录下的 `架构图.png` 和 `数据集.png`。

### 2. 动作归因图：最适合讲“奖励如何指导动作”

用途：这是当前最能说服非专业评委的机制图。图中把真实俯视图路线、距离曲线、动作、距离变化、外部奖励、门控内在奖励、PBRS 和总奖励放在同一页。

目录：

- `F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\action_attribution`

推荐优先使用：

- `action_attribution_01_case_04_C8_seed123_ep120_img5_s4_g20.png`：C8，训练进度 26.9%，本文方法到达，5/5 个对照未到达。
- `action_attribution_02_case_08_C8_seed123_ep55_img6_s4_g20.png`：C8，训练进度 12.5%，适合强调训练早期已经出现方向差异。
- `action_attribution_04_case_12_C7_seed123_ep75_img5_s4_g21.png`：C7，适合讲对照路线回退/重复，本文方法能恢复并到达。
- `action_attribution_contact_sheet.png`：只作候选总览，不建议直接放主 PPT，因为单张图太小。

配套说明：

- `F:\bishe\Undergraduate-Graduation-Project\results\reports\reward_action_attribution_cases_zh.md`

推荐讲法：这页不是测试阶段额外使用奖励，而是把训练日志中的一个真实片段拆开。上面看行为，同一起点和目标下，对照方法容易停在目标外或回退，本文方法最终到达。下面看训练信号，每一步动作的距离变化与外部奖励、门控内在奖励、PBRS 都对齐在同一列。三项信号合在一起后，策略更容易把高回报分配给连续靠近目标的动作序列。

### 3. PPT 聚焦路线与奖励曲线：主视觉案例库

用途：路线图负责让评委“一眼看懂谁到达了目标”，奖励曲线负责解释“为什么这条路线更容易被训练出来”。

目录：

- `F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases`

最推荐成对使用的案例：

- `case_04`：C8，训练进度 26.9%，本文方法 8 步到达，5/5 对照未到达。
  - 路线：`focus_case_01_case_04_C8_seed123_ep120_img5_s4_g20.png`
  - 奖励：`reward_case_01_case_04_C8_seed123_ep120_img5_s4_g20.png`
- `case_08`：C8，训练进度 12.5%，本文方法 8 步到达，5/5 对照未到达。
  - 路线：`focus_case_02_case_08_C8_seed123_ep55_img6_s4_g20.png`
  - 奖励：`reward_case_02_case_08_C8_seed123_ep55_img6_s4_g20.png`
- `case_07`：C8，对照接近目标但未完成，适合讲“接近不等于完成”。
  - 路线：`focus_case_03_case_07_C8_seed123_ep90_img15_s0_g24.png`
  - 奖励：`reward_case_03_case_07_C8_seed123_ep90_img15_s0_g24.png`
- `case_12`：C7，多个对照回退/重复，适合讲“走出循环”。
  - 路线：`focus_case_08_case_12_C7_seed123_ep75_img5_s4_g21.png`
  - 奖励：`reward_case_08_case_12_C7_seed123_ep75_img5_s4_g21.png`

总览文件：

- `focus_case_contact_sheet.png`：12 个路线候选总览。
- `reward_case_contact_sheet.png`：12 个奖励曲线候选总览。
- `F:\bishe\Undergraduate-Graduation-Project\results\tables\defense_reward_training_stage\reward_guided_ppt_focus_cases_selected.csv`：候选案例排序和元数据。
- `F:\bishe\Undergraduate-Graduation-Project\results\reports\reward_guided_ppt_focus_cases_zh.md`：每个案例的中文讲稿备注。

使用建议：PPT 正文只放 1 到 2 个案例。路线图和奖励曲线最好使用同一个 case，先路线、后曲线。contact sheet 只放备份页或交给论文 agent 选图。

### 4. 同一任务早晚期演化

用途：展示同一 C8 任务在训练早期和后期的行为变化，属于“表格无法表达的可视化结果”。

推荐文件：

- `F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\evolution_cases\same_task_evolution_c8_seed123_img20_s0_g24.png`
- `F:\bishe\Undergraduate-Graduation-Project\results\reports\same_task_evolution_visual_zh.md`

使用建议：适合备份页或机制解释页。图内英文较多、中文较少，符合用户“不在图片里放太多中文解释”的要求。

### 5. 训练趋势与收敛信号

用途：回答“训练过程中是否真的形成更好的 checkpoint”和“loss/entropy 是否收敛”。

可用文件：

- `F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_trends\figure_reward_trend_main.png`
- `F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_trends\figure_dense_training_convergence_live.png`
- `F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_trends\figure_dense_training_log_signals.png`
- `F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_trends\figure_reward_dynamic_phase.png`
- `F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_trends\figure_reward_advantage_curves.png`

配套表格：

- `F:\bishe\Undergraduate-Graduation-Project\results\tables\defense_reward_trends\dense_training_convergence_live_summary.csv`
- `F:\bishe\Undergraduate-Graduation-Project\results\tables\defense_reward_trends\dense_training_convergence_live_points.csv`
- `F:\bishe\Undergraduate-Graduation-Project\results\tables\defense_reward_trends\dense_training_log_signal_summary.csv`

使用判断：`figure_dense_training_convergence_live.png` 适合作为“训练收敛诊断”，不能作为最终排名证据，因为四个方法的验证成功率都能到 1.0。`figure_reward_dynamic_phase.png` 适合讲训练路径从低成功率/高残余距离向高成功率/低残余距离移动。

### 6. 密集 checkpoint 与续训曲线：只能作辅助

当前文件：

- `F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_trends\figure_dense_mmgag_checkpoint_curves.png`
- `F:\bishe\Undergraduate-Graduation-Project\results\tables\defense_reward_trends\dense_mmgag_checkpoint_c8_method_summary.csv`
- `F:\bishe\Undergraduate-Graduation-Project\results\tables\defense_reward_trends\dense_mmgag_checkpoint_c8_training_trend_real_points.csv`
- `F:\bishe\GeoExplorer\analysis\pipeline_20260605_dense_mmgag_checkpoint_reward_trend\mmgag_checkpoint_eval_all.csv`
- `F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_trends\figure_dense_mmgag_continuation_curves.png`
- `F:\bishe\Undergraduate-Graduation-Project\results\tables\defense_reward_trends\dense_mmgag_continuation_c8_method_summary.csv`
- `F:\bishe\Undergraduate-Graduation-Project\results\tables\defense_reward_trends\dense_mmgag_continuation_c8_final_checkpoint.csv`
- `F:\bishe\Undergraduate-Graduation-Project\results\tables\defense_reward_trends\dense_mmgag_continuation_c8_late_stage_summary.csv`
- `F:\bishe\Undergraduate-Graduation-Project\results\reports\dense_mmgag_continuation_curve_zh.md`
- `F:\bishe\GeoExplorer\analysis\pipeline_20260606_dense_mmgag_continuation_trend\mmgag_checkpoint_eval_all.csv`

已知数值：固定 seed 的 480k 密集 checkpoint 评估中，`proposed_linear_gate_pbrs` 的 C8 mean SR 最优，为 0.9262；最强基线 `external_pbrs` 为 0.8738。最弱模态成功率为 0.9234 对 0.8426，平均残余距离为 0.227 对 0.374。但 `480k -> 960k` 续训没有维持该优势：历史最佳 C8 mean SR 为 `linear_gate_no_pbrs=0.9596` 对本文方法 `0.9113`；最终 checkpoint 为 `linear_gate_no_pbrs=0.9326` 对本文方法 `0.8326`；后期平均为 `constant_gate_pbrs=0.9208` 对本文方法 `0.8631`。

风险判断：这组图不能单独作为主趋势证据。480k 密集 checkpoint 图只能说“固定 seed 下 480k 末端 checkpoint 表现最好”，不能说“全训练过程稳定领先”；续训曲线进一步说明该末端优势没有在后续连续 checkpoint 中保持。答辩中建议把它们放在备份页或内部排查材料，不放主结果页。

论文 agent 的处理规则：不要用 continuation 曲线替换当前主证据。主证据继续使用动作归因、路线案例和正式固定评估表；continuation 只用于说明“我们检查过末端优势是否延续，结果不支持持续领先”。

## 正式结果与表格依据

### 1. 主 benchmark 对比

用途：正式结果收束，支撑“本文方法总体优于外部基线”。这部分以表格为主，不建议重复画成复杂图。

文件：

- `F:\bishe\Undergraduate-Graduation-Project\results\tables\main_benchmark\paper_baseline_compare_table.csv`
- `F:\bishe\Undergraduate-Graduation-Project\results\tables\main_benchmark\paper_baseline_compare_aggregate.json`
- `F:\bishe\Undergraduate-Graduation-Project\results\reports\paper_baseline_compare_summary.md`

可引用结论：`GeoExplorer-anchor0624` 在当前统一协议的 9 个共享 benchmark 上平均 SR 为 0.5803，`GOMAA-Geo` 为 0.5186，提升约 0.0617。MM-GAG 三模态中，本文方法在 aerial、ground、text 上分别高于 GOMAA。

注意：xBD 是 OpenDataLab `paper-test800` 纸面复现实验子集，不应表述为完全等同原论文私有划分。

### 2. 奖励门控消融表

用途：正式证明 `linear gate + PBRS` 这一组合在 MM-GAG 三模态上最优。

文件：

- `F:\bishe\Undergraduate-Graduation-Project\results\tables\ablation\reward_gate_type_mmgag_only_table_with_linear.csv`
- `F:\bishe\Undergraduate-Graduation-Project\results\reports\reward_gate_type_mmgag_only_summary_zh.md`
- `F:\bishe\Undergraduate-Graduation-Project\results\reports\reward_gate_eval_protocol_audit_20260520_zh.md`

关键数值：

- `linear_0.405_pb`：MM-GAG mean SR = 0.6094。
- `external_pbrs`：MM-GAG mean SR = 0.5943。
- `constant_0.405_pb`：MM-GAG mean SR = 0.5875。
- `linear_0.405_no_pb`：MM-GAG mean SR = 0.5821。

推荐写法：线性门控与 PBRS 的组合在 MM-GAG 三模态固定评估中取得最高平均成功率，说明单纯外部奖励或固定内在权重都不如“距离相关的内在奖励调节 + 目标方向塑形”的组合。

### 3. 展示型结果图卡

用途：如果需要快速做结果页，可使用 polished 图卡。它们更适合 README 或汇报备份页，不一定适合论文正文。

目录：

- `F:\bishe\Undergraduate-Graduation-Project\results\figures\showcase\polished`

推荐文件：

- `hero_dashboard.png`：整体结果总览。
- `mmgag_modality_panel.png`：跨模态定位结果。
- `ablation_story_panel.png`：模块消融。
- `reward_design_panel.png`：奖励设计分析。
- `long_range_panel.png`：长距离和压力测试。
- `trajectory_behavior_panel.png`：轨迹行为统计。
- `reward_process_panel.png`：奖励过程分解。

使用建议：这些图卡是展示友好版本，适合 PPT 中“结果墙”或备份页。若论文 agent 写正文，应优先引用 `results/tables` 中的 CSV/JSON 作为数值来源。

## 动图与轨迹材料

用途：答辩中如果需要直观展示路线动态，优先使用三联同步 GIF。

推荐目录：

- `F:\bishe\Undergraduate-Graduation-Project\results\figures\showcase\trajectories\triptych_gifs`
- `F:\bishe\Undergraduate-Graduation-Project\results\figures\showcase\experience\trajectory_theater_gifs`
- `F:\bishe\Undergraduate-Graduation-Project\results\figures\showcase\trajectories\comparison_png`

推荐文件：

- `three_method_hardcase__img189_d6_s20_g14_r0__triptych.gif`
- `c8_anchor_success__img000_d8_s24_g00_r0__triptych.gif`
- `c8_anchor_failure_or_detour__img054_d8_s20_g04_r0__triptych.gif`
- `c6_anchor_success__img006_d6_s24_g06_r0__triptych.gif`

使用建议：动图不要承担正式实验结论，只用于直观展示“同一场景下路径更稳定、更接近目标”。如需静态打印，使用 `comparison_png` 下同名 PNG。

## 补充实验材料

用途：回答评委关于预算、长距离、压力测试、统计稳定性的追问。

图目录：

- `F:\bishe\Undergraduate-Graduation-Project\results\figures\supplement`

推荐文件：

- `mmgag_diff_ci.png`：MM-GAG 差值置信区间。
- `ultra_long_diff_ci.png`：长距离差值置信区间。
- `p0_budget_sensitivity.png`：8x8/10x10 预算敏感性。
- `p1_grid25_budget_sensitivity.png`：25x25 压力测试。
- `trajectory_behavior_metrics.png`：轨迹行为指标。
- `reward_component_traces.png`：奖励过程曲线。
- `c4_failure_profile.png`：短距离失败画像，用于解释优势主要集中在中远距离。

配套报告：

- `F:\bishe\Undergraduate-Graduation-Project\results\reports\supplement_experiment_analysis_zh.md`
- `F:\bishe\Undergraduate-Graduation-Project\results\reports\supplement_eval_overview_zh.md`
- `F:\bishe\Undergraduate-Graduation-Project\results\reports\p0_supplement_eval_summary_zh.md`
- `F:\bishe\Undergraduate-Graduation-Project\results\reports\p1_grid25_analysis_zh.md`

使用建议：补充实验不抢主线。主线讲混合奖励机制与中长距离行为优势，补充实验用于说明结果不是只在单一设置下出现。

## 不建议作为主证据的材料

- `figure_dense_mmgag_checkpoint_curves.png`：最终点很好，但趋势上中前期不占优，只能作为辅助。
- `figure_dense_training_convergence_live.png`：适合讲收敛诊断，不适合讲最终排名。
- contact sheet 总览图：适合选图和备份，不适合主 PPT 正文。
- 存在乱码的说明文件：如 `dense_mmgag_checkpoint_curve_zh.md`、`dense_reward_followup_experiment_plan_zh.md`、`docs/visualization_gallery_zh.md` 中有编码问题，论文 agent 不应直接复制其中中文，应以本交接文档和对应 CSV/图像为准。
- 任何只把表格数值改画成柱状图的材料：不符合用户要求。表格能表示的内容放表格，图应展示趋势、路线、动作归因或训练过程。

## 可复制讲稿框架

第一段：本方法中的奖励设计只作用于训练阶段。训练时，智能体每一步动作会同时受到外部任务反馈、距离门控后的内在探索反馈以及 PBRS 势函数塑形反馈。这样设计的目的不是在测试时额外引入奖励函数，而是在训练中把高回报更稳定地分配给连续接近目标的动作序列。

第二段：从典型训练案例可以看到，同一起点和目标下，对照方法并不是完全不移动，而是容易在目标附近回退、绕行或停在目标外。本文方法的蓝色路线在 C7/C8 中远距离任务上更稳定地接近目标，并最终到达。奖励分解曲线进一步说明，外部奖励负责约束任务成败，门控内在奖励保留远距离探索，PBRS 在接近目标方向上提供连续塑形。

第三段：正式评估阶段不再调用这些奖励函数，只加载对应训练 checkpoint 执行策略。最终结果仍以固定评估表格为准。在 MM-GAG 奖励门控消融中，线性门控加 PBRS 的组合取得最高三模态平均成功率；在外部 benchmark 对比中，本文改进方法相对 GOMAA-Geo 保持整体优势。

## 论文 agent 执行清单

1. 先读本文档，不要直接从乱码报告恢复中文。
2. PPT 正文优先选 `case_04` 或 `case_08` 的动作归因图、路线图和奖励曲线。
3. 趋势图只作为辅助，不能把 480k 密集曲线包装成稳定领先。
4. 正式结果数值从 `results/tables` 的 CSV/JSON 取，不从图片读数。
5. 所有关于奖励、门控、PBRS 的文字都必须加训练阶段限定。
6. 续训评估已经完成，结论不支持替换当前主趋势图；不要把 `figure_dense_mmgag_continuation_curves.png` 放入主结果页。
7. 不要在任何报告、PPT 备注或 continuity 文件中写入远程密码、token 或密钥。
