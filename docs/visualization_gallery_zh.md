# 可视化结果画廊

本文件整理项目首页、论文补充材料和答辩展示可直接使用的图件。正式数值以 `results/tables/` 为准，图件用于帮助快速理解任务、方法和结果。

## 生成方式

```bash
python code/tools/build_visual_showcase.py
python code/tools/build_supplement_experiment_analysis.py
```

核心展示图位于：

- `results/figures/defense_showcase/`
- `results/figures/acceptance_demo/`
- `results/figures/reward_cases/`
- `results/figures/showcase/`

## 首页展示素材

| 图件 | 用途 |
| --- | --- |
| `results/figures/defense_showcase/method_framework.png` | README 首屏研究框架。 |
| `results/figures/defense_showcase/short_distance_route.gif` | MASA 短距离绕路成功轨迹。 |
| `results/figures/defense_showcase/swissview_town_landmark.gif` | SwissView 城镇地标连续搜索。 |
| `results/figures/defense_showcase/c8_route_demo.gif` | SwissView 跨区域湖边长距离搜索。 |
| `results/figures/defense_showcase/mmgag_ground_target.gif` | MM-GAG 地面图像目标搜索。 |
| `results/figures/defense_showcase/multimodal_text_target.gif` | MM-GAG 文本目标搜索。 |
| `results/figures/defense_showcase/xbd_hurricane_prepost.png` | xBD 飓风灾前灾后目标对比。 |
| `results/figures/defense_showcase/xbd_disaster_route.gif` | xBD 飓风灾后搜索轨迹。 |
| `results/figures/defense_showcase/xbd_fire_prepost.png` | xBD 火灾灾前灾后目标对比。 |
| `results/figures/defense_showcase/xbd_fire_route_static.png` | xBD 火灾灾后搜索轨迹。 |
| `results/figures/defense_showcase/long_distance_grid10.gif` | 10x10 长距离固定边界搜索。 |
| `results/figures/defense_showcase/short_distance_failure.png` | 短距离失败边界。 |
| `results/figures/defense_showcase/performance_analysis.png` | 性能实验分析。 |
| `results/figures/defense_showcase/reward_distance_analysis.png` | 奖励分距离分析。 |
| `results/figures/defense_showcase/ablation_analysis.png` | 消融实验分析。 |

## 补充实验图卡

| 图件 | 说明 |
| --- | --- |
| `results/figures/showcase/polished/hero_dashboard.png` | 主结果总览。 |
| `results/figures/showcase/polished/mmgag_modality_panel.png` | 跨模态目标定位表现。 |
| `results/figures/showcase/polished/ablation_story_panel.png` | 模块消融实验。 |
| `results/figures/showcase/polished/reward_design_panel.png` | 奖励设计分析。 |
| `results/figures/showcase/polished/long_range_panel.png` | 长距离搜索与压力测试。 |
| `results/figures/showcase/polished/trajectory_behavior_panel.png` | 轨迹行为统计。 |
| `results/figures/showcase/polished/reward_process_panel.png` | 奖励过程分解。 |

## 专题目录

| 目录 | 内容 |
| --- | --- |
| `results/figures/defense_showcase/` | 最终答辩 PPT 抽取媒体。 |
| `results/figures/acceptance_demo/` | 验收演示精选动图。 |
| `results/figures/reward_cases/` | 奖励机制和动作归因案例图。 |
| `results/figures/presentation_assets/` | PPT 候选素材和说明。 |
| `results/figures/supplement/` | 置信区间、轨迹行为、短距离失败画像和奖励过程曲线。 |

## 使用建议

- README 首页优先使用 `results/figures/defense_showcase/` 中的最终答辩 PPT 展示素材。
- 答辩和演示优先使用 PNG 与 GIF，论文正文可使用 SVG/PDF 或从源表重新导出。
- 展示图只用于解释趋势，正式结论必须回到 `results/tables/` 和对应报告。
