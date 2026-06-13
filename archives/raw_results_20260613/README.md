# 原始结果归档包说明

这一目录用于说明不适合直接铺在仓库首页展示、但对毕业设计收尾和后续复查仍有价值的原始结果材料。为了不让 git 历史变得过大，仓库正文只保存 README、`manifest.csv` 和 `manifest.json`；45 个 ZIP 压缩包已上传到 GitHub Release 附件：

[raw-results-20260613](https://github.com/yangyangaaaaaa/Undergraduate-Graduation-Project/releases/tag/raw-results-20260613)

多数图片、动图、PDF 本身已经是压缩格式，所以 ZIP 后体积下降有限；压缩包的主要作用是集中保存、减少零散文件数量，并让归档结构更清楚。

每个 ZIP 包内都保留了仓库相对路径，例如 `results/figures/acceptance_demo/...`。解压后可以直接看出文件原本属于哪个实验目录。每个 ZIP 内也包含 `ARCHIVE_CONTENTS.txt`，记录该包的来源说明和文件清单。

## 归档内容

| 来源 | 分卷数 | 文件数 | 压缩后体积 | 内容说明 |
| --- | ---: | ---: | ---: | --- |
| `figures_acceptance_demo` | 4 | 53 | 243.78 MB | 验收演示阶段生成的原始动图、路线设置图、poster 图和配置清单。 |
| `figures_defense_reward_training_stage` | 16 | 204 | 455.41 MB | 答辩奖励机制分析中的训练阶段图、路线/奖励案例、动作归因、趋势图和中间可视化。 |
| `figures_ppt_candidate_pack_20260606` | 11 | 132 | 351.19 MB | 结题答辩前筛选过的 PPT 候选素材，包括方法图、数据集图、验收演示动图和实验设置图。 |
| `figures_showcase` | 6 | 142 | 198.40 MB | 可视化展示区的完整图件，包括 benchmark、轨迹、数据集、消融、奖励和补充展示图。 |
| `figures_chapter2_dataset` | 5 | 41 | 162.07 MB | 第二章数据集、搜索区域、5x5 网格、SwissView/Tseuzier 样例和人工重绘辅助素材。 |
| `figures_chapter4_trajectories` | 1 | 6 | 38.20 MB | 第四章典型轨迹相关的 PNG/SVG/PDF 高质量图件。 |
| `figures_defense_reward_trends` | 1 | 18 | 6.57 MB | 答辩奖励趋势分析相关的曲线图。 |
| `reports_tables_logs_tools` | 1 | 241 | 3.69 MB | 实验报告、结果表格、训练日志和用于生成可视化/分析结果的离线工具脚本。带服务器连接入口的远程监控脚本未纳入该归档。 |

更细的文件级清单见：

- `manifest.csv`
- `manifest.json`

## 使用建议

这些归档包主要用于保存和追溯，不作为 README 首页展示素材。需要复查某一类实验时，先根据上表选择对应分卷，再从 Release 附件或 `manifest.csv` 的 `download_url` 下载压缩包，然后查看包内的 `ARCHIVE_CONTENTS.txt`。

如果只需要快速浏览最终效果，优先查看 `results/figures/defense_readme_20260613/` 和 README 中的结果展示；如果需要追溯“这些图是从哪里来的、还有哪些备选材料”，再打开本目录的清单并下载对应压缩包。
