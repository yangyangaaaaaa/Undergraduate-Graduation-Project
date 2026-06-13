# 毕设收尾总览与经验总结

整理时间：2026-06-13。依据来源包括 `F:\bishe`、`F:\literature`、`D:\codex-home` 近几十天会话历史、项目连续性交接文件、最终结题 PPT、论文定稿 PDF 以及服务器 `/root/geoexplorer` 目录。整理过程中未在对话框中打开图片，图片只做文件级抽取、尺寸、哈希检查与 GitHub 展示压缩。

## 已浏览的历史来源

- `F:\bishe\PROJECT_STATE.md`、`PROJECT_HISTORY.md`、`PROJECT_ASSETS.md`、`PROJECT_OFFBOARDING.md`
- `F:\literature\PROJECT_STATE.md`、`PROJECT_HISTORY.md`、`PROJECT_ONBOARDING.md`、`PROJECT_ASSETS.md`
- `D:\codex-home\continuity\project_index.json`
- `D:\codex-home\sessions` 与 `D:\codex-home\archived_sessions` 中 2026-05-14 以来的相关会话
- `D:\codex-home\sessions\2026\06\08\rollout-2026-06-08T16-06-27-019ea644-cf72-76f1-9ec3-5b4a9514c19d.jsonl`
- 关键词覆盖：`bishe`、`literature`、`Undergraduate-Graduation-Project`、`GitHub`、`README`、`PPT`、`实验`、`答辩`、`验收`、`上传`、`AIGC`、`GeoExplorer`、`MM-GAG`、`xBD`
- 抽取到的相关 session/归档 session 共 66 个，重点人工核对了 5 月 17-28 日实验与仓库上传线、6 月 6-10 日答辩图包与验收线
- `019ea644...` 重点核对了 PPT 候选图包、奖励机制“分阶段直线轴”最终图、Tseuzier/SwissView 搜索区域图、C4 短距离失败案例、续训诊断结论和不在聊天框预览图片的处理约束。

## 时间线

### 论文与材料线

- 3 月至 4 月：任务书、开题报告、外文文献翻译、中期报告等过程材料形成。
- 5 月 9-14 日：`F:\literature` 中整理第二章、第三章材料包，形成 `chapter2_*`、`chapter3_*` 与 `thesis_draft.md`。
- 5 月 20-29 日：围绕 AIGC 检测、公式/OLE 抽取、Word/PDF 最终载体进行多轮修订。CNKI AIGC 从 `25.8%` 降到 `17.2%`，最终 Word/PDF 载体在 5 月 29 日至 6 月 2 日形成。
- 6 月 9-10 日：结题 PPT 逐步收口，主线转为“背景-任务-问题-框架-奖励机制-实验设置-可视化结果-实验分析”。

### 实验线

- 5 月 12-15 日：围绕 `0.624` 结果进行 after-0.624、nextfamily、anchor preserve 等探索。
- 5 月 15-17 日：完成 `anchor0624` 全因子消融与 paper-aligned generalization，确认完整分支在主泛化指标上领先。
- 5 月 17-18 日：补充 GeoExplorer-pristine 固定评估、官方源码重训对照、源代码差异审计和 Chapter 4 可视化补充。
- 5 月 19-24 日：完成附录数据集/参数/奖励门控/P0/P1/25x25 等补充实验。
- 6 月 4-6 日：制作训练阶段奖励趋势、动作归因、同任务演化、PPT 图包；同时确认 480k->960k 续训是负向/诊断结果，不进主线。
- 6 月 8-10 日：整理服务器验收演示、一键训练、一键路线包和 C1-C3 短距离完整评估。
- 6 月 8 日：在 `019ea644...` 会话中继续收口奖励机制展示图，最终保留“远距离阶段有探索正反馈、近目标阶段回退被压制、到达动作获得最高总奖励”的灰色说明，并确认该图只解释训练阶段奖励信号。

### GitHub 上传线

- 5 月 23 日：仓库目标被明确纠偏为 `yangyangaaaaaa/Undergraduate-Graduation-Project`，不是原始 `limirs/GeoExplorer`。
- 同日：本地 HTTPS git push 到 GitHub 失败，历史经验是改用 GitHub Git Data API 和环境变量 token；上传时坚持“token 不写入 remote URL、不写入文件”。
- 上传时遇到 GitHub 单文件限制，`manual_redraw_assets.zip` 因过大被拆成目录文件后成功上传。
- 5 月 23-28 日：README 被做成视觉展示页，后来按用户意见替换为更自然的中文叙事和项目图。
- 6 月 13 日：README 再次按最终答辩 PPT 重构，使用 PPT 嵌入媒体，不再沿用旧 showcase 首页图。

## 当前仓库边界

权威仓库：

```text
F:\bishe\Undergraduate-Graduation-Project
https://github.com/yangyangaaaaaa/Undergraduate-Graduation-Project
```

不要把本科毕业设计材料上传到原始 GeoExplorer 仓库。`F:\bishe\GeoExplorer` 和服务器 `/root/geoexplorer/GeoExplorer` 是实验代码与运行环境，毕业设计归档仓库是 `Undergraduate-Graduation-Project`。

当前本地 git 有两个风险：

- `main` 比 `origin/main` ahead 多个提交，远端未完全同步。
- 工作区中有大批 `geoexploxer_edit/SwissView/...` 文件显示删除，这是已有工作区状态，不能误提交删除。

本轮收尾只应选择性提交 README、文档、小型表格、验收脚本和 PPT 抽取素材，不 stage 大规模数据删除或整包图集。

## 本轮应归档内容

已整理或建议归档：

- `README.md`：按最终 PPT 主线重写。
- `docs/project_closing_summary_zh.md`：本收尾总结。
- `docs/defense_ppt_storyline_zh.md`：PPT 28 页讲述脉络。
- `docs/server_acceptance_guide_zh.md`：远端验收命令与结果边界。
- `results/figures/defense_readme_20260613/`：从最终 PPT 抽取并压缩的 22 个首页素材，覆盖 MASA、SwissView、MM-GAG、xBD、10x10 长距离和实验分析页。
- `results/figures/acceptance_demo_selected_20260613/`：从验收演示大包中压缩出的代表动图与索引图，覆盖 C4/C6/C8、三方法困难样例、MM-GAG、xBD 和长距离设置。
- `results/figures/defense_reward_selected_20260613/`：从奖励机制大目录中精选的动作归因、路线/奖励案例、同任务演化和 C4 失败边界图。
- `results/reports/short_distance_c123_summary_20260609_zh.md`：短距离边界报告。
- `results/tables/short_distance_c123/`：短距离 C1-C3 小型结果表。
- `code/tools/*acceptance*` 与 `code/tools/*c123*`：验收演示、验收训练、路线包和短距离评估脚本。
- `materials/project_admin/zhangyang_*`：任务书、开题报告、中期检查、软硬件验收表和题目变更说明等过程材料补充归档。

不建议归档：

- PPT 原件，约 105 MB，超过 GitHub 单文件限制。
- `ppt_candidate_pack_20260606` 整包，约 357 MB，适合本地/网盘，不适合整包入仓库。
- checkpoint、大规模原始数据、服务器缓存和临时日志。
- 任何密码、token、cookie 或私钥。

## 服务器价值资料

服务器 `/root/geoexplorer` 中最有价值的是可复现脚本和权威输出指针：

- `/root/geoexplorer/run_acceptance_demo`
- `/root/geoexplorer/run_acceptance_case_pack`
- `/root/geoexplorer/run_acceptance_train`
- `/root/geoexplorer/run_c123_eval`
- `/root/geoexplorer/analysis/acceptance_demo_latest`
- `/root/geoexplorer/analysis/acceptance_case_pack_latest`
- `/root/geoexplorer/analysis/acceptance_train_latest`
- `/root/geoexplorer/analysis/short_distance_c123_eval_latest`

这些内容应该以“命令、路径、结果摘要”的形式写进仓库，而不是把远端输出全集搬进仓库。

## 经验与教训

### 项目管理

1. 项目边界必须先确认。`F:\literature` 是论文写作工作区，`F:\bishe` 是实验和答辩材料工作区，`Undergraduate-Graduation-Project` 是最终归档仓库。
2. 交接文件非常有用。`PROJECT_STATE.md` 和 `PROJECT_OFFBOARDING.md` 能避免每次从几十万行历史重新找线索。
3. 文件名和版本要保守。最终论文载体已转到 5 月 29 日 Word/PDF，不应再把旧 `thesis_draft.md` 当成最终稿。
4. 图包和 PPT 素材应分层：主线只放少量强证据，完整候选包留本地，不要全塞 README。

### 实验方法

1. 训练阶段奖励和测试阶段策略必须分开讲。奖励机制解释学习过程，正式测试只执行 checkpoint。
2. 主结果、补充结果和诊断结果不能混讲。续训结果没有支持持续领先，就只能作为诊断。
3. 短距离 C1-C3 不要包装成优势。它揭示局部相似、探索偏移和短距离直接匹配的边界。
4. xBD 口径必须说清楚：灾后搜索区域，目标仍用灾前图像嵌入。
5. 消融实验应讲组合机制，而不是把每个因子都说成独立正收益。

### 仓库上传

1. 目标仓库必须再次确认：`yangyangaaaaaa/Undergraduate-Graduation-Project`。
2. GitHub 单文件 100 MB 限制要提前检查，PPT、zip、checkpoint 不适合直接上传。
3. token 只能从环境变量或安全凭据读取，不写进 remote URL、README、脚本或连续性文件。
4. HTTPS git push 不稳定时，可用 GitHub Git Data API 兜底，但应避免一次性上传过大 blob。
5. 工作区已有大量未跟踪和删除状态时，只 stage 本轮明确需要的文件。

### 答辩材料

1. README 和 PPT 不应堆旧图，应围绕最终答辩主线讲任务、方法、结果和边界。
2. README 文字应自然介绍项目，不写成“按某条命令整理”或“执行某个 agent 指令”的口吻；过程线索放入 docs，不放在首页开头。
3. 图片处理尽量使用文件级校验、尺寸、哈希和索引，不在聊天框中批量打开图片。
4. 可视化结论必须有表格或日志支撑。漂亮图只能帮助解释，不能代替正式评估。
5. 讲稿中可直接使用这句边界：混合奖励用于训练阶段优化策略，正式测试阶段只加载训练好的策略网络执行动作。
