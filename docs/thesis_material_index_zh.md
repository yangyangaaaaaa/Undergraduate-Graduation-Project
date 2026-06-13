# 论文与过程材料索引

本文件用于结题归档导航，说明论文、任务书、开题、中期、答辩等材料的保存位置、价值和上传取舍。原则是：仓库首页展示结果与复现线索，正式材料保留清晰索引，带签字、审批或个人信息的过程文件谨慎公开原件。

## 仓库内现有材料

| 位置 | 内容 | 用途 |
| --- | --- | --- |
| `thesis/张洋_本科毕业设计论文.docx` | 论文 Word 版正文 | 论文编辑与格式追溯 |
| `thesis/thesis_draft.md` | Markdown 草稿 | 核对章节、表格、公式和早期文字 |
| `materials/project_admin/08122215_张洋_任务书.docx` | 毕业设计任务书 | 说明课题来源、研究内容、进度安排 |
| `materials/project_admin/08122215_张洋_开题报告.doc` | 开题报告 | 说明选题背景、综述、技术路线 |
| `materials/project_admin/08122215_张洋_开题答辩.pptx` | 开题答辩 | 早期方案展示 |
| `materials/project_admin/08122215_张洋_中期报告.doc` | 中期报告 | 过程检查与阶段计划 |
| `materials/project_admin/08122215_张洋_外文文献翻译.docx` | 外文文献翻译 | 文献翻译与学习过程 |
| `thesis/official/zhangyang_thesis_final_20260602.pdf` | 论文定稿 PDF | 最终提交载体，45 页 |
| `materials/defense/zhangyang_final_defense_20260611.pdf` | 结题答辩 PDF | 最终答辩载体，27 页 |
| `materials/quality/aigc_reports_20260603/` | AIGC 检测报告 | 定稿阶段质量检查记录 |
| `docs/project_closing_summary_zh.md` | 结题收尾总结 | 历史记录、实验线、Git 上传经验、经验教训 |
| `docs/defense_ppt_storyline_zh.md` | 结题 PPT 汇报脉络 | 28 页答辩主线和 README 展示素材来源 |
| `docs/server_acceptance_guide_zh.md` | 服务器验收指南 | 远端脚本、输出指针和 C1-C3 边界结果 |

## 本次检查到的外部 PDF

| 外部文件 | 页数 | 大小 | 文本可提取性 | 归档价值 | 建议 |
| --- | ---: | ---: | --- | --- | --- |
| `D:\桌面\literature\张洋_本科毕业设计论文 - 定稿2 - 副本.pdf` | 45 | 3.05 MB | 可提取，约 4.7 万字符 | 论文最终 PDF 载体 | 已上传到 `thesis/official/zhangyang_thesis_final_20260602.pdf` |
| `G:\毕设答辩\08122215_张洋_结题答辩 (2).pdf` | 27 | 6.76 MB | 可提取，约 4 千字符 | 答辩展示载体 | 已上传到 `materials/defense/zhangyang_final_defense_20260611.pdf` |
| `E:\杂\bulk_download\张洋_08122215_毕业论文（设计）任务书 (2).pdf` | 5 | 0.23 MB | 可提取 | 任务书正式 PDF | 中高价值；仓库已有 DOCX，PDF 可作为正式提交版备份 |
| `E:\杂\bulk_download\zqjc.pdf` | 1 | 0.16 MB | 可提取 | 中期检查表 | 过程价值；含指导意见/签字字段，公开仓库不建议优先上传原件 |
| `E:\杂\54726c05-c215-4b17-8e90-6e50c68558d9 (3).pdf` | 11 | 0.81 MB | 可提取 | 开题报告 PDF | 中高价值；仓库已有 DOC 版，PDF 可作为正式版备份 |

## 本次新增或核验的源文件

| 外部文件 | 仓库处理 | SHA256 前 12 位 |
| --- | --- | --- |
| `D:\桌面\literature\08122215_张洋_任务书.docx` | 与 `materials/project_admin/08122215_张洋_任务书.docx` 完全一致，已在仓库中 | `416AC49B488F` |
| `D:\桌面\literature\08122215_张洋_开题报告.doc` | 与 `materials/project_admin/08122215_张洋_开题报告.doc` 完全一致，已在仓库中 | `ECA57277D9DB` |
| `E:\杂\3c4fc94e-4576-41f7-bd92-331054a3d594.doc` | 与 `materials/project_admin/08122215_张洋_中期报告.doc` 完全一致，已在仓库中 | `ED55B88B6891` |
| `E:\杂\af5a3468-1c96-4eb9-8f3c-c0e674bb7989 (1).zip` | 已上传到 `materials/quality/aigc_reports_20260603/`，并解包为独立报告 | `EFF77A8C3580` |

## 推荐公开仓库策略

优先保留在 GitHub 的内容：

1. README 展示页、最终方法图、关键结果图和压缩动图。
2. `docs/` 中的索引、复现说明、实验总结、服务器验收指南和结题经验总结。
3. 小型 CSV 表格、报告和脚本，确保结论可追溯。
4. 论文最终 PDF 和答辩 PDF，仅在确认可以公开个人信息、导师信息和学校格式后上传。

谨慎公开或只做本地备份的内容：

1. 含手写签名、审批意见、联系方式、学籍信息扩展字段的过程表。
2. 原始大图包、PPT 原件、checkpoint、完整数据集和服务器缓存。
3. 任何密码、token、cookie、私钥、服务器登录信息。

## 推荐目录映射

如果后续决定上传 PDF 原件，建议按以下目录整理：

```text
materials/project_admin/pdf/
  task_book_08122215_zhangyang.pdf
  proposal_report_08122215_zhangyang.pdf
  midterm_check_08122215_zhangyang.pdf
```

文件名建议使用英文或拼音，避免 GitHub URL 中中文和括号造成引用不便；正文索引中保留原始文件名、来源路径和日期。

## 使用优先级

论文正文引用数值时，优先使用：

1. `results/tables/`
2. `docs/experiment_summary_zh.md`
3. `results/reports/`
4. `docs/project_closing_summary_zh.md`

如果正文数值与草稿、旧报告或过程材料不一致，以 `results/tables/` 中对应 CSV 和最终论文 PDF 为准。
