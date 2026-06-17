# 论文与过程材料索引

本文件说明仓库中最终论文、答辩材料、过程文档和正式归档材料的保存位置。材料目录已经按用途整理为长期结构，日期只保留在 manifest 文件名中用于追溯来源。

## 最终论文

| 位置 | 内容 |
| --- | --- |
| `materials/thesis/final/08122215.pdf` | 最终论文 PDF。 |
| `thesis/` | 论文草稿、正文整理文件和论文相关说明。 |
| `materials/thesis/vector_figures/` | 从论文中提取或整理的 SVG 矢量图。 |

## 过程材料

| 位置 | 内容 |
| --- | --- |
| `materials/project_admin/original_forms/` | 任务书、开题报告、开题答辩、中期报告、外文文献翻译等早期原始过程材料。 |
| `materials/project_admin/signed_documents/` | 已签字或正式版的任务书、开题审查、中期检查、验收表、题目变更说明等材料。 |
| `materials/project_admin/final_documents/` | 资料盒中补充归档的正式结题材料，包括答辩资格审查、评阅、答辩小组意见、成绩组成表、外文翻译封面和中期检查等文件。 |
| `materials/defense/` | 结题答辩 PDF 与答辩材料说明。 |
| `materials/quality/` | AIGC 检测、质量检查和相关报告。 |

## 清单与校验

| 位置 | 内容 |
| --- | --- |
| `materials/manifests/final_submission_20260617.csv` | 2026-06-17 正式材料补充归档的来源、目标路径、文件大小和 SHA-256 校验值。 |
| `materials/manifests/final_submission_20260617.json` | 与 CSV 等价的结构化清单，便于程序读取。 |

## 使用建议

论文正文引用数值时，优先使用 `results/tables/` 中的 CSV/JSON；展示性结论可参考 `docs/experiment_summary_zh.md`、`docs/result_inventory_zh.md` 和 `results/reports/`。正式提交材料优先从 `materials/` 下的长期目录取用。
