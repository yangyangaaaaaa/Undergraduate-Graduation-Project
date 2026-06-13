# 服务器验收指南

本指南整理服务器 `/root/geoexplorer` 中与结题验收最相关的脚本、输出和边界。服务器登录信息来自本机环境变量或安全凭据管理器，不写入仓库。

## 远端结构

- 项目根目录：`/root/geoexplorer`
- 主代码目录：`/root/geoexplorer/GeoExplorer`
- 分析输出目录：`/root/geoexplorer/analysis`
- 验收演示代码：`/root/geoexplorer/GeoExplorer/acceptance_demo`
- 验收训练代码：`/root/geoexplorer/GeoExplorer/acceptance_train`

## 一键命令

```bash
/root/geoexplorer/run_acceptance_demo
/root/geoexplorer/run_acceptance_demo --visual-only
/root/geoexplorer/run_acceptance_case_pack
/root/geoexplorer/run_acceptance_train
/root/geoexplorer/run_acceptance_train --full
/root/geoexplorer/run_acceptance_train --status
/root/geoexplorer/run_acceptance_train --tail
/root/geoexplorer/run_c123_eval
```

## 最新远端指针

- `acceptance_demo_latest` -> `/root/geoexplorer/analysis/acceptance_demo_oneclick_20260609_022913`
- `acceptance_case_pack_latest` -> `/root/geoexplorer/analysis/acceptance_case_pack_20260609_032854`
- `acceptance_train_latest` -> `/root/geoexplorer/analysis/acceptance_train_smoke_20260608_123934`
- `short_distance_c123_eval_latest` -> `/root/geoexplorer/analysis/short_distance_c123_eval_20260609_161734`

## 已验证结果

验收演示包：

- 最新包包含路线 GIF、静态 PNG、manifest、日志和自定义图像预处理结果。
- 适合现场展示“模型能在不同目标线索和场景下连续搜索”。

验收路线包：

- `acceptance_case_pack_20260609_032854` 包含 45 张路线 PNG。
- 类别覆盖 C4/C6/C8 overhead、MM-GAG aerial/ground/text、xBD pre/post、long distance。

训练验收：

- 默认 `/root/geoexplorer/run_acceptance_train` 是安全 smoke 训练，不是长训练。
- 长训练使用 `/root/geoexplorer/run_acceptance_train --full`，输出在 `/root/geoexplorer/analysis/acceptance_train_<mode>_<timestamp>`。

短距离 C1-C3：

- 最新完整评测完成 `131/131`，失败 `0`。
- 结论是短距离不是本文方法优势区间：完整长距离 anchor 在 C1-C3 transfer mean 为 `0.2454`，同数据无新增机制控制组为 `0.2466`。
- 该结果用于解释短距离局部相似和探索过量问题，不作为主优势结果。

## 本地归档

已同步到仓库的关键小文件：

- `results/reports/short_distance_c123_summary_20260609_zh.md`
- `results/tables/short_distance_c123/comparison_method_summary.csv`
- `results/tables/short_distance_c123/comparison_method_metrics.csv`
- `results/tables/short_distance_c123/ablation_branch_summary.csv`
- `code/tools/run_acceptance_demo`
- `code/tools/run_acceptance_demo_oneclick.sh`
- `code/tools/build_acceptance_demo_visuals.py`
- `code/tools/build_acceptance_case_pack.py`
- `code/tools/run_acceptance_train`
- `code/tools/run_acceptance_train_oneclick.sh`
- `code/tools/run_c123_eval`
- `code/tools/run_short_distance_c123_eval.py`

不建议提交的内容：

- 远端 checkpoint 权重。
- 原始大规模数据集。
- 服务器临时日志、缓存、`__pycache__`。
- 任何密码、token、cookie 或私钥。
