# xBD 评测协议纠偏说明（2026-05-19）

## 结论

用户的判断是正确的：按照 GOMAA-Geo 与 GeoExplorer 原论文，xBD 的核心灾害迁移实验不是“灾前图像条件”和“灾后图像条件”两个对称任务。目标图像始终应来自灾前航拍图；真正的跨灾害迁移设置是：以灾前图像作为目标输入，在灾后图像构成的搜索区域中执行主动定位。

当前代码中的 `xbd_disaster_aerial` 配置与该协议一致：搜索数组使用 `xbd_post_grid_5.npy`，同时通过 `pre_goal_path` 指向 `xbd_pre_grid_5.npy` 作为目标图。当前问题主要出在论文表格标题和文字分析，原表述容易让读者理解成 `xBD-pre` 与 `xBD-disaster` 分别使用各自图像作为目标输入。

## 代码核对

- `xbd_pre_aerial`：搜索区域为灾前图像，目标也来自同一灾前图像。它可以作为灾前参考条件。
- `xbd_disaster_aerial`：搜索区域为灾后图像，目标来自灾前图像。它才是原论文主线中的 xBD disaster-transfer 评测。

关键配置位于：

- `F:\bishe\GeoExplorer\ab_experiments\algo_paper_generalization_20260516\anchor0624_factorial_generalization_seed321_480k\monitoring\paper_baseline_compare_supervisor.py`
- `F:\bishe\GeoExplorer\ab_experiments\algo_paper_generalization_20260516\anchor0624_factorial_generalization_seed321_480k\monitoring\paper_baseline_evaluator.py`

## 建议替换后的表4-4

表4-4 xBD 灾前目标与灾害后搜索条件下的 SR / SG 对比

| 评测任务 | GeoExplorer（原始方法） | GOMAA-Geo | 本文方法 | Random policy | DiT-AGL |
| --- | ---: | ---: | ---: | ---: | ---: |
| xBD-pre aerial（灾前目标 / 灾前搜索，参考） | 0.5080 / 1.5555 | 0.5427 / 1.4132 | 0.5852 / 1.2922 | 0.0661 / 3.9239 | 0.0131 / 3.8115 |
| xBD-disaster aerial（灾前目标 / 灾后搜索） | 0.5129 / 1.5452 | 0.5345 / 1.4194 | 0.5856 / 1.2839 | 0.0661 / 3.9239 | 0.0149 / 3.8104 |

如果希望严格贴近 GeoExplorer 论文主表，也可以只保留第二行，将第一行移至附录或作为参考条件在正文中简短说明。

## 建议替换后的分析段

xBD 评测用于考察灾害前后图像外观变化下的跨域定位能力。按照原论文设置，目标始终由灾前航拍图像给出；其中 xBD-pre 表示在灾前图像中搜索，可视为参考条件，xBD-disaster 表示在灾后图像中搜索，是更关键的灾害迁移设置。本文方法在 xBD-disaster 上取得 0.5856 的 SR 和 1.2839 的 SG，高于 GOMAA-Geo 的 0.5345 / 1.4194，也高于 GeoExplorer 原始方法的 0.5129 / 1.5452。相较 GOMAA-Geo，本文方法的 SR 提高 0.0511，SG 降低 0.1355；相较 GeoExplorer 原始方法，SR 提高 0.0727，SG 降低 0.2613。该结果表明，在目标外观来自灾前、搜索观测来自灾后的跨域条件下，本文方法仍能保持更稳定的目标接近能力。xBD-pre 结果可作为灾前参考条件，两者数值接近，但正文中不宜简单表述为“灾前/灾后两个对称图像条件”，以免误解目标输入来源。

## 对原文字的修正建议

- 原标题“xBD 灾前与灾后图像条件下的 SR / SG 对比”建议改为“xBD 灾前目标与灾害后搜索条件下的 SR / SG 对比”。
- 原句“xBD-pre 与 xBD-disaster 分别对应灾前和灾后图像条件”建议改为“xBD-pre 为灾前搜索参考条件，xBD-disaster 为灾前目标、灾后搜索的跨灾害迁移条件”。
- 原句“说明灾后外观变化没有造成明显性能下降”应降调，因为当前 paper-test800 是可复现对齐子集，不应过度推广为灾害变化完全无影响。建议写成“在该确定性对齐子集上未观察到明显下降”。
