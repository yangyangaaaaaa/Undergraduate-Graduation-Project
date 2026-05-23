# 数据集示例图手动画图素材包

用途：把之前 `chapter2_typical_dataset_scene_examples` 图中用到的图片素材单独整理出来，方便手动在 PPT、AI、PS、Visio 或 Origin 中重新排版。

## 推荐使用方式

- 先看 `00_reference_final/reference_chapter2_typical_dataset_scene_examples.png`，这是上一版整体布局参考。
- 真正重画时优先使用各数据集目录里的 `panel_ready` 图片；这些已经按上一版显示方式裁成正方形或固定比例。
- 如果需要重新裁剪或放大局部，再使用不带 `panel_ready` 的原始复制素材。
- 中文字体建议用宋体；英文数据集名和英文提示建议用 Times New Roman。

## 目录说明

- `00_reference_final`：上一版最终图的 PNG/SVG/PDF，仅作排版参考。
- `01_MASA`：MASA 航拍搜索图和目标航拍 patch。
- `02_MM-GAG`：MM-GAG 的地面目标图、航拍拼接搜索图，以及文本目标提示。
- `03_SwissView`：SwissView100 航拍样例、SwissViewMonuments 的地面目标和航拍搜索图。
- `04_xBD`：xBD 灾前目标图和灾后搜索图，以及本次抽取图像对的 metadata。
- `05_optional_overview`：更早的数据集总览图参考，不是当前四宫格图的必要素材。

## 当前四宫格建议素材

- MASA：左边放 `01_MASA/MASA_aerial_target_patch_row1_col3_panel_ready.png`，右边放 `01_MASA/MASA_aerial_search_panel_ready_square.png`。
- MM-GAG：左上放 `02_MM-GAG/MMGAG_ground_target_IMG_1704_panel_ready_square.png`，左下可手动画文字框，右边放 `02_MM-GAG/MMGAG_aerial_search_IMG_1704_panel_ready_ratio1p27.png`。
- SwissView：左边放 `03_SwissView/SwissViewMonuments_ground_target_Chillon_00_panel_ready_square.png`，右边放 `03_SwissView/SwissViewMonuments_aerial_search_Chillon_00_panel_ready_square.png`。
- xBD：左边放 `04_xBD/xBD_pre_disaster_target_panel_ready_square.png`，右边放 `04_xBD/xBD_post_disaster_search_panel_ready_square.png`。

## 标注提醒

- MASA、SwissView 和 MM-GAG 航拍搜索图上可以手动画 5x5 网格。
- 起点可用绿色框，目标可用黄色或橙色框；xBD 图对可用橙色框标出同一目标区域。
- xBD 论文表述建议写成“灾前目标图像 / 灾后搜索图像”，避免误写成灾后目标。
- `asset_manifest.json` 记录了每个生成素材的源文件和裁剪说明。
