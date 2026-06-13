# C=4 全失败徘徊路线候选

筛选条件：`distance=4`，同一个 `case_id` 下 GOMAA-Geo、GeoExplorer-pristine、GeoExplorer-anchor0624 三种方法全部失败；优先选择立即折返、重复访问、转向和最终距离较高的样例。
绘图更新：重复经过同一路径的线段做平行错位；每个方法面板下方单独拉出“重复最多模块”和“目标模块”进行局部对比；`GeoExplorer-pristine` 的可视标签改为 `GeoExplorer`。

- 总览图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\c4_all_failed_wandering_routes\c4_all_failed_wandering_contact_sheet.png`
- 候选表：`F:\bishe\Undergraduate-Graduation-Project\results\tables\trajectory_analysis\c4_all_failed_wandering_candidates.csv`

| Rank | Case | Start | Goal | Backtrack | Revisit | Avg FinalDist | Figure |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `img017_d4_s09_g17_r0` | 9 | 17 | 8 | 9 | 2.67 | `F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\c4_all_failed_wandering_routes\c4_all_failed_wandering_01_img017_d4_s09_g17_r0.png` |
| 2 | `img085_d4_s08_g10_r0` | 8 | 10 | 7 | 8 | 2.67 | `F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\c4_all_failed_wandering_routes\c4_all_failed_wandering_02_img085_d4_s08_g10_r0.png` |
| 3 | `img082_d4_s15_g07_r0` | 15 | 7 | 2 | 3 | 3.33 | `F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\c4_all_failed_wandering_routes\c4_all_failed_wandering_03_img082_d4_s15_g07_r0.png` |
| 4 | `img072_d4_s14_g22_r0` | 14 | 22 | 9 | 9 | 2.67 | `F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\c4_all_failed_wandering_routes\c4_all_failed_wandering_04_img072_d4_s14_g22_r0.png` |
| 5 | `img056_d4_s14_g06_r0` | 14 | 6 | 8 | 8 | 3.33 | `F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\c4_all_failed_wandering_routes\c4_all_failed_wandering_05_img056_d4_s14_g06_r0.png` |
| 6 | `img081_d4_s04_g06_r0` | 4 | 6 | 7 | 7 | 2.67 | `F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\c4_all_failed_wandering_routes\c4_all_failed_wandering_06_img081_d4_s04_g06_r0.png` |
| 7 | `img057_d4_s21_g07_r0` | 21 | 7 | 7 | 9 | 2.67 | `F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\c4_all_failed_wandering_routes\c4_all_failed_wandering_07_img057_d4_s21_g07_r0.png` |
| 8 | `img053_d4_s23_g03_r0` | 23 | 3 | 3 | 4 | 2.00 | `F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\c4_all_failed_wandering_routes\c4_all_failed_wandering_08_img053_d4_s23_g03_r0.png` |
