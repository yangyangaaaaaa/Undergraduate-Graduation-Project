# 验收可视化包说明

这一版按用户反馈重做：路线类 GIF 和图片统一为左侧大路线图、右侧 2x2 四宫格（目标线索、当前观察、起点位置、距离曲线），去掉左上角小字幕；路线样本同时包含平顺成功、不平顺成功和少量不平顺绕行/失败代表。

## 动态路线 GIF

- `c4_smooth_success`: case `img011_d4_s03_g11_r0`, success `True`, curve `up=0, turns=0, final_d=0`
- `c4_smooth_success`: GIF `results\figures\acceptance_demo\acceptance_c4_smooth_success.gif`；末帧 `results\figures\acceptance_demo\acceptance_c4_smooth_success_poster.png`
- `c4_wavy_success`: case `img044_d4_s07_g19_r0`, success `True`, curve `up=2, turns=4, final_d=0`
- `c4_wavy_success`: GIF `results\figures\acceptance_demo\acceptance_c4_wavy_success.gif`；末帧 `results\figures\acceptance_demo\acceptance_c4_wavy_success_poster.png`
- `c4_wavy_success_02`: case `img113_d4_s21_g13_r0`, success `True`, curve `up=2, turns=4, final_d=0`
- `c4_wavy_success_02`: GIF `results\figures\acceptance_demo\acceptance_c4_wavy_success_02.gif`；末帧 `results\figures\acceptance_demo\acceptance_c4_wavy_success_02_poster.png`
- `c4_wavy_detour`: case `img050_d4_s16_g00_r0`, success `False`, curve `up=7, turns=6, final_d=8`
- `c4_wavy_detour`: GIF `results\figures\acceptance_demo\acceptance_c4_wavy_detour.gif`；末帧 `results\figures\acceptance_demo\acceptance_c4_wavy_detour_poster.png`
- `c6_smooth_success`: case `img006_d6_s24_g06_r0`, success `True`, curve `up=0, turns=0, final_d=0`
- `c6_smooth_success`: GIF `results\figures\acceptance_demo\acceptance_c6_smooth_success.gif`；末帧 `results\figures\acceptance_demo\acceptance_c6_smooth_success_poster.png`
- `c6_wavy_success`: case `img005_d6_s19_g05_r0`, success `True`, curve `up=2, turns=4, final_d=0`
- `c6_wavy_success`: GIF `results\figures\acceptance_demo\acceptance_c6_wavy_success.gif`；末帧 `results\figures\acceptance_demo\acceptance_c6_wavy_success_poster.png`
- `c6_wavy_success_02`: case `img069_d6_s01_g19_r0`, success `True`, curve `up=2, turns=4, final_d=0`
- `c6_wavy_success_02`: GIF `results\figures\acceptance_demo\acceptance_c6_wavy_success_02.gif`；末帧 `results\figures\acceptance_demo\acceptance_c6_wavy_success_02_poster.png`
- `c6_wavy_detour`: case `img203_d6_s15_g03_r0`, success `False`, curve `up=4, turns=7, final_d=4`
- `c6_wavy_detour`: GIF `results\figures\acceptance_demo\acceptance_c6_wavy_detour.gif`；末帧 `results\figures\acceptance_demo\acceptance_c6_wavy_detour_poster.png`
- `c8_smooth_success`: case `img000_d8_s24_g00_r0`, success `True`, curve `up=0, turns=0, final_d=0`
- `c8_smooth_success`: GIF `results\figures\acceptance_demo\acceptance_c8_smooth_success.gif`；末帧 `results\figures\acceptance_demo\acceptance_c8_smooth_success_poster.png`
- `c8_wavy_success`: case `img129_d8_s20_g04_r0`, success `True`, curve `up=1, turns=2, final_d=0`
- `c8_wavy_success`: GIF `results\figures\acceptance_demo\acceptance_c8_wavy_success.gif`；末帧 `results\figures\acceptance_demo\acceptance_c8_wavy_success_poster.png`
- `c8_wavy_success_02`: case `img025_d8_s24_g00_r0`, success `True`, curve `up=1, turns=2, final_d=0`
- `c8_wavy_success_02`: GIF `results\figures\acceptance_demo\acceptance_c8_wavy_success_02.gif`；末帧 `results\figures\acceptance_demo\acceptance_c8_wavy_success_02_poster.png`
- `c8_wavy_detour`: case `img204_d8_s20_g04_r0`, success `False`, curve `up=3, turns=6, final_d=4`
- `c8_wavy_detour`: GIF `results\figures\acceptance_demo\acceptance_c8_wavy_detour.gif`；末帧 `results\figures\acceptance_demo\acceptance_c8_wavy_detour_poster.png`
- `three_method_hardcase`: case `img189_d6_s20_g14_r0`, success `True`, curve `up=2, turns=4, final_d=0`
- `three_method_hardcase`: GIF `results\figures\acceptance_demo\acceptance_three_method_hardcase.gif`；末帧 `results\figures\acceptance_demo\acceptance_three_method_hardcase_poster.png`

## 实验设置动态路线 GIF

- `xbd_pre_route_setting`: GIF `results\figures\acceptance_demo\acceptance_xbd_pre_route_setting.gif`；末帧 `results\figures\acceptance_demo\acceptance_xbd_pre_route_setting_poster.png`
- `xbd_disaster_route_setting`: GIF `results\figures\acceptance_demo\acceptance_xbd_disaster_route_setting.gif`；末帧 `results\figures\acceptance_demo\acceptance_xbd_disaster_route_setting_poster.png`
- `mmgag_aerial_route_setting`: GIF `results\figures\acceptance_demo\acceptance_mmgag_aerial_route_setting.gif`；末帧 `results\figures\acceptance_demo\acceptance_mmgag_aerial_route_setting_poster.png`
- `mmgag_ground_route_setting`: GIF `results\figures\acceptance_demo\acceptance_mmgag_ground_route_setting.gif`；末帧 `results\figures\acceptance_demo\acceptance_mmgag_ground_route_setting_poster.png`
- `mmgag_text_route_setting`: GIF `results\figures\acceptance_demo\acceptance_mmgag_text_route_setting.gif`；末帧 `results\figures\acceptance_demo\acceptance_mmgag_text_route_setting_poster.png`
- `ultralong_grid8_route_setting`: GIF `results\figures\acceptance_demo\acceptance_ultralong_grid8_route_setting.gif`；末帧 `results\figures\acceptance_demo\acceptance_ultralong_grid8_route_setting_poster.png`
- `ultralong_grid10_route_setting`: GIF `results\figures\acceptance_demo\acceptance_ultralong_grid10_route_setting.gif`；末帧 `results\figures\acceptance_demo\acceptance_ultralong_grid10_route_setting_poster.png`

## xBD 路线设置图

- `xbd_pre_route_setting`: `results\figures\acceptance_demo\acceptance_xbd_pre_route_setting.png`
- `xbd_disaster_route_setting`: `results\figures\acceptance_demo\acceptance_xbd_disaster_route_setting.png`
- 两种设置对照: `results\figures\acceptance_demo\acceptance_xbd_route_settings_compare.png`

## 其他论文实验验收图

- `route_gallery`: `results\figures\acceptance_demo\acceptance_route_gallery.png`
- `xbd_compare`: `results\figures\acceptance_demo\acceptance_xbd_route_settings_compare.png`
- `mmgag_multimodal`: `results\figures\acceptance_demo\acceptance_mmgag_multimodal_route_setting.png`
- `main_benchmark`: `results\figures\acceptance_demo\acceptance_main_benchmark_effects.png`
- `factorial`: `results\figures\acceptance_demo\acceptance_factorial_ablation_effects.png`
- `reward`: `results\figures\acceptance_demo\acceptance_reward_gate_pbrs_effects.png`
- `dataset_param`: `results\figures\acceptance_demo\acceptance_dataset_parameter_effects.png`
- `budget`: `results\figures\acceptance_demo\acceptance_budget_stress_effects.png`
- `continuation_status`: `results\figures\acceptance_demo\acceptance_continuation_status.png`
- `index`: `results\figures\acceptance_demo\acceptance_visual_pack_index.png`

## 证据边界

- xBD 图使用真实灾前/灾后影像展示 5x5 搜索设置，并引用正式评估表的 SR/SG；不冒充为未记录的逐样本策略轨迹。
- reward/gate/PBRS 只作为训练阶段机制说明；测试阶段只加载训练后的 checkpoint 做策略评估。
- continuation 页面只是状态页；必须等固定评估合并 CSV 完成后，才能画最终续训曲线并下结论。