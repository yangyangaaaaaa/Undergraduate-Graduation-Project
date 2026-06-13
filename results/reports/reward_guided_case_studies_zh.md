# 训练阶段奖励引导典型案例说明

本批图用于答辩解释混合奖励机制的训练阶段作用。所有案例均来自 `training_route_samples.csv` 的真实训练采样记录，并且比较的是同一 seed、同一 episode、同一图像、同一起点和同一目标。

图中路线叠加在实际俯视图上；底部距离曲线展示每一步是否靠近目标；奖励分解只展示本文方法在训练阶段收到的外部惩罚、门控内在奖励、PBRS 方向信号和总奖励。

重要表述：奖励、距离门控和 PBRS 只用于训练阶段指导 PPO 学习。正式测试或论文表格评估时，不再调用奖励函数，而是加载训练好的策略 checkpoint 选择动作。

- 选中案例数：24
- 图像质量筛选：整图白色空白占比 ≤ 0.01，单个网格白色空白占比 ≤ 0.08
- 选中表：`F:\bishe\Undergraduate-Graduation-Project\results\tables\defense_reward_training_stage\reward_guided_case_studies_selected.csv`
- 图像质量表：`F:\bishe\Undergraduate-Graduation-Project\results\tables\defense_reward_training_stage\reward_guided_case_studies_image_quality.csv`
- 图片目录：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\case_studies`

## 推荐讲解顺序

- `case_01`：C6，训练进度 62.0%；本文方法到达目标，5 个对照未到达，适合说明奖励信号能把中长距离动作推向正确方向。
- `case_02`：C7，训练进度 24.7%；本文方法到达目标，5 个对照未到达，适合说明奖励信号能把中长距离动作推向正确方向。
- `case_03`：C6，训练进度 33.7%；本文方法到达目标，5 个对照未到达，适合说明奖励信号能把中长距离动作推向正确方向。
- `case_04`：C8，训练进度 26.9%；本文方法到达目标，5 个对照未到达，适合说明奖励信号能把中长距离动作推向正确方向。
- `case_05`：C6，训练进度 76.2%；本文方法到达目标，5 个对照未到达，适合说明奖励信号能把中长距离动作推向正确方向。
- `case_06`：C6，训练进度 26.9%；本文方法到达目标，5 个对照未到达，适合说明奖励信号能把中长距离动作推向正确方向。
- `case_07`：C8，训练进度 20.2%；本文方法到达目标，5 个对照未到达，适合说明奖励信号能把中长距离动作推向正确方向。
- `case_08`：C8，训练进度 12.5%；本文方法到达目标，5 个对照未到达，适合说明奖励信号能把中长距离动作推向正确方向。
- `case_09`：C7，训练进度 48.8%；本文方法到达目标，5 个对照未到达，适合说明奖励信号能把中长距离动作推向正确方向。
- `case_10`：C8，训练进度 55.5%；本文方法到达目标，5 个对照未到达，适合说明奖励信号能把中长距离动作推向正确方向。
- `case_11`：C7，训练进度 25.8%；本文方法到达目标，5 个对照未到达，适合说明奖励信号能把中长距离动作推向正确方向。
- `case_12`：C7，训练进度 16.9%；本文方法到达目标，5 个对照未到达，适合说明奖励信号能把中长距离动作推向正确方向。

## 输出图片

- `F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\case_studies\case_01_seed123_ep280_img3_C6_s10_g4.png`
- `F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\case_studies\case_02_seed123_ep110_img15_C7_s23_g0.png`
- `F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\case_studies\case_03_seed321_ep150_img13_C6_s15_g3.png`
- `F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\case_studies\case_04_seed123_ep120_img5_C8_s4_g20.png`
- `F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\case_studies\case_05_seed42_ep345_img16_C6_s1_g23.png`
- `F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\case_studies\case_06_seed123_ep120_img14_C6_s22_g4.png`
- `F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\case_studies\case_07_seed123_ep90_img15_C8_s0_g24.png`
- `F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\case_studies\case_08_seed123_ep55_img6_C8_s4_g20.png`
- `F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\case_studies\case_09_seed123_ep220_img3_C7_s15_g4.png`
- `F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\case_studies\case_10_seed123_ep250_img3_C8_s4_g20.png`
- `F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\case_studies\case_11_seed123_ep115_img5_C7_s5_g24.png`
- `F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\case_studies\case_12_seed123_ep75_img5_C7_s4_g21.png`
- `F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\case_studies\case_13_seed123_ep390_img4_C6_s15_g3.png`
- `F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\case_studies\case_14_seed123_ep155_img2_C7_s4_g15.png`
- `F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\case_studies\case_15_seed123_ep385_img3_C6_s14_g20.png`
- `F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\case_studies\case_16_seed123_ep60_img1_C6_s5_g19.png`
- `F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\case_studies\case_17_seed321_ep55_img20_C6_s15_g3.png`
- `F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\case_studies\case_18_seed321_ep140_img12_C6_s16_g4.png`
- `F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\case_studies\case_19_seed321_ep290_img1_C8_s24_g0.png`
- `F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\case_studies\case_20_seed123_ep310_img6_C8_s20_g4.png`
- `F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\case_studies\case_21_seed42_ep310_img5_C7_s3_g20.png`
- `F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\case_studies\case_22_seed123_ep310_img1_C8_s20_g4.png`
- `F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\case_studies\case_23_seed123_ep100_img0_C7_s24_g5.png`
- `F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\case_studies\case_24_seed321_ep220_img16_C8_s4_g20.png`