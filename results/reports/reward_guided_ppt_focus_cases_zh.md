# PPT 聚焦版训练案例说明

本组图用于结题汇报中解释“混合奖励机制如何在训练阶段指导策略形成”。所有路线、距离和奖励分量均来自真实 `training_route_samples.csv`；每组图比较同一 seed、同一 episode、同一俯视图、同一起点和同一目标，只改变训练奖励设置。

图内文字已压缩到最少：标题、关键数字、图例和坐标轴。详细解释放在本文件中，适合直接复制到 PPT 备注或答辩讲稿。

图的阅读顺序：先看俯视图中 4 个代表方法是否形成清晰目标导向路线；再看曲线图，第一行是全部方法距离曲线，下面是同一行动步对齐的总奖励与三项奖励组成。蓝色始终表示本文方法。

重要表述：奖励、距离门控和 PBRS 只在训练阶段提供学习信号；正式测试或表格评估时只加载训练好的策略 checkpoint，不再调用奖励函数。

- 聚焦案例数：12
- 聚焦图目录：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases`
- 聚焦案例表：`F:\bishe\Undergraduate-Graduation-Project\results\tables\defense_reward_training_stage\reward_guided_ppt_focus_cases_selected.csv`
- 路线图总览：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\focus_case_contact_sheet.png`
- 曲线图总览：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\reward_case_contact_sheet.png`
- 路线图只展示 4 个代表方法：仅外部、外部+内在、线性门控、本文方法；其他对照不放在俯视图上，避免路线重复。
- 曲线图第一行保留全部 6 个方法的距离曲线；其下按同一行动步对齐展示每步总奖励、外部项、内在×系数和 PBRS。PBRS 使用独立面板，避免小数值被外部奖励淹没。

## 推荐优先放入 PPT 的案例

- `case_04`：远距离 C8，五个对照全部失败，训练早期已有方向优势。本文方法 8 步到达，对照平均终距 4.4。路线图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\focus_case_01_case_04_C8_seed123_ep120_img5_s4_g20.png`；奖励图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\reward_case_01_case_04_C8_seed123_ep120_img5_s4_g20.png`
- `case_08`：远距离 C8，五个对照全部失败，训练早期已有方向优势。本文方法 8 步到达，对照平均终距 3.6。路线图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\focus_case_02_case_08_C8_seed123_ep55_img6_s4_g20.png`；奖励图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\reward_case_02_case_08_C8_seed123_ep55_img6_s4_g20.png`
- `case_07`：远距离 C8，五个对照全部失败，对照接近目标但未完成，训练早期已有方向优势。本文方法 8 步到达，对照平均终距 2.6。路线图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\focus_case_03_case_07_C8_seed123_ep90_img15_s0_g24.png`；奖励图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\reward_case_03_case_07_C8_seed123_ep90_img15_s0_g24.png`
- `case_10`：远距离 C8，五个对照全部失败，对照接近目标但未完成。本文方法 8 步到达，对照平均终距 2.2。路线图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\focus_case_04_case_10_C8_seed123_ep250_img3_s4_g20.png`；奖励图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\reward_case_04_case_10_C8_seed123_ep250_img3_s4_g20.png`
- `case_19`：远距离 C8，4 个对照失败，多个对照出现回退/重复。本文方法 8 步到达，对照平均终距 4.0。路线图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\focus_case_05_case_19_C8_seed321_ep290_img1_s24_g0.png`；奖励图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\reward_case_05_case_19_C8_seed321_ep290_img1_s24_g0.png`
- `case_20`：远距离 C8，4 个对照失败，多个对照出现回退/重复。本文方法 8 步到达，对照平均终距 3.6。路线图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\focus_case_06_case_20_C8_seed123_ep310_img6_s20_g4.png`；奖励图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\reward_case_06_case_20_C8_seed123_ep310_img6_s20_g4.png`
- `case_02`：C7，五个对照全部失败，多个对照出现回退/重复，训练早期已有方向优势。本文方法 7 步到达，对照平均终距 4.2。路线图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\focus_case_07_case_02_C7_seed123_ep110_img15_s23_g0.png`；奖励图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\reward_case_07_case_02_C7_seed123_ep110_img15_s23_g0.png`
- `case_12`：C7，五个对照全部失败，多个对照出现回退/重复，训练早期已有方向优势。本文方法 9 步到达，对照平均终距 4.4。路线图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\focus_case_08_case_12_C7_seed123_ep75_img5_s4_g21.png`；奖励图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\reward_case_08_case_12_C7_seed123_ep75_img5_s4_g21.png`

## 单页讲解备注

### 01. case_04（C8）

- 路线图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\focus_case_01_case_04_C8_seed123_ep120_img5_s4_g20.png`
- 奖励图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\reward_case_01_case_04_C8_seed123_ep120_img5_s4_g20.png`
- 真实条件：seed=123，episode=120，训练进度 26.9%，起点=4，目标=20。
- 图上结论：本文方法 8 步到达目标，回退 0 次，重复访问 1 次；5/5 个对照方法未到达，对照平均终距 4.4。
- 行为证据：路线图只保留 4 个代表方法并做平行偏移，蓝色路线可直接看到连续接近目标；曲线图第一行保留所有方法，本文方法最终降到 0。对照路线停在目标外，且有 2 个对照出现明显回退或重复访问。
- 奖励解释：曲线图在同一行动步上对齐距离、总奖励、外部项、内在×系数和 PBRS。外部项主要惩罚无效移动并在到达/接近时给出正反馈；内在项提供探索收益；PBRS 虽然数值小，但在接近目标方向上持续给出形状信号。三者相加后，本文方法更容易把高奖励动作集中到“继续靠近目标”的动作序列上。
- 可强调点：这是最远距离桶 C8，更适合说明中长距离导航中的训练指导优势。
- 可强调点：训练早期已经出现方向差异，说明奖励设计不是只在后期挑 checkpoint 才显得好，而是在训练过程中就提供了更清晰的学习信号。
- 建议讲法：同一个起点和目标下，对照方法并不是完全不会移动，而是容易在目标附近回退、绕行或停住；本文方法的优势在于把探索和方向约束结合起来，使策略形成可连续执行的目标导向路线。
- 避免表述：不要说测试阶段继续使用 PBRS 或奖励函数；应说这些信号只参与训练，测试阶段只执行训练好的策略。

### 02. case_08（C8）

- 路线图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\focus_case_02_case_08_C8_seed123_ep55_img6_s4_g20.png`
- 奖励图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\reward_case_02_case_08_C8_seed123_ep55_img6_s4_g20.png`
- 真实条件：seed=123，episode=55，训练进度 12.5%，起点=4，目标=20。
- 图上结论：本文方法 8 步到达目标，回退 0 次，重复访问 1 次；5/5 个对照方法未到达，对照平均终距 3.6。
- 行为证据：路线图只保留 4 个代表方法并做平行偏移，蓝色路线可直接看到连续接近目标；曲线图第一行保留所有方法，本文方法最终降到 0。对照路线停在目标外，且有 1 个对照出现明显回退或重复访问。
- 奖励解释：曲线图在同一行动步上对齐距离、总奖励、外部项、内在×系数和 PBRS。外部项主要惩罚无效移动并在到达/接近时给出正反馈；内在项提供探索收益；PBRS 虽然数值小，但在接近目标方向上持续给出形状信号。三者相加后，本文方法更容易把高奖励动作集中到“继续靠近目标”的动作序列上。
- 可强调点：这是最远距离桶 C8，更适合说明中长距离导航中的训练指导优势。
- 可强调点：训练早期已经出现方向差异，说明奖励设计不是只在后期挑 checkpoint 才显得好，而是在训练过程中就提供了更清晰的学习信号。
- 建议讲法：同一个起点和目标下，对照方法并不是完全不会移动，而是容易在目标附近回退、绕行或停住；本文方法的优势在于把探索和方向约束结合起来，使策略形成可连续执行的目标导向路线。
- 避免表述：不要说测试阶段继续使用 PBRS 或奖励函数；应说这些信号只参与训练，测试阶段只执行训练好的策略。

### 03. case_07（C8）

- 路线图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\focus_case_03_case_07_C8_seed123_ep90_img15_s0_g24.png`
- 奖励图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\reward_case_03_case_07_C8_seed123_ep90_img15_s0_g24.png`
- 真实条件：seed=123，episode=90，训练进度 20.2%，起点=0，目标=24。
- 图上结论：本文方法 8 步到达目标，回退 0 次，重复访问 1 次；5/5 个对照方法未到达，对照平均终距 2.6。
- 行为证据：路线图只保留 4 个代表方法并做平行偏移，蓝色路线可直接看到连续接近目标；曲线图第一行保留所有方法，本文方法最终降到 0。对照路线停在目标外，且有 1 个对照出现明显回退或重复访问。
- 奖励解释：曲线图在同一行动步上对齐距离、总奖励、外部项、内在×系数和 PBRS。外部项主要惩罚无效移动并在到达/接近时给出正反馈；内在项提供探索收益；PBRS 虽然数值小，但在接近目标方向上持续给出形状信号。三者相加后，本文方法更容易把高奖励动作集中到“继续靠近目标”的动作序列上。
- 可强调点：有 3 个对照已经接近目标但没有真正到达，适合说明“接近目标”和“完成任务”之间仍需要稳定的方向塑形。
- 可强调点：这是最远距离桶 C8，更适合说明中长距离导航中的训练指导优势。
- 可强调点：训练早期已经出现方向差异，说明奖励设计不是只在后期挑 checkpoint 才显得好，而是在训练过程中就提供了更清晰的学习信号。
- 建议讲法：同一个起点和目标下，对照方法并不是完全不会移动，而是容易在目标附近回退、绕行或停住；本文方法的优势在于把探索和方向约束结合起来，使策略形成可连续执行的目标导向路线。
- 避免表述：不要说测试阶段继续使用 PBRS 或奖励函数；应说这些信号只参与训练，测试阶段只执行训练好的策略。

### 04. case_10（C8）

- 路线图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\focus_case_04_case_10_C8_seed123_ep250_img3_s4_g20.png`
- 奖励图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\reward_case_04_case_10_C8_seed123_ep250_img3_s4_g20.png`
- 真实条件：seed=123，episode=250，训练进度 55.5%，起点=4，目标=20。
- 图上结论：本文方法 8 步到达目标，回退 0 次，重复访问 1 次；5/5 个对照方法未到达，对照平均终距 2.2。
- 行为证据：路线图只保留 4 个代表方法并做平行偏移，蓝色路线可直接看到连续接近目标；曲线图第一行保留所有方法，本文方法最终降到 0。对照路线停在目标外，且有 1 个对照出现明显回退或重复访问。
- 奖励解释：曲线图在同一行动步上对齐距离、总奖励、外部项、内在×系数和 PBRS。外部项主要惩罚无效移动并在到达/接近时给出正反馈；内在项提供探索收益；PBRS 虽然数值小，但在接近目标方向上持续给出形状信号。三者相加后，本文方法更容易把高奖励动作集中到“继续靠近目标”的动作序列上。
- 可强调点：有 3 个对照已经接近目标但没有真正到达，适合说明“接近目标”和“完成任务”之间仍需要稳定的方向塑形。
- 可强调点：这是最远距离桶 C8，更适合说明中长距离导航中的训练指导优势。
- 建议讲法：同一个起点和目标下，对照方法并不是完全不会移动，而是容易在目标附近回退、绕行或停住；本文方法的优势在于把探索和方向约束结合起来，使策略形成可连续执行的目标导向路线。
- 避免表述：不要说测试阶段继续使用 PBRS 或奖励函数；应说这些信号只参与训练，测试阶段只执行训练好的策略。

### 05. case_19（C8）

- 路线图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\focus_case_05_case_19_C8_seed321_ep290_img1_s24_g0.png`
- 奖励图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\reward_case_05_case_19_C8_seed321_ep290_img1_s24_g0.png`
- 真实条件：seed=321，episode=290，训练进度 64.5%，起点=24，目标=0。
- 图上结论：本文方法 8 步到达目标，回退 0 次，重复访问 1 次；4/5 个对照方法未到达，对照平均终距 4.0。
- 行为证据：路线图只保留 4 个代表方法并做平行偏移，蓝色路线可直接看到连续接近目标；曲线图第一行保留所有方法，本文方法最终降到 0。对照路线停在目标外，且有 3 个对照出现明显回退或重复访问。
- 奖励解释：曲线图在同一行动步上对齐距离、总奖励、外部项、内在×系数和 PBRS。外部项主要惩罚无效移动并在到达/接近时给出正反馈；内在项提供探索收益；PBRS 虽然数值小，但在接近目标方向上持续给出形状信号。三者相加后，本文方法更容易把高奖励动作集中到“继续靠近目标”的动作序列上。
- 可强调点：这是最远距离桶 C8，更适合说明中长距离导航中的训练指导优势。
- 建议讲法：同一个起点和目标下，对照方法并不是完全不会移动，而是容易在目标附近回退、绕行或停住；本文方法的优势在于把探索和方向约束结合起来，使策略形成可连续执行的目标导向路线。
- 避免表述：不要说测试阶段继续使用 PBRS 或奖励函数；应说这些信号只参与训练，测试阶段只执行训练好的策略。

### 06. case_20（C8）

- 路线图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\focus_case_06_case_20_C8_seed123_ep310_img6_s20_g4.png`
- 奖励图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\reward_case_06_case_20_C8_seed123_ep310_img6_s20_g4.png`
- 真实条件：seed=123，episode=310，训练进度 68.6%，起点=20，目标=4。
- 图上结论：本文方法 8 步到达目标，回退 0 次，重复访问 1 次；4/5 个对照方法未到达，对照平均终距 3.6。
- 行为证据：路线图只保留 4 个代表方法并做平行偏移，蓝色路线可直接看到连续接近目标；曲线图第一行保留所有方法，本文方法最终降到 0。对照路线停在目标外，且有 3 个对照出现明显回退或重复访问。
- 奖励解释：曲线图在同一行动步上对齐距离、总奖励、外部项、内在×系数和 PBRS。外部项主要惩罚无效移动并在到达/接近时给出正反馈；内在项提供探索收益；PBRS 虽然数值小，但在接近目标方向上持续给出形状信号。三者相加后，本文方法更容易把高奖励动作集中到“继续靠近目标”的动作序列上。
- 可强调点：这是最远距离桶 C8，更适合说明中长距离导航中的训练指导优势。
- 建议讲法：同一个起点和目标下，对照方法并不是完全不会移动，而是容易在目标附近回退、绕行或停住；本文方法的优势在于把探索和方向约束结合起来，使策略形成可连续执行的目标导向路线。
- 避免表述：不要说测试阶段继续使用 PBRS 或奖励函数；应说这些信号只参与训练，测试阶段只执行训练好的策略。

### 07. case_02（C7）

- 路线图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\focus_case_07_case_02_C7_seed123_ep110_img15_s23_g0.png`
- 奖励图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\reward_case_07_case_02_C7_seed123_ep110_img15_s23_g0.png`
- 真实条件：seed=123，episode=110，训练进度 24.7%，起点=23，目标=0。
- 图上结论：本文方法 7 步到达目标，回退 0 次，重复访问 1 次；5/5 个对照方法未到达，对照平均终距 4.2。
- 行为证据：路线图只保留 4 个代表方法并做平行偏移，蓝色路线可直接看到连续接近目标；曲线图第一行保留所有方法，本文方法最终降到 0。对照路线停在目标外，且有 3 个对照出现明显回退或重复访问。
- 奖励解释：曲线图在同一行动步上对齐距离、总奖励、外部项、内在×系数和 PBRS。外部项主要惩罚无效移动并在到达/接近时给出正反馈；内在项提供探索收益；PBRS 虽然数值小，但在接近目标方向上持续给出形状信号。三者相加后，本文方法更容易把高奖励动作集中到“继续靠近目标”的动作序列上。
- 可强调点：训练早期已经出现方向差异，说明奖励设计不是只在后期挑 checkpoint 才显得好，而是在训练过程中就提供了更清晰的学习信号。
- 建议讲法：同一个起点和目标下，对照方法并不是完全不会移动，而是容易在目标附近回退、绕行或停住；本文方法的优势在于把探索和方向约束结合起来，使策略形成可连续执行的目标导向路线。
- 避免表述：不要说测试阶段继续使用 PBRS 或奖励函数；应说这些信号只参与训练，测试阶段只执行训练好的策略。

### 08. case_12（C7）

- 路线图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\focus_case_08_case_12_C7_seed123_ep75_img5_s4_g21.png`
- 奖励图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\reward_case_08_case_12_C7_seed123_ep75_img5_s4_g21.png`
- 真实条件：seed=123，episode=75，训练进度 16.9%，起点=4，目标=21。
- 图上结论：本文方法 9 步到达目标，回退 1 次，重复访问 1 次；5/5 个对照方法未到达，对照平均终距 4.4。
- 行为证据：路线图只保留 4 个代表方法并做平行偏移，蓝色路线可直接看到连续接近目标；曲线图第一行保留所有方法，本文方法最终降到 0。对照路线停在目标外，且有 5 个对照出现明显回退或重复访问。
- 奖励解释：曲线图在同一行动步上对齐距离、总奖励、外部项、内在×系数和 PBRS。外部项主要惩罚无效移动并在到达/接近时给出正反馈；内在项提供探索收益；PBRS 虽然数值小，但在接近目标方向上持续给出形状信号。三者相加后，本文方法更容易把高奖励动作集中到“继续靠近目标”的动作序列上。
- 可强调点：训练早期已经出现方向差异，说明奖励设计不是只在后期挑 checkpoint 才显得好，而是在训练过程中就提供了更清晰的学习信号。
- 建议讲法：同一个起点和目标下，对照方法并不是完全不会移动，而是容易在目标附近回退、绕行或停住；本文方法的优势在于把探索和方向约束结合起来，使策略形成可连续执行的目标导向路线。
- 避免表述：不要说测试阶段继续使用 PBRS 或奖励函数；应说这些信号只参与训练，测试阶段只执行训练好的策略。

### 09. case_01（C6）

- 路线图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\focus_case_09_case_01_C6_seed123_ep280_img3_s10_g4.png`
- 奖励图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\reward_case_09_case_01_C6_seed123_ep280_img3_s10_g4.png`
- 真实条件：seed=123，episode=280，训练进度 62.0%，起点=10，目标=4。
- 图上结论：本文方法 6 步到达目标，回退 0 次，重复访问 1 次；5/5 个对照方法未到达，对照平均终距 4.2。
- 行为证据：路线图只保留 4 个代表方法并做平行偏移，蓝色路线可直接看到连续接近目标；曲线图第一行保留所有方法，本文方法最终降到 0。对照路线停在目标外，且有 5 个对照出现明显回退或重复访问。
- 奖励解释：曲线图在同一行动步上对齐距离、总奖励、外部项、内在×系数和 PBRS。外部项主要惩罚无效移动并在到达/接近时给出正反馈；内在项提供探索收益；PBRS 虽然数值小，但在接近目标方向上持续给出形状信号。三者相加后，本文方法更容易把高奖励动作集中到“继续靠近目标”的动作序列上。
- 建议讲法：同一个起点和目标下，对照方法并不是完全不会移动，而是容易在目标附近回退、绕行或停住；本文方法的优势在于把探索和方向约束结合起来，使策略形成可连续执行的目标导向路线。
- 避免表述：不要说测试阶段继续使用 PBRS 或奖励函数；应说这些信号只参与训练，测试阶段只执行训练好的策略。

### 10. case_03（C6）

- 路线图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\focus_case_10_case_03_C6_seed321_ep150_img13_s15_g3.png`
- 奖励图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\reward_case_10_case_03_C6_seed321_ep150_img13_s15_g3.png`
- 真实条件：seed=321，episode=150，训练进度 33.7%，起点=15，目标=3。
- 图上结论：本文方法 6 步到达目标，回退 0 次，重复访问 1 次；5/5 个对照方法未到达，对照平均终距 4.4。
- 行为证据：路线图只保留 4 个代表方法并做平行偏移，蓝色路线可直接看到连续接近目标；曲线图第一行保留所有方法，本文方法最终降到 0。对照路线停在目标外，且有 4 个对照出现明显回退或重复访问。
- 奖励解释：曲线图在同一行动步上对齐距离、总奖励、外部项、内在×系数和 PBRS。外部项主要惩罚无效移动并在到达/接近时给出正反馈；内在项提供探索收益；PBRS 虽然数值小，但在接近目标方向上持续给出形状信号。三者相加后，本文方法更容易把高奖励动作集中到“继续靠近目标”的动作序列上。
- 建议讲法：同一个起点和目标下，对照方法并不是完全不会移动，而是容易在目标附近回退、绕行或停住；本文方法的优势在于把探索和方向约束结合起来，使策略形成可连续执行的目标导向路线。
- 避免表述：不要说测试阶段继续使用 PBRS 或奖励函数；应说这些信号只参与训练，测试阶段只执行训练好的策略。

### 11. case_05（C6）

- 路线图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\focus_case_11_case_05_C6_seed42_ep345_img16_s1_g23.png`
- 奖励图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\reward_case_11_case_05_C6_seed42_ep345_img16_s1_g23.png`
- 真实条件：seed=42，episode=345，训练进度 76.2%，起点=1，目标=23。
- 图上结论：本文方法 6 步到达目标，回退 0 次，重复访问 1 次；5/5 个对照方法未到达，对照平均终距 3.4。
- 行为证据：路线图只保留 4 个代表方法并做平行偏移，蓝色路线可直接看到连续接近目标；曲线图第一行保留所有方法，本文方法最终降到 0。对照路线停在目标外，且有 4 个对照出现明显回退或重复访问。
- 奖励解释：曲线图在同一行动步上对齐距离、总奖励、外部项、内在×系数和 PBRS。外部项主要惩罚无效移动并在到达/接近时给出正反馈；内在项提供探索收益；PBRS 虽然数值小，但在接近目标方向上持续给出形状信号。三者相加后，本文方法更容易把高奖励动作集中到“继续靠近目标”的动作序列上。
- 可强调点：有 1 个对照已经接近目标但没有真正到达，适合说明“接近目标”和“完成任务”之间仍需要稳定的方向塑形。
- 建议讲法：同一个起点和目标下，对照方法并不是完全不会移动，而是容易在目标附近回退、绕行或停住；本文方法的优势在于把探索和方向约束结合起来，使策略形成可连续执行的目标导向路线。
- 避免表述：不要说测试阶段继续使用 PBRS 或奖励函数；应说这些信号只参与训练，测试阶段只执行训练好的策略。

### 12. case_18（C6）

- 路线图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\focus_case_12_case_18_C6_seed321_ep140_img12_s16_g4.png`
- 奖励图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\reward_case_12_case_18_C6_seed321_ep140_img12_s16_g4.png`
- 真实条件：seed=321，episode=140，训练进度 31.5%，起点=16，目标=4。
- 图上结论：本文方法 8 步到达目标，回退 1 次，重复访问 2 次；5/5 个对照方法未到达，对照平均终距 3.8。
- 行为证据：路线图只保留 4 个代表方法并做平行偏移，蓝色路线可直接看到连续接近目标；曲线图第一行保留所有方法，本文方法最终降到 0。对照路线停在目标外，且有 5 个对照出现明显回退或重复访问。
- 奖励解释：曲线图在同一行动步上对齐距离、总奖励、外部项、内在×系数和 PBRS。外部项主要惩罚无效移动并在到达/接近时给出正反馈；内在项提供探索收益；PBRS 虽然数值小，但在接近目标方向上持续给出形状信号。三者相加后，本文方法更容易把高奖励动作集中到“继续靠近目标”的动作序列上。
- 建议讲法：同一个起点和目标下，对照方法并不是完全不会移动，而是容易在目标附近回退、绕行或停住；本文方法的优势在于把探索和方向约束结合起来，使策略形成可连续执行的目标导向路线。
- 避免表述：不要说测试阶段继续使用 PBRS 或奖励函数；应说这些信号只参与训练，测试阶段只执行训练好的策略。


## 汇报话术

“这几张图不是测试阶段额外使用奖励函数，而是把训练日志中的典型片段可视化出来。每个案例分成两页：第一页只看俯视图路线，第二页把距离曲线和奖励组成按行动步对齐。路线图只保留四个代表方法，并用平行线把重叠路线分开；曲线图仍保留所有方法。可以看到本文方法的蓝色路线持续接近目标，距离曲线下降到 0；而对照方法往往在目标附近回退、重复访问，或者停在目标外。奖励曲线进一步说明，本文方法不是简单把某一项奖励放大，而是把外部惩罚、门控内在奖励和 PBRS 方向塑形组合起来，使训练阶段更容易把高奖励动作分配给连续靠近目标的动作序列。”

## 全部输出

- 路线图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\focus_case_01_case_04_C8_seed123_ep120_img5_s4_g20.png`
- 奖励图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\reward_case_01_case_04_C8_seed123_ep120_img5_s4_g20.png`
- 路线图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\focus_case_02_case_08_C8_seed123_ep55_img6_s4_g20.png`
- 奖励图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\reward_case_02_case_08_C8_seed123_ep55_img6_s4_g20.png`
- 路线图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\focus_case_03_case_07_C8_seed123_ep90_img15_s0_g24.png`
- 奖励图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\reward_case_03_case_07_C8_seed123_ep90_img15_s0_g24.png`
- 路线图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\focus_case_04_case_10_C8_seed123_ep250_img3_s4_g20.png`
- 奖励图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\reward_case_04_case_10_C8_seed123_ep250_img3_s4_g20.png`
- 路线图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\focus_case_05_case_19_C8_seed321_ep290_img1_s24_g0.png`
- 奖励图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\reward_case_05_case_19_C8_seed321_ep290_img1_s24_g0.png`
- 路线图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\focus_case_06_case_20_C8_seed123_ep310_img6_s20_g4.png`
- 奖励图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\reward_case_06_case_20_C8_seed123_ep310_img6_s20_g4.png`
- 路线图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\focus_case_07_case_02_C7_seed123_ep110_img15_s23_g0.png`
- 奖励图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\reward_case_07_case_02_C7_seed123_ep110_img15_s23_g0.png`
- 路线图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\focus_case_08_case_12_C7_seed123_ep75_img5_s4_g21.png`
- 奖励图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\reward_case_08_case_12_C7_seed123_ep75_img5_s4_g21.png`
- 路线图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\focus_case_09_case_01_C6_seed123_ep280_img3_s10_g4.png`
- 奖励图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\reward_case_09_case_01_C6_seed123_ep280_img3_s10_g4.png`
- 路线图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\focus_case_10_case_03_C6_seed321_ep150_img13_s15_g3.png`
- 奖励图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\reward_case_10_case_03_C6_seed321_ep150_img13_s15_g3.png`
- 路线图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\focus_case_11_case_05_C6_seed42_ep345_img16_s1_g23.png`
- 奖励图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\reward_case_11_case_05_C6_seed42_ep345_img16_s1_g23.png`
- 路线图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\focus_case_12_case_18_C6_seed321_ep140_img12_s16_g4.png`
- 奖励图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases\reward_case_12_case_18_C6_seed321_ep140_img12_s16_g4.png`