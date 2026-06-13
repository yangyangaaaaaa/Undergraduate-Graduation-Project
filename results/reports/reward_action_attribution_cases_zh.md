# 训练动作归因可视化说明

这组图补充现有路线图和奖励曲线，目标是让评委直接看到：同一个训练样本中，本文方法为什么更容易把动作序列推向目标。

新版图内只保留识别所需的短标签、数字和缩写，不放解释性长句；详细解释放在本文件中，适合作为 PPT 备注。

重要表述：奖励、距离门控和 PBRS 只在训练阶段提供学习信号；正式测试或论文表格评估时只加载训练好的策略 checkpoint，不再调用奖励函数。

## 读图顺序

1. 页眉只保留 `Case / C 距离 / Ours 1 / Ctrl 0-5 / 训练进度`。其中 `Ours 1` 表示本文方法到达，`Ctrl 0/5` 表示五个对照方法均未到达。
2. 先看左上角真实俯视图：蓝色实线是本文方法，对照方法为虚线；起点为绿色圆点，目标为黄色星标，未到达终点用红色叉号标出。
3. 再看右上角 `Dist` 曲线：蓝线是否持续下降到 0，用来判断路线是否真正形成目标导向。
4. 最后看下方每步奖励账本：`Act` 为动作，`Dist` 为距离变化，`Ext` 为外部奖励，`Int*g` 为门控后的内在奖励，`PBRS` 为势函数塑形项，`Total` 为总奖励。绿色表示靠近目标或正反馈，红色表示回退或惩罚。

## 推荐使用方式

- PPT 主线：先放训练趋势图说明整体效果，再放 1-2 页动作归因图说明机制，最后放路线图/表格补充结果。
- 最推荐优先使用 `case_04` 或 `case_08`：它们是 C8 远距离样例，五个对照均未到达，本文方法到达。
- 若需要说明“走出回退/循环”，使用 `case_12`：对照方法多次回退，本文方法虽然有一次回退但能恢复并到达。

## 输出文件

- 总览图：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\action_attribution\action_attribution_contact_sheet.png`
- `case_04`：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\action_attribution\action_attribution_01_case_04_C8_seed123_ep120_img5_s4_g20.png`；C8，训练进度 26.9%，本文方法到达，5/5 对照未到达。
- `case_08`：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\action_attribution\action_attribution_02_case_08_C8_seed123_ep55_img6_s4_g20.png`；C8，训练进度 12.5%，本文方法到达，5/5 对照未到达。
- `case_07`：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\action_attribution\action_attribution_03_case_07_C8_seed123_ep90_img15_s0_g24.png`；C8，训练进度 20.2%，本文方法到达，5/5 对照未到达。
- `case_12`：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\action_attribution\action_attribution_04_case_12_C7_seed123_ep75_img5_s4_g21.png`；C7，训练进度 16.9%，本文方法到达，5/5 对照未到达。
- `case_02`：`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\action_attribution\action_attribution_05_case_02_C7_seed123_ep110_img15_s23_g0.png`；C7，训练进度 24.7%，本文方法到达，5/5 对照未到达。

## 评委视角的证据链

1. 趋势层：训练检查点曲线证明本文方法在 C8 中长距离任务上形成更好的模型检查点。
2. 机制层：动作归因图证明三项奖励不是抽象公式，而是在每一步训练样本上共同改变动作反馈。
3. 行为层：真实俯视图路线证明这种反馈最终表现为更连续的目标接近轨迹。
4. 结果层：正式表格仍作为最终性能依据，避免把训练阶段奖励图误解成测试阶段额外信息。

## 答辩话术

“这页不是测试阶段额外使用奖励，而是把训练日志中的一个真实片段拆开。上面先看行为：同一起点和目标下，对照方法容易停在目标外或回退，本文方法最终到达。下面再看训练信号：每一步动作的距离变化与外部奖励、门控内在奖励、PBRS 都对齐在同一列。可以看到，外部奖励负责惩罚无效移动和奖励接近目标，门控内在奖励保留远距离探索信号，PBRS 给接近目标的移动提供连续塑形。三项信号合在一起后，策略更容易把高回报分配给连续靠近目标的动作序列。正式测试时不再调用这些奖励函数，只执行已经学习好的策略。”
