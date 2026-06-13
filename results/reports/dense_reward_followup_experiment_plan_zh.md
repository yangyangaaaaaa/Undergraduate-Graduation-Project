# 密集奖励趋势补充实验建议

## 2026-06-06 最新判断

当前 `480k` 密集 checkpoint 固定评估已经完成，4 个方法各有 `46` 个真实 checkpoint 点。本文方法在 MM-GAG C=8 上取得最佳 mean SR `92.62%`，最强基线为外部+PBRS `87.38%`；最弱模态成功率本文方法 `92.34%`，最强基线 `84.26%`；平均剩余距离本文方法 `0.227` 格，也优于所有基线。

但这条曲线存在明显展示风险：本文方法主要在最后 checkpoint 超过基线，中前期并不占优，评委可能认为最后上升是偶然点。因此已经部署 `480k -> 960k` 续训实验，而不是继续包装当前曲线。续训实验保持训练集、seed、四个方法不变，从各方法 480k 末端 `geoexplorer_450.pt` 继续训练，目标是检验本文方法优势是否能在后续连续 checkpoint 中保持。

当前远程续训实验：

- 远程实验根目录：`/root/geoexplorer/ab_experiments/defense_reward_continuation_20260606/mmgag_c8_from480k_to960k_seed321`
- 远程 checkpoint 根目录：`/root/geoexplorer/results/checkpoint/defense_reward_continuation_20260606/mmgag_c8_from480k_to960k`
- 远程固定评估输出：`/root/geoexplorer/analysis/pipeline_20260606_dense_mmgag_continuation_trend`
- 本地监控脚本：`F:\bishe\Undergraduate-Graduation-Project\code\tools\run_dense_continuation_monitor_once.py`
- 本地续训曲线脚本：`F:\bishe\Undergraduate-Graduation-Project\code\tools\build_dense_mmgag_continuation_curves.py`
- 自动化：小时级本地 cron `续训 checkpoint 监控`

## 当前 480k 实验口径

当前已完成的 `mmgag_c8_dense_checkpoint_seed321_480k` 训练使用 4 个 reward-gate 方法、同一 seed、同一训练集、同一验证/测试拆分，并每 10 个 episode 保存一个 checkpoint。它适合回答：

- 不同奖励机制在训练过程中什么时候形成中长距离能力。
- 本文方法是否更快到达较高成功率区间。
- loss、entropy、KL 是否显示稳定收敛。

注意：训练日志里的 validation success 不能单独作为最终方法排名。最终主趋势仍以固定 checkpoint 的 MM-GAG C8 评估为准。

## 是否拉长训练步长

最新判断下，已启动同训练集续训，不先扩大训练集。下面的 720k/960k 从头训练方案降级为备用材料。

推荐优先级：

1. `720k`：成本较低，足够观察 480k 以后是否继续提升或进入平台期。
2. `960k`：更适合画完整 loss 收敛和后期稳定性，但时间更长，过拟合段需要在图中弱化。

图形口径：

- 成功率主图使用 best-so-far envelope，突出“达到最高值前的上升速度”和“最高值”。
- 后期若有下降或震荡，不作为主结论，只放在 loss/entropy/KL 辅助图里解释稳定性或过拟合风险。
- 不把 720k/960k 结果和论文正式表格混为同一口径，定位为答辩可视化补充实验。

## 是否扩大训练集

当前训练集已经是 184 个图像的 MASA+MM-GAG 混合集：

- MASA train：137 个图像。
- MM-GAG：47 个图像。
- MASA validation：4 个图像。
- MASA test：10 个图像。

因此扩大训练集时不能把 validation/test 加入训练，也不能重复添加已有 MM-GAG。建议采用以下两种安全方案之一：

### 方案 A：保持训练集不变，只拉长步长

这是最稳妥的方案。变量只有训练长度，图形解释最干净。

适合目标：

- 画更完整的收敛曲线。
- 观察本文方法是否更早达到最优 checkpoint。
- 避免评委追问“训练集变了是否影响可比性”。

### 方案 B：构造扩展训练集，只加入额外训练来源

可以新增非测试来源的训练图像，例如额外 SwissView/SwissViewMonuments 训练素材或未进入验证/测试的 MASA/MM-GAG 样本。前提是生成 manifest，逐项记录：

- 每个样本来源。
- 是否属于 train/val/test。
- 与当前 validation/test 的 key 是否重叠。
- 扩展后的图像数量。

适合目标：

- 强化“训练样本更多时，混合奖励机制仍能稳定指导中长距离行动”。
- 作为答辩备用材料，不建议替代当前 480k 主趋势。

## 推荐执行组合

推荐先做 `720k_same_train`：

- 4 个方法不变：`external_pbrs`、`linear_gate_no_pbrs`、`constant_gate_pbrs`、`proposed_linear_gate_pbrs`。
- 训练集不变：继续使用当前 184 图像 MASA+MM-GAG 混合集。
- checkpoint 间隔不变：每 10 episode 保存。
- 评估不变：MM-GAG aerial/ground/text，C=8，budget=10。

如果 720k 曲线仍然不够突出，再做 `720k_expanded_train`：

- 使用同样 4 个方法。
- 使用扩展训练集。
- 必须输出训练集 manifest 和 val/test 重叠检查报告。
- 图形标题中明确标注为 expanded-train follow-up，避免和主实验混淆。

## 答辩展示建议

主图使用固定 checkpoint 评估曲线；训练日志图只解释训练信号。

推荐展示顺序：

1. 密集 checkpoint 成功率趋势：说明本文方法在训练过程中形成更好的 C8 能力。
2. best-so-far envelope：说明不是某一个随机点，而是训练过程中逐步达到更高上界。
3. loss/entropy/KL：说明训练后期趋于稳定。
4. 典型路线图：说明奖励信号如何把行动从绕圈或偏离引导到目标方向。
