# reward-gate 消融测试协议核查

## 结论

当前 reward-gate/PBRS 消融表的测试流程是 evaluation-only：测试阶段只加载 policy checkpoint 和 LLM checkpoint，然后使用 greedy policy 选择动作。训练期的奖励门控、PBRS、entropy loss、validation-distance 选择等模块不在测试时调用。

因此，对 `G/P/E/V` 和 `reward_gate_type` 这类训练期消融而言，使用不同 checkpoint 进行同协议测试是合理的；这些模块的影响通过训练后的策略权重体现，而不是通过测试时开关体现。

## 代码依据

- `paper_geo_evaluator.py` 的 `load_bundle()` 只设置测试数据、设备、checkpoint、budget，并加载 `PPO` state dict。
- `paper_geo_evaluator.py` 的 `evaluate_tasks()` 调用 `model_action()`，后者使用 `select_greedy_action()`。
- 正式 evaluator 中没有调用 `train.py` 里的 `gate_weight()`、`pbrs_bonus()`、`reward_ex/reward_in`、`finish_bonus()` 或 `GEOEXPLORER_VAL_DISTS`。
- `models/ppo.py` 的旧 `validate()` 系列函数会计算 reward trace，但当前正式表没有走这条路径。

## 写作备注

建议在论文消融表备注中写：

> 所有消融模型在测试阶段均使用相同的 greedy evaluation protocol。奖励门控和 PBRS 等模块仅在训练阶段改变策略学习过程，测试时不额外注入奖励信号；因此表中差异反映不同训练设计得到的 checkpoint 策略差异。
