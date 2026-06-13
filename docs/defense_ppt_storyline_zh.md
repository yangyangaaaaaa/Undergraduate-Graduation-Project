# 结题答辩 PPT 汇报脉络

依据文件：`D:\桌面\08122215_张洋_结题答辩 (2).pptx`。本仓库未直接提交 PPT 原件，因为该文件约 105 MB，超过 GitHub 普通单文件限制；首页素材改为抽取 PPT 中实际嵌入的关键媒体，GIF 另做 GitHub 展示压缩。

## 28 页结构

1. 封面：好奇心驱动的无人机主动定位目标导航方法。
2. 选题背景：低空经济、搜救巡检、无人机智能升级。
3. 任务设置：5x5 搜索区域、图像/语义目标引导、上下左右动作空间。
4. 现存问题：引导不足、反馈稀疏、局部最优。
5. 研究框架：目标线索统一编码，历史动作与观测建模，混合奖励和 PPO 训练。
6. 混合奖励机制：外在奖励、好奇心内在奖励、距离调节奖励。
7. 实验设置：GPU、网格、搜索预算、距离桶、训练/验证/测试数据集。
8-24. 实验结果可视化：短距离、C6/C8、MM-GAG 三模态、xBD 灾前灾后、10x10 长距离、短距离失败案例。
25. 性能实验对比：整体成功率、中远距离优势、短距离波动。
26. 奖励分距离分析：训练阶段接近/偏离动作的平均单步奖励。
27. 消融实验对比：外在奖励、好奇心奖励、距离调节函数、本文完整方法。
28. 致谢页。

## 首页抽取素材

`results/figures/defense_readme_20260613/` 中的文件均依据 PPT 的 `ppt/media/*` 整理；其中 GIF 为压缩展示版：

| 文件 | PPT 来源 | 用途 |
| --- | --- | --- |
| `method_framework.png` | slide 5 | 研究框架 |
| `short_distance_route.gif` | slide 8 | 短距离绕路成功轨迹 |
| `c8_route_demo.gif` | slide 12 | C8 连续搜索轨迹 |
| `multimodal_text_target.gif` | slide 16 | MM-GAG 文本目标 |
| `xbd_disaster_route.gif` | slide 19 | xBD 灾后搜索 |
| `long_distance_grid10.gif` | slide 23 | 10x10 长距离搜索 |
| `short_distance_failure.png` | slide 24 | 短距离失败边界 |
| `performance_analysis.png` | slide 25 | 性能实验分析 |
| `reward_distance_analysis.png` | slide 26 | 奖励分距离分析 |
| `ablation_analysis.png` | slide 27 | 消融实验分析 |

## 讲述边界

- 奖励、门控和 PBRS 是训练阶段信号；测试阶段不调用奖励函数。
- 主结论放在 MM-GAG、多模态、长距离和正式固定评估上。
- 短距离 C1-C3 结果是边界分析，不应包装成本文方法优势。
- 续训曲线是诊断材料，不用于证明本文方法持续领先。
- xBD-disaster 口径为灾后搜索区域，目标仍使用灾前图像嵌入。
