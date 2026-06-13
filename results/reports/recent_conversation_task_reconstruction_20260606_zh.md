# 近期对话与任务脉络梳理

整理时间：2026-06-06 21:35（Asia/Shanghai）

本文用于把 2026-06-06 前后频繁切换的几段 Codex 对话重新串起来。依据是 `F:\bishe\PROJECT_STATE.md`、`PROJECT_OFFBOARDING.md` 和 `D:\codex-home\sessions\2026\06\06` 中的结构化会话记录。本文不记录远程密码、API token 或私有凭据。

## 一、总主线

最近这段时间的工作已经从“继续做实验”逐步转成“把已有实验、可视化和答辩材料收口”。现在最重要的主线不是继续追求新的训练结果，而是把已有证据组织成答辩 PPT 和论文叙述：

1. 正式数值结果用固定评估表格和已经完成的 paper-aligned 评估支撑。
2. 答辩可视化主证据用动作归因、训练路线案例、奖励曲线和验收动图支撑。
3. 续训曲线只能作为诊断/备选，不再作为“本文方法持续领先”的主证据。
4. 奖励、距离门控、PBRS 都是训练阶段机制；测试阶段只加载 checkpoint 执行动作，不能写成测试时调用奖励函数。
5. 近期图片/动图任务容易因为聊天框预览或一次性上传大量图片导致会话出错，后续处理图片应尽量走本地文件、脚本校验、少量人工抽查。

## 二、近期会话时间线

### 1. `019e6d25` / `019e9bc4`：续训实验与验收图包的起点

用户先要求接手 `019e6d25-09cd-73a1-bd8c-7c15e6b2857d` 的最后任务和实验部分。这个阶段有两条事情交织：

- 续训实验：为了解决 dense checkpoint 曲线只在末尾超过基线的问题，部署了 `480k -> 960k` same-train continuation。
- 画图任务：开始把已有实验做成验收展示图，包括路线 GIF、xBD/MM-GAG/超远距离设置图、轨迹与目标线索展示。

早期图包的主要设计目标是：使用真实航拍图和轨迹，不做空洞示意图；右侧展示目标/当前观测/起点等 cue；路线图要更像照片布局，避免旧版蓝框。

### 2. `019e9bce`：验收演示包第一轮大改

用户指出第一版不好看，主要问题包括：

- 目标观测和起点应该竖着放在右侧。
- 不要蓝框，不要装饰性框。
- demo 背景要白色。
- 图片不能变形，要对齐。
- 左边需要保留数据/路线信息。
- xBD 也要做成路线图，不只是示意图。
- MM-GAG、xBD、超远距离等能可视化的实验都应做验收图。
- 主要素材在服务器，缺素材时应自己找，但不能把凭据写入文件。

这一轮生成了更完整的 acceptance demo 包：主路线 GIF、xBD/MM-GAG/超远距离动态设置 GIF，以及结果/索引页。后续又发现字体、右侧列位置、标题遮挡等问题。

### 3. `019e9c03`：字体、对齐和遮挡修复

用户接手 `019e9bce` 后继续要求：

- 动图英文换中文宋体。
- 右侧列靠下，需要和中间路线图上下对齐。
- 标题下面的小字不能遮挡路线图。

这一轮主要是修复 `build_acceptance_demo_visuals.py` 的字体入口、布局坐标、标题区域和右侧 cue grid。之后重新生成图包并抽查代表图。

### 4. `019e9c1a`：MM-GAG 文本目标面板修复

用户继续接手 `019e9c03`，并提出更细的图面问题：

- 方法名称改为“本文方法”。
- MM-GAG 文本目标描述不能被压缩成短标签。
- 目标文字应占据原先图片位置，不要超出。
- 目标图必须保持正方形，不能变形。
- 不要一次性上传很多图片，容易导致会话损坏。

这一阶段做了小范围重生成，重点修复 MM-GAG text route setting poster/GIF。后续用户澄清“不是要删掉距离曲线，而是文字目标描述要更完整”，这点在下一段对话里彻底收口。

### 5. `019e9c48`：验收图包最终收口

用户接手 `019e9c1a` 后继续要求：

- “不不不，我是指文字目标描述。”
- “距离曲线保留。”
- 英文用 Times New Roman，其余按该要求整体修改。
- 多抽几张、多做几组 GIF。
- 缺素材可以去服务器找。

最终完成的验收演示包位于：

`F:\bishe\Undergraduate-Graduation-Project\results\figures\acceptance_demo`

关键产物：

- `13` 个 route GIF。
- `7` 个 route-setting GIF。
- `10` 张 evidence/page 图。
- `2` 张 xBD setting page。
- manifest：`acceptance_demo_manifest.json`。
- 报告：`F:\bishe\Undergraduate-Graduation-Project\results\reports\acceptance_demo_visuals_zh.md`。

最终版特征：

- 左侧是路线图，右侧是 2x2 cue grid。
- 右下角保留 distance curve。
- MM-GAG text target panel 保留完整文本目标描述。
- 英文字体在生成器可控范围内使用 Times New Roman。
- 路线 GIF 尺寸为 1920x1080。
- 使用真实轨迹和真实航拍/目标图，不是纯示意图。
- 清理了 `build_acceptance_demo_visuals.py` 中早期 unreachable legacy layout 代码。

### 6. `019e9cba`：续训实验最终判定

用户要求接手 `019e6d25` 续训实验，并判断实验结束后怎么办。

最终状态：

- 远端 continuation fixed eval 已完成 `12/12`。
- 下载了 `38` 个评估文件到：
  `F:\bishe\GeoExplorer\analysis\pipeline_20260606_dense_mmgag_continuation_trend`
- 合并表 `552` 行：4 个方法 x 46 个 checkpoint x 3 个 MM-GAG 模态。
- 生成图：
  `F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_trends\figure_dense_mmgag_continuation_curves.png`
  和 `.svg`。
- 生成报告：
  `F:\bishe\Undergraduate-Graduation-Project\results\reports\dense_mmgag_continuation_curve_zh.md`

关键结论是负向/诊断性：

- 历史最佳 C8 mean SR：`linear_gate_no_pbrs=0.9596`，本文方法 `0.9113`。
- 最终 checkpoint：`linear_gate_no_pbrs=0.9326`，本文方法 `0.8326`。
- 后期平均：`constant_gate_pbrs=0.9208`，本文方法 `0.8631`。

因此这条续训曲线不能作为“本文方法持续领先”的主证据。后续答辩或论文应把它放在诊断/备选位置，主证据应回到正式固定评估表、动作归因图、路线案例和奖励曲线。

### 7. `019e9cbb`：PPT 图片包整理与搜索区域图

用户说现在有点凌乱，希望把当前 PPT 能放的图片都整理出来，数量要多一些，每类都要有，附说明书。

完成的 PPT 候选图片包位于：

`F:\bishe\Undergraduate-Graduation-Project\results\figures\ppt_candidate_pack_20260606`

最初整理出 `98` 个候选媒体文件，分类包括：

- `01_开场_方法与数据集`
- `02_验收演示_动图与实验设置`
- `03_奖励机制_动作归因与GP故事`
- `04_训练案例_PPT路线与奖励曲线`
- `05_训练案例_完整单页备用`
- `06_趋势曲线_训练过程`
- `07_结果图卡_汇报总览`
- `08_附录备用_演化与旧轨迹`

说明书：

`F:\bishe\Undergraduate-Graduation-Project\results\figures\ppt_candidate_pack_20260606\00_说明书\PPT图片包说明书.md`

索引：

`F:\bishe\Undergraduate-Graduation-Project\results\figures\ppt_candidate_pack_20260606\00_说明书\图片索引.csv`

之后用户又要求生成一张搜索区域图：用数据集的一张图，虚线分成 5x5 区域。第一次生成的是 Masa 城市航拍图：

- 主输出：
  `F:\bishe\Undergraduate-Graduation-Project\results\figures\chapter2_dataset\search_area_5x5_grid_masa.png`
- PPT 包副本：
  `F:\bishe\Undergraduate-Graduation-Project\results\figures\ppt_candidate_pack_20260606\01_开场_方法与数据集\search_area_5x5_grid_masa.png`

加入这张后，PPT 包变为 `99` 个媒体文件。

### 8. `019e9d00`：本轮接手，搜索区域图补画与历史梳理

本轮先接手 `019e9cbb`，完成 Masa 5x5 搜索区域图。随后用户又给了一张高山水库/大坝场景截图，要求找出对应数据集图，本地没有再去服务器，环境变量里有连接信息。

后续用户提醒：不要在聊天框里打开/阅读图片，容易搞坏。因此本轮后续只用文件列表、元数据、图像统计脚本和机械校验，不再做聊天内图片预览。

本地成功找到源图，不需要上服务器：

`F:\bishe\geoexploxer_edit\SwissView\data\SwissViewMonuments\aerial_view\Tseuzier_03.png`

生成两张 SwissView/Tseuzier 搜索区域图：

- 带黄色目标点：
  `F:\bishe\Undergraduate-Graduation-Project\results\figures\chapter2_dataset\search_area_5x5_grid_swissview_tseuzier.png`
- 无目标点：
  `F:\bishe\Undergraduate-Graduation-Project\results\figures\chapter2_dataset\search_area_5x5_grid_swissview_tseuzier_no_target.png`

两张都复制进 PPT 包 `01_开场_方法与数据集`。机械校验通过：

- 尺寸均为 `1500x1500`。
- 5x5 dashed grid 的线条命中存在。
- 带目标点版本中心 RGB 为黄色。

加入这两张后，PPT 候选包当前共有 `101` 个媒体文件，`图片索引.csv` 也为 `101` 行。

### 9. `019e9d11`：另一条论文写作工作区接手

当天还有一条独立会话 `019e9d11-e855-7643-801c-212c6267b912`，用户要求接手 `019e3580-3621-7f12-af10-015f45ce3f0a`，这条不是当前 PPT/画图主线，而是 `F:\literature` 论文写作/终稿审查工作区的历史恢复。

该线索的边界：

- `F:\literature` 是论文写作、终稿审查、AIGC 修改、公式/Word 抽取核查工作区。
- `F:\bishe` 是 GeoExplorer 实验、可视化、答辩材料整理工作区。

后续不要把 `F:\literature` 的论文终稿任务和 `F:\bishe` 的答辩图片包任务混在一起。若继续论文写作，应先读 `F:\literature\PROJECT_STATE.md`。

## 三、现在最清楚的答辩材料主线

如果要做答辩 PPT，建议按以下顺序组织：

1. 方法/任务总览：用方法框架图和数据集总览图。
2. 搜索区域定义：用 `search_area_5x5_grid_swissview_tseuzier.png` 或 Masa 版本解释 5x5 搜索空间。
3. 实验设置与验收动图：用 acceptance demo 中的 route GIF、MM-GAG setting、xBD setting、超远距离 setting。
4. 训练阶段机制：用动作归因图解释外部奖励、门控内在奖励和 PBRS 如何影响动作反馈。
5. 真实路线案例：用 `ppt_focus_cases` 的 route page 展示同一起终点下本文方法和对照方法的路线差异。
6. 奖励曲线案例：用对应 reward curve page 展示距离变化和奖励组成如何对齐。
7. 正式结果：回到正式 fixed-eval 表格或 polished result cards。
8. 附录/追问：放完整 case study、旧轨迹、续训诊断曲线。

核心表述边界：

- 可以说：训练阶段混合奖励改变学习信号，使策略更容易形成中远距离连续接近目标的行为。
- 不要说：测试阶段调用奖励函数、PBRS 或 gate。
- 可以说：续训实验说明 dense checkpoint 曲线不能作为持续领先主证据。
- 不要说：480k 后本文方法一直领先。

## 四、当前关键文件地图

PPT 候选包：

`F:\bishe\Undergraduate-Graduation-Project\results\figures\ppt_candidate_pack_20260606`

PPT 图片说明书：

`F:\bishe\Undergraduate-Graduation-Project\results\figures\ppt_candidate_pack_20260606\00_说明书\PPT图片包说明书.md`

PPT 图片索引：

`F:\bishe\Undergraduate-Graduation-Project\results\figures\ppt_candidate_pack_20260606\00_说明书\图片索引.csv`

验收演示包：

`F:\bishe\Undergraduate-Graduation-Project\results\figures\acceptance_demo`

验收演示报告：

`F:\bishe\Undergraduate-Graduation-Project\results\reports\acceptance_demo_visuals_zh.md`

续训诊断报告：

`F:\bishe\Undergraduate-Graduation-Project\results\reports\dense_mmgag_continuation_curve_zh.md`

答辩材料 handoff：

`F:\bishe\Undergraduate-Graduation-Project\results\reports\defense_materials_handoff_for_paper_agent_zh.md`

训练阶段路线/奖励案例：

`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\ppt_focus_cases`

动作归因图：

`F:\bishe\Undergraduate-Graduation-Project\results\figures\defense_reward_training_stage\action_attribution`

搜索区域图：

`F:\bishe\Undergraduate-Graduation-Project\results\figures\chapter2_dataset\search_area_5x5_grid_masa.png`

`F:\bishe\Undergraduate-Graduation-Project\results\figures\chapter2_dataset\search_area_5x5_grid_swissview_tseuzier.png`

`F:\bishe\Undergraduate-Graduation-Project\results\figures\chapter2_dataset\search_area_5x5_grid_swissview_tseuzier_no_target.png`

## 五、后续接手注意事项

1. 先读 `F:\bishe\PROJECT_STATE.md`，再读本文件。
2. 如果继续论文正文而不是答辩材料，切到 `F:\literature` 并读那里的 `PROJECT_STATE.md`。
3. 图片任务不要在聊天框里反复打开/预览大图；优先用脚本生成和机械校验，必要时只做少量本地检查。
4. 远程连接信息可从环境变量或凭据管理器读取，但不要写入报告、代码、PROJECT_STATE 或聊天总结。
5. 不要把续训曲线作为主结论；它现在是负向诊断。
6. 当前最稳的 PPT 叙事是：正式表格给结果，动作归因和路线/奖励案例解释机制，验收动图展示实验可视化和任务设置。
