#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build action-level attribution figures for reward-guided training cases.

These figures bridge the gap between route visualizations and reward curves:
each page keeps the real overhead route context, then aligns the proposed
method's action, distance transition, reward components, and total reward for
every training step.

All values come from real ``training_route_samples.csv`` records.  The reward
mechanism is described as training-stage signal only.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image

import build_reward_guided_case_studies as base


OUT_DIR = base.RESULTS / "figures" / "defense_reward_training_stage" / "action_attribution"
TABLE_DIR = base.TABLES
REPORT_DIR = base.REPORTS
FOCUS_TABLE = TABLE_DIR / "reward_guided_ppt_focus_cases_selected.csv"
REPORT_PATH = REPORT_DIR / "reward_action_attribution_cases_zh.md"

CASE_IDS = ["case_04", "case_08", "case_07", "case_12", "case_02"]

ROUTE_METHODS = [
    "external_only",
    "mixed_no_gate_no_pbrs",
    "mixed_gate_only",
    "proposed_linear_gate_pbrs",
]

METHOD_LABEL = {
    "external_only": "Ext",
    "intrinsic_only": "Int",
    "mixed_no_gate_no_pbrs": "Ext+Int",
    "mixed_gate_only": "Gate",
    "mixed_pbrs_only": "PBRS",
    "proposed_linear_gate_pbrs": "Ours",
}

METHOD_LINESTYLE = {
    "external_only": (0, (8, 4)),
    "intrinsic_only": (0, (2, 2)),
    "mixed_no_gate_no_pbrs": (0, (6, 2, 1, 2)),
    "mixed_gate_only": (0, (7, 3)),
    "mixed_pbrs_only": (0, (1, 2)),
    "proposed_linear_gate_pbrs": "solid",
}

ROUTE_OFFSETS = {
    "external_only": -0.18,
    "mixed_no_gate_no_pbrs": -0.06,
    "mixed_gate_only": 0.06,
    "proposed_linear_gate_pbrs": 0.18,
}

ACTION_LABEL = {
    "up": "↑",
    "right": "→",
    "down": "↓",
    "left": "←",
}


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def clear_outputs() -> None:
    for pattern in ["action_attribution_*.png", "action_attribution_*.svg", "action_attribution_contact_sheet.png"]:
        for path in OUT_DIR.glob(pattern):
            if path.is_file():
                path.unlink()


def numeric_focus_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if col in {"case_id", "distance_bucket", "dataset", "image_id", "image_path"} or col.endswith("_path"):
            continue
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


def case_mask(routes: pd.DataFrame, case_row: pd.Series) -> pd.Series:
    mask = pd.Series(True, index=routes.index)
    for col in ["seed", "episode", "image_index", "distance_bucket", "initial_patch", "goal_patch"]:
        mask &= routes[col].astype(str).eq(str(case_row[col]))
    return mask


def route_goal_and_path(row: pd.Series) -> tuple[int, list[int]]:
    seq = [int(x) for x in base.parse_list(row.get("patch_sequence", ""))]
    if len(seq) >= 2:
        return int(seq[0]), [int(x) for x in seq[1:]]
    return int(row["goal_patch"]), [int(row["initial_patch"]), int(row["final_patch"])]


def offset_route_xy(xy: list[tuple[float, float]], method: str) -> list[tuple[float, float]]:
    offset = ROUTE_OFFSETS.get(method, 0.0)
    return [(x + offset, y + offset) for x, y in xy]


def draw_route(ax: plt.Axes, row: pd.Series, method: str) -> None:
    _, path = route_goal_and_path(row)
    if len(path) < 1:
        return
    xy = offset_route_xy([base.patch_xy(p) for p in path], method)
    xs, ys = zip(*xy)
    key = method == "proposed_linear_gate_pbrs"
    color = base.METHOD_STYLE[method]["color"]
    ax.plot(
        xs,
        ys,
        color=color,
        linewidth=5.5 if key else 4.0,
        linestyle="solid" if key else METHOD_LINESTYLE[method],
        alpha=1.0 if key else 0.88,
        solid_capstyle="round",
        zorder=14 if key else 9,
        path_effects=[pe.Stroke(linewidth=8.5 if key else 6.8, foreground="white", alpha=0.78), pe.Normal()],
    )
    final_x, final_y = xy[-1]
    success = int(row["success"]) == 1
    ax.scatter(
        [final_x],
        [final_y],
        s=135 if key else 105,
        marker="o" if success else "X",
        color="#10B981" if success else "#EF4444",
        edgecolor="white",
        linewidth=1.6,
        zorder=18,
    )


def draw_overhead_panel(ax: plt.Axes, image: Image.Image, rows_by_method: dict[str, pd.Series]) -> None:
    ax.imshow(image, extent=(-0.5, base.PATCH_SIZE - 0.5, base.PATCH_SIZE - 0.5, -0.5), zorder=0)
    ax.set_xlim(-0.5, base.PATCH_SIZE - 0.5)
    ax.set_ylim(base.PATCH_SIZE - 0.5, -0.5)
    ax.set_aspect("equal")
    base.draw_grid(ax)

    for method in ROUTE_METHODS:
        if method != "proposed_linear_gate_pbrs":
            draw_route(ax, rows_by_method[method], method)
    draw_route(ax, rows_by_method["proposed_linear_gate_pbrs"], "proposed_linear_gate_pbrs")

    proposed = rows_by_method["proposed_linear_gate_pbrs"]
    goal, path = route_goal_and_path(proposed)
    if path:
        sx, sy = base.patch_xy(path[0])
        ax.scatter([sx], [sy], s=220, marker="o", color="#10B981", edgecolor="white", linewidth=2.2, zorder=22)
    gx, gy = base.patch_xy(goal)
    ax.scatter([gx], [gy], s=310, marker="*", color=base.YELLOW, edgecolor=base.INK, linewidth=1.4, zorder=22)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_distance_panel(ax: plt.Axes, rows_by_method: dict[str, pd.Series]) -> None:
    for method in ROUTE_METHODS:
        row = rows_by_method[method]
        dist = [float(x) for x in base.parse_list(row.get("dist_sequence", ""))]
        if not dist:
            continue
        x = np.arange(0, len(dist))
        key = method == "proposed_linear_gate_pbrs"
        ax.plot(
            x,
            dist,
            color=base.METHOD_STYLE[method]["color"],
            linewidth=4.2 if key else 2.4,
            linestyle="solid" if key else METHOD_LINESTYLE[method],
            marker="o" if key else None,
            markersize=5.6,
            alpha=1.0 if key else 0.72,
            label=METHOD_LABEL[method],
            zorder=9 if key else 5,
        )
    ax.set_xlabel("Step")
    ax.set_ylabel("Dist")
    ax.set_ylim(-0.25, 8.6)
    ax.set_yticks(range(0, 9, 2))
    ax.grid(axis="y", color=base.GRID, linewidth=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        ncol=4,
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.02),
        borderaxespad=0.0,
        fontsize=12.2,
        handlelength=2.8,
        columnspacing=1.2,
    )


def reward_arrays(row: pd.Series) -> dict[str, np.ndarray]:
    return {
        "external": np.array([float(x) for x in base.parse_list(row.get("step_reward_ex", ""))], dtype=float),
        "intrinsic": np.array([float(x) for x in base.parse_list(row.get("step_reward_in_gated", ""))], dtype=float),
        "pbrs": np.array([float(x) for x in base.parse_list(row.get("step_pbrs_bonus", ""))], dtype=float),
        "total": np.array([float(x) for x in base.parse_list(row.get("step_reward_total", ""))], dtype=float),
    }


def blend_with_white(hex_color: str, strength: float) -> tuple[float, float, float]:
    strength = float(np.clip(strength, 0.0, 1.0))
    hex_color = hex_color.lstrip("#")
    rgb = np.array([int(hex_color[i : i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0
    out = 1.0 - (1.0 - rgb) * strength
    return tuple(out.tolist())


def value_cell_color(value: float, max_abs: float, positive: str = "#10B981", negative: str = "#DC2626") -> tuple[float, float, float]:
    if not np.isfinite(value) or max_abs <= 0:
        return (1.0, 1.0, 1.0)
    strength = min(abs(value) / max_abs, 1.0) * 0.80 + 0.12
    return blend_with_white(positive if value >= 0 else negative, strength)


def draw_reward_ledger(ax: plt.Axes, row: pd.Series) -> None:
    arrays = reward_arrays(row)
    total = arrays["total"]
    n = int(total.size)
    actions = [str(x) for x in base.parse_list(row.get("action_sequence", ""))][:n]
    dist = [float(x) for x in base.parse_list(row.get("dist_sequence", ""))]
    if len(dist) < n + 1:
        dist = dist + [np.nan] * (n + 1 - len(dist))

    labels = ["Act", "Dist", "Ext", "Int*g", "PBRS", "Total"]
    data_rows: list[list[str]] = []
    color_rows: list[list[tuple[float, float, float]]] = []

    data_rows.append([ACTION_LABEL.get(a, a) for a in actions])
    color_rows.append([(1.0, 1.0, 1.0) for _ in range(n)])

    dist_texts: list[str] = []
    dist_colors: list[tuple[float, float, float]] = []
    for i in range(n):
        before, after = dist[i], dist[i + 1]
        if np.isfinite(before) and np.isfinite(after):
            delta = after - before
            dist_texts.append(f"{int(before)}→{int(after)}")
            if delta < 0:
                dist_colors.append(blend_with_white("#10B981", 0.78))
            elif delta > 0:
                dist_colors.append(blend_with_white("#DC2626", 0.78))
            else:
                dist_colors.append(blend_with_white("#6B7280", 0.45))
        else:
            dist_texts.append("")
            dist_colors.append((1.0, 1.0, 1.0))
    data_rows.append(dist_texts)
    color_rows.append(dist_colors)

    row_specs = [
        ("external", "{:+.1f}", "#10B981", "#DC2626"),
        ("intrinsic", "{:+.2f}", "#059669", "#DC2626"),
        ("pbrs", "{:+.3f}", "#1764AB", "#DC2626"),
        ("total", "{:+.1f}", "#10B981", "#DC2626"),
    ]
    for key, fmt, pos, neg in row_specs:
        values = arrays[key][:n]
        max_abs = max(float(np.nanmax(np.abs(values))) if values.size else 0.0, 0.02 if key == "pbrs" else 0.5)
        data_rows.append([fmt.format(float(v)) for v in values])
        color_rows.append([value_cell_color(float(v), max_abs, pos, neg) for v in values])

    ax.set_axis_off()
    ax.set_xlim(0, n + 1.6)
    ax.set_ylim(0, len(labels) + 0.35)

    label_w = 1.05
    cell_w = 1.0
    cell_h = 0.82
    y_top = len(labels) - 0.15
    for r, label in enumerate(labels):
        y = y_top - r
        ax.text(0.0, y + cell_h / 2, label, fontsize=13.8, fontweight="bold", ha="left", va="center")
        for c in range(n):
            x = label_w + c * cell_w
            ax.add_patch(Rectangle((x, y), cell_w - 0.045, cell_h, facecolor=color_rows[r][c], edgecolor="#E5E7EB", linewidth=1.0))
            text = data_rows[r][c]
            weight = "bold" if r in {1, 5} else "normal"
            ax.text(x + (cell_w - 0.045) / 2, y + cell_h / 2, text, fontsize=12.2, fontweight=weight, ha="center", va="center", color=base.INK)

    for c in range(n):
        x = label_w + c * cell_w
        ax.text(x + (cell_w - 0.045) / 2, 0.20, str(c + 1), fontsize=10.5, color=base.MUTED, ha="center", va="center")
    ax.text(label_w - 0.08, 0.20, "Step", fontsize=10.5, color=base.MUTED, ha="right", va="center")


def add_header(fig: plt.Figure, case_index: int, case_row: pd.Series) -> None:
    ctrl_success = max(0, 5 - int(case_row["fail_count"]))
    title = f"Case {case_index:02d} | {case_row['distance_bucket']} | Ours 1 | Ctrl {ctrl_success}/5 | {float(case_row['run_progress']) * 100:.1f}%"
    fig.text(0.035, 0.955, title, ha="left", va="top", fontsize=24, fontweight="bold", color=base.INK)
    fig.lines.append(Line2D([0.035, 0.965], [0.905, 0.905], transform=fig.transFigure, color="#CBD5E1", lw=1.3))


def add_route_legend(fig: plt.Figure) -> None:
    handles = []
    for method in ROUTE_METHODS:
        key = method == "proposed_linear_gate_pbrs"
        handles.append(
            Line2D(
                [0],
                [0],
                color=base.METHOD_STYLE[method]["color"],
                linewidth=4.8 if key else 3.0,
                linestyle="solid" if key else METHOD_LINESTYLE[method],
                label=METHOD_LABEL[method],
            )
        )
    fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.068, 0.487),
        ncol=4,
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#D1D5DB",
        fontsize=12.2,
        handlelength=2.8,
        columnspacing=1.4,
    )


def build_case_figure(case_index: int, case_row: pd.Series, rows_by_method: dict[str, pd.Series]) -> Path:
    asset = base.ImageAsset(
        str(case_row.get("dataset", "")),
        str(case_row.get("image_id", "")),
        Path(str(case_row["image_path"])),
        f"{case_row.get('dataset', '')} overhead",
    )
    image = base.open_overhead_image(asset)

    fig = plt.figure(figsize=(16, 9))
    add_header(fig, case_index, case_row)
    gs = fig.add_gridspec(
        2,
        2,
        left=0.045,
        right=0.965,
        top=0.835,
        bottom=0.060,
        width_ratios=[0.88, 1.22],
        height_ratios=[1.12, 1.00],
        wspace=0.14,
        hspace=0.28,
    )

    ax_map = fig.add_subplot(gs[0, 0])
    draw_overhead_panel(ax_map, image, rows_by_method)

    ax_dist = fig.add_subplot(gs[0, 1])
    plot_distance_panel(ax_dist, rows_by_method)

    ax_ledger = fig.add_subplot(gs[1, :])
    draw_reward_ledger(ax_ledger, rows_by_method["proposed_linear_gate_pbrs"])

    stem = (
        f"action_attribution_{case_index:02d}_{case_row['case_id']}_"
        f"{case_row['distance_bucket']}_seed{int(case_row['seed'])}_ep{int(case_row['episode'])}_"
        f"img{int(case_row['image_index'])}_s{int(case_row['initial_patch'])}_g{int(case_row['goal_patch'])}"
    )
    png_path = OUT_DIR / f"{stem}.png"
    svg_path = OUT_DIR / f"{stem}.svg"
    fig.savefig(png_path, dpi=240)
    fig.savefig(svg_path)
    plt.close(fig)
    return png_path


def make_contact_sheet(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    thumb_w, thumb_h = 640, 360
    cols = 2
    rows = math.ceil(len(paths) / cols)
    sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), "#F7F9FC")
    for i, path in enumerate(paths):
        img = Image.open(path).convert("RGB")
        img.thumbnail((thumb_w - 14, thumb_h - 14), Image.Resampling.LANCZOS)
        x = (i % cols) * thumb_w + (thumb_w - img.width) // 2
        y = (i // cols) * thumb_h + (thumb_h - img.height) // 2
        sheet.paste(img, (x, y))
    out = OUT_DIR / "action_attribution_contact_sheet.png"
    sheet.save(out, quality=95)
    return out


def write_report(selected_rows: list[pd.Series], paths: list[Path], contact_sheet: Path | None) -> None:
    lines = [
        "# 训练动作归因可视化说明",
        "",
        "这组图补充现有路线图和奖励曲线，目标是让评委直接看到：同一个训练样本中，本文方法为什么更容易把动作序列推向目标。",
        "",
        "新版图内只保留识别所需的短标签、数字和缩写，不放解释性长句；详细解释放在本文件中，适合作为 PPT 备注。",
        "",
        "重要表述：奖励、距离门控和 PBRS 只在训练阶段提供学习信号；正式测试或论文表格评估时只加载训练好的策略 checkpoint，不再调用奖励函数。",
        "",
        "## 读图顺序",
        "",
        "1. 页眉只保留 `Case / C 距离 / Ours 1 / Ctrl 0-5 / 训练进度`。其中 `Ours 1` 表示本文方法到达，`Ctrl 0/5` 表示五个对照方法均未到达。",
        "2. 先看左上角真实俯视图：蓝色实线是本文方法，对照方法为虚线；起点为绿色圆点，目标为黄色星标，未到达终点用红色叉号标出。",
        "3. 再看右上角 `Dist` 曲线：蓝线是否持续下降到 0，用来判断路线是否真正形成目标导向。",
        "4. 最后看下方每步奖励账本：`Act` 为动作，`Dist` 为距离变化，`Ext` 为外部奖励，`Int*g` 为门控后的内在奖励，`PBRS` 为势函数塑形项，`Total` 为总奖励。绿色表示靠近目标或正反馈，红色表示回退或惩罚。",
        "",
        "## 推荐使用方式",
        "",
        "- PPT 主线：先放训练趋势图说明整体效果，再放 1-2 页动作归因图说明机制，最后放路线图/表格补充结果。",
        "- 最推荐优先使用 `case_04` 或 `case_08`：它们是 C8 远距离样例，五个对照均未到达，本文方法到达。",
        "- 若需要说明“走出回退/循环”，使用 `case_12`：对照方法多次回退，本文方法虽然有一次回退但能恢复并到达。",
        "",
        "## 输出文件",
        "",
    ]
    if contact_sheet is not None:
        lines.append(f"- 总览图：`{contact_sheet}`")
    for row, path in zip(selected_rows, paths):
        lines.append(
            f"- `{row['case_id']}`：`{path}`；{row['distance_bucket']}，训练进度 {float(row['run_progress']) * 100:.1f}%，"
            f"本文方法到达，{int(row['fail_count'])}/5 对照未到达。"
        )
    lines.extend(
        [
            "",
            "## 评委视角的证据链",
            "",
            "1. 趋势层：训练检查点曲线证明本文方法在 C8 中长距离任务上形成更好的模型检查点。",
            "2. 机制层：动作归因图证明三项奖励不是抽象公式，而是在每一步训练样本上共同改变动作反馈。",
            "3. 行为层：真实俯视图路线证明这种反馈最终表现为更连续的目标接近轨迹。",
            "4. 结果层：正式表格仍作为最终性能依据，避免把训练阶段奖励图误解成测试阶段额外信息。",
            "",
            "## 答辩话术",
            "",
            "“这页不是测试阶段额外使用奖励，而是把训练日志中的一个真实片段拆开。上面先看行为：同一起点和目标下，对照方法容易停在目标外或回退，本文方法最终到达。下面再看训练信号：每一步动作的距离变化与外部奖励、门控内在奖励、PBRS 都对齐在同一列。可以看到，外部奖励负责惩罚无效移动和奖励接近目标，门控内在奖励保留远距离探索信号，PBRS 给接近目标的移动提供连续塑形。三项信号合在一起后，策略更容易把高回报分配给连续靠近目标的动作序列。正式测试时不再调用这些奖励函数，只执行已经学习好的策略。”",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> None:
    ensure_dirs()
    clear_outputs()
    base.setup_style()

    focus = numeric_focus_table(pd.read_csv(FOCUS_TABLE))
    routes = base.read_routes()

    selected_rows: list[pd.Series] = []
    paths: list[Path] = []
    for idx, case_id in enumerate(CASE_IDS, start=1):
        match = focus[focus["case_id"].astype(str).eq(case_id)]
        if match.empty:
            continue
        case_row = match.iloc[0]
        group = routes[case_mask(routes, case_row)]
        rows_by_method = {method: group[group["method"].eq(method)].iloc[0] for method in base.METHODS if not group[group["method"].eq(method)].empty}
        if not all(method in rows_by_method for method in ROUTE_METHODS + ["proposed_linear_gate_pbrs"]):
            continue
        path = build_case_figure(idx, case_row, rows_by_method)
        selected_rows.append(case_row)
        paths.append(path)

    contact_sheet = make_contact_sheet(paths)
    write_report(selected_rows, paths, contact_sheet)

    manifest = {
        "figure_count": len(paths),
        "figures": [str(p) for p in paths],
        "contact_sheet": str(contact_sheet) if contact_sheet else None,
        "report": str(REPORT_PATH),
    }
    (OUT_DIR / "action_attribution_manifest.json").write_text(pd.Series(manifest).to_json(force_ascii=False, indent=2), encoding="utf-8")
    print(manifest)


if __name__ == "__main__":
    main()
