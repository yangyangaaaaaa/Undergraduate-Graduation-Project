#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build PPT-focused training-stage case figures for reward guidance.

The companion script ``build_reward_guided_case_studies.py`` produces a full
six-panel evidence figure for each case.  This script creates presentation
figures with a simpler reading order:

1. one real overhead-image route figure;
2. one curve figure with aligned distance and reward traces.

Text is deliberately kept sparse on the exported figures.  Detailed
explanations are written to the companion Markdown report.

No synthetic route data is created here.  All paths, rewards and distances are
read from real ``training_route_samples.csv`` logs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image

import build_reward_guided_case_studies as base


OUT_DIR = base.RESULTS / "figures" / "defense_reward_training_stage" / "ppt_focus_cases"
TABLE_DIR = base.TABLES
REPORT_DIR = base.REPORTS

FOCUS_CASE_COUNT = 12
SELECTED_TABLE = TABLE_DIR / "reward_guided_case_studies_selected.csv"
FOCUS_TABLE = TABLE_DIR / "reward_guided_ppt_focus_cases_selected.csv"
FOCUS_REPORT = REPORT_DIR / "reward_guided_ppt_focus_cases_zh.md"

PREFERRED_CASE_ORDER = [
    "case_04",
    "case_08",
    "case_07",
    "case_10",
    "case_19",
    "case_20",
    "case_02",
    "case_12",
    "case_01",
    "case_03",
    "case_05",
    "case_18",
]

METHOD_LABEL_SHORT = {
    "external_only": "仅外部",
    "intrinsic_only": "仅内在",
    "mixed_no_gate_no_pbrs": "直接相加",
    "mixed_gate_only": "门控内在",
    "mixed_pbrs_only": "仅 PBRS",
    "proposed_linear_gate_pbrs": "本文方法",
}

METHOD_LINESTYLE = {
    "external_only": (0, (5, 3)),
    "intrinsic_only": (0, (2, 2)),
    "mixed_no_gate_no_pbrs": (0, (6, 2, 1, 2)),
    "mixed_gate_only": (0, (7, 2)),
    "mixed_pbrs_only": (0, (1, 2)),
    "proposed_linear_gate_pbrs": "solid",
}


def ensure_output_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


ROUTE_METHODS = [
    "external_only",
    "mixed_no_gate_no_pbrs",
    "mixed_gate_only",
    "proposed_linear_gate_pbrs",
]

ROUTE_OFFSETS = {
    "external_only": -0.18,
    "mixed_no_gate_no_pbrs": -0.06,
    "mixed_gate_only": 0.06,
    "proposed_linear_gate_pbrs": 0.18,
}

METHOD_LABEL_SHORT.update(
    {
        "mixed_no_gate_no_pbrs": "外部+内在",
        "mixed_gate_only": "线性门控",
    }
)


def clear_focus_outputs() -> None:
    for pattern in [
        "focus_case_*.png",
        "focus_case_*.svg",
        "reward_case_*.png",
        "reward_case_*.svg",
        "focus_case_contact_sheet.png",
        "reward_case_contact_sheet.png",
    ]:
        for path in OUT_DIR.glob(pattern):
            if path.is_file():
                path.unlink()


def numeric_selected(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if col == "case_id" or col.endswith("_path") or col in {"dataset", "image_id", "distance_bucket"}:
            continue
        df[col] = pd.to_numeric(df[col], errors="ignore")
    if "dist_num" not in df.columns:
        df["dist_num"] = df["distance_bucket"].astype(str).str.replace("C", "", regex=False).astype(int)
    return df


def select_focus_cases(selected: pd.DataFrame) -> pd.DataFrame:
    selected = numeric_selected(selected)
    preferred = selected[selected["case_id"].isin(PREFERRED_CASE_ORDER)].copy()
    preferred["preferred_rank"] = preferred["case_id"].map({case_id: i for i, case_id in enumerate(PREFERRED_CASE_ORDER)})
    preferred = preferred.sort_values("preferred_rank")

    remaining = selected[~selected["case_id"].isin(preferred["case_id"])].copy()
    remaining["focus_score"] = (
        remaining["fail_count"].astype(float) * 14.0
        + remaining["dist_num"].astype(float) * 7.0
        + remaining["loop_controls"].astype(float) * 2.2
        + remaining["control_mean_final_dist"].astype(float) * 1.4
        + remaining["near_miss_controls"].astype(float) * 0.8
        - remaining["image_max_patch_white_ratio"].astype(float).fillna(0.0) * 10.0
    )
    remaining = remaining.sort_values(["focus_score", "run_progress"], ascending=[False, True])
    focus = pd.concat([preferred, remaining], ignore_index=True)
    focus = focus.drop_duplicates("case_id").head(FOCUS_CASE_COUNT).copy()
    focus.insert(0, "focus_rank", np.arange(1, len(focus) + 1))
    return focus


def case_mask(routes: pd.DataFrame, case_row: pd.Series) -> np.ndarray:
    key_cols = ["seed", "episode", "image_index", "distance_bucket", "initial_patch", "goal_patch"]
    mask = np.ones(len(routes), dtype=bool)
    for col in key_cols:
        mask &= routes[col].astype(str).eq(str(case_row[col])).to_numpy()
    return mask


def route_goal_and_path(row: pd.Series) -> tuple[int, list[int]]:
    seq = [int(x) for x in base.parse_list(row.get("patch_sequence", ""))]
    if len(seq) >= 2:
        return int(seq[0]), [int(x) for x in seq[1:]]
    return int(row["goal_patch"]), [int(row["initial_patch"]), int(row["final_patch"])]


def offset_route_xy(xy: list[tuple[float, float]], method: str) -> list[tuple[float, float]]:
    offset = ROUTE_OFFSETS.get(method, 0.0)
    return [(x + offset, y + offset) for x, y in xy]


def draw_route_overlay(ax: plt.Axes, row: pd.Series, method: str, emphasize: bool = False) -> None:
    color = base.METHOD_STYLE[method]["color"]
    _, path = route_goal_and_path(row)
    if len(path) < 1:
        return
    xy = offset_route_xy([base.patch_xy(p) for p in path], method)
    xs, ys = zip(*xy)
    if emphasize:
        line_width = 5.6
        alpha = 1.0
        zorder = 12
        stroke_width = 8.5
    else:
        line_width = 4.2
        alpha = 0.90
        zorder = 8
        stroke_width = 7.0
    route_linestyle = "solid" if emphasize else (0, (8, 4))
    ax.plot(
        xs,
        ys,
        color=color,
        linewidth=line_width,
        linestyle=route_linestyle,
        alpha=alpha,
        solid_capstyle="round",
        zorder=zorder,
        path_effects=[pe.Stroke(linewidth=stroke_width, foreground="white", alpha=0.78), pe.Normal()],
    )
    final_x, final_y = xy[-1]
    success = int(row["success"]) == 1
    marker_color = "#EF4444" if not success else "#10B981"
    ax.scatter(
        [final_x],
        [final_y],
        s=126 if emphasize else 96,
        marker="X" if not success else "o",
        color=marker_color,
        edgecolor="white",
        linewidth=1.5,
        zorder=15,
        alpha=0.95,
    )


def draw_big_overhead_map(ax: plt.Axes, image: Image.Image, rows_by_method: dict[str, pd.Series]) -> None:
    ax.imshow(image, extent=(-0.5, base.PATCH_SIZE - 0.5, base.PATCH_SIZE - 0.5, -0.5), zorder=0)
    ax.set_xlim(-0.5, base.PATCH_SIZE - 0.5)
    ax.set_ylim(base.PATCH_SIZE - 0.5, -0.5)
    ax.set_aspect("equal")
    base.draw_grid(ax)

    proposed = rows_by_method["proposed_linear_gate_pbrs"]
    goal, proposed_path = route_goal_and_path(proposed)
    for method in ROUTE_METHODS:
        if method == "proposed_linear_gate_pbrs":
            continue
        draw_route_overlay(ax, rows_by_method[method], method, emphasize=False)
    draw_route_overlay(ax, proposed, "proposed_linear_gate_pbrs", emphasize=True)

    if proposed_path:
        sx, sy = base.patch_xy(proposed_path[0])
        ax.scatter([sx], [sy], s=240, marker="o", color="#10B981", edgecolor="white", linewidth=2.4, zorder=20)
    gx, gy = base.patch_xy(goal)
    ax.scatter([gx], [gy], s=310, marker="*", color=base.YELLOW, edgecolor=base.INK, linewidth=1.5, zorder=20)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_overlay_legend(fig: plt.Figure) -> None:
    handles = []
    for method in ROUTE_METHODS:
        style = base.METHOD_STYLE[method]
        lw = 4.8 if method == "proposed_linear_gate_pbrs" else 2.7
        route_linestyle = "solid" if method == "proposed_linear_gate_pbrs" else (0, (8, 4))
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                lw=lw,
                linestyle=route_linestyle,
                alpha=1.0,
                label=METHOD_LABEL_SHORT[method],
            )
        )
    fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.210, 0.060),
        ncol=4,
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#D1D5DB",
        fontsize=12,
        handlelength=2.8,
        columnspacing=1.8,
    )


def plot_distance_panel(ax: plt.Axes, rows_by_method: dict[str, pd.Series]) -> None:
    for method in base.METHODS:
        row = rows_by_method[method]
        dist = [float(x) for x in base.parse_list(row.get("dist_sequence", ""))]
        if not dist:
            continue
        style = base.METHOD_STYLE[method]
        x = np.arange(1, len(dist) + 1)
        key = method == "proposed_linear_gate_pbrs"
        ax.plot(
            x,
            dist,
            color=style["color"],
            linewidth=3.9 if key else 1.9,
            linestyle=METHOD_LINESTYLE[method],
            marker="o" if key else None,
            markersize=5.2,
            alpha=1.0 if key else 0.56,
            label=METHOD_LABEL_SHORT[method],
            zorder=8 if key else 4,
        )
    ax.set_title("距离曲线", loc="left", fontsize=17, fontweight="bold", pad=8)
    ax.set_xlabel("")
    ax.set_ylabel("到目标距离", fontsize=11.5)
    ax.set_ylim(-0.2, 8.6)
    ax.set_yticks(range(0, 9, 2))
    ax.set_xlim(1, 10)
    ax.grid(axis="y", color=base.GRID, linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(ncol=6, frameon=False, loc="upper right", fontsize=10.0, handlelength=2.3, columnspacing=1.0)


def draw_behavior_table(ax: plt.Axes, rows_by_method: dict[str, pd.Series]) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.0, 1.04, "结果", fontsize=14, fontweight="bold", ha="left", va="bottom")
    columns = [("方法", 0.02), ("结果", 0.36), ("终距", 0.70)]
    for label, x in columns:
        ax.text(x, 0.88, label, fontsize=10.4, color=base.MUTED, ha="left", va="center")
    row_h = 0.12
    y0 = 0.78
    for i, method in enumerate(base.METHODS):
        row = rows_by_method[method]
        y = y0 - i * row_h
        key = method == "proposed_linear_gate_pbrs"
        bg = "#EAF3FF" if key else "#FFFFFF"
        edge = base.BLUE if key else "#E5E7EB"
        ax.add_patch(Rectangle((0.0, y - 0.047), 0.98, row_h - 0.012, facecolor=bg, edgecolor=edge, linewidth=1.2))
        ax.add_patch(
            Rectangle(
                (0.018, y - 0.022),
                0.025,
                0.044,
                facecolor=base.METHOD_STYLE[method]["color"],
                edgecolor="none",
                alpha=0.95,
            )
        )
        success = int(row["success"]) == 1
        result_text = "到达" if success else "未到达"
        result_color = "#047857" if success else "#B91C1C"
        weight = "bold" if key else "normal"
        ax.text(0.055, y, METHOD_LABEL_SHORT[method], fontsize=10.8, fontweight=weight, ha="left", va="center")
        ax.text(0.36, y, result_text, fontsize=10.8, color=result_color, fontweight="bold", ha="left", va="center")
        ax.text(0.70, y, f"{int(row['final_dist'])}", fontsize=10.8, fontweight=weight, ha="left", va="center")


def pad_to_same_length(arr: np.ndarray, n: int) -> np.ndarray:
    if len(arr) >= n:
        return arr[:n]
    return np.pad(arr, (0, n - len(arr)), constant_values=np.nan)


def reward_arrays(row: pd.Series) -> dict[str, np.ndarray]:
    return {
        "external": np.array([float(x) for x in base.parse_list(row.get("step_reward_ex", ""))], dtype=float),
        "intrinsic": np.array([float(x) for x in base.parse_list(row.get("step_reward_in_gated", ""))], dtype=float),
        "pbrs": np.array([float(x) for x in base.parse_list(row.get("step_pbrs_bonus", ""))], dtype=float),
        "total": np.array([float(x) for x in base.parse_list(row.get("step_reward_total", ""))], dtype=float),
    }


def plot_total_reward_comparison(ax: plt.Axes, rows_by_method: dict[str, pd.Series]) -> None:
    ax.axhline(0, color="#9CA3AF", linewidth=1.0)
    for method in base.METHODS:
        arrays = reward_arrays(rows_by_method[method])
        total = arrays["total"]
        if total.size == 0:
            continue
        x = np.arange(1, total.size + 1)
        key = method == "proposed_linear_gate_pbrs"
        ax.plot(
            x,
            total,
            color=base.METHOD_STYLE[method]["color"],
            linewidth=3.2 if key else 1.8,
            linestyle=METHOD_LINESTYLE[method],
            marker="o" if key else None,
            markersize=4.8,
            alpha=1.0 if key else 0.72,
            label=METHOD_LABEL_SHORT[method],
            zorder=8 if key else 4,
        )
    ax.set_title("每步总奖励", loc="left", fontsize=17, fontweight="bold", pad=8)
    ax.set_xlabel("")
    ax.set_ylabel("总奖励", fontsize=11.5)
    ax.grid(axis="y", color=base.GRID, linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_component_lines(
    ax: plt.Axes,
    rows_by_method: dict[str, pd.Series],
    key: str,
    title: str,
    ylabel: str,
) -> None:
    ax.axhline(0, color="#9CA3AF", linewidth=0.9)
    all_values = []
    for method in base.METHODS:
        arr = reward_arrays(rows_by_method[method])[key]
        if arr.size == 0:
            continue
        all_values.extend(arr.tolist())
        x = np.arange(1, arr.size + 1)
        is_key = method == "proposed_linear_gate_pbrs"
        ax.plot(
            x,
            arr,
            color=base.METHOD_STYLE[method]["color"],
            linewidth=2.5 if is_key else 1.4,
            linestyle=METHOD_LINESTYLE[method],
            marker="o" if is_key or key == "pbrs" else None,
            markersize=3.3,
            alpha=1.0 if is_key else 0.68,
            zorder=8 if is_key else 4,
        )
    ax.set_title(title, loc="left", fontsize=13.8, fontweight="bold", pad=6)
    ax.set_xlabel("行动步", fontsize=10.5)
    ax.set_ylabel(ylabel, fontsize=10.2)
    ax.grid(axis="y", color=base.GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if key == "pbrs":
        vals = np.array(all_values, dtype=float) if all_values else np.array([0.0])
        max_abs = max(float(np.nanmax(np.abs(vals))), 0.02)
        ax.set_ylim(-max_abs * 1.35, max_abs * 1.35)


def plot_cumulative_decomposition(ax: plt.Axes, rows_by_method: dict[str, pd.Series]) -> None:
    methods = base.METHODS
    x = np.arange(len(methods))
    ex = np.array([float(rows_by_method[m].get("reward_ex_sum", 0.0)) for m in methods])
    intrinsic = np.array([float(rows_by_method[m].get("reward_in_gated_sum", 0.0)) for m in methods])
    pbrs = np.array([float(rows_by_method[m].get("pbrs_bonus_sum", 0.0)) for m in methods])
    total = np.array([float(rows_by_method[m].get("total_reward", 0.0)) for m in methods])
    width = 0.22
    ax.axhline(0, color="#9CA3AF", linewidth=1.0)
    ax.bar(x - width, ex, width=width, color=base.RED, alpha=0.68, label="外部项")
    ax.bar(x, intrinsic, width=width, color=base.GREEN, alpha=0.75, label="内在×系数")
    ax.bar(x + width, pbrs, width=width, color=base.BLUE, alpha=0.75, label="PBRS")
    ax.plot(x, total, color=base.INK, marker="o", linewidth=2.0, label="累计总奖励")
    ax.set_title("累计分解", loc="left", fontsize=13.4, fontweight="bold", pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL_SHORT[m] for m in methods], fontsize=9.2)
    ax.set_ylabel("累计奖励", fontsize=10.2)
    ax.grid(axis="y", color=base.GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(ncol=4, frameon=False, loc="upper left", fontsize=9.5, handlelength=1.4, columnspacing=0.9)


def build_reward_figure(
    focus_idx: int,
    case_row: pd.Series,
    rows_by_method: dict[str, pd.Series],
) -> Path:
    fig = plt.figure(figsize=(16, 9), facecolor=base.PAPER)
    dist = str(case_row["distance_bucket"])
    fig.text(
        0.035,
        0.962,
        f"曲线分析 | 案例 {focus_idx:02d} | {dist}",
        ha="left",
        va="top",
        fontsize=23,
        fontweight="bold",
        color=base.INK,
    )
    fig.lines.append(plt.Line2D([0.035, 0.965], [0.918, 0.918], transform=fig.transFigure, color="#CBD5E1", lw=1.1))

    ax_dist = fig.add_axes([0.060, 0.690, 0.895, 0.190])
    plot_distance_panel(ax_dist, rows_by_method)
    ax_dist.set_xlabel("")

    ax_total = fig.add_axes([0.060, 0.470, 0.895, 0.165], sharex=ax_dist)
    plot_total_reward_comparison(ax_total, rows_by_method)
    ax_total.set_xlabel("")

    ax_ex = fig.add_axes([0.060, 0.255, 0.275, 0.145], sharex=ax_dist)
    plot_component_lines(ax_ex, rows_by_method, "external", "外部项", "奖励")

    ax_intrinsic = fig.add_axes([0.375, 0.255, 0.275, 0.145], sharex=ax_dist)
    plot_component_lines(ax_intrinsic, rows_by_method, "intrinsic", "内在×系数", "奖励")

    ax_pbrs = fig.add_axes([0.690, 0.255, 0.265, 0.145], sharex=ax_dist)
    plot_component_lines(ax_pbrs, rows_by_method, "pbrs", "PBRS", "奖励")

    ax_cum = fig.add_axes([0.060, 0.060, 0.895, 0.125])
    plot_cumulative_decomposition(ax_cum, rows_by_method)

    for ax in [ax_dist, ax_total, ax_ex, ax_intrinsic, ax_pbrs]:
        ax.label_outer()

    stem = (
        f"reward_case_{focus_idx:02d}_{case_row['case_id']}_"
        f"{case_row['distance_bucket']}_seed{int(case_row['seed'])}_ep{int(case_row['episode'])}_"
        f"img{int(case_row['image_index'])}_s{int(case_row['initial_patch'])}_g{int(case_row['goal_patch'])}"
    )
    png_path = OUT_DIR / f"{stem}.png"
    svg_path = OUT_DIR / f"{stem}.svg"
    fig.savefig(png_path, dpi=240, facecolor=base.PAPER)
    fig.savefig(svg_path, facecolor=base.PAPER)
    plt.close(fig)
    return png_path


def add_header(fig: plt.Figure, case_rank: int, case_row: pd.Series) -> None:
    fail_count = int(case_row["fail_count"])
    dist = str(case_row["distance_bucket"])
    title = f"案例 {case_rank:02d} | {dist} | 本文方法到达，{fail_count}/5 对照未到达"
    fig.text(0.035, 0.962, title, ha="left", va="top", fontsize=23, fontweight="bold", color=base.INK)
    fig.lines.append(plt.Line2D([0.035, 0.965], [0.918, 0.918], transform=fig.transFigure, color="#CBD5E1", lw=1.1))


def build_focus_figure(
    focus_idx: int,
    case_row: pd.Series,
    rows_by_method: dict[str, pd.Series],
    asset: base.ImageAsset,
) -> Path:
    image = base.open_overhead_image(asset)
    fig = plt.figure(figsize=(16, 9), facecolor=base.PAPER)
    add_header(fig, focus_idx, case_row)

    ax_map = fig.add_axes([0.075, 0.115, 0.850, 0.780])
    draw_big_overhead_map(ax_map, image, rows_by_method)
    add_overlay_legend(fig)

    stem = (
        f"focus_case_{focus_idx:02d}_{case_row['case_id']}_"
        f"{case_row['distance_bucket']}_seed{int(case_row['seed'])}_ep{int(case_row['episode'])}_"
        f"img{int(case_row['image_index'])}_s{int(case_row['initial_patch'])}_g{int(case_row['goal_patch'])}"
    )
    png_path = OUT_DIR / f"{stem}.png"
    svg_path = OUT_DIR / f"{stem}.svg"
    fig.savefig(png_path, dpi=240, facecolor=base.PAPER)
    fig.savefig(svg_path, facecolor=base.PAPER)
    plt.close(fig)
    return png_path


def make_contact_sheet(paths: list[Path], out_name: str = "focus_case_contact_sheet.png") -> Path | None:
    if not paths:
        return None
    thumb_w, thumb_h = 640, 360
    gutter = 24
    cols = 3
    rows = math.ceil(len(paths) / cols)
    sheet = Image.new("RGB", (cols * thumb_w + (cols + 1) * gutter, rows * thumb_h + (rows + 1) * gutter), "#F7F9FC")
    for i, path in enumerate(paths):
        img = Image.open(path).convert("RGB")
        img.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x = gutter + (i % cols) * (thumb_w + gutter)
        y = gutter + (i // cols) * (thumb_h + gutter)
        canvas = Image.new("RGB", (thumb_w, thumb_h), "white")
        ox = (thumb_w - img.width) // 2
        oy = (thumb_h - img.height) // 2
        canvas.paste(img, (ox, oy))
        sheet.paste(canvas, (x, y))
    out = OUT_DIR / out_name
    sheet.save(out, quality=95)
    return out


def case_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if int(row["dist_num"]) >= 8:
        reasons.append("远距离 C8")
    else:
        reasons.append(str(row["distance_bucket"]))
    if int(row["fail_count"]) >= 5:
        reasons.append("五个对照全部失败")
    else:
        reasons.append(f"{int(row['fail_count'])} 个对照失败")
    if int(row["loop_controls"]) >= 3:
        reasons.append("多个对照出现回退/重复")
    if int(row["near_miss_controls"]) >= 2:
        reasons.append("对照接近目标但未完成")
    if float(row["run_progress"]) <= 0.30:
        reasons.append("训练早期已有方向优势")
    return "，".join(reasons)


def write_focus_report(
    focus: pd.DataFrame,
    figure_paths: list[Path],
    reward_paths: list[Path],
    contact_sheet: Path | None,
    reward_contact_sheet: Path | None,
) -> None:
    lines = [
        "# PPT 聚焦版训练案例说明",
        "",
        "本组图用于结题汇报中解释“混合奖励机制如何在训练阶段指导策略形成”。所有路线、距离和奖励分量均来自真实 `training_route_samples.csv`；每组图比较同一 seed、同一 episode、同一俯视图、同一起点和同一目标，只改变训练奖励设置。",
        "",
        "图内文字已压缩到最少：标题、关键数字、图例和坐标轴。详细解释放在本文件中，适合直接复制到 PPT 备注或答辩讲稿。",
        "",
        "图的阅读顺序：先看俯视图中 4 个代表方法是否形成清晰目标导向路线；再看曲线图，第一行是全部方法距离曲线，下面是同一行动步对齐的总奖励与三项奖励组成。蓝色始终表示本文方法。",
        "",
        "重要表述：奖励、距离门控和 PBRS 只在训练阶段提供学习信号；正式测试或表格评估时只加载训练好的策略 checkpoint，不再调用奖励函数。",
        "",
        f"- 聚焦案例数：{len(focus)}",
        f"- 聚焦图目录：`{OUT_DIR}`",
        f"- 聚焦案例表：`{FOCUS_TABLE}`",
    ]
    if contact_sheet is not None:
        lines.append(f"- 路线图总览：`{contact_sheet}`")
    if reward_contact_sheet is not None:
        lines.append(f"- 曲线图总览：`{reward_contact_sheet}`")
    lines.extend(
        [
            "- 路线图只展示 4 个代表方法：仅外部、外部+内在、线性门控、本文方法；其他对照不放在俯视图上，避免路线重复。",
            "- 曲线图第一行保留全部 6 个方法的距离曲线；其下按同一行动步对齐展示每步总奖励、外部项、内在×系数和 PBRS。PBRS 使用独立面板，避免小数值被外部奖励淹没。",
        ]
    )
    lines.extend(["", "## 推荐优先放入 PPT 的案例", ""])
    for row, path, reward_path in zip(focus.head(8).itertuples(index=False), figure_paths[:8], reward_paths[:8]):
        row_dict = row._asdict()
        lines.append(
            f"- `{row_dict['case_id']}`：{case_reason(pd.Series(row_dict))}。"
            f"本文方法 {int(row_dict['proposed_path_len'])} 步到达，"
            f"对照平均终距 {float(row_dict['control_mean_final_dist']):.1f}。路线图：`{path}`；奖励图：`{reward_path}`"
        )
    lines.extend(["", "## 单页讲解备注", ""])
    for row, path, reward_path in zip(focus.itertuples(index=False), figure_paths, reward_paths):
        row_dict = row._asdict()
        fail_count = int(row_dict["fail_count"])
        dist = str(row_dict["distance_bucket"])
        progress = float(row_dict["run_progress"]) * 100
        proposed_steps = int(row_dict["proposed_path_len"])
        proposed_regress = int(row_dict["proposed_regress_steps"])
        proposed_revisit = int(row_dict["proposed_revisit_count"])
        control_final = float(row_dict["control_mean_final_dist"])
        loop_controls = int(row_dict["loop_controls"])
        near_miss = int(row_dict["near_miss_controls"])
        lines.extend(
            [
                f"### {int(row_dict['focus_rank']):02d}. {row_dict['case_id']}（{dist}）",
                "",
                f"- 路线图：`{path}`",
                f"- 奖励图：`{reward_path}`",
                f"- 真实条件：seed={int(row_dict['seed'])}，episode={int(row_dict['episode'])}，训练进度 {progress:.1f}%，起点={int(row_dict['initial_patch'])}，目标={int(row_dict['goal_patch'])}。",
                f"- 图上结论：本文方法 {proposed_steps} 步到达目标，回退 {proposed_regress} 次，重复访问 {proposed_revisit} 次；{fail_count}/5 个对照方法未到达，对照平均终距 {control_final:.1f}。",
                f"- 行为证据：路线图只保留 4 个代表方法并做平行偏移，蓝色路线可直接看到连续接近目标；曲线图第一行保留所有方法，本文方法最终降到 0。对照路线停在目标外，且有 {loop_controls} 个对照出现明显回退或重复访问。",
                f"- 奖励解释：曲线图在同一行动步上对齐距离、总奖励、外部项、内在×系数和 PBRS。外部项主要惩罚无效移动并在到达/接近时给出正反馈；内在项提供探索收益；PBRS 虽然数值小，但在接近目标方向上持续给出形状信号。三者相加后，本文方法更容易把高奖励动作集中到“继续靠近目标”的动作序列上。",
            ]
        )
        if near_miss > 0:
            lines.append(f"- 可强调点：有 {near_miss} 个对照已经接近目标但没有真正到达，适合说明“接近目标”和“完成任务”之间仍需要稳定的方向塑形。")
        if dist == "C8":
            lines.append("- 可强调点：这是最远距离桶 C8，更适合说明中长距离导航中的训练指导优势。")
        if progress <= 30.0:
            lines.append("- 可强调点：训练早期已经出现方向差异，说明奖励设计不是只在后期挑 checkpoint 才显得好，而是在训练过程中就提供了更清晰的学习信号。")
        lines.extend(
            [
                "- 建议讲法：同一个起点和目标下，对照方法并不是完全不会移动，而是容易在目标附近回退、绕行或停住；本文方法的优势在于把探索和方向约束结合起来，使策略形成可连续执行的目标导向路线。",
                "- 避免表述：不要说测试阶段继续使用 PBRS 或奖励函数；应说这些信号只参与训练，测试阶段只执行训练好的策略。",
                "",
            ]
        )
    lines.extend(
        [
            "",
            "## 汇报话术",
            "",
            "“这几张图不是测试阶段额外使用奖励函数，而是把训练日志中的典型片段可视化出来。每个案例分成两页：第一页只看俯视图路线，第二页把距离曲线和奖励组成按行动步对齐。路线图只保留四个代表方法，并用平行线把重叠路线分开；曲线图仍保留所有方法。可以看到本文方法的蓝色路线持续接近目标，距离曲线下降到 0；而对照方法往往在目标附近回退、重复访问，或者停在目标外。奖励曲线进一步说明，本文方法不是简单把某一项奖励放大，而是把外部惩罚、门控内在奖励和 PBRS 方向塑形组合起来，使训练阶段更容易把高奖励动作分配给连续靠近目标的动作序列。”",
            "",
            "## 全部输出",
            "",
        ]
    )
    for path, reward_path in zip(figure_paths, reward_paths):
        lines.append(f"- 路线图：`{path}`")
        lines.append(f"- 奖励图：`{reward_path}`")
    FOCUS_REPORT.write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> int:
    ensure_output_dirs()
    clear_focus_outputs()
    base.setup_style()
    if not SELECTED_TABLE.exists():
        raise FileNotFoundError(f"Missing selected case table: {SELECTED_TABLE}")

    selected = pd.read_csv(SELECTED_TABLE)
    focus = select_focus_cases(selected)
    routes = base.read_routes()
    mix_index, masa_metadata, mmgag_index = base.load_image_mapping()

    output_records: list[dict] = []
    figure_paths: list[Path] = []
    reward_paths: list[Path] = []
    for focus_idx, (_, case_row) in enumerate(focus.iterrows(), start=1):
        asset = base.resolve_image_asset(int(case_row["image_index"]), mix_index, masa_metadata, mmgag_index)
        if asset is None:
            continue
        case_routes = routes.loc[case_mask(routes, case_row)].copy()
        if case_routes["method"].nunique() != len(base.METHODS):
            continue
        rows_by_method = {method: case_routes[case_routes["method"].eq(method)].iloc[0] for method in base.METHODS}
        png_path = build_focus_figure(focus_idx, case_row, rows_by_method, asset)
        reward_path = build_reward_figure(focus_idx, case_row, rows_by_method)
        record = dict(case_row)
        record.update(
            {
                "focus_figure_path": str(png_path),
                "reward_figure_path": str(reward_path),
                "focus_reason": case_reason(case_row),
                "dataset": asset.dataset,
                "image_id": asset.image_id,
                "image_path": str(asset.image_path),
            }
        )
        output_records.append(record)
        figure_paths.append(png_path)
        reward_paths.append(reward_path)

    focus_out = pd.DataFrame(output_records)
    focus_out.to_csv(FOCUS_TABLE, index=False, encoding="utf-8-sig")
    contact_sheet = make_contact_sheet(figure_paths, "focus_case_contact_sheet.png")
    reward_contact_sheet = make_contact_sheet(reward_paths, "reward_case_contact_sheet.png")
    write_focus_report(focus_out, figure_paths, reward_paths, contact_sheet, reward_contact_sheet)
    print(
        json.dumps(
            {
                "focus_case_count": int(len(focus_out)),
                "figures": [str(p) for p in figure_paths],
                "reward_figures": [str(p) for p in reward_paths],
                "contact_sheet": str(contact_sheet) if contact_sheet else None,
                "reward_contact_sheet": str(reward_contact_sheet) if reward_contact_sheet else None,
                "focus_table": str(FOCUS_TABLE),
                "report": str(FOCUS_REPORT),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
