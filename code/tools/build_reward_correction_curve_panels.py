#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build a correction-ability figure by placing each case's two curve panels together."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import build_reward_guided_case_studies as base  # noqa: E402
import build_reward_mechanism_single_case_refined as refined  # noqa: E402


SMOOTH_CASE_ID = "case_04"
CORRECTION_CASE_ID = "case_12"

OUT_DIR = base.RESULTS / "figures" / "defense_reward_training_stage" / "single_reward_mechanism"
PPT_PACK_DIR = (
    base.RESULTS
    / "figures"
    / "ppt_candidate_pack_20260606"
    / "03_奖励机制_动作归因与GP故事"
)


def load_case(case_id: str) -> dict[str, pd.Series]:
    selected = pd.read_csv(refined.SELECTED_TABLE)
    match = selected[selected["case_id"].astype(str).eq(case_id)]
    if match.empty:
        raise RuntimeError(f"Cannot find {case_id} in {refined.SELECTED_TABLE}")
    case_row = match.iloc[0]

    routes = base.read_routes()
    group = routes[refined.case_mask(routes, case_row)].copy()
    rows_by_method = {
        method: group[group["method"].eq(method)].iloc[0]
        for method in refined.METHOD_ORDER
        if not group[group["method"].eq(method)].empty
    }
    missing = [method for method in refined.METHOD_ORDER if method not in rows_by_method]
    if missing:
        raise RuntimeError(f"Missing methods for {case_id}: {missing}")
    return rows_by_method


def style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=refined.GRID, linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=11.5)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(refined.NUM_FONT)


def max_dist(rows_by_case: list[dict[str, pd.Series]]) -> float:
    values: list[float] = []
    for rows_by_method in rows_by_case:
        for method in refined.PLOT_METHOD_ORDER:
            values.extend(float(x) for x in base.parse_list(rows_by_method[method].get("dist_sequence", "")))
    return max(values) if values else 0.0


def max_dist_step(rows_by_case: list[dict[str, pd.Series]]) -> int:
    out = 0
    for rows_by_method in rows_by_case:
        for method in refined.PLOT_METHOD_ORDER:
            dist = base.parse_list(rows_by_method[method].get("dist_sequence", ""))
            if dist:
                out = max(out, len(dist) - 1)
    return out


def reward_limits(rows_by_case: list[dict[str, pd.Series]]) -> tuple[float, float, int]:
    values: list[float] = []
    max_step = 0
    for rows_by_method in rows_by_case:
        for method in refined.PLOT_METHOD_ORDER:
            total = [float(x) for x in base.parse_list(rows_by_method[method].get("step_reward_total", ""))]
            values.extend(total)
            max_step = max(max_step, len(total))
    vals = np.array(values, dtype=float) if values else np.array([0.0])
    ymin = min(-1.35, float(np.nanmin(vals)) - 0.20)
    ymax = max(2.45, float(np.nanmax(vals)) + 0.20)
    return ymin, ymax, max_step


def plot_distance_panel(
    ax: plt.Axes,
    rows_by_method: dict[str, pd.Series],
    x_max: int,
    y_max: float,
    mode: str,
) -> None:
    for method in refined.PLOT_METHOD_ORDER:
        row = rows_by_method[method]
        dist = [float(x) for x in base.parse_list(row.get("dist_sequence", ""))]
        if not dist:
            continue
        x = np.arange(0, len(dist))
        key = method == "proposed_linear_gate_pbrs"
        color = base.METHOD_STYLE[method]["color"]
        ax.plot(
            x,
            dist,
            color=color,
            linewidth=4.0 if key else 1.9,
            linestyle=refined.LINESTYLE[method],
            marker="o" if key else None,
            markersize=5.4,
            alpha=1.0 if key else 0.62,
            zorder=12 if key else 6,
        )
        final_marker = "o" if int(row["success"]) == 1 else "X"
        ax.scatter([x[-1]], [dist[-1]], s=84 if key else 54, marker=final_marker, color=color, edgecolor="white", linewidth=1.1, zorder=14)

    ax.axhline(0, color="#94A3B8", linewidth=1.0)
    ax.set_xlim(-0.15, x_max + 0.25)
    ax.set_ylim(-0.35, y_max + 0.55)
    ax.set_yticks(range(0, int(y_max) + 1, 2))
    ax.set_xlabel("行动步", fontproperties=refined.CN_FONT, fontsize=12.5)
    ax.set_ylabel("到目标距离", fontproperties=refined.CN_FONT, fontsize=12.5)
    style_axes(ax)

    if mode != "smooth":
        ax.axvspan(5.65, 6.35, color=refined.ORANGE, alpha=0.070, zorder=0)
        ax.annotate(
            "短暂偏离",
            xy=(6, 3),
            xytext=(6.45, 4.15),
            fontproperties=refined.CN_FONT,
            fontsize=12.5,
            color=refined.ORANGE,
            arrowprops=dict(arrowstyle="-|>", color=refined.ORANGE, lw=1.6, shrinkA=2, shrinkB=2),
        )
        ax.annotate(
            "纠偏到达",
            xy=(9, 0),
            xytext=(6.95, 0.78),
            fontproperties=refined.CN_FONT,
            fontsize=12.5,
            color=refined.BLUE,
            arrowprops=dict(arrowstyle="-|>", color=refined.BLUE, lw=1.6, shrinkA=2, shrinkB=2),
        )


def plot_reward_panel(
    ax: plt.Axes,
    rows_by_method: dict[str, pd.Series],
    x_max: int,
    y_lim: tuple[float, float],
    mode: str,
) -> None:
    for method in refined.PLOT_METHOD_ORDER:
        row = rows_by_method[method]
        total = [float(x) for x in base.parse_list(row.get("step_reward_total", ""))]
        if not total:
            continue
        x = np.arange(1, len(total) + 1)
        key = method == "proposed_linear_gate_pbrs"
        color = base.METHOD_STYLE[method]["color"]
        ax.plot(
            x,
            total,
            color=color,
            linewidth=3.7 if key else 1.8,
            linestyle=refined.LINESTYLE[method],
            marker="o" if key else None,
            markersize=5.0,
            alpha=1.0 if key else 0.62,
            zorder=12 if key else 6,
        )

    ax.axhline(0, color="#94A3B8", linewidth=1.0)
    ax.set_xlim(0.85, x_max + 0.25)
    ax.set_ylim(*y_lim)
    ax.set_yticks([-1, 0, 1, 2])
    ax.set_xlabel("行动步", fontproperties=refined.CN_FONT, fontsize=12.5)
    ax.set_ylabel("总奖励", fontproperties=refined.CN_FONT, fontsize=12.5)
    style_axes(ax)

    if mode != "smooth":
        ax.axvspan(5.65, 6.35, color=refined.ORANGE, alpha=0.070, zorder=0)
        ax.annotate(
            "偏离受惩罚",
            xy=(6, -0.75),
            xytext=(3.95, -1.08),
            fontproperties=refined.CN_FONT,
            fontsize=12.5,
            color=refined.ORANGE,
            arrowprops=dict(arrowstyle="-|>", color=refined.ORANGE, lw=1.5, shrinkA=2, shrinkB=2),
        )


def draw_method_legend(fig: plt.Figure) -> None:
    handles = []
    for method in refined.PLOT_METHOD_ORDER:
        key = method == "proposed_linear_gate_pbrs"
        handles.append(
            Line2D(
                [0],
                [0],
                color=base.METHOD_STYLE[method]["color"],
                lw=4.4 if key else 2.2,
                linestyle=refined.LINESTYLE[method],
                label=refined.METHOD_CN[method],
            )
        )
    leg = fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.500, 0.895),
        ncol=5,
        frameon=False,
        fontsize=11.5,
        handlelength=2.35,
        columnspacing=1.25,
    )
    for text in leg.get_texts():
        text.set_fontproperties(refined.CN_FONT)
        text.set_fontsize(11.5)


def build_figure() -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PPT_PACK_DIR.mkdir(parents=True, exist_ok=True)
    refined.setup_style()

    smooth_rows = load_case(SMOOTH_CASE_ID)
    correction_rows = load_case(CORRECTION_CASE_ID)
    cases = [smooth_rows, correction_rows]

    dist_y_max = max_dist(cases)
    dist_x_max = max_dist_step(cases)
    reward_y_min, reward_y_max, reward_x_max = reward_limits(cases)

    fig = plt.figure(figsize=(16, 9), facecolor=refined.PAPER)
    fig.text(
        0.050,
        0.958,
        "本文方法的稳定收敛与纠偏能力",
        ha="left",
        va="top",
        fontsize=24,
        fontproperties=refined.CN_FONT,
        color=refined.INK,
    )
    fig.text(
        0.050,
        0.922,
        "直接抽取两个案例图右侧曲线：平顺到达 vs 短暂偏离后恢复",
        ha="left",
        va="top",
        fontsize=13.2,
        fontproperties=refined.CN_FONT,
        color=refined.MUTED,
    )
    draw_method_legend(fig)
    fig.lines.append(Line2D([0.050, 0.950], [0.852, 0.852], transform=fig.transFigure, color="#CBD5E1", lw=1.1))
    fig.lines.append(Line2D([0.500, 0.500], [0.105, 0.805], transform=fig.transFigure, color="#D9E0EA", lw=1.0))

    fig.text(0.265, 0.820, "平顺到达案例", ha="center", va="center", fontsize=16.5, fontproperties=refined.CN_FONT, color=refined.GREEN)
    fig.text(0.735, 0.820, "短暂偏离后纠偏案例", ha="center", va="center", fontsize=16.5, fontproperties=refined.CN_FONT, color=refined.BLUE)

    ax_smooth_dist = fig.add_axes([0.060, 0.500, 0.390, 0.285])
    ax_correct_dist = fig.add_axes([0.550, 0.500, 0.390, 0.285])
    ax_smooth_reward = fig.add_axes([0.060, 0.125, 0.390, 0.285])
    ax_correct_reward = fig.add_axes([0.550, 0.125, 0.390, 0.285])

    plot_distance_panel(ax_smooth_dist, smooth_rows, dist_x_max, dist_y_max, "smooth")
    plot_distance_panel(ax_correct_dist, correction_rows, dist_x_max, dist_y_max, "correction")
    plot_reward_panel(ax_smooth_reward, smooth_rows, reward_x_max, (reward_y_min, reward_y_max), "smooth")
    plot_reward_panel(ax_correct_reward, correction_rows, reward_x_max, (reward_y_min, reward_y_max), "correction")

    stem = "18_本文方法纠偏能力_双案例右侧曲线拼接_删注释版"
    out_png = OUT_DIR / f"{stem}.png"
    pack_png = PPT_PACK_DIR / f"{stem}.png"
    fig.savefig(out_png, dpi=240, facecolor=refined.PAPER)
    plt.close(fig)
    shutil.copy2(out_png, pack_png)
    return out_png, pack_png


def main() -> int:
    out_png, pack_png = build_figure()
    print(f"saved: {out_png}")
    print(f"copied: {pack_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
