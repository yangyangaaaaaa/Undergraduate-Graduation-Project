#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build a two-panel PPT figure for the smoother reward-guidance case."""

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


CASE_ID = "case_04"

OUT_DIR = base.RESULTS / "figures" / "defense_reward_training_stage" / "single_reward_mechanism"
PPT_PACK_DIR = (
    base.RESULTS
    / "figures"
    / "ppt_candidate_pack_20260606"
    / "03_奖励机制_动作归因与GP故事"
)


def load_case() -> tuple[pd.Series, dict[str, pd.Series]]:
    selected = pd.read_csv(refined.SELECTED_TABLE)
    match = selected[selected["case_id"].astype(str).eq(CASE_ID)]
    if match.empty:
        raise RuntimeError(f"Cannot find {CASE_ID} in {refined.SELECTED_TABLE}")
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
        raise RuntimeError(f"Missing methods for {CASE_ID}: {missing}")
    return case_row, rows_by_method


def style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=refined.GRID, linewidth=0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=12)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(refined.NUM_FONT)


def draw_distance_curve(ax: plt.Axes, rows_by_method: dict[str, pd.Series]) -> None:
    max_step = 0
    max_dist = 0.0
    for method in refined.PLOT_METHOD_ORDER:
        row = rows_by_method[method]
        dist = [float(x) for x in base.parse_list(row.get("dist_sequence", ""))]
        if not dist:
            continue
        x = np.arange(0, len(dist))
        max_step = max(max_step, int(x[-1]))
        max_dist = max(max_dist, max(dist))
        key = method == "proposed_linear_gate_pbrs"
        color = base.METHOD_STYLE[method]["color"]
        ax.plot(
            x,
            dist,
            color=color,
            linewidth=4.6 if key else 2.2,
            linestyle=refined.LINESTYLE[method],
            marker="o" if key else None,
            markersize=6.4,
            alpha=1.0 if key else 0.62,
            zorder=12 if key else 6,
        )
        final_marker = "o" if int(row["success"]) == 1 else "X"
        ax.scatter(
            [x[-1]],
            [dist[-1]],
            s=108 if key else 64,
            marker=final_marker,
            color=color,
            edgecolor="white",
            linewidth=1.2,
            zorder=14,
        )

    ax.axhline(0, color="#94A3B8", linewidth=1.0)
    ax.set_xlim(-0.18, max_step + 0.25)
    ax.set_ylim(-0.35, max_dist + 0.65)
    ax.set_yticks(range(0, int(max_dist) + 1, 2))
    ax.set_xlabel("行动步", fontproperties=refined.CN_FONT, fontsize=14)
    ax.set_ylabel("到目标距离", fontproperties=refined.CN_FONT, fontsize=14)
    style_axes(ax)
    ax.annotate(
        "持续接近目标",
        xy=(max_step, 0),
        xytext=(max_step - 2.6, 1.05),
        fontproperties=refined.CN_FONT,
        fontsize=15,
        color=refined.BLUE,
        arrowprops=dict(arrowstyle="-|>", color=refined.BLUE, lw=1.8, shrinkA=2, shrinkB=2),
    )


def draw_reward_curve(ax: plt.Axes, rows_by_method: dict[str, pd.Series]) -> None:
    all_values: list[float] = []
    max_step = 0
    proposed_final: tuple[int, float] | None = None
    for method in refined.PLOT_METHOD_ORDER:
        row = rows_by_method[method]
        total = [float(x) for x in base.parse_list(row.get("step_reward_total", ""))]
        if not total:
            continue
        x = np.arange(1, len(total) + 1)
        max_step = max(max_step, int(x[-1]))
        all_values.extend(total)
        key = method == "proposed_linear_gate_pbrs"
        color = base.METHOD_STYLE[method]["color"]
        ax.plot(
            x,
            total,
            color=color,
            linewidth=4.2 if key else 2.1,
            linestyle=refined.LINESTYLE[method],
            marker="o" if key else None,
            markersize=6.0,
            alpha=1.0 if key else 0.62,
            zorder=12 if key else 6,
        )
        if key:
            proposed_final = (int(x[-1]), float(total[-1]))

    vals = np.array(all_values, dtype=float) if all_values else np.array([0.0])
    ymin = min(-1.25, float(np.nanmin(vals)) - 0.25)
    ymax = max(2.45, float(np.nanmax(vals)) + 0.25)
    ax.axhline(0, color="#94A3B8", linewidth=1.0)
    if proposed_final is not None:
        ax.axvspan(proposed_final[0] - 0.35, proposed_final[0] + 0.35, color=refined.BLUE, alpha=0.055, zorder=0)
    ax.set_xlim(0.85, max_step + 0.25)
    ax.set_ylim(ymin, ymax)
    ax.set_yticks([-1, 0, 1, 2])
    ax.set_xlabel("行动步", fontproperties=refined.CN_FONT, fontsize=14)
    ax.set_ylabel("总奖励", fontproperties=refined.CN_FONT, fontsize=14)
    style_axes(ax)
    if proposed_final is not None:
        ax.annotate(
            "到达强化",
            xy=proposed_final,
            xytext=(max_step - 2.25, proposed_final[1] - 0.65),
            fontproperties=refined.CN_FONT,
            fontsize=15,
            color=refined.BLUE,
            arrowprops=dict(arrowstyle="-|>", color=refined.BLUE, lw=1.8, shrinkA=2, shrinkB=2),
        )


def draw_legend(fig: plt.Figure) -> None:
    handles = []
    for method in refined.PLOT_METHOD_ORDER:
        key = method == "proposed_linear_gate_pbrs"
        handles.append(
            Line2D(
                [0],
                [0],
                color=base.METHOD_STYLE[method]["color"],
                lw=4.8 if key else 2.4,
                linestyle=refined.LINESTYLE[method],
                label=refined.METHOD_CN[method],
            )
        )
    leg = fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.500, 0.925),
        ncol=5,
        frameon=False,
        fontsize=13,
        handlelength=2.5,
        columnspacing=1.45,
    )
    for text in leg.get_texts():
        text.set_fontproperties(refined.CN_FONT)
        text.set_fontsize(13)


def build_figure() -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PPT_PACK_DIR.mkdir(parents=True, exist_ok=True)
    refined.setup_style()

    _, rows_by_method = load_case()

    fig = plt.figure(figsize=(16, 9), facecolor=refined.PAPER)
    draw_legend(fig)
    fig.lines.append(Line2D([0.055, 0.945], [0.865, 0.865], transform=fig.transFigure, color="#CBD5E1", lw=1.1))

    ax_dist = fig.add_axes([0.065, 0.150, 0.415, 0.705])
    draw_distance_curve(ax_dist, rows_by_method)
    ax_reward = fig.add_axes([0.545, 0.150, 0.415, 0.705])
    draw_reward_curve(ax_reward, rows_by_method)

    stem = "14_奖励机制_case04_距离奖励双曲线对比"
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
