#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build a smooth-case reward figure with method-level total-reward comparison."""

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
import build_reward_smooth_and_correction_comparison as smooth_fig  # noqa: E402


CASE_ID = "case_04"

OUT_DIR = base.RESULTS / "figures" / "defense_reward_training_stage" / "single_reward_mechanism"
PPT_PACK_DIR = (
    base.RESULTS
    / "figures"
    / "ppt_candidate_pack_20260606"
    / "03_奖励机制_动作归因与GP故事"
)


def draw_method_total_comparison(ax: plt.Axes, rows_by_method: dict[str, pd.Series]) -> None:
    methods = refined.PLOT_METHOD_ORDER
    labels = [refined.METHOD_CN[m] for m in methods]
    totals = np.array([float(rows_by_method[m]["total_reward"]) for m in methods], dtype=float)
    finals = [int(rows_by_method[m]["final_dist"]) for m in methods]
    successes = [int(rows_by_method[m]["success"]) == 1 for m in methods]
    colors = [base.METHOD_STYLE[m]["color"] for m in methods]

    y = np.arange(len(methods))
    ax.axvline(0, color="#94A3B8", linewidth=1.1, zorder=0)
    bars = ax.barh(y, totals, height=0.56, color=colors, alpha=0.88, edgecolor="white", linewidth=1.0, zorder=3)

    ax.set_xlim(-8.7, 4.1)
    ax.set_ylim(-0.65, len(methods) - 0.35)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontproperties=refined.CN_FONT, fontsize=12.2)
    ax.invert_yaxis()
    ax.set_xlabel("累计总奖励", fontproperties=refined.CN_FONT, fontsize=11.8, color=refined.MUTED)
    ax.set_title("各方法累计总奖励与结果", loc="left", fontsize=15.5, fontproperties=refined.CN_FONT, color=refined.INK, pad=8)
    ax.grid(axis="x", color=refined.GRID, linewidth=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for label in ax.get_xticklabels():
        label.set_fontproperties(refined.NUM_FONT)
        label.set_fontsize(10.8)

    for i, (bar, total, final, success, method) in enumerate(zip(bars, totals, finals, successes, methods)):
        x_text = total + 0.18 if total >= 0 else total - 0.18
        ha = "left" if total >= 0 else "right"
        ax.text(
            x_text,
            bar.get_y() + bar.get_height() / 2,
            f"{total:+.2f}",
            ha=ha,
            va="center",
            fontsize=11.4,
            fontproperties=refined.NUM_FONT,
            color=refined.INK,
        )
        result = "到达" if success else "未达"
        result_color = "#047857" if success else "#B91C1C"
        suffix = "  终距 0" if success else f"  终距 {final}"
        weight = "bold" if method == "proposed_linear_gate_pbrs" else "normal"
        ax.text(
            4.22,
            i,
            result + suffix,
            ha="left",
            va="center",
            fontsize=11.6,
            fontproperties=refined.CN_FONT,
            fontweight=weight,
            color=result_color,
            clip_on=False,
        )

    ax.text(
        4.22,
        -0.62,
        "结果",
        ha="left",
        va="center",
        fontsize=11.6,
        fontproperties=refined.CN_FONT,
        color=refined.MUTED,
        clip_on=False,
    )


def build_figure() -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PPT_PACK_DIR.mkdir(parents=True, exist_ok=True)
    refined.setup_style()

    _, rows_by_method, asset = smooth_fig.load_case(CASE_ID)
    image = base.open_overhead_image(asset)

    fig = plt.figure(figsize=(16, 9), facecolor=refined.PAPER)
    fig.text(
        0.035,
        0.970,
        "混合奖励机制如何指导训练动作",
        ha="left",
        va="top",
        fontsize=25,
        fontproperties=refined.CN_FONT,
        color=refined.INK,
    )
    fig.text(
        0.035,
        0.940,
        "平顺到达案例：距离持续收敛，比较不同奖励设置下的累计反馈",
        ha="left",
        va="top",
        fontsize=13.5,
        fontproperties=refined.CN_FONT,
        color=refined.MUTED,
    )
    fig.lines.append(Line2D([0.035, 0.965], [0.888, 0.888], transform=fig.transFigure, color="#CBD5E1", lw=1.1))

    ax_map = fig.add_axes([0.045, 0.365, 0.340, 0.500])
    refined.draw_route_panel(ax_map, image, rows_by_method)
    ax_dist = fig.add_axes([0.430, 0.625, 0.535, 0.240])
    smooth_fig.draw_smooth_distance_curve(ax_dist, rows_by_method)
    ax_reward = fig.add_axes([0.430, 0.365, 0.535, 0.220])
    smooth_fig.draw_smooth_reward_curve(ax_reward, rows_by_method)
    refined.draw_legends(fig)

    ax_total = fig.add_axes([0.075, 0.070, 0.760, 0.235])
    draw_method_total_comparison(ax_total, rows_by_method)

    stem = "19_奖励机制典型案例_case04_平顺版_累计总奖励对比"
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
