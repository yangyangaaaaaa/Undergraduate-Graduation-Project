#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build smooth-case reward figure and correction-ability comparison figure."""

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


def load_case(case_id: str) -> tuple[pd.Series, dict[str, pd.Series], base.ImageAsset]:
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

    asset = base.ImageAsset(
        str(case_row.get("dataset", "")),
        str(case_row.get("image_id", "")),
        Path(str(case_row["image_path"])),
        f"{case_row.get('dataset', '')} overhead",
    )
    return case_row, rows_by_method, asset


def style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=refined.GRID, linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(refined.NUM_FONT)


def draw_smooth_distance_curve(ax: plt.Axes, rows_by_method: dict[str, pd.Series]) -> None:
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
            linewidth=4.2 if key else 2.0,
            linestyle=refined.LINESTYLE[method],
            marker="o" if key else None,
            markersize=5.8,
            alpha=1.0 if key else 0.64,
            zorder=12 if key else 6,
        )
        final_marker = "o" if int(row["success"]) == 1 else "X"
        ax.scatter([x[-1]], [dist[-1]], s=92 if key else 58, marker=final_marker, color=color, edgecolor="white", linewidth=1.1, zorder=14)

    ax.axhline(0, color="#94A3B8", linewidth=1.0)
    ax.set_xlim(-0.15, max_step + 0.25)
    ax.set_ylim(-0.35, max_dist + 0.55)
    ax.set_yticks(range(0, int(max_dist) + 1))
    ax.set_xlabel("行动步", fontproperties=refined.CN_FONT, fontsize=12)
    ax.set_ylabel("到目标距离", fontproperties=refined.CN_FONT, fontsize=12)
    style_axes(ax)
    ax.annotate(
        "稳定接近",
        xy=(max_step, 0),
        xytext=(max_step - 2.35, 0.92),
        fontproperties=refined.CN_FONT,
        fontsize=13,
        color=refined.BLUE,
        arrowprops=dict(arrowstyle="-|>", color=refined.BLUE, lw=1.7, shrinkA=2, shrinkB=2),
    )


def draw_smooth_reward_curve(ax: plt.Axes, rows_by_method: dict[str, pd.Series]) -> None:
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
            linewidth=3.8 if key else 1.9,
            linestyle=refined.LINESTYLE[method],
            marker="o" if key else None,
            markersize=5.2,
            alpha=1.0 if key else 0.64,
            zorder=12 if key else 6,
        )
        if key:
            proposed_final = (int(x[-1]), float(total[-1]))

    vals = np.array(all_values, dtype=float) if all_values else np.array([0.0])
    ymin = min(-1.35, float(np.nanmin(vals)) - 0.25)
    ymax = max(2.45, float(np.nanmax(vals)) + 0.25)
    ax.axhline(0, color="#94A3B8", linewidth=1.0)
    ax.set_xlim(0.85, max_step + 0.25)
    ax.set_ylim(ymin, ymax)
    ax.set_yticks([-1, 0, 1, 2])
    ax.set_xlabel("行动步", fontproperties=refined.CN_FONT, fontsize=12)
    ax.set_ylabel("总奖励", fontproperties=refined.CN_FONT, fontsize=12)
    style_axes(ax)
    if proposed_final is not None:
        ax.annotate(
            "到达奖励最高",
            xy=proposed_final,
            xytext=(max_step - 2.55, proposed_final[1] - 0.62),
            fontproperties=refined.CN_FONT,
            fontsize=12.5,
            color=refined.BLUE,
            arrowprops=dict(arrowstyle="-|>", color=refined.BLUE, lw=1.5, shrinkA=2, shrinkB=2),
        )


def build_smooth_full_figure() -> tuple[Path, Path]:
    _, rows_by_method, asset = load_case(SMOOTH_CASE_ID)
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
        "平顺到达案例：距离持续收敛，关键进展获得奖励强化",
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
    draw_smooth_distance_curve(ax_dist, rows_by_method)
    ax_reward = fig.add_axes([0.430, 0.365, 0.535, 0.220])
    draw_smooth_reward_curve(ax_reward, rows_by_method)
    refined.draw_legends(fig)

    ax_table = fig.add_axes([0.035, 0.040, 0.930, 0.280])
    refined.draw_reward_ledger(ax_table, rows_by_method["proposed_linear_gate_pbrs"])

    stem = "15_奖励机制典型案例_case04_平顺版"
    out_png = OUT_DIR / f"{stem}.png"
    pack_png = PPT_PACK_DIR / f"{stem}.png"
    fig.savefig(out_png, dpi=240, facecolor=refined.PAPER)
    plt.close(fig)
    shutil.copy2(out_png, pack_png)
    return out_png, pack_png


def proposed_dist(rows_by_method: dict[str, pd.Series]) -> list[float]:
    row = rows_by_method["proposed_linear_gate_pbrs"]
    return [float(x) for x in base.parse_list(row.get("dist_sequence", ""))]


def proposed_reward(rows_by_method: dict[str, pd.Series]) -> list[float]:
    row = rows_by_method["proposed_linear_gate_pbrs"]
    return [float(x) for x in base.parse_list(row.get("step_reward_total", ""))]


def draw_comparison_legend(fig: plt.Figure) -> None:
    handles = [
        Line2D([0], [0], color=refined.GREEN, lw=4.3, linestyle=(0, (6, 3)), marker="o", markersize=6, label="平顺到达"),
        Line2D([0], [0], color=refined.BLUE, lw=4.6, linestyle="solid", marker="o", markersize=6, label="偏离后纠偏"),
    ]
    leg = fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.500, 0.898),
        ncol=2,
        frameon=False,
        fontsize=14,
        handlelength=2.8,
        columnspacing=2.0,
    )
    for text in leg.get_texts():
        text.set_fontproperties(refined.CN_FONT)
        text.set_fontsize(14)


def build_correction_comparison_figure() -> tuple[Path, Path]:
    _, smooth_rows, _ = load_case(SMOOTH_CASE_ID)
    _, correction_rows, _ = load_case(CORRECTION_CASE_ID)

    smooth_dist = proposed_dist(smooth_rows)
    correction_dist = proposed_dist(correction_rows)
    smooth_reward = proposed_reward(smooth_rows)
    correction_reward = proposed_reward(correction_rows)

    fig = plt.figure(figsize=(16, 9), facecolor=refined.PAPER)
    fig.text(
        0.055,
        0.955,
        "本文方法的纠偏能力对比",
        ha="left",
        va="top",
        fontsize=25,
        fontproperties=refined.CN_FONT,
        color=refined.INK,
    )
    fig.text(
        0.055,
        0.918,
        "同一算法在平顺路径与短暂偏离路径中的距离变化和奖励反馈",
        ha="left",
        va="top",
        fontsize=13.5,
        fontproperties=refined.CN_FONT,
        color=refined.MUTED,
    )
    draw_comparison_legend(fig)
    fig.lines.append(Line2D([0.055, 0.945], [0.852, 0.852], transform=fig.transFigure, color="#CBD5E1", lw=1.1))

    ax_dist = fig.add_axes([0.065, 0.145, 0.420, 0.690])
    sx = np.arange(0, len(smooth_dist))
    cx = np.arange(0, len(correction_dist))
    ax_dist.plot(sx, smooth_dist, color=refined.GREEN, linewidth=4.3, linestyle=(0, (6, 3)), marker="o", markersize=6.2, zorder=8)
    ax_dist.plot(cx, correction_dist, color=refined.BLUE, linewidth=4.6, linestyle="solid", marker="o", markersize=6.4, zorder=10)
    ax_dist.axhline(0, color="#94A3B8", linewidth=1.0)
    ax_dist.axvspan(5.65, 6.35, color=refined.ORANGE, alpha=0.075, zorder=0)
    ax_dist.set_xlim(-0.18, max(len(smooth_dist), len(correction_dist)) - 0.75)
    ax_dist.set_ylim(-0.35, max(max(smooth_dist), max(correction_dist)) + 0.65)
    ax_dist.set_yticks(range(0, int(max(max(smooth_dist), max(correction_dist))) + 1, 2))
    ax_dist.set_xlabel("行动步", fontproperties=refined.CN_FONT, fontsize=14)
    ax_dist.set_ylabel("到目标距离", fontproperties=refined.CN_FONT, fontsize=14)
    style_axes(ax_dist)
    ax_dist.annotate(
        "短暂偏离",
        xy=(6, correction_dist[6]),
        xytext=(6.45, correction_dist[6] + 1.15),
        fontproperties=refined.CN_FONT,
        fontsize=14,
        color=refined.ORANGE,
        arrowprops=dict(arrowstyle="-|>", color=refined.ORANGE, lw=1.7, shrinkA=2, shrinkB=2),
    )
    ax_dist.annotate(
        "纠偏到达",
        xy=(len(correction_dist) - 1, correction_dist[-1]),
        xytext=(len(correction_dist) - 3.2, 0.86),
        fontproperties=refined.CN_FONT,
        fontsize=14,
        color=refined.BLUE,
        arrowprops=dict(arrowstyle="-|>", color=refined.BLUE, lw=1.7, shrinkA=2, shrinkB=2),
    )

    ax_reward = fig.add_axes([0.545, 0.145, 0.420, 0.690])
    srx = np.arange(1, len(smooth_reward) + 1)
    crx = np.arange(1, len(correction_reward) + 1)
    ax_reward.plot(srx, smooth_reward, color=refined.GREEN, linewidth=4.3, linestyle=(0, (6, 3)), marker="o", markersize=6.2, zorder=8)
    ax_reward.plot(crx, correction_reward, color=refined.BLUE, linewidth=4.6, linestyle="solid", marker="o", markersize=6.4, zorder=10)
    vals = np.array(smooth_reward + correction_reward, dtype=float)
    ax_reward.axhline(0, color="#94A3B8", linewidth=1.0)
    ax_reward.axvspan(5.65, 6.35, color=refined.ORANGE, alpha=0.075, zorder=0)
    ax_reward.set_xlim(0.85, max(len(smooth_reward), len(correction_reward)) + 0.25)
    ax_reward.set_ylim(min(-1.35, float(vals.min()) - 0.25), max(2.45, float(vals.max()) + 0.25))
    ax_reward.set_yticks([-1, 0, 1, 2])
    ax_reward.set_xlabel("行动步", fontproperties=refined.CN_FONT, fontsize=14)
    ax_reward.set_ylabel("总奖励", fontproperties=refined.CN_FONT, fontsize=14)
    style_axes(ax_reward)
    ax_reward.annotate(
        "偏离受惩罚",
        xy=(6, correction_reward[5]),
        xytext=(3.95, correction_reward[5] - 0.48),
        fontproperties=refined.CN_FONT,
        fontsize=14,
        color=refined.ORANGE,
        arrowprops=dict(arrowstyle="-|>", color=refined.ORANGE, lw=1.7, shrinkA=2, shrinkB=2),
    )
    ax_reward.annotate(
        "恢复后强化",
        xy=(len(correction_reward), correction_reward[-1]),
        xytext=(len(correction_reward) - 3.05, correction_reward[-1] - 0.62),
        fontproperties=refined.CN_FONT,
        fontsize=14,
        color=refined.BLUE,
        arrowprops=dict(arrowstyle="-|>", color=refined.BLUE, lw=1.7, shrinkA=2, shrinkB=2),
    )

    stem = "16_本文方法纠偏能力_距离奖励对比"
    out_png = OUT_DIR / f"{stem}.png"
    pack_png = PPT_PACK_DIR / f"{stem}.png"
    fig.savefig(out_png, dpi=240, facecolor=refined.PAPER)
    plt.close(fig)
    shutil.copy2(out_png, pack_png)
    return out_png, pack_png


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PPT_PACK_DIR.mkdir(parents=True, exist_ok=True)
    refined.setup_style()
    outputs = [build_smooth_full_figure(), build_correction_comparison_figure()]
    for out_png, pack_png in outputs:
        print(f"saved: {out_png}")
        print(f"copied: {pack_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
