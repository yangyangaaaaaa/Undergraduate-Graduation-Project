#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build an intuitive PPT figure for global stepwise reward attribution."""

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
from matplotlib.patches import Rectangle

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import build_reward_guided_case_studies as base  # noqa: E402
import build_reward_mechanism_single_case_refined as refined  # noqa: E402


METHODS = [
    "external_only",
    "intrinsic_only",
    "mixed_no_gate_no_pbrs",
    "mixed_pbrs_only",
    "proposed_linear_gate_pbrs",
]

METHOD_CN = {
    "external_only": "仅外在",
    "intrinsic_only": "仅好奇心",
    "mixed_no_gate_no_pbrs": "直接相加",
    "mixed_pbrs_only": "仅塑形",
    "proposed_linear_gate_pbrs": "本文方法",
}

DIAGNOSIS = {
    "external_only": "中间接近仍偏负",
    "intrinsic_only": "回退也被鼓励",
    "mixed_no_gate_no_pbrs": "好奇心削弱惩罚",
    "mixed_pbrs_only": "塑形不足以收敛",
    "proposed_linear_gate_pbrs": "反馈对齐目标",
}

TABLE_DIR = base.TABLES
OUT_DIR = base.RESULTS / "figures" / "defense_reward_training_stage" / "single_reward_mechanism"
PPT_PACK_DIR = (
    base.RESULTS
    / "figures"
    / "ppt_candidate_pack_20260606"
    / "03_奖励机制_动作归因与GP故事"
)


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    method = pd.read_csv(TABLE_DIR / "reward_stepwise_global_method_summary.csv")
    component = pd.read_csv(TABLE_DIR / "reward_stepwise_global_component_by_move.csv")
    method = method[method["method"].isin(METHODS)].copy()
    method["method"] = pd.Categorical(method["method"], categories=METHODS, ordered=True)
    method = method.sort_values("method")
    component = component[component["method"].isin(METHODS)].copy()
    return method, component


def tile_color(value: float) -> tuple[float, float, float]:
    color = "#16A34A" if value >= 0 else "#DC2626"
    strength = 0.12 + 0.30 * min(abs(value) / 2.4, 1.0)
    return refined.blend(color, strength)


def draw_text(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 12,
    color: str = refined.INK,
    ha: str = "center",
    va: str = "center",
    font=None,
    weight: str = "normal",
) -> None:
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=size,
        color=color,
        fontproperties=font or refined.CN_FONT,
        fontweight=weight,
    )


def draw_feedback_matrix(ax: plt.Axes, method: pd.DataFrame) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    left = 0.015
    top = 0.825
    bottom = 0.070
    method_w = 0.155
    cell_w = 0.158
    diag_w = 0.245
    gap = 0.010
    row_h = (top - bottom) / len(method)

    headers = [
        ("方法", method_w),
        ("接近/到达\n应为正", cell_w),
        ("回退\n应为负", cell_w),
        ("最终到达\n应强正", cell_w),
        ("动作反馈诊断", diag_w),
    ]
    x = left
    for label, width in headers:
        draw_text(ax, x + width / 2, top + 0.055, label, size=12.5, color=refined.MUTED)
        x += width + gap

    for r, (_, row) in enumerate(method.iterrows()):
        method_key = str(row["method"])
        y = top - (r + 1) * row_h
        is_key = method_key == "proposed_linear_gate_pbrs"
        method_color = base.METHOD_STYLE[method_key]["color"]
        values = [
            float(row["接近/到达步平均总奖励"]),
            float(row["回退步平均总奖励"]),
            float(row["到达步平均总奖励"]),
        ]

        x = left
        ax.add_patch(Rectangle((x, y), method_w, row_h - 0.010, facecolor=refined.blend(method_color, 0.10 if not is_key else 0.18), edgecolor="#CBD5E1", linewidth=0.9))
        ax.add_patch(Rectangle((x, y), 0.010, row_h - 0.010, facecolor=method_color, edgecolor=method_color, linewidth=0))
        draw_text(ax, x + method_w * 0.54, y + row_h / 2, METHOD_CN[method_key], size=13.2, weight="bold" if is_key else "normal")
        x += method_w + gap

        for c, value in enumerate(values):
            edge = "#E2E8F0"
            lw = 0.9
            if c == 1 and value > 0:
                edge = "#B91C1C"
                lw = 2.2
            if is_key and c in [1, 2]:
                edge = refined.BLUE if c == 2 else "#0F766E"
                lw = 1.7
            ax.add_patch(Rectangle((x, y), cell_w, row_h - 0.010, facecolor=tile_color(value), edgecolor=edge, linewidth=lw))
            draw_text(ax, x + cell_w / 2, y + row_h * 0.58, f"{value:+.2f}", size=15.2, font=refined.NUM_FONT, weight="bold" if is_key else "normal")
            if c == 1 and value > 0:
                draw_text(ax, x + cell_w / 2, y + row_h * 0.28, "误奖励", size=10.7, color="#B91C1C", weight="bold")
            elif c == 2:
                draw_text(ax, x + cell_w / 2, y + row_h * 0.28, "强正" if value > 1.5 else "偏弱", size=10.7, color=refined.MUTED)
            x += cell_w + gap

        face = refined.blend(refined.BLUE, 0.08) if is_key else "#FFFFFF"
        ax.add_patch(Rectangle((x, y), diag_w, row_h - 0.010, facecolor=face, edgecolor=refined.BLUE if is_key else "#D8DEE9", linewidth=1.5 if is_key else 0.9))
        draw_text(ax, x + 0.022, y + row_h / 2, DIAGNOSIS[method_key], size=12.7, ha="left", weight="bold" if is_key else "normal", color=refined.BLUE if is_key else refined.INK)

    ax.text(
        left + method_w + gap + cell_w * 1.5 + gap,
        0.018,
        "单元格为全量动作步的平均总奖励；绿色表示正反馈，红色表示负反馈",
        ha="center",
        va="bottom",
        fontsize=11.4,
        fontproperties=refined.CN_FONT,
        color=refined.MUTED,
    )


def setup_numeric_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(refined.NUM_FONT)


def draw_wrong_reward_rate(ax: plt.Axes, method: pd.DataFrame) -> None:
    labels = [METHOD_CN[m] for m in method["method"].astype(str)]
    values = method["回退步正奖励率%"].astype(float).to_numpy()
    colors = [base.METHOD_STYLE[m]["color"] for m in method["method"].astype(str)]
    y = np.arange(len(method))

    ax.barh(y, values, height=0.54, color=colors, alpha=0.86, zorder=3)
    ax.set_xlim(0, 105)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontproperties=refined.CN_FONT, fontsize=11.5)
    ax.invert_yaxis()
    ax.grid(axis="x", color=refined.GRID, linewidth=0.8)
    ax.set_xlabel("正奖励率（%）", fontproperties=refined.CN_FONT, fontsize=11.2, color=refined.MUTED)
    ax.set_title("回退动作被正向奖励的比例", loc="left", fontproperties=refined.CN_FONT, fontsize=15.5, color=refined.INK, pad=8)
    setup_numeric_axes(ax)
    for label in ax.get_yticklabels():
        label.set_fontproperties(refined.CN_FONT)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    for yi, value in zip(y, values):
        ax.text(min(value + 2, 101.5), yi, f"{value:.2f}%", ha="left", va="center", fontproperties=refined.NUM_FONT, fontsize=10.8, color=refined.INK)
    ax.text(102, -0.82, "越低越好", ha="right", va="center", fontproperties=refined.CN_FONT, fontsize=11.0, color=refined.MUTED)


def draw_proposed_components(ax: plt.Axes, component: pd.DataFrame) -> None:
    proposed = component[component["method"].eq("proposed_linear_gate_pbrs")].copy()
    proposed["动作类型"] = pd.Categorical(proposed["动作类型"], categories=["progress", "regress", "goal"], ordered=True)
    proposed = proposed.sort_values("动作类型")

    labels = {"progress": "接近", "regress": "回退", "goal": "到达"}
    x = np.arange(len(proposed))
    parts = [
        ("reward_ex_mean", "外在", refined.ORANGE),
        ("reward_in_gated_mean", "好奇心×门控", refined.PURPLE),
        ("pbrs_bonus_mean", "塑形", refined.BLUE),
    ]
    pos_bottom = np.zeros(len(proposed))
    neg_bottom = np.zeros(len(proposed))
    for col, label, color in parts:
        vals = proposed[col].astype(float).to_numpy()
        bottoms = np.where(vals >= 0, pos_bottom, neg_bottom)
        ax.bar(x, vals, width=0.54, bottom=bottoms, color=color, alpha=0.84, edgecolor="white", linewidth=0.8, label=label, zorder=3)
        pos_bottom += np.where(vals >= 0, vals, 0)
        neg_bottom += np.where(vals < 0, vals, 0)

    totals = proposed["reward_total_mean"].astype(float).to_numpy()
    ax.scatter(x, totals, s=78, marker="D", color=refined.INK, edgecolor="white", linewidth=0.8, label="总奖励", zorder=5)
    for xi, total in zip(x, totals):
        ax.text(xi, total + (0.16 if total >= 0 else -0.16), f"{total:+.2f}", ha="center", va="bottom" if total >= 0 else "top", fontproperties=refined.NUM_FONT, fontsize=10.8, color=refined.INK)

    ax.axhline(0, color="#94A3B8", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[str(t)] for t in proposed["动作类型"].astype(str)], fontproperties=refined.CN_FONT, fontsize=11.7)
    ax.set_ylim(-1.25, 2.45)
    ax.set_ylabel("平均奖励", fontproperties=refined.CN_FONT, fontsize=11.2, color=refined.MUTED)
    ax.set_title("本文方法的奖励组成", loc="left", fontproperties=refined.CN_FONT, fontsize=15.5, color=refined.INK, pad=8)
    ax.grid(axis="y", color=refined.GRID, linewidth=0.8)
    setup_numeric_axes(ax)
    for label in ax.get_xticklabels():
        label.set_fontproperties(refined.CN_FONT)
    leg = ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=10.3, handlelength=1.2, columnspacing=0.8)
    for text in leg.get_texts():
        text.set_fontproperties(refined.CN_FONT)


def draw_bottom_sentence(ax: plt.Axes) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(Rectangle((0.015, 0.18), 0.970, 0.62, facecolor=refined.blend(refined.BLUE, 0.06), edgecolor="#D8DEE9", linewidth=1.0))
    ax.add_patch(Rectangle((0.015, 0.18), 0.010, 0.62, facecolor=refined.BLUE, edgecolor=refined.BLUE, linewidth=0))
    ax.text(
        0.045,
        0.50,
        "结论：本文方法不是让所有奖励变大，而是让回退保持负反馈、让到达动作获得强正反馈，从而把探索信号对齐到目标收敛。",
        ha="left",
        va="center",
        fontsize=13.2,
        fontproperties=refined.CN_FONT,
        color=refined.INK,
    )


def build_figure() -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PPT_PACK_DIR.mkdir(parents=True, exist_ok=True)
    refined.setup_style()

    method, component = load_tables()
    n_steps = len(pd.read_csv(TABLE_DIR / "reward_stepwise_global_steps.csv", usecols=["method"]))

    fig = plt.figure(figsize=(16, 9), facecolor=refined.PAPER)
    fig.text(0.038, 0.960, "全量逐步奖励归因：奖励是否给到正确动作", ha="left", va="top", fontsize=24.5, fontproperties=refined.CN_FONT, color=refined.INK)
    fig.text(0.038, 0.922, f"训练路线采样 {n_steps:,} 个动作步；比较每类动作获得的总奖励反馈", ha="left", va="top", fontsize=13.2, fontproperties=refined.CN_FONT, color=refined.MUTED)
    fig.lines.append(Line2D([0.038, 0.962], [0.880, 0.880], transform=fig.transFigure, color="#CBD5E1", lw=1.1))

    ax_matrix = fig.add_axes([0.035, 0.235, 0.610, 0.650])
    draw_feedback_matrix(ax_matrix, method)

    ax_wrong = fig.add_axes([0.685, 0.530, 0.285, 0.330])
    draw_wrong_reward_rate(ax_wrong, method)

    ax_comp = fig.add_axes([0.685, 0.205, 0.285, 0.290])
    draw_proposed_components(ax_comp, component)

    ax_bottom = fig.add_axes([0.050, 0.040, 0.910, 0.120])
    draw_bottom_sentence(ax_bottom)

    stem = "23_全量逐步奖励归因_直观反馈带"
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
