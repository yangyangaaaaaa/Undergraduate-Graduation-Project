#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build one PPT figure summarizing global stepwise reward attribution."""

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

TABLE_DIR = base.TABLES
OUT_DIR = base.RESULTS / "figures" / "defense_reward_training_stage" / "single_reward_mechanism"
PPT_PACK_DIR = (
    base.RESULTS
    / "figures"
    / "ppt_candidate_pack_20260606"
    / "03_奖励机制_动作归因与GP故事"
)


def setup_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for label in ax.get_xticklabels():
        label.set_fontproperties(refined.NUM_FONT)


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    method = pd.read_csv(TABLE_DIR / "reward_stepwise_global_method_summary.csv")
    component = pd.read_csv(TABLE_DIR / "reward_stepwise_global_component_by_move.csv")
    method = method[method["method"].isin(METHODS)].copy()
    method["method"] = pd.Categorical(method["method"], categories=METHODS, ordered=True)
    method = method.sort_values("method")
    component = component[component["method"].isin(METHODS)].copy()
    return method, component


def draw_reward_alignment(ax: plt.Axes, method: pd.DataFrame) -> None:
    y = np.arange(len(method))
    progress = method["接近/到达步平均总奖励"].astype(float).to_numpy()
    regress = method["回退步平均总奖励"].astype(float).to_numpy()
    labels = [METHOD_CN[m] for m in method["method"].astype(str)]

    ax.axvline(0, color="#94A3B8", linewidth=1.1, zorder=0)
    ax.barh(y - 0.18, progress, height=0.30, color="#16A34A", alpha=0.86, label="接近/到达步", zorder=3)
    ax.barh(y + 0.18, regress, height=0.30, color="#DC2626", alpha=0.78, label="回退步", zorder=3)

    key_idx = labels.index("本文方法")
    ax.add_patch(Rectangle((-1.20, key_idx - 0.48), 1.98, 0.96, facecolor=refined.blend(refined.BLUE, 0.08), edgecolor=refined.BLUE, linewidth=1.4, zorder=1))

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontproperties=refined.CN_FONT, fontsize=12.8)
    ax.set_xlim(-1.12, 0.68)
    ax.invert_yaxis()
    ax.grid(axis="x", color=refined.GRID, linewidth=0.85)
    ax.set_xlabel("平均总奖励", fontproperties=refined.CN_FONT, fontsize=12.2, color=refined.MUTED)
    ax.set_title("正确动作与回退动作的奖励分离", loc="left", fontproperties=refined.CN_FONT, fontsize=17, color=refined.INK, pad=10)
    setup_axes(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    for yi, p, r in zip(y, progress, regress):
        ax.text(p + 0.035 if p >= 0 else p - 0.035, yi - 0.18, f"{p:+.2f}", ha="left" if p >= 0 else "right", va="center", fontproperties=refined.NUM_FONT, fontsize=11.2, color=refined.INK)
        ax.text(r + 0.035 if r >= 0 else r - 0.035, yi + 0.18, f"{r:+.2f}", ha="left" if r >= 0 else "right", va="center", fontproperties=refined.NUM_FONT, fontsize=11.2, color=refined.INK)

    leg = ax.legend(loc="lower right", frameon=False, fontsize=11.4, handlelength=1.5)
    for text in leg.get_texts():
        text.set_fontproperties(refined.CN_FONT)


def draw_regress_positive_rate(ax: plt.Axes, method: pd.DataFrame) -> None:
    labels = [METHOD_CN[m] for m in method["method"].astype(str)]
    values = method["回退步正奖励率%"].astype(float).to_numpy()
    y = np.arange(len(method))
    colors = [base.METHOD_STYLE[m]["color"] for m in method["method"].astype(str)]

    ax.barh(y, values, height=0.50, color=colors, alpha=0.82, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontproperties=refined.CN_FONT, fontsize=11.4)
    ax.set_xlim(0, 105)
    ax.invert_yaxis()
    ax.grid(axis="x", color=refined.GRID, linewidth=0.8)
    ax.set_xlabel("正奖励率（%）", fontproperties=refined.CN_FONT, fontsize=11.3, color=refined.MUTED)
    ax.set_title("回退动作是否被误奖励", loc="left", fontproperties=refined.CN_FONT, fontsize=15.5, color=refined.INK, pad=9)
    setup_axes(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    for yi, value in zip(y, values):
        label = f"{value:.2f}%"
        ax.text(min(value + 2.0, 101.5), yi, label, ha="left", va="center", fontproperties=refined.NUM_FONT, fontsize=10.8, color=refined.INK)


def draw_proposed_components(ax: plt.Axes, component: pd.DataFrame) -> None:
    proposed = component[component["method"].eq("proposed_linear_gate_pbrs")].copy()
    proposed["动作类型"] = pd.Categorical(proposed["动作类型"], categories=["progress", "regress", "goal"], ordered=True)
    proposed = proposed.sort_values("动作类型")

    labels = {"progress": "接近", "regress": "回退", "goal": "到达"}
    x = np.arange(len(proposed))
    components = [
        ("reward_ex_mean", "外在", refined.ORANGE),
        ("reward_in_gated_mean", "好奇心×门控", refined.PURPLE),
        ("pbrs_bonus_mean", "塑形", refined.BLUE),
    ]
    pos_bottom = np.zeros(len(proposed))
    neg_bottom = np.zeros(len(proposed))
    for col, name, color in components:
        vals = proposed[col].astype(float).to_numpy()
        bottoms = np.where(vals >= 0, pos_bottom, neg_bottom)
        ax.bar(x, vals, width=0.54, bottom=bottoms, color=color, alpha=0.82, label=name, edgecolor="white", linewidth=0.8, zorder=3)
        pos_bottom += np.where(vals >= 0, vals, 0)
        neg_bottom += np.where(vals < 0, vals, 0)

    totals = proposed["reward_total_mean"].astype(float).to_numpy()
    ax.scatter(x, totals, s=80, marker="D", color=refined.INK, edgecolor="white", linewidth=0.8, zorder=5, label="总奖励")
    for xi, total in zip(x, totals):
        ax.text(xi, total + (0.16 if total >= 0 else -0.16), f"{total:+.2f}", ha="center", va="bottom" if total >= 0 else "top", fontproperties=refined.NUM_FONT, fontsize=11.0, color=refined.INK)

    ax.axhline(0, color="#94A3B8", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[str(t)] for t in proposed["动作类型"].astype(str)], fontproperties=refined.CN_FONT, fontsize=11.8)
    ax.set_ylim(-1.25, 2.45)
    ax.set_ylabel("平均奖励", fontproperties=refined.CN_FONT, fontsize=11.3, color=refined.MUTED)
    ax.set_title("本文方法：奖励项如何合成动作反馈", loc="left", fontproperties=refined.CN_FONT, fontsize=15.5, color=refined.INK, pad=9)
    ax.grid(axis="y", color=refined.GRID, linewidth=0.8)
    setup_axes(ax)
    for label in ax.get_xticklabels():
        label.set_fontproperties(refined.CN_FONT)
    leg = ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=10.6, handlelength=1.4, columnspacing=1.0)
    for text in leg.get_texts():
        text.set_fontproperties(refined.CN_FONT)


def draw_conclusion(ax: plt.Axes) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    items = [
        ("仅好奇心", "回退步也几乎全为正奖励，探索没有目标对齐", refined.PURPLE),
        ("仅外在", "接近/到达步均值仍为负，训练信号稀疏", refined.ORANGE),
        ("本文方法", "回退保持负反馈，到达动作获得强正反馈", refined.BLUE),
    ]
    for i, (title, body, color) in enumerate(items):
        x = 0.020 + i * 0.326
        ax.add_patch(Rectangle((x, 0.16), 0.302, 0.68, facecolor=refined.blend(color, 0.08), edgecolor="#D8DEE9", linewidth=1.0))
        ax.add_patch(Rectangle((x, 0.16), 0.010, 0.68, facecolor=color, edgecolor=color, linewidth=0))
        ax.text(x + 0.026, 0.62, title, ha="left", va="center", fontproperties=refined.CN_FONT, fontsize=12.8, color=refined.INK, fontweight="bold")
        ax.text(x + 0.026, 0.39, body, ha="left", va="center", fontproperties=refined.CN_FONT, fontsize=11.5, color=refined.INK)


def build_figure() -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PPT_PACK_DIR.mkdir(parents=True, exist_ok=True)
    refined.setup_style()

    method, component = load_tables()
    steps_path = TABLE_DIR / "reward_stepwise_global_steps.csv"
    n_steps = len(pd.read_csv(steps_path, usecols=["method"]))

    fig = plt.figure(figsize=(16, 9), facecolor=refined.PAPER)
    fig.text(0.040, 0.958, "全量逐步奖励归因：正反馈是否落在正确动作上", ha="left", va="top", fontsize=24.5, fontproperties=refined.CN_FONT, color=refined.INK)
    fig.text(0.040, 0.920, f"训练路线采样 {n_steps:,} 个动作步；按动作级奖励统计不同机制的反馈差异", ha="left", va="top", fontsize=13.2, fontproperties=refined.CN_FONT, color=refined.MUTED)
    fig.lines.append(Line2D([0.040, 0.960], [0.875, 0.875], transform=fig.transFigure, color="#CBD5E1", lw=1.1))

    ax_align = fig.add_axes([0.060, 0.280, 0.525, 0.590])
    draw_reward_alignment(ax_align, method)

    ax_regress = fig.add_axes([0.635, 0.535, 0.320, 0.315])
    draw_regress_positive_rate(ax_regress, method)

    ax_comp = fig.add_axes([0.635, 0.180, 0.320, 0.310])
    draw_proposed_components(ax_comp, component)

    ax_bottom = fig.add_axes([0.055, 0.035, 0.900, 0.120])
    draw_conclusion(ax_bottom)

    stem = "22_全量逐步奖励归因_动作反馈对齐"
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
