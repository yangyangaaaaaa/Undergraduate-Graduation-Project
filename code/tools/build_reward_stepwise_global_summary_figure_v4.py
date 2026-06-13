#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build an intuitive reward-fingerprint figure for global stepwise attribution."""

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

SHORT_NOTE = {
    "external_only": "中间反馈稀疏",
    "intrinsic_only": "回退也被鼓励",
    "mixed_no_gate_no_pbrs": "惩罚被好奇心削弱",
    "mixed_pbrs_only": "方向信号仍不足",
    "proposed_linear_gate_pbrs": "回退负、到达强正",
}

TABLE_DIR = base.TABLES
OUT_DIR = base.RESULTS / "figures" / "defense_reward_training_stage" / "single_reward_mechanism"
PPT_PACK_DIR = (
    base.RESULTS
    / "figures"
    / "ppt_candidate_pack_20260606"
    / "03_奖励机制_动作归因与GP故事"
)


def load_method_summary() -> pd.DataFrame:
    df = pd.read_csv(TABLE_DIR / "reward_stepwise_global_method_summary.csv")
    df = df[df["method"].isin(METHODS)].copy()
    df["method"] = pd.Categorical(df["method"], categories=METHODS, ordered=True)
    return df.sort_values("method")


def setup_style() -> None:
    refined.setup_style()
    plt.rcParams.update({"axes.unicode_minus": False})


def draw_reward_fingerprint(ax: plt.Axes, df: pd.DataFrame) -> None:
    methods = [str(x) for x in df["method"]]
    labels = [METHOD_CN[m] for m in methods]
    y = np.arange(len(methods))[::-1]

    regress = df["回退步平均总奖励"].astype(float).to_numpy()
    progress = df["接近/到达步平均总奖励"].astype(float).to_numpy()
    goal = df["到达步平均总奖励"].astype(float).to_numpy()
    wrong = df["回退步正奖励率%"].astype(float).to_numpy()

    ax.set_xlim(-1.20, 3.12)
    ax.set_ylim(-0.75, len(methods) - 0.25)
    ax.axvspan(-1.20, 0.0, color="#FEE2E2", alpha=0.52, zorder=0)
    ax.axvspan(0.0, 3.12, color="#DCFCE7", alpha=0.36, zorder=0)
    ax.axvline(0, color="#64748B", linewidth=1.2, zorder=1)
    ax.grid(axis="x", color="#CBD5E1", linewidth=0.8, alpha=0.75)

    key_idx = methods.index("proposed_linear_gate_pbrs")
    ax.axhspan(y[key_idx] - 0.38, y[key_idx] + 0.38, color=refined.blend(refined.BLUE, 0.10), zorder=0)

    for i, method in enumerate(methods):
        yi = y[i]
        method_color = base.METHOD_STYLE[method]["color"]
        ax.plot([regress[i], goal[i]], [yi, yi], color="#94A3B8", linewidth=2.0, alpha=0.55, zorder=2)
        ax.scatter([regress[i]], [yi], s=150, color="#DC2626", edgecolor="white", linewidth=1.5, zorder=5)
        ax.scatter([progress[i]], [yi], s=150, color="#16A34A", edgecolor="white", linewidth=1.5, zorder=6)
        ax.scatter([goal[i]], [yi], s=230, marker="*", color=method_color if method != "proposed_linear_gate_pbrs" else refined.BLUE, edgecolor="white", linewidth=1.3, zorder=7)

        ax.text(regress[i] - 0.055, yi - 0.22, f"{regress[i]:+.2f}", ha="right", va="center", fontsize=10.8, fontproperties=refined.NUM_FONT, color="#991B1B")
        ax.text(progress[i] + 0.055, yi + 0.20, f"{progress[i]:+.2f}", ha="left", va="center", fontsize=10.8, fontproperties=refined.NUM_FONT, color="#166534")
        ax.text(goal[i] + 0.060, yi, f"{goal[i]:+.2f}", ha="left", va="center", fontsize=11.2, fontproperties=refined.NUM_FONT, color=refined.INK)

        wrong_color = "#B91C1C" if wrong[i] > 50 else refined.MUTED
        ax.text(2.98, yi + 0.17, f"误奖 {wrong[i]:.2f}%", ha="right", va="center", fontsize=11.0, fontproperties=refined.CN_FONT, color=wrong_color)
        ax.text(2.98, yi - 0.17, SHORT_NOTE[method], ha="right", va="center", fontsize=10.8, fontproperties=refined.CN_FONT, color=wrong_color if wrong[i] > 50 else refined.MUTED)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontproperties=refined.CN_FONT, fontsize=13.8)
    for tick, method in zip(ax.get_yticklabels(), methods):
        tick.set_color(base.METHOD_STYLE[method]["color"])
        if method == "proposed_linear_gate_pbrs":
            tick.set_fontweight("bold")

    ax.set_xlabel("该类动作的平均总奖励", fontproperties=refined.CN_FONT, fontsize=13, color=refined.MUTED)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=11.5)
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(refined.NUM_FONT)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#CBD5E1")

    ax.text(-0.72, len(methods) - 0.12, "负反馈区\n回退应在这里", ha="center", va="top", fontsize=11.5, fontproperties=refined.CN_FONT, color="#991B1B")
    ax.text(2.18, len(methods) - 0.12, "正反馈区\n到达应在这里", ha="center", va="top", fontsize=11.5, fontproperties=refined.CN_FONT, color="#166534")


def draw_legend(fig: plt.Figure) -> None:
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#DC2626", markeredgecolor="white", markersize=10, label="回退动作"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#16A34A", markeredgecolor="white", markersize=10, label="接近/到达动作"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=refined.BLUE, markeredgecolor="white", markersize=14, label="最终到达动作"),
    ]
    leg = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.535, 0.885), ncol=3, frameon=False, fontsize=12.2, handletextpad=0.45, columnspacing=1.6)
    for text in leg.get_texts():
        text.set_fontproperties(refined.CN_FONT)


def draw_bottom_sentence(fig: plt.Figure) -> None:
    fig.text(
        0.070,
        0.090,
        "读图方式：红点越靠左越好，蓝星越靠右越好；纯好奇心的红点落在正反馈区，说明回退也被奖励。",
        ha="left",
        va="center",
        fontsize=13.2,
        fontproperties=refined.CN_FONT,
        color=refined.INK,
    )
    fig.text(
        0.070,
        0.052,
        "本文方法的关键不是总奖励最大，而是保持回退负反馈，同时让最终到达获得强正反馈。",
        ha="left",
        va="center",
        fontsize=13.2,
        fontproperties=refined.CN_FONT,
        color=refined.BLUE,
    )


def build_figure() -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PPT_PACK_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    df = load_method_summary()
    n_steps = len(pd.read_csv(TABLE_DIR / "reward_stepwise_global_steps.csv", usecols=["method"]))

    fig = plt.figure(figsize=(16, 9), facecolor=refined.PAPER)
    fig.text(0.045, 0.960, "全量动作级奖励指纹：奖励有没有给到正确动作", ha="left", va="top", fontsize=25, fontproperties=refined.CN_FONT, color=refined.INK)
    fig.text(0.045, 0.922, f"训练路线采样 {n_steps:,} 个动作步；每行展示三类动作获得的平均总奖励", ha="left", va="top", fontsize=13.5, fontproperties=refined.CN_FONT, color=refined.MUTED)
    fig.lines.append(Line2D([0.045, 0.955], [0.875, 0.875], transform=fig.transFigure, color="#CBD5E1", lw=1.1))
    draw_legend(fig)

    ax = fig.add_axes([0.115, 0.165, 0.815, 0.650])
    draw_reward_fingerprint(ax, df)
    draw_bottom_sentence(fig)

    stem = "25_全量动作级奖励指纹_直观版"
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
