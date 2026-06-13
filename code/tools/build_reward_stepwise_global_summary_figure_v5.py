#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build a mid-stage curiosity-focused reward attribution figure."""

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

NOTE = {
    "external_only": "中间接近仍偏负",
    "intrinsic_only": "回退也给正反馈",
    "mixed_no_gate_no_pbrs": "惩罚被好奇心削弱",
    "mixed_pbrs_only": "补方向但不门控",
    "proposed_linear_gate_pbrs": "接近被补偿，回退仍为负",
}

TABLE_DIR = base.TABLES
OUT_DIR = base.RESULTS / "figures" / "defense_reward_training_stage" / "single_reward_mechanism"
PPT_PACK_DIR = (
    base.RESULTS
    / "figures"
    / "ppt_candidate_pack_20260606"
    / "03_奖励机制_动作归因与GP故事"
)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    comp = pd.read_csv(TABLE_DIR / "reward_stepwise_global_component_by_move.csv")
    comp = comp[comp["method"].isin(METHODS)].copy()
    comp["method"] = pd.Categorical(comp["method"], categories=METHODS, ordered=True)
    comp = comp.sort_values(["method", "动作类型"])
    steps = pd.read_csv(TABLE_DIR / "reward_stepwise_global_steps.csv", usecols=["method"])
    return comp, steps


def setup_style() -> None:
    refined.setup_style()
    plt.rcParams.update({"axes.unicode_minus": False})


def draw_midstage_fingerprint(ax: plt.Axes, comp: pd.DataFrame) -> None:
    rows = []
    for method in METHODS:
        hit = comp[comp["method"].astype(str).eq(method)]
        progress = hit[hit["动作类型"].eq("progress")].iloc[0]
        regress = hit[hit["动作类型"].eq("regress")].iloc[0]
        rows.append(
            {
                "method": method,
                "progress_total": float(progress["reward_total_mean"]),
                "regress_total": float(regress["reward_total_mean"]),
                "progress_n": int(progress["步数"]),
                "regress_n": int(regress["步数"]),
                "regress_pos": float(regress["正总奖励率%"]),
            }
        )

    y = np.arange(len(rows))[::-1]
    ax.set_xlim(-1.18, 0.72)
    ax.set_ylim(-0.70, len(rows) - 0.20)
    ax.axvspan(-1.18, 0, color="#FEE2E2", alpha=0.55, zorder=0)
    ax.axvspan(0, 0.72, color="#DCFCE7", alpha=0.42, zorder=0)
    ax.axvline(0, color="#64748B", linewidth=1.1, zorder=1)
    ax.grid(axis="x", color="#CBD5E1", linewidth=0.8, alpha=0.75)

    key_idx = METHODS.index("proposed_linear_gate_pbrs")
    ax.axhspan(y[key_idx] - 0.38, y[key_idx] + 0.38, color=refined.blend(refined.BLUE, 0.10), zorder=0)

    for i, row in enumerate(rows):
        method = row["method"]
        yi = y[i]
        method_color = base.METHOD_STYLE[method]["color"]
        p = row["progress_total"]
        r = row["regress_total"]
        ax.plot([r, p], [yi, yi], color="#94A3B8", linewidth=2.0, alpha=0.55, zorder=2)
        ax.scatter([r], [yi], s=180, color="#DC2626", edgecolor="white", linewidth=1.5, zorder=5)
        ax.scatter([p], [yi], s=180, color="#16A34A", edgecolor="white", linewidth=1.5, zorder=6)

        ax.text(r - 0.045 if r < 0 else r + 0.045, yi - 0.21, f"{r:+.2f}", ha="right" if r < 0 else "left", va="center", fontsize=11.0, fontproperties=refined.NUM_FONT, color="#991B1B")
        ax.text(p + 0.045 if p >= 0 else p - 0.045, yi + 0.20, f"{p:+.2f}", ha="left" if p >= 0 else "right", va="center", fontsize=11.0, fontproperties=refined.NUM_FONT, color="#166534")

        note_color = "#B91C1C" if row["regress_pos"] > 50 else (refined.BLUE if method == "proposed_linear_gate_pbrs" else refined.MUTED)
        ax.text(0.69, yi + 0.16, NOTE[method], ha="right", va="center", fontsize=10.9, fontproperties=refined.CN_FONT, color=note_color, fontweight="bold" if method == "proposed_linear_gate_pbrs" else "normal")
        ax.text(0.69, yi - 0.17, f"回退正奖励 {row['regress_pos']:.2f}%", ha="right", va="center", fontsize=10.7, fontproperties=refined.CN_FONT, color=note_color)

    ax.set_yticks(y)
    ax.set_yticklabels([METHOD_CN[m] for m in METHODS], fontproperties=refined.CN_FONT, fontsize=13.8)
    for tick, method in zip(ax.get_yticklabels(), METHODS):
        tick.set_color(base.METHOD_STYLE[method]["color"])
        if method == "proposed_linear_gate_pbrs":
            tick.set_fontweight("bold")

    ax.set_xlabel("中间动作平均总奖励（不含最终到达步）", fontproperties=refined.CN_FONT, fontsize=13, color=refined.MUTED)
    ax.set_title("中间阶段：接近动作是否被补偿，回退动作是否被压制", loc="left", fontproperties=refined.CN_FONT, fontsize=17, color=refined.INK, pad=10)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=11.5)
    for label in ax.get_xticklabels():
        label.set_fontproperties(refined.NUM_FONT)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#CBD5E1")

    ax.text(-0.63, len(rows) - 0.05, "负反馈区\n回退应在这里", ha="center", va="top", fontsize=11.4, fontproperties=refined.CN_FONT, color="#991B1B")
    ax.text(0.36, len(rows) - 0.05, "正反馈区\n接近可被鼓励", ha="center", va="top", fontsize=11.4, fontproperties=refined.CN_FONT, color="#166534")


def draw_proposed_decomposition(ax: plt.Axes, comp: pd.DataFrame) -> None:
    proposed = comp[comp["method"].astype(str).eq("proposed_linear_gate_pbrs")].copy()
    proposed = proposed[proposed["动作类型"].isin(["progress", "regress"])].copy()
    proposed["动作类型"] = pd.Categorical(proposed["动作类型"], categories=["progress", "regress"], ordered=True)
    proposed = proposed.sort_values("动作类型")

    labels = {"progress": "接近动作", "regress": "回退动作"}
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
        ax.bar(x, vals, width=0.50, bottom=bottoms, color=color, alpha=0.84, edgecolor="white", linewidth=0.8, label=label, zorder=3)
        pos_bottom += np.where(vals >= 0, vals, 0)
        neg_bottom += np.where(vals < 0, vals, 0)

    totals = proposed["reward_total_mean"].astype(float).to_numpy()
    ax.scatter(x, totals, s=90, marker="D", color=refined.INK, edgecolor="white", linewidth=0.8, label="总奖励", zorder=5)
    for xi, total in zip(x, totals):
        ax.text(xi, total + (0.12 if total >= 0 else -0.12), f"{total:+.2f}", ha="center", va="bottom" if total >= 0 else "top", fontproperties=refined.NUM_FONT, fontsize=11.4, color=refined.INK)

    ax.axhline(0, color="#94A3B8", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[str(t)] for t in proposed["动作类型"].astype(str)], fontproperties=refined.CN_FONT, fontsize=12.2)
    ax.set_ylim(-1.15, 0.72)
    ax.set_ylabel("平均奖励", fontproperties=refined.CN_FONT, fontsize=11.4, color=refined.MUTED)
    ax.set_title("本文方法：好奇心补偿中间接近动作", loc="left", fontproperties=refined.CN_FONT, fontsize=16.2, color=refined.INK, pad=10)
    ax.grid(axis="y", color=refined.GRID, linewidth=0.8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for label in ax.get_yticklabels():
        label.set_fontproperties(refined.NUM_FONT)

    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.50, -0.12), ncol=2, frameon=False, fontsize=10.7, handlelength=1.2, columnspacing=1.1)
    for text in leg.get_texts():
        text.set_fontproperties(refined.CN_FONT)


def draw_legend(fig: plt.Figure) -> None:
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#16A34A", markeredgecolor="white", markersize=10, label="接近动作"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#DC2626", markeredgecolor="white", markersize=10, label="回退动作"),
    ]
    leg = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.430, 0.882), ncol=2, frameon=False, fontsize=12.2, handletextpad=0.45, columnspacing=1.6)
    for text in leg.get_texts():
        text.set_fontproperties(refined.CN_FONT)


def draw_bottom_sentence(fig: plt.Figure) -> None:
    fig.text(
        0.055,
        0.085,
        "核心：到达奖励本来就是外在 +2，真正需要解释的是到达前的中间阶段。",
        ha="left",
        va="center",
        fontsize=13.3,
        fontproperties=refined.CN_FONT,
        color=refined.INK,
    )
    fig.text(
        0.055,
        0.050,
        "本文方法用好奇心门控补偿中间接近动作的探索价值，同时保留外在惩罚，使回退动作仍为负反馈。",
        ha="left",
        va="center",
        fontsize=13.3,
        fontproperties=refined.CN_FONT,
        color=refined.BLUE,
    )


def build_figure() -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PPT_PACK_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    comp, steps = load_data()
    n_steps = len(steps)

    fig = plt.figure(figsize=(16, 9), facecolor=refined.PAPER)
    fig.text(0.045, 0.960, "中间阶段奖励归因：好奇心如何补充探索", ha="left", va="top", fontsize=25, fontproperties=refined.CN_FONT, color=refined.INK)
    fig.text(0.045, 0.922, f"训练路线采样 {n_steps:,} 个动作步；统计不含最终到达步的接近/回退动作反馈", ha="left", va="top", fontsize=13.4, fontproperties=refined.CN_FONT, color=refined.MUTED)
    fig.lines.append(Line2D([0.045, 0.955], [0.875, 0.875], transform=fig.transFigure, color="#CBD5E1", lw=1.1))
    draw_legend(fig)

    ax_left = fig.add_axes([0.095, 0.175, 0.570, 0.650])
    draw_midstage_fingerprint(ax_left, comp)

    ax_right = fig.add_axes([0.725, 0.235, 0.230, 0.520])
    draw_proposed_decomposition(ax_right, comp)

    draw_bottom_sentence(fig)

    stem = "26_中间阶段奖励归因_突出好奇心"
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
