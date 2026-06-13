#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build a clean single-table PPT figure for global stepwise reward attribution."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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

JUDGEMENT = {
    "external_only": "中间接近动作\n仍缺少正反馈",
    "intrinsic_only": "回退也几乎\n被当作好动作",
    "mixed_no_gate_no_pbrs": "好奇心补探索\n但回退惩罚变弱",
    "mixed_pbrs_only": "方向信号存在\n但收敛仍不足",
    "proposed_linear_gate_pbrs": "回退压低\n到达强强化",
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


def reward_face(value: float, scale: float = 2.4) -> tuple[float, float, float]:
    color = "#16A34A" if value >= 0 else "#DC2626"
    strength = 0.12 + 0.30 * min(abs(value) / scale, 1.0)
    return refined.blend(color, strength)


def rate_face(value: float) -> tuple[float, float, float]:
    if value >= 50:
        return refined.blend("#DC2626", 0.36)
    if value > 0:
        return refined.blend("#F97316", 0.22)
    return refined.blend("#16A34A", 0.18)


def draw_feedback_table(ax: plt.Axes, df: pd.DataFrame) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    left = 0.030
    top = 0.820
    bottom = 0.090
    gap = 0.010
    widths = [0.145, 0.170, 0.150, 0.165, 0.145, 0.230]
    headers = [
        "方法",
        "接近/到达动作\n平均总奖励",
        "回退动作\n平均总奖励",
        "最终到达动作\n平均总奖励",
        "回退误奖励率",
        "动作反馈判断",
    ]
    row_h = (top - bottom) / len(df)

    x = left
    for header, width in zip(headers, widths):
        draw_text(ax, x + width / 2, top + 0.060, header, size=12.5, color=refined.MUTED)
        x += width + gap

    for r, (_, row) in enumerate(df.iterrows()):
        method = str(row["method"])
        is_key = method == "proposed_linear_gate_pbrs"
        method_color = base.METHOD_STYLE[method]["color"]
        y = top - (r + 1) * row_h

        # Row guide line: no box around the method column.
        ax.plot([left, 0.970], [y - 0.004, y - 0.004], color="#E2E8F0", lw=0.8, zorder=0)
        if is_key:
            ax.add_patch(Rectangle((left - 0.008, y + 0.004), 0.948, row_h - 0.018, facecolor=refined.blend(refined.BLUE, 0.055), edgecolor=refined.BLUE, linewidth=1.7, zorder=0))

        x = left
        draw_text(ax, x + widths[0] / 2, y + row_h / 2, METHOD_CN[method], size=14.0, color=method_color, weight="bold" if is_key else "normal")
        x += widths[0] + gap

        values = [
            float(row["接近/到达步平均总奖励"]),
            float(row["回退步平均总奖励"]),
            float(row["到达步平均总奖励"]),
        ]
        for c, value in enumerate(values):
            width = widths[c + 1]
            ax.add_patch(Rectangle((x, y + 0.014), width, row_h - 0.030, facecolor=reward_face(value), edgecolor="#D8DEE9", linewidth=0.9))
            draw_text(ax, x + width / 2, y + row_h * 0.60, f"{value:+.2f}", size=17.0, font=refined.NUM_FONT, weight="bold" if is_key else "normal")
            sub = "应为正" if c == 0 else ("应为负" if c == 1 else "应强正")
            draw_text(ax, x + width / 2, y + row_h * 0.31, sub, size=10.4, color=refined.MUTED)
            x += width + gap

        wrong_rate = float(row["回退步正奖励率%"])
        width = widths[4]
        ax.add_patch(Rectangle((x, y + 0.014), width, row_h - 0.030, facecolor=rate_face(wrong_rate), edgecolor="#D8DEE9", linewidth=0.9))
        draw_text(ax, x + width / 2, y + row_h * 0.58, f"{wrong_rate:.2f}%", size=15.8, font=refined.NUM_FONT, weight="bold" if wrong_rate > 50 or is_key else "normal")
        draw_text(ax, x + width / 2, y + row_h * 0.31, "越低越好", size=10.4, color=refined.MUTED)
        x += width + gap

        width = widths[5]
        ax.add_patch(Rectangle((x, y + 0.014), width, row_h - 0.030, facecolor="#FFFFFF", edgecolor=refined.BLUE if is_key else "#D8DEE9", linewidth=1.4 if is_key else 0.9))
        draw_text(ax, x + width / 2, y + row_h / 2, JUDGEMENT[method], size=12.3, color=refined.BLUE if is_key else refined.INK, weight="bold" if is_key else "normal")


def draw_bottom_cards(ax: plt.Axes) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    cards = [
        ("看奖励落点", "不是奖励越大越好，而是正反馈是否落在正确动作上。", refined.GREEN),
        ("纯好奇心问题", "回退动作正奖励率接近 99%，探索没有目标约束。", refined.PURPLE),
        ("本文方法结论", "回退保持负反馈，到达动作获得强正反馈。", refined.BLUE),
    ]
    for i, (title, body, color) in enumerate(cards):
        x = 0.012 + i * 0.330
        ax.add_patch(Rectangle((x, 0.14), 0.315, 0.70, facecolor=refined.blend(color, 0.075), edgecolor="#D8DEE9", linewidth=1.0))
        ax.add_patch(Rectangle((x, 0.14), 0.010, 0.70, facecolor=color, edgecolor=color, linewidth=0))
        draw_text(ax, x + 0.028, 0.61, title, size=12.7, ha="left", weight="bold")
        draw_text(ax, x + 0.028, 0.38, body, size=11.4, ha="left")


def build_figure() -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PPT_PACK_DIR.mkdir(parents=True, exist_ok=True)
    refined.setup_style()

    df = load_method_summary()
    n_steps = len(pd.read_csv(TABLE_DIR / "reward_stepwise_global_steps.csv", usecols=["method"]))

    fig = plt.figure(figsize=(16, 9), facecolor=refined.PAPER)
    fig.text(0.040, 0.960, "全量逐步奖励归因：奖励是否给到正确动作", ha="left", va="top", fontsize=24.5, fontproperties=refined.CN_FONT, color=refined.INK)
    fig.text(0.040, 0.922, f"训练路线采样 {n_steps:,} 个动作步；按动作类型汇总每一步总奖励", ha="left", va="top", fontsize=13.2, fontproperties=refined.CN_FONT, color=refined.MUTED)
    fig.lines.append(Line2D([0.040, 0.960], [0.878, 0.878], transform=fig.transFigure, color="#CBD5E1", lw=1.1))

    ax_table = fig.add_axes([0.025, 0.210, 0.950, 0.690])
    draw_feedback_table(ax_table, df)

    ax_cards = fig.add_axes([0.045, 0.035, 0.910, 0.145])
    draw_bottom_cards(ax_cards)

    stem = "24_全量逐步奖励归因_反馈对齐表"
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
