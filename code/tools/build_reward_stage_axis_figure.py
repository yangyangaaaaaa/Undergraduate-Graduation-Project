#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build a PPT-ready phase-axis figure for the curiosity reward mechanism."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import build_reward_guided_case_studies as base  # noqa: E402
import build_reward_mechanism_single_case_refined as refined  # noqa: E402


TABLE_DIR = base.TABLES
OUT_DIR = base.RESULTS / "figures" / "defense_reward_training_stage" / "single_reward_mechanism"
PPT_PACK_DIR = (
    base.RESULTS
    / "figures"
    / "ppt_candidate_pack_20260606"
    / "03_奖励机制_动作归因与GP故事"
)

METHODS = [
    "external_only",
    "intrinsic_only",
    "mixed_no_gate_no_pbrs",
    "proposed_linear_gate_pbrs",
]

METHOD_CN = {
    "external_only": "仅外在",
    "intrinsic_only": "仅好奇心",
    "mixed_no_gate_no_pbrs": "内外奖励相加",
    "proposed_linear_gate_pbrs": "本文方法",
}

WHITE = "#FFFFFF"
PHASES = [
    ("far", "远距离探索", r"$d\geq6$", refined.PURPLE),
    ("mid", "中距离筛选", r"$3\leq d\leq5$", refined.GREEN),
    ("near", "近目标收敛", r"$d\leq2$", refined.BLUE),
]
AXIS_VMIN = -1.1
AXIS_VMAX = 1.4
AXIS_TICKS = [-1.0, 0.0, 1.0]
FONT_SCALE = 1.14
YAHEI_BOLD_PATH = Path("C:/Windows/Fonts/msyhbd.ttc")
YAHEI_BOLD = FontProperties(fname=str(YAHEI_BOLD_PATH)) if YAHEI_BOLD_PATH.exists() else refined.CN_FONT


def scaled_font(size: float) -> float:
    return size * FONT_SCALE


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
    weight: str = "normal",
    font=None,
    **kwargs,
) -> None:
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=scaled_font(size),
        color=color,
        fontproperties=font or YAHEI_BOLD,
        fontweight="bold",
        **kwargs,
    )


def load_steps() -> pd.DataFrame:
    return pd.read_csv(TABLE_DIR / "reward_stepwise_global_steps.csv")


def phase_code(prev_dist: float) -> str:
    if prev_dist >= 6:
        return "far"
    if prev_dist >= 3:
        return "mid"
    return "near"


def proposed_phase_stats(steps: pd.DataFrame) -> dict[str, dict[str, float]]:
    proposed = steps[steps["method"].eq("proposed_linear_gate_pbrs")].copy()
    move = proposed[proposed["move_type"].isin(["progress", "regress"])].copy()
    move["phase"] = move["prev_dist"].apply(phase_code)

    stats: dict[str, dict[str, float]] = {}
    for phase in ["far", "mid", "near"]:
        hit = move[move["phase"].eq(phase)]
        progress = hit[hit["move_type"].eq("progress")]
        regress = hit[hit["move_type"].eq("regress")]
        stats[phase] = {
            "n": float(len(hit)),
            "ex": float(hit["reward_ex"].mean()),
            "curiosity": float(hit["reward_in_gated"].mean()),
            "pbrs": float(hit["pbrs_bonus"].mean()),
            "total": float(hit["reward_total"].mean()),
            "progress_total": float(progress["reward_total"].mean()) if len(progress) else np.nan,
            "regress_total": float(regress["reward_total"].mean()) if len(regress) else np.nan,
        }

    goal = proposed[proposed["move_type"].eq("goal")]
    stats["goal"] = {
        "n": float(len(goal)),
        "total": float(goal["reward_total"].mean()) if len(goal) else np.nan,
        "ex": float(goal["reward_ex"].mean()) if len(goal) else np.nan,
    }
    return stats


def method_alignment_stats(steps: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for method in METHODS:
        hit = steps[steps["method"].eq(method)]
        move = hit[hit["move_type"].isin(["progress", "regress"])].copy()
        move["phase"] = move["prev_dist"].apply(phase_code)
        out[method] = {}
        for phase, _, _, _ in PHASES:
            phase_hit = move[move["phase"].eq(phase)]
            progress = phase_hit[phase_hit["move_type"].eq("progress")]
            regress = phase_hit[phase_hit["move_type"].eq("regress")]
            out[method][phase] = {
                "progress_total": float(progress["reward_total"].mean()) if len(progress) else np.nan,
                "regress_total": float(regress["reward_total"].mean()) if len(regress) else np.nan,
                "regress_pos": 100.0 * float(regress["reward_positive"].mean()) if len(regress) else np.nan,
            }
    return out


def reward_to_x(value: float, x0: float, x1: float, vmin: float = AXIS_VMIN, vmax: float = AXIS_VMAX) -> float:
    value = min(max(value, vmin), vmax)
    return x0 + (value - vmin) / (vmax - vmin) * (x1 - x0)


def draw_stage_axis(ax: plt.Axes, phase_stats: dict[str, dict[str, float]]) -> None:
    y = 0.690
    segments = [
        (
            "远距离探索",
            r"$d\geq6$",
            "允许试探",
            f"动作反馈：阶段均值 {phase_stats['far']['total']:+.2f}",
            0.080,
            0.350,
            refined.PURPLE,
        ),
        (
            "中距离筛选",
            r"$3\leq d\leq5$",
            "筛出方向",
            f"动作反馈：接近 {phase_stats['mid']['progress_total']:+.2f}，偏离 {phase_stats['mid']['regress_total']:+.2f}",
            0.350,
            0.665,
            refined.GREEN,
        ),
        (
            "近目标收敛",
            r"$d\leq2$",
            "目标接管",
            f"动作反馈：接近 {phase_stats['near']['progress_total']:+.2f}，偏离 {phase_stats['near']['regress_total']:+.2f}",
            0.665,
            0.925,
            refined.BLUE,
        ),
    ]

    ax.plot([0.050, 0.955], [y, y], color="#CBD5E1", linewidth=2.0, zorder=1)
    for title, dist_label, role, evidence, x0, x1, color in segments:
        cx = (x0 + x1) / 2
        ax.plot([x0, x1], [y, y], color=color, linewidth=17, solid_capstyle="butt", alpha=0.90, zorder=2)
        ax.scatter([x0, x1], [y, y], s=54, color="white", edgecolor=color, linewidth=2.0, zorder=3)
        draw_text(ax, cx, 0.815, title, size=15.0, color=color, weight="bold")
        ax.text(
            cx,
            0.765,
            dist_label,
            ha="center",
            va="center",
            fontsize=scaled_font(12.2),
            color=refined.MUTED,
            fontproperties=YAHEI_BOLD,
            fontweight="bold",
        )
        draw_text(ax, cx, 0.618, role, size=13.0, color=refined.INK, weight="bold")
        draw_text(ax, cx, 0.565, evidence, size=11.8, color=refined.MUTED)

    draw_text(ax, 0.050, 0.642, "起点", size=11.4, color=refined.MUTED)
    draw_text(ax, 0.955, 0.642, "目标", size=11.4, color=refined.MUTED)


def draw_alignment_axis(
    ax: plt.Axes,
    stats: dict[str, dict[str, float]],
) -> None:
    col_bounds = {
        "far": (0.200, 0.405),
        "mid": (0.448, 0.653),
        "near": (0.695, 0.900),
    }
    header_y = 0.475
    tick_y = 0.418

    draw_text(ax, 0.060, header_y, "方法对比", size=14.0, color=refined.INK, ha="left", weight="bold")
    ax.scatter([0.735], [0.515], s=58, color=refined.GREEN, edgecolor="white", linewidth=1.0, zorder=5)
    draw_text(ax, 0.755, 0.515, "接近", size=10.8, color=refined.GREEN, ha="left", weight="bold")
    ax.scatter([0.810], [0.515], s=58, color=refined.RED, edgecolor="white", linewidth=1.0, zorder=5)
    draw_text(ax, 0.830, 0.515, "偏离", size=10.8, color=refined.RED, ha="left", weight="bold")

    for phase, title, _, phase_color in PHASES:
        x0, x1 = col_bounds[phase]
        cx = (x0 + x1) / 2
        ax.add_patch(
            Rectangle(
                (x0 - 0.017, 0.095),
                (x1 - x0) + 0.034,
                0.355,
                facecolor=phase_color,
                edgecolor="none",
                alpha=0.050,
                zorder=0,
            )
        )
        draw_text(ax, cx, header_y, title, size=13.0, color=phase_color, weight="bold")
        ax.plot([x0, x1], [tick_y, tick_y], color="#94A3B8", linewidth=1.0)
        zero_x = reward_to_x(0.0, x0, x1)
        ax.plot([zero_x, zero_x], [0.105, tick_y + 0.018], color="#64748B", linewidth=1.0, linestyle=(0, (4, 3)), zorder=0)
        for value in AXIS_TICKS:
            tx = reward_to_x(value, x0, x1)
            ax.plot([tx, tx], [tick_y - 0.010, tick_y + 0.010], color="#94A3B8", linewidth=0.9)
            ax.text(
                tx,
                tick_y + 0.028,
                "0" if value == 0.0 else f"{value:+.1f}",
                ha="center",
                va="center",
                fontsize=scaled_font(8.9),
                color=refined.MUTED,
                fontproperties=YAHEI_BOLD,
                fontweight="bold",
            )

    ys = np.linspace(0.375, 0.135, len(METHODS))
    for y, method in zip(ys, METHODS):
        color = base.METHOD_STYLE[method]["color"]
        is_key = method == "proposed_linear_gate_pbrs"
        draw_text(ax, 0.060, y, METHOD_CN[method], size=12.0, color=color, ha="left", weight="bold" if is_key else "normal")
        if is_key:
            draw_text(
                ax,
                0.060,
                y - 0.052,
                "绿色/红色点分别表示接近/偏离动作的平均单步总奖励，越靠右表示该动作越容易被策略强化。",
                size=9.6,
                color=refined.MUTED,
                ha="left",
            )
        for phase, _, _, _ in PHASES:
            x0, x1 = col_bounds[phase]
            phase_stats = stats[method][phase]
            progress = phase_stats["progress_total"]
            regress = phase_stats["regress_total"]
            if np.isnan(progress) or np.isnan(regress):
                continue
            px = reward_to_x(progress, x0, x1)
            rx = reward_to_x(regress, x0, x1)

            ax.plot([x0, x1], [y, y], color="#E2E8F0", linewidth=1.0, zorder=1)
            ax.plot([rx, px], [y, y], color=color, linewidth=3.0 if is_key else 1.8, alpha=0.72, zorder=2)
            ax.scatter([px], [y], s=92 if is_key else 62, color=refined.GREEN, edgecolor="white", linewidth=1.0, zorder=4)
            ax.scatter([rx], [y], s=92 if is_key else 62, color=refined.RED, edgecolor="white", linewidth=1.0, zorder=4)

            if is_key:
                ax.text(
                    px,
                    y + 0.025,
                    f"{progress:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=scaled_font(8.9),
                    color=refined.GREEN,
                    fontproperties=YAHEI_BOLD,
                    fontweight="bold",
                )
                ax.text(
                    rx,
                    y - 0.026,
                    f"{regress:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=scaled_font(8.9),
                    color=refined.RED,
                    fontproperties=YAHEI_BOLD,
                    fontweight="bold",
                )


def draw_action_rule(ax: plt.Axes) -> None:
    draw_text(ax, 0.060, 0.925, "动作归因", size=13.2, color=refined.INK, ha="left", weight="bold")
    items = [
        ("接近", r"$D_{t+1}<D_t,\ D_{t+1}>0$", refined.GREEN),
        ("偏离", r"$D_{t+1}>D_t$", refined.RED),
        ("到达", r"$D_{t+1}=0$", refined.BLUE),
    ]
    xs = [0.185, 0.425, 0.620]
    for x, (name, formula, color) in zip(xs, items):
        ax.scatter([x], [0.925], s=52, color=color, edgecolor="white", linewidth=1.2)
        draw_text(ax, x + 0.018, 0.925, name, size=12.0, color=color, ha="left", weight="bold")
        ax.text(
            x + 0.068,
            0.925,
            formula,
            ha="left",
            va="center",
            fontsize=scaled_font(11.7),
            color=refined.INK,
            fontproperties=YAHEI_BOLD,
            fontweight="bold",
        )
    draw_text(ax, 0.945, 0.925, "仅用于解释每一步奖励是否对齐目标方向", size=11.3, color=refined.MUTED, ha="right")


def build_figure() -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PPT_PACK_DIR.mkdir(parents=True, exist_ok=True)
    refined.setup_style()
    plt.rcParams.update(
        {
            "figure.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "font.family": "Microsoft YaHei",
            "font.sans-serif": ["Microsoft YaHei", "Microsoft YaHei UI", "SimHei", "DejaVu Sans"],
            "font.weight": "bold",
            "mathtext.default": "regular",
        }
    )

    steps = load_steps()
    phase_stats = proposed_phase_stats(steps)
    alignment_stats = method_alignment_stats(steps)
    n_steps = len(steps)

    fig = plt.figure(figsize=(16, 9), facecolor=WHITE)
    fig.text(
        0.040,
        0.963,
        "到达前看奖励差异：好奇心如何被目标约束对齐",
        ha="left",
        va="top",
        fontsize=scaled_font(25.0),
        fontproperties=YAHEI_BOLD,
        color=refined.INK,
        fontweight="bold",
    )
    fig.text(
        0.040,
        0.918,
        f"基于训练日志 {n_steps:,} 个动作步；单步奖励由外在目标项、门控好奇心和势函数塑形组成，到达时外在目标项给出 +2",
        ha="left",
        va="top",
        fontsize=scaled_font(13.3),
        fontproperties=YAHEI_BOLD,
        color=refined.MUTED,
        fontweight="bold",
    )
    fig.lines.append(Line2D([0.040, 0.960], [0.884, 0.884], transform=fig.transFigure, color="#CBD5E1", lw=1.0))

    ax = fig.add_axes([0.040, 0.045, 0.920, 0.835])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    draw_action_rule(ax)
    draw_stage_axis(ax, phase_stats)
    draw_alignment_axis(ax, alignment_stats)

    stem = "27_奖励机制分阶段直线轴_好奇心作用"
    out_png = OUT_DIR / f"{stem}.png"
    pack_png = PPT_PACK_DIR / f"{stem}.png"
    fig.savefig(out_png, dpi=240, facecolor=WHITE)
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
