#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build non-table visual evidence for defense reward analysis.

The goal is not to redraw summary tables.  This script visualizes aspects that
tables express poorly: phase-space movement, modality spread over training, and
late-stage convergence behavior.

Inputs are real fixed-checkpoint MM-GAG C8 evaluation points generated for the
defense reward-trend package.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables" / "defense_reward_trends"
FIGURES = RESULTS / "figures" / "defense_reward_trends"
REPORTS = RESULTS / "reports"

TREND_TABLE = TABLES / "mmgag_checkpoint_c8_training_trend_real_points.csv"
REPORT_PATH = REPORTS / "defense_non_table_visuals_zh.md"

INK = "#17212F"
MUTED = "#5B6777"
PAPER = "#F7F9FC"
CARD = "#FFFFFF"
GRID = "#D8E0EA"
BLUE = "#1764AB"
ORANGE = "#D27A20"
GREEN = "#168A63"
PURPLE = "#7C5CC4"

METHOD_ORDER = [
    "linear_gate_no_pbrs",
    "external_pbrs",
    "constant_gate_pbrs",
    "proposed_linear_gate_pbrs",
]

METHOD_STYLE = {
    "linear_gate_no_pbrs": {"label": "Gate", "color": GREEN, "ls": (0, (6, 3))},
    "external_pbrs": {"label": "Ext+PBRS", "color": ORANGE, "ls": (0, (7, 3))},
    "constant_gate_pbrs": {"label": "Const+PBRS", "color": PURPLE, "ls": (0, (4, 2, 1, 2))},
    "proposed_linear_gate_pbrs": {"label": "Ours", "color": BLUE, "ls": "solid"},
}


def setup_style() -> None:
    for font in [
        r"C:\Windows\Fonts\times.ttf",
        r"C:\Windows\Fonts\timesbd.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]:
        path = Path(font)
        if path.exists():
            font_manager.fontManager.addfont(str(path))
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "axes.facecolor": CARD,
            "font.family": ["Times New Roman", "SimSun"],
            "font.serif": ["Times New Roman", "SimSun"],
            "font.sans-serif": ["Times New Roman", "SimSun"],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "axes.edgecolor": "#C9D2DE",
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 12.5,
            "axes.labelsize": 10.6,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "legend.fontsize": 9.6,
            "lines.linewidth": 2.4,
        }
    )


def clean_axes(ax: plt.Axes, grid_axis: str | None = "both") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#C9D2DE")
    ax.spines["bottom"].set_color("#C9D2DE")
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.72)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(-0.10, 1.08, text, transform=ax.transAxes, fontsize=15, fontweight="bold", ha="left", va="top")


def draw_phase_path(ax: plt.Axes, trend: pd.DataFrame) -> None:
    ax.add_patch(
        Rectangle(
            (0.90, 0.0),
            0.10,
            0.35,
            facecolor="#DBEAFE",
            edgecolor="none",
            alpha=0.75,
            zorder=0,
        )
    )

    for method in METHOD_ORDER:
        df = trend[trend["method"].eq(method)].sort_values("run_progress")
        if df.empty:
            continue
        style = METHOD_STYLE[method]
        x = df["c8_success_mean"].astype(float).to_numpy()
        y = df["c8_sg_mean"].astype(float).to_numpy()
        progress = df["run_progress"].astype(float).to_numpy()
        key = method == "proposed_linear_gate_pbrs"
        ax.plot(
            x,
            y,
            color=style["color"],
            linestyle=style["ls"],
            linewidth=3.4 if key else 2.1,
            alpha=1.0 if key else 0.78,
            zorder=4 if key else 2,
        )
        sizes = 30 + 70 * progress
        ax.scatter(
            x,
            y,
            s=sizes,
            color=style["color"],
            edgecolor="white",
            linewidth=1.0,
            alpha=0.98 if key else 0.82,
            zorder=5 if key else 3,
        )
        for i in range(len(x) - 1):
            if i % 2 == 1 and i != len(x) - 2:
                continue
            arrow = FancyArrowPatch(
                (x[i], y[i]),
                (x[i + 1], y[i + 1]),
                arrowstyle="-|>",
                mutation_scale=11 if key else 8,
                linewidth=0,
                color=style["color"],
                alpha=0.90 if key else 0.58,
                zorder=6 if key else 3,
            )
            ax.add_patch(arrow)
        ax.scatter(
            [x[-1]],
            [y[-1]],
            s=170 if key else 110,
            marker="*" if key else "o",
            color=style["color"],
            edgecolor="white",
            linewidth=1.2,
            zorder=8,
        )

    ax.set_xlim(0.08, 0.98)
    ax.set_ylim(4.75, 0.05)
    ax.set_xlabel("C8 SR")
    ax.set_ylabel("Residual")
    panel_label(ax, "A")
    clean_axes(ax)


def draw_modality_band(ax: plt.Axes, trend: pd.DataFrame) -> None:
    modalities = ["mmgag_aerial", "mmgag_ground", "mmgag_text"]
    for method in METHOD_ORDER:
        df = trend[trend["method"].eq(method)].sort_values("run_progress")
        if df.empty:
            continue
        style = METHOD_STYLE[method]
        x = df["run_progress"].astype(float).to_numpy()
        mean = df["c8_success_mean"].astype(float).to_numpy()
        y_min = df[modalities].astype(float).min(axis=1).to_numpy()
        y_max = df[modalities].astype(float).max(axis=1).to_numpy()
        key = method == "proposed_linear_gate_pbrs"
        ax.fill_between(x, y_min, y_max, color=style["color"], alpha=0.08 if key else 0.055, linewidth=0)
        ax.plot(
            x,
            mean,
            color=style["color"],
            linestyle=style["ls"],
            linewidth=3.3 if key else 2.0,
            alpha=1.0 if key else 0.74,
        )
        ax.scatter(x, mean, s=24 if key else 17, color=style["color"], edgecolor="white", linewidth=0.8, zorder=5)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0.05, 0.98)
    ax.set_xlabel("Progress")
    ax.set_ylabel("C8 SR")
    panel_label(ax, "B")
    clean_axes(ax, "y")


def draw_late_convergence(ax: plt.Axes, trend: pd.DataFrame) -> None:
    for method in METHOD_ORDER:
        df = trend[trend["method"].eq(method)].sort_values("run_progress").tail(5)
        if df.empty:
            continue
        style = METHOD_STYLE[method]
        x = df["run_progress"].astype(float).to_numpy()
        y = df["c8_sg_mean"].astype(float).to_numpy()
        key = method == "proposed_linear_gate_pbrs"
        ax.plot(
            x,
            y,
            color=style["color"],
            linestyle=style["ls"],
            linewidth=3.4 if key else 2.1,
            alpha=1.0 if key else 0.75,
            marker="o",
            markersize=5.4 if key else 4.0,
        )
    ax.axhspan(0.0, 0.35, color="#DBEAFE", alpha=0.70, zorder=0)
    ax.set_xlim(0.52, 1.02)
    ax.set_ylim(1.7, 0.05)
    ax.set_xlabel("Progress")
    ax.set_ylabel("Residual")
    panel_label(ax, "C")
    clean_axes(ax, "y")


def build_figure(trend: pd.DataFrame) -> Path:
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.04, 0.955, "Training Dynamics: C8", fontsize=24, fontweight="bold", ha="left", va="top")
    fig.text(
        0.04,
        0.918,
        "Not a table redraw: phase path, modality spread, and late-stage convergence",
        fontsize=11.8,
        color=MUTED,
        ha="left",
        va="top",
    )
    fig.lines.append(Line2D([0.04, 0.965], [0.885, 0.885], transform=fig.transFigure, color="#CCD6E2", lw=1.2))

    gs = fig.add_gridspec(
        2,
        2,
        left=0.065,
        right=0.965,
        top=0.825,
        bottom=0.090,
        width_ratios=[1.05, 1.0],
        height_ratios=[1.0, 0.88],
        wspace=0.22,
        hspace=0.36,
    )
    ax_phase = fig.add_subplot(gs[:, 0])
    ax_band = fig.add_subplot(gs[0, 1])
    ax_late = fig.add_subplot(gs[1, 1])

    draw_phase_path(ax_phase, trend)
    draw_modality_band(ax_band, trend)
    draw_late_convergence(ax_late, trend)

    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_STYLE[m]["color"],
            linestyle=METHOD_STYLE[m]["ls"],
            linewidth=3.2 if m == "proposed_linear_gate_pbrs" else 2.2,
            label=METHOD_STYLE[m]["label"],
        )
        for m in METHOD_ORDER
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.965, 0.960), ncol=4, frameon=False, handlelength=2.8)

    out = FIGURES / "figure_reward_dynamic_phase.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.14)
    fig.savefig(FIGURES / "figure_reward_dynamic_phase.svg", bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    return out


def write_report(out: Path, trend: pd.DataFrame) -> None:
    ours = trend[trend["method"].eq("proposed_linear_gate_pbrs")].sort_values("run_progress").iloc[-1]
    lines = [
        "# 非表格式训练动态可视化说明",
        "",
        "本页只用于展示表格难以表达的信息，不替代正式结果表格。",
        "",
        f"- 输出图：`{out}`",
        f"- SVG：`{out.with_suffix('.svg')}`",
        "- 数据来源：`results/tables/defense_reward_trends/mmgag_checkpoint_c8_training_trend_real_points.csv`",
        "",
        "## 为什么这不是表格重复",
        "",
        "1. A 图是训练相图：横轴是 C8 成功率，纵轴是残余距离，箭头表示 checkpoint 的时间顺序。它展示方法在训练中如何移动，而不是只给最后一行数值。",
        "2. B 图展示三种 MM-GAG 模态的训练期展开范围。浅色带越窄，说明同一方法在 aerial/ground/text 三种输入上越稳定；这类波动结构用表格不直观。",
        "3. C 图只看后半程收敛：它用曲线形状展示后期是否继续向低残余距离区域移动，避免只盯最终 checkpoint。",
        "",
        "## 建议讲法",
        "",
        "这页可以放在主趋势图之后。主趋势图说明本文方法最后达到更高 C8 成功率；这页进一步说明优势不是一个孤立终点，而是训练轨迹逐步进入“高成功率、低残余距离”的区域。图中蓝色轨迹代表本文方法，最终进入左图右上方高成功率且低残余距离的浅蓝目标区域；右侧两图说明这种优势在三种 MM-GAG 模态上也保持一致，并在训练后期继续收敛。",
        "",
        "## 表述边界",
        "",
        "奖励、距离门控和 PBRS 仍然只解释训练阶段的学习信号；正式测试阶段只加载 checkpoint 执行动作。",
        "",
        "## 当前关键事实",
        "",
        f"- 本文方法末端 C8 平均成功率为 {float(ours['c8_success_mean']) * 100:.2f}%，平均残余距离为 {float(ours['c8_sg_mean']):.3f}。",
        "- 这些数值可放进表格或口头说明，不建议直接堆到图面上。",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    setup_style()
    trend = pd.read_csv(TREND_TABLE)
    trend = trend[trend["method"].isin(METHOD_ORDER)].copy()
    out = build_figure(trend)
    write_report(out, trend)
    print({"figure": str(out), "svg": str(out.with_suffix(".svg")), "report": str(REPORT_PATH)})


if __name__ == "__main__":
    main()
