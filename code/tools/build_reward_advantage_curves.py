#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build trend-first reward advantage curves for defense slides.

This figure keeps curves because they show training dynamics that tables cannot:
when the proposed method catches up, how the final advantage appears, whether
the advantage is robust across MM-GAG modalities, and whether distance residuals
continue to shrink.

For defense clarity the main panels use best-so-far envelopes over observed
checkpoints. This suppresses checkpoint noise while still using only real
observed checkpoint values.
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
from scipy.interpolate import PchipInterpolator


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables" / "defense_reward_trends"
FIGURES = RESULTS / "figures" / "defense_reward_trends"
REPORTS = RESULTS / "reports"

TREND_TABLE = TABLES / "mmgag_checkpoint_c8_training_trend_real_points.csv"
REPORT_PATH = REPORTS / "reward_advantage_curve_design_zh.md"

INK = "#17212F"
MUTED = "#5B6777"
PAPER = "#F7F9FC"
CARD = "#FFFFFF"
GRID = "#D8E0EA"
BLUE = "#1764AB"
ORANGE = "#D27A20"
GREEN = "#168A63"
PURPLE = "#7C5CC4"
GRAY = "#7A8699"

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

MODALITY_COLS = ["mmgag_aerial", "mmgag_ground", "mmgag_text"]


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
            "axes.titlesize": 12.4,
            "axes.labelsize": 10.4,
            "xtick.labelsize": 9.1,
            "ytick.labelsize": 9.1,
            "legend.fontsize": 9.4,
            "lines.linewidth": 2.3,
        }
    )


def clean_axes(ax: plt.Axes, grid_axis: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#C9D2DE")
    ax.spines["bottom"].set_color("#C9D2DE")
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.74)
    ax.set_axisbelow(True)


def smooth_xy(x: np.ndarray, y: np.ndarray, n: int = 180) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or len(np.unique(x)) < 3:
        return x, y
    order = np.argsort(x)
    x, y = x[order], y[order]
    xs = np.linspace(float(x.min()), float(x.max()), n)
    ys = PchipInterpolator(x, y)(xs)
    return xs, ys


def panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(-0.08, 1.08, text, transform=ax.transAxes, fontsize=14.5, fontweight="bold", ha="left", va="top")


def draw_method_curves(ax: plt.Axes, trend: pd.DataFrame, y_col: str, ylabel: str, title: str) -> None:
    for method in METHOD_ORDER:
        df = trend[trend["method"].eq(method)].sort_values("run_progress")
        style = METHOD_STYLE[method]
        x = df["run_progress"].astype(float).to_numpy()
        y = df[y_col].astype(float).to_numpy() * 100.0
        xs, ys = smooth_xy(x, y)
        key = method == "proposed_linear_gate_pbrs"
        ax.plot(
            xs,
            ys,
            color=style["color"],
            linestyle=style["ls"],
            linewidth=3.2 if key else 2.0,
            alpha=1.0 if key else 0.74,
        )
        ax.scatter(
            x,
            y,
            s=32 if key else 22,
            color=style["color"],
            edgecolor="white",
            linewidth=0.85,
            alpha=1.0 if key else 0.82,
            zorder=4,
        )
    ax.set_title(title, loc="left", pad=7)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Progress")
    ax.set_ylabel(ylabel)
    clean_axes(ax)


def build_aligned_margins(trend: pd.DataFrame) -> pd.DataFrame:
    data = trend.copy()
    data["checkpoint_key"] = data["episode"].astype(int)

    rows = []
    for checkpoint, group in data.groupby("checkpoint_key"):
        ours = group[group["method"].eq("proposed_linear_gate_pbrs")]
        ctrl = group[~group["method"].eq("proposed_linear_gate_pbrs")]
        if ours.empty or ctrl.empty:
            continue
        ours = ours.iloc[0]
        rows.append(
            {
                "checkpoint": int(checkpoint),
                "progress": float(group["run_progress"].astype(float).mean()),
                "mean_sr_margin": float(ours["c8_success_envelope"] - ctrl["c8_success_envelope"].max()) * 100.0,
                "worst_sr_margin": float(ours["worst_modality_sr_envelope"] - ctrl["worst_modality_sr_envelope"].max()) * 100.0,
                "closeness_margin": float(ours["c8_closeness_envelope"] - ctrl["c8_closeness_envelope"].max()) * 100.0,
            }
        )
    return pd.DataFrame(rows).sort_values("progress")


def draw_margin_panel(ax: plt.Axes, margins: pd.DataFrame) -> None:
    late = margins[margins["progress"].ge(0.55)].copy()
    x = late["progress"].astype(float).to_numpy()
    series = [
        ("mean_sr_margin", "Mean SR", BLUE, 3.2),
        ("worst_sr_margin", "Worst SR", "#0F766E", 2.6),
        ("closeness_margin", "Closeness", "#7C3AED", 2.2),
    ]
    for col, label, color, lw in series:
        y = late[col].astype(float).to_numpy()
        xs, ys = smooth_xy(x, y)
        ax.plot(xs, ys, color=color, linewidth=lw, label=label)
        ax.scatter(x, y, s=28, color=color, edgecolor="white", linewidth=0.85, zorder=4)
    ax.axhline(0, color=INK, linewidth=1.0, alpha=0.70)
    mean_margin = late["mean_sr_margin"].to_numpy()
    ax.fill_between(x, 0, mean_margin, where=mean_margin >= 0, color=BLUE, alpha=0.14)
    ax.fill_between(x, 0, mean_margin, where=mean_margin < 0, color=GRAY, alpha=0.10)
    ax.set_title("Late Advantage", loc="left", pad=7)
    ax.set_xlim(0.54, 1.02)
    ax.set_ylim(-16, 12)
    ax.set_xlabel("Progress")
    ax.set_ylabel("Adv. (pp)")
    ax.legend(frameon=False, loc="lower right", ncol=3, handlelength=2.0, columnspacing=0.8)
    clean_axes(ax)


def build_figure(trend: pd.DataFrame, margins: pd.DataFrame) -> Path:
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.035, 0.955, "C8 Trend Curves", fontsize=23.5, fontweight="bold", ha="left", va="top")
    fig.text(0.035, 0.918, "Best-so-far envelopes from real checkpoints; markers are observed values", fontsize=11.5, color=MUTED, ha="left", va="top")
    fig.lines.append(Line2D([0.035, 0.985], [0.888, 0.888], transform=fig.transFigure, color="#CBD5E1", lw=1.1))

    gs = fig.add_gridspec(2, 2, left=0.060, right=0.985, top=0.835, bottom=0.075, wspace=0.16, hspace=0.33)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    draw_method_curves(ax_a, trend, "c8_success_envelope", "C8 SR (%)", "Best Mean SR")
    draw_method_curves(ax_b, trend, "worst_modality_sr_envelope", "C8 SR (%)", "Best Worst-Modality SR")
    draw_method_curves(ax_c, trend, "c8_closeness_envelope", "Closeness (%)", "Best Closeness")
    draw_margin_panel(ax_d, margins)

    for label, ax in zip("ABCD", [ax_a, ax_b, ax_c, ax_d]):
        panel_label(ax, label)

    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_STYLE[m]["color"],
            linestyle=METHOD_STYLE[m]["ls"],
            linewidth=3.1 if m == "proposed_linear_gate_pbrs" else 2.1,
            label=METHOD_STYLE[m]["label"],
        )
        for m in METHOD_ORDER
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.963), ncol=4, frameon=False, handlelength=2.7)

    out = FIGURES / "figure_reward_advantage_curves.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.10)
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    return out


def write_report(out: Path, trend: pd.DataFrame, margins: pd.DataFrame) -> None:
    final = margins.sort_values("progress").iloc[-1]
    lines = [
        "# 奖励机制优势趋势曲线设计说明",
        "",
        "这张图保留曲线，但避免把表格内容简单图形化。它展示的是训练过程中的最好已达到能力、模态稳定性、距离收敛和后期相对优势形成过程。",
        "",
        f"- 输出图：`{out}`",
        f"- SVG：`{out.with_suffix('.svg')}`",
        "- 数据来源：`results/tables/defense_reward_trends/mmgag_checkpoint_c8_training_trend_real_points.csv`",
        "",
        "## 四个子图的作用",
        "",
        "1. `Best Mean SR`：到当前训练进度为止，平均 C8 成功率已经达到过的最好水平。",
        "2. `Best Worst-Modality SR`：三种 MM-GAG 输入模态中最差模态的 best-so-far 成功率，用于说明不是只在某一种模态上好。",
        "3. `Best Closeness`：由残余距离换算成接近目标程度，越高表示路线最终离目标越近。",
        "4. `Late Advantage`：55% 训练进度之后，本文方法相对同一 checkpoint 最强对照的优势差值，单位是百分点。零线以上表示本文方法领先。",
        "",
        "## 为什么适合答辩",
        "",
        "这张图回答的是“优势在训练过程中怎样形成”，不是重复最终表格。A/B/C 使用 best-so-far envelope，能避免中间 checkpoint 抖动干扰；D 图聚焦后期优势窗口，能把评委注意力拉到最终阶段本文方法如何超过最强对照。",
        "",
        "## 当前最终优势",
        "",
        f"- 平均 C8 成功率优势：{float(final['mean_sr_margin']):+.2f} 个百分点。",
        f"- 最差模态 C8 成功率优势：{float(final['worst_sr_margin']):+.2f} 个百分点。",
        f"- 接近目标程度优势：{float(final['closeness_margin']):+.2f} 个百分点。",
        "",
        "## 表述边界",
        "",
        "曲线来自固定 checkpoint 评估和训练阶段日志整理。奖励、距离门控和 PBRS 仍然只解释训练阶段学习信号；正式测试阶段只加载 checkpoint 执行动作。",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    setup_style()
    trend = pd.read_csv(TREND_TABLE)
    trend = trend[trend["method"].isin(METHOD_ORDER)].copy()
    trend["worst_modality_sr"] = trend[MODALITY_COLS].astype(float).min(axis=1)
    trend = trend.sort_values(["method", "run_progress"])
    trend["worst_modality_sr_envelope"] = trend.groupby("method")["worst_modality_sr"].cummax()
    margins = build_aligned_margins(trend)
    out = build_figure(trend, margins)
    write_report(out, trend, margins)
    print({"figure": str(out), "svg": str(out.with_suffix(".svg")), "report": str(REPORT_PATH)})


if __name__ == "__main__":
    main()
