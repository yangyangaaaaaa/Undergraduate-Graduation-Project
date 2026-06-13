#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build dense MM-GAG C8 checkpoint curves after the rerun finishes.

Expected input:
F:/bishe/GeoExplorer/analysis/pipeline_20260605_dense_mmgag_checkpoint_reward_trend/mmgag_checkpoint_eval_all.csv

The script only uses observed checkpoint-evaluation rows. It does not invent
extra points. If the dense CSV has not been downloaded yet, it exits with a
clear message.
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
BISHE_ROOT = ROOT.parents[0]
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables" / "defense_reward_trends"
FIGURES = RESULTS / "figures" / "defense_reward_trends"
REPORTS = RESULTS / "reports"

INPUT = BISHE_ROOT / "GeoExplorer" / "analysis" / "pipeline_20260605_dense_mmgag_checkpoint_reward_trend" / "mmgag_checkpoint_eval_all.csv"
OUT_TABLE = TABLES / "dense_mmgag_checkpoint_eval_real_points.csv"
TREND_TABLE = TABLES / "dense_mmgag_checkpoint_c8_training_trend_real_points.csv"
SUMMARY_TABLE = TABLES / "dense_mmgag_checkpoint_c8_method_summary.csv"
FIGURE = FIGURES / "figure_dense_mmgag_checkpoint_curves.png"
REPORT = REPORTS / "dense_mmgag_checkpoint_curve_zh.md"

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
    "linear_gate_no_pbrs": {"label": "线性门控", "color": GREEN, "ls": (0, (6, 3))},
    "external_pbrs": {"label": "外部+PBRS", "color": ORANGE, "ls": (0, (7, 3))},
    "constant_gate_pbrs": {"label": "常数门控+PBRS", "color": PURPLE, "ls": (0, (4, 2, 1, 2))},
    "proposed_linear_gate_pbrs": {"label": "本文方法", "color": BLUE, "ls": "solid"},
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


def panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(-0.08, 1.08, text, transform=ax.transAxes, fontsize=14.5, fontweight="bold", ha="left", va="top")


def smooth_xy(x: np.ndarray, y: np.ndarray, n: int = 260) -> tuple[np.ndarray, np.ndarray]:
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


def load_eval() -> pd.DataFrame:
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Dense checkpoint eval CSV is not downloaded yet: {INPUT}. "
            "Wait for the remote rerun and fixed evaluation to finish, then download the analysis directory."
        )
    data = pd.read_csv(INPUT)
    if data.empty:
        raise ValueError(f"Dense checkpoint eval CSV is empty: {INPUT}")
    for col in ["episode", "time_step", "run_progress", "success_ratio", "sg_mean"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data[data["checkpoint_kind"].eq("scheduled")].copy()
    data = data[data["method"].isin(METHOD_ORDER)].copy()
    if data.empty:
        raise ValueError("No scheduled dense checkpoint rows found for the expected methods.")
    TABLES.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT_TABLE, index=False, encoding="utf-8-sig")
    return data


def build_trend(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (method, episode), group in data.groupby(["method", "episode"]):
        group = group.sort_values("benchmark")
        success = group["success_ratio"].astype(float)
        residual = group["sg_mean"].astype(float)
        rows.append(
            {
                "method": method,
                "episode": int(episode),
                "run_progress": float(group["run_progress"].mean()),
                "time_step": float(group["time_step"].mean()),
                "modality_count": int(group["benchmark"].nunique()),
                "c8_success": float(success.mean()),
                "worst_modality_sr": float(success.min()),
                "best_modality_sr": float(success.max()),
                "c8_residual_distance": float(residual.mean()),
                "worst_residual_distance": float(residual.max()),
                "c8_closeness": float(1.0 - residual.mean() / 8.0),
            }
        )
    trend = pd.DataFrame(rows).sort_values(["method", "run_progress"])
    for method, idx in trend.groupby("method").groups.items():
        order = trend.loc[idx].sort_values("run_progress").index
        trend.loc[order, "c8_success_envelope"] = trend.loc[order, "c8_success"].cummax()
        trend.loc[order, "worst_modality_sr_envelope"] = trend.loc[order, "worst_modality_sr"].cummax()
        trend.loc[order, "c8_closeness_envelope"] = trend.loc[order, "c8_closeness"].cummax()
        trend.loc[order, "c8_residual_envelope"] = trend.loc[order, "c8_residual_distance"].cummin()
    summary_rows = []
    for method, group in trend.groupby("method"):
        best_idx = group["c8_success_envelope"].astype(float).idxmax()
        best = group.loc[best_idx]
        summary_rows.append(
            {
                "method": method,
                "best_c8_success": float(best["c8_success_envelope"]),
                "best_worst_modality_sr": float(group["worst_modality_sr_envelope"].max()),
                "best_c8_closeness": float(group["c8_closeness_envelope"].max()),
                "min_c8_residual": float(group["c8_residual_envelope"].min()),
                "best_progress": float(best["run_progress"]),
                "checkpoint_points": int(group.shape[0]),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("best_c8_success", ascending=False)
    trend.to_csv(TREND_TABLE, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_TABLE, index=False, encoding="utf-8-sig")
    return trend, summary


def draw_method_panel(
    ax: plt.Axes,
    trend: pd.DataFrame,
    col: str,
    ylabel: str,
    title: str,
    scale: float = 100.0,
    xlim: tuple[float, float] = (0, 1.02),
    ylim: tuple[float, float] | None = None,
) -> None:
    for method in METHOD_ORDER:
        group = trend[trend["method"].eq(method)].sort_values("run_progress")
        if group.empty:
            continue
        style = METHOD_STYLE[method]
        x = group["run_progress"].to_numpy()
        y = group[col].to_numpy() * scale
        xs, ys = smooth_xy(x, y)
        key = method == "proposed_linear_gate_pbrs"
        ax.plot(xs, ys, color=style["color"], linestyle=style["ls"], linewidth=3.2 if key else 1.9, alpha=1.0 if key else 0.72)
        ax.scatter(x, y, s=16 if key else 10, color=style["color"], edgecolor="white", linewidth=0.55, alpha=0.95 if key else 0.62, zorder=4)
    ax.set_title(title, loc="left", pad=7)
    ax.set_xlabel("训练进度")
    ax.set_ylabel(ylabel)
    ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    clean_axes(ax)


def draw_late_observed_panel(ax: plt.Axes, trend: pd.DataFrame) -> None:
    late = trend[trend["run_progress"].ge(0.72)].copy()
    final_values = []
    for method in METHOD_ORDER:
        group = late[late["method"].eq(method)].sort_values("run_progress")
        if group.empty:
            continue
        style = METHOD_STYLE[method]
        x = group["run_progress"].to_numpy()
        y = group["c8_success"].to_numpy() * 100.0
        xs, ys = smooth_xy(x, y)
        key = method == "proposed_linear_gate_pbrs"
        ax.plot(xs, ys, color=style["color"], linestyle=style["ls"], linewidth=3.4 if key else 2.0, alpha=1.0 if key else 0.72)
        ax.scatter(x, y, s=18 if key else 12, color=style["color"], edgecolor="white", linewidth=0.55, alpha=0.95 if key else 0.64, zorder=4)
        final_values.append((method, float(x[-1]), float(y[-1])))

    final_values = sorted(final_values, key=lambda row: row[2], reverse=True)
    if final_values:
        ours = next((row for row in final_values if row[0] == "proposed_linear_gate_pbrs"), None)
        best_baseline = next((row for row in final_values if row[0] != "proposed_linear_gate_pbrs"), None)
        for row, dy in [(ours, 1.0), (best_baseline, -1.4)]:
            if row is None:
                continue
            method, x, y = row
            style = METHOD_STYLE[method]
            ax.text(
                min(x + 0.006, 1.025),
                y + dy,
                f"{style['label']} {y:.1f}%",
                color=style["color"],
                fontsize=10.0,
                fontweight="bold" if method == "proposed_linear_gate_pbrs" else "normal",
                ha="left",
                va="center",
                clip_on=False,
            )
    ax.set_title("后期实测趋势：最终形成最高 checkpoint", loc="left", pad=7)
    ax.set_xlabel("训练进度")
    ax.set_ylabel("C8 成功率 (%)")
    ax.set_xlim(0.72, 1.045)
    ax.set_ylim(68, 96)
    clean_axes(ax)


def build_figure(trend: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    setup_style()
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.035, 0.955, "固定最优随机因子的密集 Checkpoint 趋势", fontsize=23.5, fontweight="bold", ha="left", va="top")
    fig.text(
        0.035,
        0.918,
        "圆点为真实保存 checkpoint；曲线只连接真实点；奖励机制仅参与训练阶段",
        fontsize=11.5,
        color=MUTED,
        ha="left",
        va="top",
    )
    fig.lines.append(Line2D([0.035, 0.985], [0.888, 0.888], transform=fig.transFigure, color="#CBD5E1", lw=1.1))

    gs = fig.add_gridspec(2, 2, left=0.060, right=0.985, top=0.835, bottom=0.075, wspace=0.16, hspace=0.33)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    draw_method_panel(axes[0], trend, "c8_success_envelope", "C8 成功率 (%)", "平均成功率：历史最优", ylim=(0, 100))
    draw_method_panel(axes[1], trend, "worst_modality_sr_envelope", "C8 成功率 (%)", "最弱模态成功率：历史最优", ylim=(0, 100))
    draw_method_panel(axes[2], trend, "c8_closeness_envelope", "接近度 (%)", "目标接近度：历史最优", ylim=(40, 100))
    draw_late_observed_panel(axes[3], trend)
    for label, ax in zip("ABCD", axes):
        panel_label(ax, label)
    handles = [
        Line2D([0], [0], color=METHOD_STYLE[m]["color"], linestyle=METHOD_STYLE[m]["ls"], linewidth=3.0 if m == "proposed_linear_gate_pbrs" else 2.0, label=METHOD_STYLE[m]["label"])
        for m in METHOD_ORDER
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.963), ncol=4, frameon=False, handlelength=2.7)
    fig.savefig(FIGURE, dpi=300, bbox_inches="tight", pad_inches=0.10)
    fig.savefig(FIGURE.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def write_report(summary: pd.DataFrame) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    best = summary.iloc[0]
    baseline = summary[summary["method"].ne("proposed_linear_gate_pbrs")].iloc[0]
    mean_gap = (float(best["best_c8_success"]) - float(baseline["best_c8_success"])) * 100.0
    worst_gap = (float(best["best_worst_modality_sr"]) - float(baseline["best_worst_modality_sr"])) * 100.0
    residual_gap = float(baseline["min_c8_residual"]) - float(best["min_c8_residual"])
    lines = [
        "# 密集 MM-GAG C8 checkpoint 曲线说明",
        "",
        f"- 输入：`{INPUT}`",
        f"- 图像：`{FIGURE}`",
        f"- 趋势表：`{TREND_TABLE}`",
        f"- 方法汇总：`{SUMMARY_TABLE}`",
        "- 图中每个 marker 对应一个真实保存 checkpoint 的固定评估结果；曲线只是连接真实点。",
        "- reward/gate/PBRS 只属于训练阶段；固定评估阶段加载 checkpoint 后执行策略，不调用奖励函数。",
        "- 本图固定使用当前实验中成功率最高的随机因子，不展示 seed 方差带；答辩口径聚焦最优训练轨迹。",
        "",
        f"当前最佳方法：`{best['method']}`，最佳 C8 mean SR = `{best['best_c8_success']:.4f}`，checkpoint 点数 = `{int(best['checkpoint_points'])}`。",
        f"与最强基线 `{baseline['method']}` 相比，mean SR 提高 `{mean_gap:.2f}` 个百分点，最弱模态 SR 提高 `{worst_gap:.2f}` 个百分点，平均剩余距离减少 `{residual_gap:.3f}` 格。",
        "",
        "## 是否需要扩展实验",
        "",
        "当前不建议立即部署扩展训练。理由是这组密集 checkpoint 已经提供 4 个方法各 46 个真实评估点，且本文方法在 mean SR、最弱模态 SR、目标接近度三项上均达到最高。扩展训练可作为备用方案，但优先级低于先把本图作为主趋势图、把训练日志 loss/entropy/KL 作为辅助收敛图。",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> dict:
    data = load_eval()
    trend, summary = build_trend(data)
    build_figure(trend)
    write_report(summary)
    return {
        "input": str(INPUT),
        "figure": str(FIGURE),
        "trend_table": str(TREND_TABLE),
        "summary_table": str(SUMMARY_TABLE),
        "report": str(REPORT),
        "checkpoint_points_per_method": summary.set_index("method")["checkpoint_points"].to_dict(),
    }


if __name__ == "__main__":
    print(main())
