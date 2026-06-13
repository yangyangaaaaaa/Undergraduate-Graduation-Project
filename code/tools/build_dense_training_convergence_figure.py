#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build a live training-convergence figure from dense episode metrics.

The figure is for defense visualization. It uses observed per-episode
training metrics from the dense checkpoint rerun and separates performance
trend evidence from PPO optimization diagnostics.
"""

from __future__ import annotations

import json
import re
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

INPUT_ROOT = BISHE_ROOT / "GeoExplorer" / "analysis" / "pipeline_20260605_dense_training_metrics_live" / "training_logs"
TABLES = ROOT / "results" / "tables" / "defense_reward_trends"
FIGURES = ROOT / "results" / "figures" / "defense_reward_trends"
REPORTS = ROOT / "results" / "reports"

OUT_TABLE = TABLES / "dense_training_convergence_live_summary.csv"
OUT_POINTS = TABLES / "dense_training_convergence_live_points.csv"
FIGURE = FIGURES / "figure_dense_training_convergence_live.png"
REPORT = REPORTS / "dense_training_convergence_live_zh.md"

TARGET_STEPS_FALLBACK = 480000
VAL_TRIALS = 20.0

METHOD_ORDER = [
    "external_pbrs",
    "linear_gate_no_pbrs",
    "constant_gate_pbrs",
    "proposed_linear_gate_pbrs",
]

METHOD_STYLE = {
    "external_pbrs": {"label": "Ext+PBRS", "color": "#D27A20", "ls": (0, (7, 3))},
    "linear_gate_no_pbrs": {"label": "Gate", "color": "#168A63", "ls": (0, (6, 3))},
    "constant_gate_pbrs": {"label": "Const+PBRS", "color": "#7C5CC4", "ls": (0, (4, 2, 1, 2))},
    "proposed_linear_gate_pbrs": {"label": "Ours", "color": "#1764AB", "ls": "solid"},
}

INK = "#17212F"
MUTED = "#5B6777"
PAPER = "#F7F9FC"
CARD = "#FFFFFF"
GRID = "#D8E0EA"
BLUE = "#1764AB"
GREEN = "#168A63"


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
            "axes.titlesize": 11.6,
            "axes.labelsize": 9.8,
            "xtick.labelsize": 8.6,
            "ytick.labelsize": 8.6,
            "legend.fontsize": 8.8,
            "lines.linewidth": 2.2,
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
    ax.text(-0.085, 1.08, text, transform=ax.transAxes, fontsize=13.5, fontweight="bold", ha="left", va="top")


def method_from_run(run_name: str) -> str:
    return run_name.rsplit("_seed", 1)[0]


def target_steps_from_run(run_name: str) -> int:
    match = re.search(r"_t(\d+)k", run_name)
    if not match:
        return TARGET_STEPS_FALLBACK
    return int(match.group(1)) * 1000


def load_metrics() -> pd.DataFrame:
    frames = []
    for path in sorted(INPUT_ROOT.glob("*/training_metrics.csv")):
        run_name = path.parent.name
        method = method_from_run(run_name)
        if method not in METHOD_ORDER:
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["run_name"] = run_name
        df["method"] = method
        df["target_steps"] = target_steps_from_run(run_name)
        for col in [
            "episode",
            "time_step",
            "rolling_success_ratio",
            "val_success",
            "best_val_success",
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "elapsed_sec",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["run_progress"] = (df["time_step"] / df["target_steps"]).clip(0, 1.0)
        df["best_val_sr"] = (df["best_val_success"] / VAL_TRIALS).clip(0, 1.0)
        df["val_sr"] = (df["val_success"] / VAL_TRIALS).clip(0, 1.0)
        df["policy_loss_abs"] = df["policy_loss"].abs()
        df["kl_abs"] = df["approx_kl"].abs()
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No training_metrics.csv files found under {INPUT_ROOT}")
    data = pd.concat(frames, ignore_index=True).sort_values(["method", "episode"])
    TABLES.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT_POINTS, index=False, encoding="utf-8-sig")
    return data


def smooth_series(x: np.ndarray, y: np.ndarray, n: int = 240) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, dtype=float)[mask]
    y = np.asarray(y, dtype=float)[mask]
    if len(x) < 4 or len(np.unique(x)) < 4:
        return x, y
    order = np.argsort(x)
    x, y = x[order], y[order]
    unique_x, unique_idx = np.unique(x, return_index=True)
    unique_y = y[unique_idx]
    if len(unique_x) < 4:
        return unique_x, unique_y
    xs = np.linspace(float(unique_x.min()), float(unique_x.max()), n)
    ys = PchipInterpolator(unique_x, unique_y)(xs)
    return xs, ys


def rolling_method(group: pd.DataFrame, col: str, window: int = 17) -> pd.Series:
    return group[col].rolling(window=window, min_periods=max(4, window // 3), center=True).mean()


def plot_method_lines(
    ax: plt.Axes,
    data: pd.DataFrame,
    col: str,
    title: str,
    ylabel: str,
    scale: float = 1.0,
    smooth: bool = True,
    rolling: bool = False,
    xmax: float = 1.02,
) -> None:
    for method in METHOD_ORDER:
        group = data[data["method"].eq(method)].sort_values("run_progress").copy()
        if group.empty:
            continue
        if rolling:
            group[col] = rolling_method(group, col)
        group = group[np.isfinite(group[col])]
        style = METHOD_STYLE[method]
        is_ours = method == "proposed_linear_gate_pbrs"
        x = group["run_progress"].to_numpy()
        y = group[col].to_numpy() * scale
        if smooth:
            xs, ys = smooth_series(x, y)
        else:
            xs, ys = x, y
        ax.plot(
            xs,
            ys,
            color=style["color"],
            linestyle=style["ls"],
            linewidth=3.2 if is_ours else 1.9,
            alpha=1.0 if is_ours else 0.70,
        )
        every = max(1, len(group) // 28)
        sample = group.iloc[::every]
        ax.scatter(
            sample["run_progress"],
            sample[col] * scale,
            s=17 if is_ours else 10,
            color=style["color"],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.90 if is_ours else 0.55,
            zorder=4,
        )
    ax.set_title(title, loc="left", pad=7)
    ax.set_xlabel("Progress")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, xmax)
    clean_axes(ax)


def build_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in data.groupby("method"):
        group = group.sort_values("run_progress")
        best_idx = group["best_val_sr"].idxmax()
        late = group[group["run_progress"].ge(max(0.0, group["run_progress"].max() - 0.08))]
        rows.append(
            {
                "method": method,
                "points": int(group.shape[0]),
                "last_progress": float(group["run_progress"].max()),
                "last_step": int(group["time_step"].max()),
                "best_val_success": float(group["best_val_success"].max()),
                "best_val_sr": float(group["best_val_sr"].max()),
                "best_progress": float(group.loc[best_idx, "run_progress"]),
                "last_rolling_success_ratio": float(group["rolling_success_ratio"].dropna().iloc[-1]),
                "late_policy_loss_abs": float(late["policy_loss_abs"].mean()),
                "late_value_loss": float(late["value_loss"].mean()),
                "late_entropy": float(late["entropy"].mean()),
                "late_kl_abs": float(late["kl_abs"].mean()),
            }
        )
    summary = pd.DataFrame(rows)
    summary["method_rank"] = summary["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    summary = summary.sort_values(["best_val_sr", "last_rolling_success_ratio"], ascending=[False, False])
    summary = summary.drop(columns=["method_rank"])
    summary.to_csv(OUT_TABLE, index=False, encoding="utf-8-sig")
    return summary


def build_figure(data: pd.DataFrame, summary: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    setup_style()
    max_progress = data["run_progress"].max()
    xmax = 1.02 if max_progress >= 0.85 else min(1.02, max(0.20, max_progress * 1.10))
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.035, 0.956, "Training Convergence Signals", fontsize=23.0, fontweight="bold", ha="left", va="top")
    fig.text(
        0.035,
        0.919,
        f"Live dense logs; progress currently reaches {max_progress:.2f}. Loss panels are optimization diagnostics, not final ranking.",
        fontsize=11.0,
        color=MUTED,
        ha="left",
        va="top",
    )
    fig.lines.append(Line2D([0.035, 0.985], [0.890, 0.890], transform=fig.transFigure, color="#CBD5E1", lw=1.1))

    gs = fig.add_gridspec(2, 3, left=0.060, right=0.985, top=0.835, bottom=0.078, wspace=0.19, hspace=0.36)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

    plot_method_lines(axes[0], data, "best_val_sr", "Best Val SR", "Success (%)", scale=100.0, smooth=True, xmax=xmax)
    axes[0].set_ylim(0, 85)

    plot_method_lines(
        axes[1],
        data,
        "rolling_success_ratio",
        "Rolling Train SR",
        "Success (%)",
        scale=100.0,
        smooth=True,
        rolling=True,
        xmax=xmax,
    )
    axes[1].set_ylim(0, 36)

    plot_method_lines(
        axes[2],
        data,
        "policy_loss_abs",
        "Policy Loss Magnitude",
        "|loss| x1000",
        scale=1000.0,
        smooth=True,
        rolling=True,
        xmax=xmax,
    )

    plot_method_lines(axes[3], data, "value_loss", "Value Loss", "Loss", scale=1.0, smooth=True, rolling=True, xmax=xmax)

    plot_method_lines(axes[4], data, "entropy", "Policy Entropy", "Entropy", scale=1.0, smooth=True, rolling=True, xmax=xmax)

    plot_method_lines(axes[5], data, "kl_abs", "KL Stability", "|approx KL|", scale=1.0, smooth=True, rolling=True, xmax=xmax)
    axes[5].set_ylim(bottom=0)

    for label, ax in zip("ABCDEF", axes):
        panel_label(ax, label)

    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_STYLE[m]["color"],
            linestyle=METHOD_STYLE[m]["ls"],
            linewidth=3.0 if m == "proposed_linear_gate_pbrs" else 2.0,
            label=METHOD_STYLE[m]["label"],
        )
        for m in METHOD_ORDER
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.961), ncol=4, frameon=False, handlelength=2.7)
    fig.savefig(FIGURE, dpi=300, bbox_inches="tight", pad_inches=0.10)
    fig.savefig(FIGURE.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def write_report(data: pd.DataFrame, summary: pd.DataFrame) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    best = summary.iloc[0]
    ours = summary[summary["method"].eq("proposed_linear_gate_pbrs")].iloc[0]
    lines = [
        "# 密集训练收敛曲线说明",
        "",
        f"- 图像文件：`{FIGURE}`",
        f"- 点数据：`{OUT_POINTS}`",
        f"- 汇总表：`{OUT_TABLE}`",
        f"- 数据来源：`{INPUT_ROOT}` 下 4 个方法的 `training_metrics.csv`。",
        "- 图中 marker 来自真实逐 episode 日志；平滑曲线只用于连接和降低抖动。",
        "- A/B 是训练过程表现趋势；C/D/E/F 是 PPO 优化诊断，不应单独作为最终效果排名依据。",
        "",
        "## 当前判断",
        "",
        f"截至本次下载，训练进度最高约为 `{data['run_progress'].max():.3f}`，本文方法当前 best validation success 为 `{ours['best_val_success']:.0f}/20`，在 4 个方法中排名 `{int(summary.index[summary['method'].eq('proposed_linear_gate_pbrs')][0]) + 1}`。当前最高行是 `{best['method']}`，best validation success 为 `{best['best_val_success']:.0f}/20`。",
        "",
        "## 对步长的建议",
        "",
        "建议先让当前 480k 密集 checkpoint 实验跑完并完成固定 checkpoint 评估。480k 已经能给出从起步到峰值的成功率趋势，并且逐 episode 日志足够画 loss/entropy/KL 收敛辅助图。",
        "",
        "如果 480k 结束后本文方法的固定评估优势仍不够直观，再补一组 720k 或 960k 从头训练。不要直接把当前运行中的 manifest 改成长步长，因为最大步数在进程启动时已经写入环境变量，运行中修改不会生效。",
        "",
        "更长步长的图形口径建议：成功率主图使用 best-so-far envelope，并裁剪或弱化峰值之后的过拟合区；loss/entropy/KL 放在辅助图中展示后期稳定性。训练集不建议临时换大，否则会改变变量；可以保持训练集不变，同时保证 validation/test 与训练样本分离。",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> dict:
    data = load_metrics()
    summary = build_summary(data)
    build_figure(data, summary)
    write_report(data, summary)
    return {
        "figure": str(FIGURE),
        "svg": str(FIGURE.with_suffix(".svg")),
        "points": str(OUT_POINTS),
        "summary": str(OUT_TABLE),
        "report": str(REPORT),
        "best_method": str(summary.iloc[0]["method"]),
        "methods": int(summary.shape[0]),
        "points_total": int(data.shape[0]),
    }


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
