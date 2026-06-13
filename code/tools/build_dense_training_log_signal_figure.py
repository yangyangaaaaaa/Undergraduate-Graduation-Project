#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build a dense, real training-log signal figure for the reward mechanism.

This figure is deliberately different from the sparse fixed-checkpoint
evaluation curves. It uses per-episode training logs to explain how the
proposed training-stage reward signal changes behavior. It should not be used
as a formal test-time ranking figure.
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


ROOT = Path(__file__).resolve().parents[2]
BISHE_ROOT = ROOT.parents[0]
LOG_ROOT = BISHE_ROOT / "GeoExplorer" / "analysis" / "pipeline_20260603_defense_reward_training_curves" / "training_logs"
FIGURES = ROOT / "results" / "figures" / "defense_reward_trends"
REPORTS = ROOT / "results" / "reports"
TABLES = ROOT / "results" / "tables" / "defense_reward_trends"

OUT = FIGURES / "figure_dense_training_log_signals.png"
REPORT = REPORTS / "dense_training_log_signal_zh.md"
SUMMARY = TABLES / "dense_training_log_signal_summary.csv"

METHOD = "proposed_linear_gate_pbrs"
INK = "#17212F"
MUTED = "#5B6777"
PAPER = "#F7F9FC"
CARD = "#FFFFFF"
GRID = "#D8E0EA"
BLUE = "#1764AB"
GREEN = "#168A63"
ORANGE = "#D27A20"
PURPLE = "#7C5CC4"
RED = "#C44536"
GRAY = "#7A8699"


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
            "axes.titlesize": 12.3,
            "axes.labelsize": 10.4,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 9.3,
            "lines.linewidth": 2.4,
        }
    )


def clean_axes(ax: plt.Axes, grid_axis: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#C9D2DE")
    ax.spines["bottom"].set_color("#C9D2DE")
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.72)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(-0.075, 1.085, text, transform=ax.transAxes, fontsize=14.5, fontweight="bold", ha="left", va="top")


def read_proposed_logs() -> pd.DataFrame:
    frames = []
    for path in sorted(LOG_ROOT.glob(f"{METHOD}_seed*_t480k/training_reward_components.csv")):
        seed_text = path.parent.name.split("_seed", 1)[1].split("_", 1)[0]
        df = pd.read_csv(path)
        df["seed"] = int(seed_text)
        df["run_name"] = path.parent.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No proposed training logs under {LOG_ROOT}")
    data = pd.concat(frames, ignore_index=True)
    numeric_cols = [
        "episode",
        "run_progress",
        "C8_success_rate",
        "C8_mean_final_dist",
        "C8_mean_reward_ex_sum",
        "C8_mean_reward_in_gated_sum",
        "C8_mean_pbrs_bonus_sum",
        "C8_mean_total_reward",
        "progress_step_ratio",
        "regress_step_ratio",
        "final_dist_mean",
        "min_dist_mean",
    ]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data.sort_values(["seed", "episode"])


def rolling_by_seed(data: pd.DataFrame, columns: list[str], window: int = 17) -> pd.DataFrame:
    rows = []
    for seed, group in data.groupby("seed"):
        group = group.sort_values("episode").copy()
        rolled = group[columns].rolling(window=window, min_periods=max(5, window // 3), center=True).mean()
        out = group[["seed", "episode", "run_progress"]].copy()
        for col in columns:
            out[col] = rolled[col]
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def mean_curve(rolled: pd.DataFrame, col: str) -> pd.DataFrame:
    return (
        rolled.groupby("episode", as_index=False)
        .agg(run_progress=("run_progress", "mean"), mean=(col, "mean"), std=(col, "std"), n=(col, "count"))
        .sort_values("run_progress")
    )


def draw_seed_lines(ax: plt.Axes, rolled: pd.DataFrame, col: str, color: str, scale: float = 1.0) -> pd.DataFrame:
    for seed, group in rolled.groupby("seed"):
        ax.plot(group["run_progress"], group[col] * scale, color=color, alpha=0.18, linewidth=1.2)
    curve = mean_curve(rolled, col)
    y = curve["mean"] * scale
    std = curve["std"].fillna(0) * scale
    ax.plot(curve["run_progress"], y, color=color, linewidth=3.0)
    ax.fill_between(curve["run_progress"], y - std, y + std, color=color, alpha=0.11, linewidth=0)
    return curve


def draw_raw_points(ax: plt.Axes, data: pd.DataFrame, col: str, color: str, scale: float = 1.0, every: int = 5) -> None:
    sample = data[(data["episode"].astype(int) % every).eq(0)].copy()
    ax.scatter(sample["run_progress"], sample[col] * scale, s=8, color=color, alpha=0.12, linewidth=0)


def build_summary(data: pd.DataFrame, rolled: pd.DataFrame) -> pd.DataFrame:
    late = data[data["run_progress"].ge(0.75)].copy()
    rows = []
    for seed, group in late.groupby("seed"):
        rows.append(
            {
                "method": METHOD,
                "seed": int(seed),
                "episode_points": int((data["seed"].eq(seed)).sum()),
                "late_C8_success_rate": float(group["C8_success_rate"].mean()),
                "late_C8_final_distance": float(group["C8_mean_final_dist"].mean()),
                "late_progress_step_ratio": float(group["progress_step_ratio"].mean()),
                "late_regress_step_ratio": float(group["regress_step_ratio"].mean()),
                "late_C8_external_sum": float(group["C8_mean_reward_ex_sum"].mean()),
                "late_C8_intrinsic_gated_sum": float(group["C8_mean_reward_in_gated_sum"].mean()),
                "late_C8_pbrs_sum": float(group["C8_mean_pbrs_bonus_sum"].mean()),
                "late_C8_total_reward": float(group["C8_mean_total_reward"].mean()),
            }
        )
    summary = pd.DataFrame(rows)
    avg = {"method": METHOD, "seed": -1, "episode_points": int(data.shape[0])}
    for col in summary.columns:
        if col not in {"method", "seed", "episode_points"}:
            avg[col] = float(summary[col].mean())
    summary = pd.concat([summary, pd.DataFrame([avg])], ignore_index=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY, index=False, encoding="utf-8-sig")
    return summary


def build_figure(data: pd.DataFrame, rolled: pd.DataFrame, summary: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    setup_style()
    fig = plt.figure(figsize=(16, 9))
    fig.text(0.035, 0.955, "Dense Training Signals", fontsize=23.5, fontweight="bold", ha="left", va="top")
    fig.text(
        0.035,
        0.918,
        "Ours only; all points come from per-episode training logs, not interpolated checkpoint eval",
        fontsize=11.5,
        color=MUTED,
        ha="left",
        va="top",
    )
    fig.lines.append(Line2D([0.035, 0.985], [0.888, 0.888], transform=fig.transFigure, color="#CBD5E1", lw=1.1))

    gs = fig.add_gridspec(2, 2, left=0.060, right=0.985, top=0.835, bottom=0.075, wspace=0.16, hspace=0.33)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    draw_raw_points(ax_a, data, "C8_mean_final_dist", BLUE, every=4)
    draw_seed_lines(ax_a, rolled, "C8_mean_final_dist", BLUE)
    ax_a.set_title("C8 Distance Falls", loc="left", pad=7)
    ax_a.set_xlabel("Progress")
    ax_a.set_ylabel("Final distance")
    ax_a.set_xlim(0, 1.02)
    ax_a.set_ylim(0, 8.2)
    clean_axes(ax_a)

    draw_raw_points(ax_b, data, "C8_success_rate", GREEN, scale=100.0, every=4)
    draw_seed_lines(ax_b, rolled, "C8_success_rate", GREEN, scale=100.0)
    ax_b.set_title("C8 Success Forms", loc="left", pad=7)
    ax_b.set_xlabel("Progress")
    ax_b.set_ylabel("Success rate (%)")
    ax_b.set_xlim(0, 1.02)
    ax_b.set_ylim(0, 62)
    clean_axes(ax_b)

    progress = mean_curve(rolled, "progress_step_ratio")
    regress = mean_curve(rolled, "regress_step_ratio")
    ax_c.plot(progress["run_progress"], progress["mean"] * 100, color=BLUE, linewidth=3.0, label="Progress")
    ax_c.plot(regress["run_progress"], regress["mean"] * 100, color=RED, linewidth=2.4, linestyle=(0, (6, 3)), label="Regress")
    ax_c.fill_between(
        progress["run_progress"],
        regress["mean"] * 100,
        progress["mean"] * 100,
        where=(progress["mean"] >= regress["mean"]).to_numpy(),
        color=BLUE,
        alpha=0.14,
        linewidth=0,
    )
    ax_c.set_title("Action Direction Gap", loc="left", pad=7)
    ax_c.set_xlabel("Progress")
    ax_c.set_ylabel("Step share (%)")
    ax_c.set_xlim(0, 1.02)
    ax_c.set_ylim(24, 78)
    ax_c.legend(frameon=False, loc="lower right")
    clean_axes(ax_c)

    ex = mean_curve(rolled, "C8_mean_reward_ex_sum")
    intr = mean_curve(rolled, "C8_mean_reward_in_gated_sum")
    pbrs = mean_curve(rolled, "C8_mean_pbrs_bonus_sum")
    total = mean_curve(rolled, "C8_mean_total_reward")
    ax_d.plot(ex["run_progress"], ex["mean"], color=ORANGE, linewidth=2.4, linestyle=(0, (6, 3)), label="Ext")
    ax_d.plot(intr["run_progress"], intr["mean"], color=PURPLE, linewidth=2.6, label="Int*g")
    ax_d.plot(total["run_progress"], total["mean"], color=BLUE, linewidth=3.0, label="Total")
    ax_d.axhline(0, color=INK, linewidth=1.0, alpha=0.70)
    ax_d2 = ax_d.twinx()
    ax_d2.plot(pbrs["run_progress"], pbrs["mean"], color=GREEN, linewidth=2.2, label="PBRS")
    ax_d2.spines["top"].set_visible(False)
    ax_d2.spines["right"].set_color("#C9D2DE")
    ax_d2.tick_params(axis="y", colors=MUTED, labelsize=9.0)
    ax_d2.set_ylabel("PBRS")
    ax_d.set_title("Reward Components", loc="left", pad=7)
    ax_d.set_xlabel("Progress")
    ax_d.set_ylabel("C8 reward sum")
    ax_d.set_xlim(0, 1.02)
    ax_d.set_ylim(-8.8, 6.2)
    ax_d2.set_ylim(-0.03, 0.14)
    h1, l1 = ax_d.get_legend_handles_labels()
    h2, l2 = ax_d2.get_legend_handles_labels()
    ax_d.legend(h1 + h2, l1 + l2, frameon=False, loc="lower right", ncol=4, handlelength=2.0, columnspacing=0.8)
    clean_axes(ax_d)

    for label, ax in zip("ABCD", [ax_a, ax_b, ax_c, ax_d]):
        panel_label(ax, label)

    fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.10)
    fig.savefig(OUT.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def write_report(summary: pd.DataFrame) -> None:
    avg = summary[summary["seed"].eq(-1)].iloc[0]
    REPORTS.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 密集训练日志信号图说明",
        "",
        f"- 图像文件：`{OUT}`",
        f"- 数据汇总：`{SUMMARY}`",
        "- 数据来源：`pipeline_20260603_defense_reward_training_curves/training_logs` 中本文方法 3 个 seed 的 `training_reward_components.csv`。",
        "- 这张图只解释训练阶段奖励如何改变行为，不作为测试阶段排名证据；正式排名仍应使用固定 checkpoint 评估。",
        "- 图中曲线由逐 episode 真实日志做滚动均值，浅色点/浅色线来自真实 episode 记录，没有把 10 个 checkpoint 插值成更多点。",
        "",
        "## 可用于答辩的解释",
        "",
        f"1. C8 最终距离在训练中持续下降，末期三 seed 均值约为 `{avg['late_C8_final_distance']:.3f}`，说明奖励信号逐步把策略推向中长距离目标。",
        f"2. C8 成功率从训练初期接近 0 逐步形成，末期三 seed 均值约为 `{avg['late_C8_success_rate']:.3f}`；它是训练采样信号，不应和固定测试成功率混用。",
        f"3. 前进一步比例末期约为 `{avg['late_progress_step_ratio']:.3f}`，回退比例约为 `{avg['late_regress_step_ratio']:.3f}`，可以说明奖励信号在动作层面减少无效回退。",
        f"4. 奖励分量图显示外部奖励、门控内在奖励和 PBRS 在同一训练阶段共同作用；PBRS 数值量级较小，所以图中使用单独右轴显示，避免被总奖励曲线吞掉。",
        "",
        "## 和密集 checkpoint 实验的关系",
        "",
        "这张图解决“训练日志点数不够”的解释性问题；正在补跑的密集 checkpoint 实验解决“固定评估曲线点数不够”的排名问题。两者不要混成同一类证据。",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> dict:
    data = read_proposed_logs()
    columns = [
        "C8_success_rate",
        "C8_mean_final_dist",
        "C8_mean_reward_ex_sum",
        "C8_mean_reward_in_gated_sum",
        "C8_mean_pbrs_bonus_sum",
        "C8_mean_total_reward",
        "progress_step_ratio",
        "regress_step_ratio",
    ]
    rolled = rolling_by_seed(data, columns)
    summary = build_summary(data, rolled)
    build_figure(data, rolled, summary)
    write_report(summary)
    return {"figure": str(OUT), "svg": str(OUT.with_suffix(".svg")), "report": str(REPORT), "summary": str(SUMMARY)}


if __name__ == "__main__":
    print(main())
