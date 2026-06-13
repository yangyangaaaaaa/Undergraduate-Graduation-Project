#!/usr/bin/env python
"""Build final defense figures for training-stage mixed reward analysis.

The package separates three claims:
1. Reward and gate/PBRS are training-time signals.
2. Formal "best method" evidence comes from the fixed MM-GAG reward-gate table.
3. Route figures explain the learned policy behavior after training.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
REPORTS = RESULTS / "reports"
FIGURES = RESULTS / "figures" / "defense_reward_final"
SUMMARY_TABLES = TABLES / "defense_reward_final"

TRAIN_LOG_ROOT = Path(r"F:\bishe\GeoExplorer\analysis\pipeline_20260603_defense_reward_training_curves\training_logs")

GATE_TABLE = TABLES / "ablation" / "reward_gate_type_mmgag_only_table_with_linear.csv"
ROUTE_TABLE = TABLES / "trajectory_analysis" / "trajectory_behavior_by_distance.csv"
GENERALIZATION_TABLE = TABLES / "ablation" / "anchor0624_generalization_table.csv"

ROUTE_THREE_METHOD = RESULTS / "figures" / "chapter4_trajectories" / "figure4_y_three_method_hardcase_revised.png"
ROUTE_C4_C6_C8 = RESULTS / "figures" / "chapter4_trajectories" / "figure4_x_anchor_typical_c4_c6_c8.png"
ROUTE_GP_2X2 = RESULTS / "figures" / "showcase" / "reward_story" / "figure_b_gp_2x2_paths_img189.png"


INK = "#17212F"
MUTED = "#5B6777"
GRID = "#D8E0EA"
PAPER = "#F7F9FC"
CARD = "#FFFFFF"
BLUE = "#1764AB"
BLUE_LIGHT = "#DCEBFA"
ORANGE = "#D27A20"
GREEN = "#168A63"
TEAL = "#2098A3"
RED = "#B84A48"
PURPLE = "#7C5CC4"
GRAY = "#808A98"
GRAY_LIGHT = "#E7ECF2"
YELLOW = "#F4C542"

METHODS = {
    "external_only": {"label": "仅外部奖励", "short": "外部奖励", "color": ORANGE},
    "intrinsic_only": {"label": "仅内在奖励", "short": "内在奖励", "color": PURPLE},
    "mixed_no_gate_no_pbrs": {"label": "外部+内在直接相加", "short": "直接相加", "color": GREEN},
    "mixed_gate_only": {"label": "外部+门控内在", "short": "门控内在", "color": TEAL},
    "mixed_pbrs_only": {"label": "外部+内在+PBRS", "short": "仅加 PBRS", "color": RED},
    "proposed_linear_gate_pbrs": {"label": "本文方法：线性门控+PBRS", "short": "本文方法", "color": BLUE},
}

METHOD_ORDER = [
    "external_only",
    "intrinsic_only",
    "mixed_no_gate_no_pbrs",
    "mixed_gate_only",
    "mixed_pbrs_only",
    "proposed_linear_gate_pbrs",
]

RANK_LABELS = {
    "linear_0.405_pb": "本文方法\n线性门控+PBRS",
    "external_pbrs": "外部奖励+PBRS",
    "constant_0.405_pb": "常数门控+PBRS",
    "linear_0.405_no_pb": "线性门控\n无 PBRS",
    "blend_lp_pb": "线性-幂次门控+PBRS",
    "blend_lp_no_pb": "线性-幂次门控\n无 PBRS",
    "power2_pb": "二次门控+PBRS",
    "constant_0.405_no_pb": "常数门控\n无 PBRS",
    "power2_no_pb": "二次门控\n无 PBRS",
    "sine_pb": "正弦门控+PBRS",
    "sine_no_pb": "正弦门控\n无 PBRS",
}


def ensure_dirs() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    SUMMARY_TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)


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
            "axes.titlesize": 13.5,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "lines.linewidth": 2.2,
        }
    )


def save_figure(fig: plt.Figure, stem: str, dpi: int = 300) -> None:
    fig.savefig(FIGURES / f"{stem}.png", dpi=dpi, bbox_inches="tight", pad_inches=0.14)
    fig.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)


def clean_axes(ax: plt.Axes, grid_axis: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#C9D2DE")
    ax.spines["bottom"].set_color("#C9D2DE")
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.72)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.085,
        1.075,
        label,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        ha="left",
        va="top",
        color=INK,
    )


def add_header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.04, 0.968, title, fontsize=24, fontweight="bold", ha="left", va="top", color=INK)
    fig.text(0.04, 0.928, subtitle, fontsize=11.8, ha="left", va="top", color=MUTED)
    fig.lines.append(
        plt.Line2D([0.04, 0.965], [0.895, 0.895], transform=fig.transFigure, color="#CCD6E2", lw=1.2)
    )


def add_note(ax: plt.Axes, text: str, xy: tuple[float, float] = (0.02, 0.03), width: float = 0.92) -> None:
    ax.text(
        xy[0],
        xy[1],
        text,
        transform=ax.transAxes,
        fontsize=9.5,
        color=MUTED,
        va="bottom",
        ha="left",
        linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.15", facecolor="#F8FAFC", edgecolor="#D9E1EC"),
        wrap=True,
    )


def moving_average(values: pd.Series, window: int = 17) -> pd.Series:
    return values.astype(float).rolling(window=window, min_periods=1, center=True).mean()


def method_from_run(run_name: str) -> str | None:
    for method in METHOD_ORDER:
        if run_name.startswith(method + "_seed"):
            return method
    return None


def seed_from_run(run_name: str) -> int | None:
    if "_seed" not in run_name:
        return None
    text = run_name.split("_seed", 1)[1].split("_", 1)[0]
    try:
        return int(text)
    except ValueError:
        return None


def read_training_logs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_rows: list[pd.DataFrame] = []
    component_rows: list[pd.DataFrame] = []
    route_rows: list[pd.DataFrame] = []
    if not TRAIN_LOG_ROOT.exists():
        raise FileNotFoundError(TRAIN_LOG_ROOT)

    for run_dir in sorted(TRAIN_LOG_ROOT.iterdir()):
        if not run_dir.is_dir():
            continue
        method = method_from_run(run_dir.name)
        seed = seed_from_run(run_dir.name)
        if method is None or seed is None:
            continue
        for filename, rows in [
            ("training_metrics.csv", metrics_rows),
            ("training_reward_components.csv", component_rows),
            ("training_route_samples.csv", route_rows),
        ]:
            path = run_dir / filename
            if not path.exists() or path.stat().st_size == 0:
                continue
            df = pd.read_csv(path)
            if df.empty:
                continue
            df["run_name"] = run_dir.name
            df["method"] = method
            df["seed"] = seed
            df["method_label"] = METHODS[method]["label"]
            df["method_short"] = METHODS[method]["short"]
            if "run_progress" not in df and "time_step" in df:
                df["run_progress"] = df["time_step"].astype(float) / 480000.0
            rows.append(df)

    metrics = pd.concat(metrics_rows, ignore_index=True)
    components = pd.concat(component_rows, ignore_index=True)
    routes = pd.concat(route_rows, ignore_index=True)
    return metrics, components, routes


def interpolate_by_run(df: pd.DataFrame, y_col: str, points: int = 121, smooth: int = 13) -> pd.DataFrame:
    rows: list[dict] = []
    grid = np.linspace(0.0, min(1.0, float(df["run_progress"].max())), points)
    for (method, seed, run_name), sub in df.groupby(["method", "seed", "run_name"]):
        if y_col not in sub:
            continue
        sub = sub.sort_values("run_progress")
        x = sub["run_progress"].astype(float).to_numpy()
        y = moving_average(sub[y_col], smooth).astype(float).to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if len(x) < 2:
            continue
        valid_grid = grid[(grid >= x.min()) & (grid <= x.max())]
        if len(valid_grid) == 0:
            continue
        vals = np.interp(valid_grid, x, y)
        rows.extend(
            {
                "method": method,
                "seed": seed,
                "run_name": run_name,
                "run_progress": gx,
                y_col: gy,
            }
            for gx, gy in zip(valid_grid, vals)
        )
    return pd.DataFrame(rows)


def mean_band(df: pd.DataFrame, y_col: str) -> pd.DataFrame:
    return (
        df.groupby(["method", "run_progress"], as_index=False)
        .agg(mean=(y_col, "mean"), std=(y_col, "std"), n=(y_col, "count"))
        .assign(std=lambda x: x["std"].fillna(0.0))
    )


def plot_mean_band(
    ax: plt.Axes,
    source: pd.DataFrame,
    y_col: str,
    methods: Iterable[str],
    scale: float = 1.0,
    alpha: float = 0.10,
    lw_key: float = 3.0,
) -> None:
    interp = interpolate_by_run(source, y_col)
    band = mean_band(interp, y_col)
    for method in methods:
        sub = band[band["method"] == method].sort_values("run_progress")
        if sub.empty:
            continue
        x = sub["run_progress"].to_numpy() * 100
        mean = sub["mean"].to_numpy() * scale
        std = sub["std"].to_numpy() * scale
        color = METHODS[method]["color"]
        lw = lw_key if method == "proposed_linear_gate_pbrs" else 2.0
        z = 5 if method == "proposed_linear_gate_pbrs" else 3
        ax.plot(x, mean, color=color, lw=lw, label=METHODS[method]["short"], zorder=z)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=alpha, linewidth=0, zorder=z - 1)


def add_c6_c8_fields(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = metrics.copy()
    counts = pd.Series(0.0, index=metrics.index)
    successes = pd.Series(0.0, index=metrics.index)
    final_num = pd.Series(0.0, index=metrics.index)
    for dist in [6, 7, 8]:
        count_col = f"C{dist}_trajectory_count"
        success_col = f"C{dist}_success_count"
        final_col = f"C{dist}_mean_final_dist"
        if count_col in metrics:
            c = metrics[count_col].astype(float).fillna(0.0)
            counts += c
            if success_col in metrics:
                successes += metrics[success_col].astype(float).fillna(0.0)
            if final_col in metrics:
                final_num += metrics[final_col].astype(float).fillna(0.0) * c
    metrics["c6_c8_success_rate"] = successes / counts.replace(0, np.nan)
    metrics["c6_c8_final_dist"] = final_num / counts.replace(0, np.nan)
    return metrics


def gate_weight_curve(distance: np.ndarray, gate_floor: float = 0.405, optimal_steps: int = 8) -> np.ndarray:
    # Mirrors the linear training gate used by the experiment logs: far distance keeps full intrinsic signal,
    # and the gate decays toward a floor when the policy is near the target.
    ratio = np.clip(distance / max(1, optimal_steps), 0.0, 1.0)
    return gate_floor + (1.0 - gate_floor) * ratio


def pbrs_curve(prev_distance: np.ndarray, coef: float = 0.1, optimal_steps: int = 8) -> np.ndarray:
    improved = np.maximum(prev_distance - np.maximum(prev_distance - 1, 0), 0)
    normalized = improved / max(1, optimal_steps)
    return coef * (0.1 + normalized)


def label_gate_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("mmgag_mean_sr", ascending=False).copy()
    df["rank"] = np.arange(1, len(df) + 1)
    df["label"] = df["value"].map(RANK_LABELS).fillna(df["value"])
    df["pb_enabled"] = df["value"].str.endswith("_pb") | df["value"].eq("external_pbrs")
    return df


def draw_main_evidence(metrics: pd.DataFrame) -> None:
    metrics = add_c6_c8_fields(metrics)
    gate = label_gate_values(pd.read_csv(GATE_TABLE))
    top4 = gate.head(4).copy()
    ours = gate[gate["value"].eq("linear_0.405_pb")].iloc[0]
    external = gate[gate["value"].eq("external_pbrs")].iloc[0]
    no_pb = gate[gate["value"].eq("linear_0.405_no_pb")].iloc[0]

    fig = plt.figure(figsize=(15.6, 8.8))
    add_header(
        fig,
        "训练阶段奖励设计如何形成最优 MM-GAG 策略",
        "左侧解释奖励怎样指导 PPO 学习；右侧使用同协议 reward-gate/PBRS 正式表证明本文方法排名第一。",
    )
    gs = GridSpec(2, 3, figure=fig, left=0.055, right=0.965, top=0.84, bottom=0.09, width_ratios=[1.05, 1.0, 1.32], hspace=0.36, wspace=0.32)
    ax_curve = fig.add_subplot(gs[0, 0])
    ax_gate = fig.add_subplot(gs[1, 0])
    ax_rank = fig.add_subplot(gs[:, 1])
    ax_modal = fig.add_subplot(gs[0, 2])
    ax_message = fig.add_subplot(gs[1, 2])

    training_methods = ["external_only", "mixed_no_gate_no_pbrs", "mixed_gate_only", "mixed_pbrs_only", "proposed_linear_gate_pbrs"]
    plot_mean_band(ax_curve, metrics, "c6_c8_success_rate", training_methods, scale=100)
    ax_curve.set_title("训练曲线：中长距离样本逐步学会到达目标")
    ax_curve.set_xlabel("训练进度（%）")
    ax_curve.set_ylabel("C=6-8 训练样本成功率（%）")
    ax_curve.set_ylim(0, 58)
    clean_axes(ax_curve)
    ax_curve.legend(frameon=False, ncol=2, loc="upper left")
    add_note(ax_curve, "阴影为 3 个 seed 的标准差；曲线用于说明训练过程，不作为最终排名依据。", xy=(0.03, 0.04))

    d = np.arange(0, 9)
    gate_y = gate_weight_curve(d)
    ax_gate.plot(d, gate_y, color=BLUE, lw=3.0, marker="o", ms=6, label="线性门控权重 G(d)")
    ax_gate.fill_between(d, 0, gate_y, color=BLUE_LIGHT, alpha=0.9)
    ax_gate.axhline(0.405, color=GRAY, ls="--", lw=1.2)
    ax_gate.text(0.15, 0.425, "近目标保留 0.405", color=MUTED, fontsize=9.5)
    ax_gate.annotate(
        "距离远：保持探索信号",
        xy=(8, gate_y[-1]),
        xytext=(4.6, 0.87),
        arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.4),
        fontsize=10,
        color=BLUE,
    )
    ax_gate.annotate(
        "距离近：降低内在奖励干扰",
        xy=(1, gate_y[1]),
        xytext=(2.2, 0.56),
        arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.4),
        fontsize=10,
        color=BLUE,
    )
    ax_gate.set_title("奖励机制：按距离调节训练信号")
    ax_gate.set_xlabel("当前位置到目标的网格距离 d")
    ax_gate.set_ylabel("内在奖励权重")
    ax_gate.set_ylim(0.34, 1.06)
    ax_gate.set_xticks(d)
    clean_axes(ax_gate)

    rank = top4.iloc[::-1]
    y = np.arange(len(rank))
    colors = [BLUE if v == "linear_0.405_pb" else "#B8C2CF" for v in rank["value"]]
    ax_rank.barh(y, rank["mmgag_mean_sr"] * 100, color=colors, height=0.58)
    ax_rank.set_yticks(y)
    ax_rank.set_yticklabels(rank["label"])
    ax_rank.set_xlim(56.5, 61.8)
    ax_rank.set_xlabel("MM-GAG 三模态平均 SR（%）")
    ax_rank.set_title("正式评估排名：本文方法第一")
    clean_axes(ax_rank, "x")
    for yi, (_, row) in enumerate(rank.iterrows()):
        value = row["mmgag_mean_sr"] * 100
        label = f"{value:.2f}%"
        color = BLUE if row["value"] == "linear_0.405_pb" else MUTED
        ax_rank.text(value + 0.08, yi, label, va="center", ha="left", fontsize=11.5, fontweight="bold", color=color)
    ax_rank.text(
        0.02,
        -0.11,
        "同一 MM-GAG 任务协议；评估阶段只加载 checkpoint，不调用训练奖励函数。",
        transform=ax_rank.transAxes,
        fontsize=9.5,
        color=MUTED,
        ha="left",
        va="top",
    )

    modalities = ["mmgag_aerial", "mmgag_ground", "mmgag_text"]
    labels = ["Aerial", "Ground", "Text"]
    compare = gate[gate["value"].isin(["linear_0.405_pb", "external_pbrs", "linear_0.405_no_pb"])].set_index("value")
    x = np.arange(len(labels))
    width = 0.24
    series = [
        ("linear_0.405_pb", "本文方法", BLUE),
        ("external_pbrs", "外部+PBRS", ORANGE),
        ("linear_0.405_no_pb", "无 PBRS", GRAY),
    ]
    for i, (key, label, color) in enumerate(series):
        vals = [compare.loc[key, col] * 100 for col in modalities]
        ax_modal.bar(x + (i - 1) * width, vals, width=width, color=color, label=label, alpha=0.95)
    ax_modal.set_xticks(x)
    ax_modal.set_xticklabels(labels)
    ax_modal.set_ylabel("SR（%）")
    ax_modal.set_ylim(54, 63.5)
    ax_modal.set_title("三种输入模态上表现更均衡")
    clean_axes(ax_modal)
    ax_modal.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))

    ax_message.axis("off")
    boxes = [
        ("训练阶段", "混合奖励进入 PPO 更新，改变策略网络权重。", BLUE),
        ("不是测试规则", "正式测试时使用 greedy policy；奖励函数不再被调用。", RED),
        ("结果解释", f"本文方法 {ours['mmgag_mean_sr']*100:.2f}% ，比外部+PBRS 高 {(ours['mmgag_mean_sr']-external['mmgag_mean_sr'])*100:.2f} 个百分点，比无 PBRS 高 {(ours['mmgag_mean_sr']-no_pb['mmgag_mean_sr'])*100:.2f} 个百分点。", GREEN),
    ]
    for i, (head, body, color) in enumerate(boxes):
        y0 = 0.69 - i * 0.29
        patch = FancyBboxPatch(
            (0.02, y0),
            0.95,
            0.22,
            boxstyle="round,pad=0.018,rounding_size=0.02",
            transform=ax_message.transAxes,
            facecolor=CARD,
            edgecolor=color,
            linewidth=1.6,
        )
        ax_message.add_patch(patch)
        ax_message.text(0.07, y0 + 0.145, head, transform=ax_message.transAxes, color=color, fontsize=14, fontweight="bold", ha="left", va="center")
        ax_message.text(0.07, y0 + 0.070, body, transform=ax_message.transAxes, color=INK, fontsize=11, ha="left", va="center", wrap=True)

    for label, ax in zip(["A", "B", "C", "D", "E"], [ax_curve, ax_gate, ax_rank, ax_modal, ax_message]):
        panel_label(ax, label)

    top4[["rank", "value", "label", "mmgag_mean_sr", "mmgag_aerial", "mmgag_ground", "mmgag_text"]].to_csv(
        SUMMARY_TABLES / "main_reward_gate_top4.csv", index=False
    )
    save_figure(fig, "figure1_reward_training_to_best_policy")


def draw_training_signal(metrics: pd.DataFrame, components: pd.DataFrame) -> None:
    metrics = add_c6_c8_fields(metrics)
    proposed = metrics[metrics["method"].eq("proposed_linear_gate_pbrs")].copy()
    proposed_components = components[components["method"].eq("proposed_linear_gate_pbrs")].copy()

    fig = plt.figure(figsize=(15.6, 8.8))
    add_header(
        fig,
        "训练信号可视化：混合奖励怎样指导中长距离行动",
        "重点看训练阶段日志：门控内在奖励保持探索，PBRS 提供朝向目标的方向塑形，外部奖励负责目标到达反馈。",
    )
    gs = GridSpec(2, 3, figure=fig, left=0.055, right=0.965, top=0.84, bottom=0.09, width_ratios=[1.05, 1.05, 1.2], hspace=0.36, wspace=0.30)
    ax_success = fig.add_subplot(gs[0, 0])
    ax_final = fig.add_subplot(gs[1, 0])
    ax_gate = fig.add_subplot(gs[0, 1])
    ax_share = fig.add_subplot(gs[1, 1])
    ax_distance = fig.add_subplot(gs[:, 2])

    focus_methods = ["external_only", "mixed_no_gate_no_pbrs", "proposed_linear_gate_pbrs", "intrinsic_only"]
    plot_mean_band(ax_success, metrics, "c6_c8_success_rate", focus_methods, scale=100)
    ax_success.set_title("中长距离训练成功率")
    ax_success.set_xlabel("训练进度（%）")
    ax_success.set_ylabel("C=6-8 成功率（%）")
    ax_success.set_ylim(0, 58)
    clean_axes(ax_success)
    ax_success.legend(frameon=False, ncol=2, loc="upper left")

    plot_mean_band(ax_final, metrics, "c6_c8_final_dist", focus_methods, scale=1.0)
    ax_final.set_title("中长距离最终距离")
    ax_final.set_xlabel("训练进度（%）")
    ax_final.set_ylabel("最终到目标距离（越低越好）")
    ax_final.set_ylim(0.5, 6.6)
    clean_axes(ax_final)
    add_note(ax_final, "仅内在奖励长期停留在较大最终距离，说明单独探索信号无法稳定完成目标定位。", xy=(0.03, 0.04))

    for dist, color in [(4, GREEN), (6, TEAL), (8, BLUE)]:
        col = f"C{dist}_mean_gate_weight"
        if col in proposed_components:
            tmp = interpolate_by_run(proposed_components, col, smooth=9)
            band = mean_band(tmp, col)
            sub = band[band["method"].eq("proposed_linear_gate_pbrs")].sort_values("run_progress")
            ax_gate.plot(sub["run_progress"] * 100, sub["mean"], color=color, lw=2.4, label=f"C={dist}")
    ax_gate.set_title("距离门控：远距离保留更多探索")
    ax_gate.set_xlabel("训练进度（%）")
    ax_gate.set_ylabel("平均门控权重")
    ax_gate.set_ylim(0.50, 1.04)
    clean_axes(ax_gate)
    ax_gate.legend(frameon=False, ncol=3, loc="lower right")

    share_cols = [
        ("abs_reward_ex_share", "外部目标反馈", ORANGE),
        ("abs_reward_in_gated_share", "门控内在探索", GREEN),
        ("abs_pbrs_bonus_share", "PBRS 方向塑形", BLUE),
    ]
    for col, label, color in share_cols:
        tmp = interpolate_by_run(proposed_components, col, smooth=13)
        band = mean_band(tmp, col)
        sub = band[band["method"].eq("proposed_linear_gate_pbrs")].sort_values("run_progress")
        ax_share.plot(sub["run_progress"] * 100, sub["mean"] * 100, color=color, lw=2.5, label=label)
    ax_share.set_title("本文方法的奖励组成比例")
    ax_share.set_xlabel("训练进度（%）")
    ax_share.set_ylabel("绝对奖励占比（%）")
    ax_share.set_ylim(0, 74)
    clean_axes(ax_share)
    ax_share.legend(frameon=False, loc="upper right")

    distances = np.arange(1, 9)
    x = np.arange(len(distances))
    width = 0.24
    gate_vals = gate_weight_curve(distances)
    pbrs_vals = 0.1 * (0.1 + 1 / 8)
    ax_distance.bar(x - width, gate_vals, width=width, color=BLUE, label="门控权重 G(d)")
    ax_distance.bar(x, np.repeat(0.405, len(x)), width=width, color=GRAY_LIGHT, edgecolor=GRAY, label="最低探索保留")
    ax_distance.plot(x + width, np.repeat(pbrs_vals, len(x)), color=ORANGE, lw=3.0, marker="o", label="向目标前进一步的 PBRS")
    ax_distance.set_xticks(x)
    ax_distance.set_xticklabels([str(d) for d in distances])
    ax_distance.set_title("把奖励机制翻译成直观动作指导")
    ax_distance.set_xlabel("当前位置到目标距离 d")
    ax_distance.set_ylabel("训练信号强度")
    ax_distance.set_ylim(0, 1.10)
    clean_axes(ax_distance)
    ax_distance.legend(frameon=False, loc="upper left")
    ax_distance.text(
        0.04,
        0.08,
        "读图方式：距离越远，门控越接近 1，策略被鼓励继续探索；\n每次向目标靠近，PBRS 给出小而连续的正反馈，帮助学习中长距离路径。",
        transform=ax_distance.transAxes,
        fontsize=11,
        color=MUTED,
        linespacing=1.45,
        bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.16", facecolor="#F8FAFC", edgecolor="#D9E1EC"),
    )

    for label, ax in zip(["A", "B", "C", "D", "E"], [ax_success, ax_final, ax_gate, ax_share, ax_distance]):
        panel_label(ax, label)

    save_figure(fig, "figure2_training_signal_decomposition")


def draw_route_evidence() -> None:
    route = pd.read_csv(ROUTE_TABLE)
    sub = route[route["distance"].isin([6, 8])].copy()
    method_order = ["GeoExplorer-pristine", "GOMAA-Geo", "GeoExplorer-anchor0624"]
    labels = {
        "GeoExplorer-pristine": "GeoExplorer",
        "GOMAA-Geo": "GOMAA-Geo",
        "GeoExplorer-anchor0624": "本文方法",
    }
    colors = {
        "GeoExplorer-pristine": GRAY,
        "GOMAA-Geo": ORANGE,
        "GeoExplorer-anchor0624": BLUE,
    }

    fig = plt.figure(figsize=(15.6, 8.8))
    add_header(
        fig,
        "路线可视化：训练后的策略更适合中长距离搜索",
        "路线图是 checkpoint 的行为结果，用于解释为什么训练阶段的混合奖励能改善中长距离行动。",
    )
    gs = GridSpec(2, 3, figure=fig, left=0.055, right=0.965, top=0.84, bottom=0.08, width_ratios=[0.9, 1.05, 1.55], hspace=0.35, wspace=0.28)
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_prog = fig.add_subplot(gs[1, 0])
    ax_case = fig.add_subplot(gs[:, 1:])

    x = np.arange(2)
    width = 0.23
    for i, method in enumerate(method_order):
        vals = sub[sub["method"].eq(method)].set_index("distance").loc[[6, 8]]
        ax_bar.bar(x + (i - 1) * width, vals["success_rate"] * 100, width=width, color=colors[method], label=labels[method])
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(["C=6", "C=8"])
    ax_bar.set_ylabel("成功率（%）")
    ax_bar.set_ylim(0, 100)
    ax_bar.set_title("中长距离路线成功率")
    clean_axes(ax_bar)
    ax_bar.legend(frameon=False, loc="upper left")

    for i, method in enumerate(method_order):
        vals = sub[sub["method"].eq(method)].set_index("distance").loc[[6, 8]]
        ax_prog.plot(x, vals["progress_ratio"] * 100, color=colors[method], lw=2.8, marker="o", ms=7, label=labels[method])
    ax_prog.set_xticks(x)
    ax_prog.set_xticklabels(["C=6", "C=8"])
    ax_prog.set_ylabel("接近目标进度（%）")
    ax_prog.set_ylim(70, 100)
    ax_prog.set_title("路径是否持续接近目标")
    clean_axes(ax_prog)
    add_note(ax_prog, "C=8 上本文方法成功率 85.0%，GOMAA-Geo 为 68.3%。", xy=(0.03, 0.05))

    image_path = ROUTE_THREE_METHOD if ROUTE_THREE_METHOD.exists() else ROUTE_C4_C6_C8
    img = Image.open(image_path).convert("RGB")
    ax_case.imshow(img)
    ax_case.set_xticks([])
    ax_case.set_yticks([])
    ax_case.set_title("同一任务下的路线对比：本文方法更快贴近目标", pad=10)
    for spine in ax_case.spines.values():
        spine.set_visible(True)
        spine.set_color("#CCD6E2")
        spine.set_linewidth(1.2)

    for label, ax in zip(["A", "B", "C"], [ax_bar, ax_prog, ax_case]):
        panel_label(ax, label)

    sub.to_csv(SUMMARY_TABLES / "route_behavior_c6_c8.csv", index=False)
    save_figure(fig, "figure3_training_result_route_evidence", dpi=260)


def draw_gp_ablation_card(metrics: pd.DataFrame) -> None:
    metrics = add_c6_c8_fields(metrics)
    ab = pd.read_csv(GENERALIZATION_TABLE)
    gp_rows = ab[ab["branch"].isin(["g0_p0_e1_v1", "g1_p0_e1_v1", "g0_p1_e1_v1", "g1_p1_e1_v1"])].copy()
    order = ["g0_p0_e1_v1", "g1_p0_e1_v1", "g0_p1_e1_v1", "g1_p1_e1_v1"]
    branch_to_method = {
        "g0_p0_e1_v1": "mixed_no_gate_no_pbrs",
        "g1_p0_e1_v1": "mixed_gate_only",
        "g0_p1_e1_v1": "mixed_pbrs_only",
        "g1_p1_e1_v1": "proposed_linear_gate_pbrs",
    }
    labels = {
        "g0_p0_e1_v1": "G 关 / P 关",
        "g1_p0_e1_v1": "只开 G",
        "g0_p1_e1_v1": "只开 P",
        "g1_p1_e1_v1": "G+P 全开",
    }
    colors = {
        "g0_p0_e1_v1": GRAY,
        "g1_p0_e1_v1": TEAL,
        "g0_p1_e1_v1": RED,
        "g1_p1_e1_v1": BLUE,
    }

    fig = plt.figure(figsize=(15.6, 8.8))
    add_header(
        fig,
        "消融可视化：门控 G 与 PBRS P 需要组合使用",
        "固定 E=1、V=1，只改变 G/P 训练配置；曲线看训练过程，柱图和路线看训练后的策略结果。",
    )
    gs = GridSpec(2, 3, figure=fig, left=0.055, right=0.965, top=0.84, bottom=0.08, width_ratios=[1.0, 1.0, 1.46], hspace=0.35, wspace=0.28)
    ax_curve = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[1, 0])
    ax_mmgag = fig.add_subplot(gs[:, 1])
    ax_img = fig.add_subplot(gs[:, 2])

    for branch in order:
        method = branch_to_method[branch]
        interp = interpolate_by_run(metrics[metrics["method"].eq(method)], "c6_c8_success_rate")
        band = mean_band(interp, "c6_c8_success_rate")
        sub = band[band["method"].eq(method)].sort_values("run_progress")
        if sub.empty:
            continue
        ax_curve.plot(sub["run_progress"] * 100, sub["mean"] * 100, color=colors[branch], lw=3.0 if branch == "g1_p1_e1_v1" else 2.0, label=labels[branch])
    ax_curve.set_title("训练过程：G+P 组合更稳定")
    ax_curve.set_xlabel("训练进度（%）")
    ax_curve.set_ylabel("C=6-8 成功率（%）")
    ax_curve.set_ylim(0, 58)
    clean_axes(ax_curve)
    ax_curve.legend(frameon=False, ncol=2, loc="upper left")

    gp_rows = gp_rows.set_index("branch").loc[order].reset_index()
    x = np.arange(len(gp_rows))
    ax_bar.bar(x, gp_rows["primary_generalization_mean"] * 100, color=[colors[b] for b in gp_rows["branch"]])
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([labels[b].replace(" / ", "\n") for b in gp_rows["branch"]])
    ax_bar.set_ylim(54, 64)
    ax_bar.set_ylabel("Primary mean SR（%）")
    ax_bar.set_title("训练后泛化结果")
    clean_axes(ax_bar)
    for i, row in gp_rows.iterrows():
        ax_bar.text(i, row["primary_generalization_mean"] * 100 + 0.35, f"{row['primary_generalization_mean']*100:.1f}%", ha="center", fontsize=10.5, fontweight="bold")

    mmgag_means = gp_rows[["mmgag_aerial", "mmgag_ground", "mmgag_text"]].mean(axis=1) * 100
    ax_mmgag.plot(x, mmgag_means, color=BLUE, lw=3.0, marker="o", ms=8)
    ax_mmgag.set_xticks(x)
    ax_mmgag.set_xticklabels([labels[b].replace(" / ", "\n") for b in gp_rows["branch"]])
    ax_mmgag.set_ylim(57, 64.5)
    ax_mmgag.set_ylabel("MM-GAG 平均 SR（%）")
    ax_mmgag.set_title("跨模态定位：G+P 全开最高")
    clean_axes(ax_mmgag)
    for i, val in enumerate(mmgag_means):
        ax_mmgag.text(i, val + 0.25, f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold", color=BLUE if i == 3 else MUTED)

    image_path = ROUTE_GP_2X2 if ROUTE_GP_2X2.exists() else ROUTE_THREE_METHOD
    img = Image.open(image_path).convert("RGB")
    ax_img.imshow(img)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_title("同一样例路线：G+P 组合更容易到达目标", pad=10)
    for spine in ax_img.spines.values():
        spine.set_visible(True)
        spine.set_color("#CCD6E2")
        spine.set_linewidth(1.2)

    for label, ax in zip(["A", "B", "C", "D"], [ax_curve, ax_bar, ax_mmgag, ax_img]):
        panel_label(ax, label)

    gp_rows[["branch", "primary_generalization_mean", "mmgag_aerial", "mmgag_ground", "mmgag_text"]].to_csv(
        SUMMARY_TABLES / "gp_ablation_summary.csv", index=False
    )
    save_figure(fig, "figure4_gp_ablation_route_card", dpi=260)


def draw_talk_card() -> None:
    gate = label_gate_values(pd.read_csv(GATE_TABLE))
    ours = gate[gate["value"].eq("linear_0.405_pb")].iloc[0]
    external = gate[gate["value"].eq("external_pbrs")].iloc[0]
    no_pb = gate[gate["value"].eq("linear_0.405_no_pb")].iloc[0]

    fig = plt.figure(figsize=(15.6, 8.8))
    fig.text(0.055, 0.92, "答辩时这一页怎么讲", fontsize=28, fontweight="bold", ha="left", va="top", color=INK)
    fig.text(
        0.055,
        0.86,
        "建议 30-45 秒讲完：先界定训练阶段，再说明机制，最后落到正式结果。",
        fontsize=14,
        ha="left",
        va="top",
        color=MUTED,
    )
    ax = fig.add_axes([0.055, 0.10, 0.89, 0.68])
    ax.axis("off")

    cards = [
        ("第一句：界定位置", "混合奖励不是测试阶段的规则，而是在训练阶段进入 PPO 更新，用来塑造策略网络。", BLUE),
        ("第二句：解释机制", "距离远时保留更多内在探索；接近目标后降低内在奖励干扰；PBRS 给每一步朝向目标的连续反馈。", GREEN),
        ("第三句：给出证据", f"在同协议 MM-GAG reward-gate/PBRS 表中，本文方法达到 {ours['mmgag_mean_sr']*100:.2f}% ，排名第一。", ORANGE),
        ("第四句：连接路线", "路线图展示的是训练后 checkpoint 的行为：中长距离任务中，策略更能持续接近目标并减少回退。", PURPLE),
    ]
    for i, (head, body, color) in enumerate(cards):
        y = 0.78 - i * 0.22
        patch = FancyBboxPatch(
            (0.02, y),
            0.78,
            0.16,
            boxstyle="round,pad=0.02,rounding_size=0.018",
            transform=ax.transAxes,
            facecolor=CARD,
            edgecolor=color,
            linewidth=1.6,
        )
        ax.add_patch(patch)
        ax.text(0.055, y + 0.105, head, transform=ax.transAxes, fontsize=15, fontweight="bold", color=color, va="center")
        ax.text(0.055, y + 0.052, body, transform=ax.transAxes, fontsize=12.2, color=INK, va="center", wrap=True)

    summary = FancyBboxPatch(
        (0.835, 0.22),
        0.14,
        0.56,
        boxstyle="round,pad=0.02,rounding_size=0.018",
        transform=ax.transAxes,
        facecolor="#EEF4FF",
        edgecolor=BLUE,
        linewidth=1.6,
    )
    ax.add_patch(summary)
    ax.text(0.905, 0.70, "核心数字", transform=ax.transAxes, fontsize=16, fontweight="bold", color=BLUE, ha="center")
    ax.text(0.905, 0.59, f"{ours['mmgag_mean_sr']*100:.2f}%", transform=ax.transAxes, fontsize=28, fontweight="bold", color=BLUE, ha="center")
    ax.text(0.905, 0.49, "MM-GAG\n三模态平均 SR", transform=ax.transAxes, fontsize=11.5, color=INK, ha="center", linespacing=1.3)
    ax.text(0.905, 0.36, f"+{(ours['mmgag_mean_sr']-external['mmgag_mean_sr'])*100:.2f}", transform=ax.transAxes, fontsize=20, fontweight="bold", color=GREEN, ha="center")
    ax.text(0.905, 0.30, "百分点\n相对外部+PBRS", transform=ax.transAxes, fontsize=10.8, color=MUTED, ha="center", linespacing=1.25)
    ax.text(0.905, 0.21, f"+{(ours['mmgag_mean_sr']-no_pb['mmgag_mean_sr'])*100:.2f}", transform=ax.transAxes, fontsize=20, fontweight="bold", color=GREEN, ha="center")
    ax.text(0.905, 0.15, "百分点\n相对无 PBRS", transform=ax.transAxes, fontsize=10.8, color=MUTED, ha="center", linespacing=1.25)

    fig.text(
        0.055,
        0.045,
        "严谨表述：测试阶段只加载训练后的 checkpoint，不调用 gate_weight、pbrs_bonus 或奖励组合函数；机制影响通过已学习的策略权重体现。",
        fontsize=11.5,
        color=MUTED,
        ha="left",
    )
    save_figure(fig, "figure5_defense_talk_card")


def copy_reference_images() -> None:
    for src in [ROUTE_THREE_METHOD, ROUTE_C4_C6_C8, ROUTE_GP_2X2]:
        if src.exists():
            shutil.copy2(src, FIGURES / src.name)


def write_report(metrics: pd.DataFrame) -> None:
    gate = label_gate_values(pd.read_csv(GATE_TABLE))
    ours = gate[gate["value"].eq("linear_0.405_pb")].iloc[0]
    external = gate[gate["value"].eq("external_pbrs")].iloc[0]
    constant = gate[gate["value"].eq("constant_0.405_pb")].iloc[0]
    no_pb = gate[gate["value"].eq("linear_0.405_no_pb")].iloc[0]
    route = pd.read_csv(ROUTE_TABLE)
    route_ours_d8 = route[(route["method"].eq("GeoExplorer-anchor0624")) & (route["distance"].eq(8))].iloc[0]
    route_gomaa_d8 = route[(route["method"].eq("GOMAA-Geo")) & (route["distance"].eq(8))].iloc[0]

    run_count = metrics[metrics["method"].eq("proposed_linear_gate_pbrs")]["seed"].nunique()

    text = f"""# 结题汇报可视化说明：训练阶段混合奖励与路线行为

## 结论先行

本文方法的“最好”应使用正式同协议评估表来证明，而不是只看训练采样曲线。  
在 MM-GAG reward-gate/PBRS 补充表中，`linear_0.405_pb` 即本文“线性门控+PBRS”方法达到 **{ours['mmgag_mean_sr']*100:.2f}%** 的三模态平均 SR，排名第一。

对比关系：

- 本文方法 `linear_0.405_pb`：{ours['mmgag_mean_sr']:.4f}
- 外部奖励+PBRS `external_pbrs`：{external['mmgag_mean_sr']:.4f}
- 常数门控+PBRS `constant_0.405_pb`：{constant['mmgag_mean_sr']:.4f}
- 线性门控但无 PBRS `linear_0.405_no_pb`：{no_pb['mmgag_mean_sr']:.4f}

因此可以在答辩中说：本文混合奖励不是简单把奖励相加，而是通过“距离门控 G + PBRS 方向塑形”在训练阶段形成更好的跨模态搜索策略。

## 必须讲清楚的严谨点

混合奖励机制属于训练阶段设计。训练时，奖励信号进入 PPO 更新并改变 actor/critic 权重；测试时只加载训练完成后的 policy checkpoint，使用 greedy action 输出路线，不再调用 `gate_weight()`、`pbrs_bonus()`、外部/内在奖励组合函数或 validation-distance checkpoint-selection 逻辑。

所以：

- 训练曲线用于解释“奖励如何指导学习过程”。
- reward-gate/PBRS 正式表用于证明“训练出的策略谁最好”。
- 路线图用于解释“训练后策略的行为差异”。

## 训练阶段解释

本次图包读取了 {run_count} 个 seed 的本文方法训练日志，并对不同奖励配置进行同一训练进度下的均值和标准差展示。训练日志说明：

- 仅内在奖励可以提供探索信号，但缺少明确目标约束，训练后的中长距离完成能力弱。
- 外部奖励能提供目标到达反馈，但在中长距离搜索中早期引导不足。
- 直接相加内在奖励和外部奖励并不等于最优，内在奖励在近目标阶段仍可能干扰收敛。
- 线性门控让远距离阶段保留探索，近距离阶段降低内在奖励干扰；PBRS 负责给“向目标靠近”的连续正反馈。

## 路线行为解释

路线图不是训练阶段奖励本身，而是训练后 checkpoint 的行为结果。  
在可视化路线统计中，本文方法在 C=8 中长距离任务上的成功率为 **{route_ours_d8['success_rate']*100:.1f}%**，GOMAA-Geo 为 **{route_gomaa_d8['success_rate']*100:.1f}%**。这可以作为答辩时的直观解释：混合奖励在训练阶段帮助策略学到更稳定的中长距离接近目标行为。

## 输出文件

- `figure1_reward_training_to_best_policy.png/svg`：主图，训练过程解释 + 正式 MM-GAG 排名第一。
- `figure2_training_signal_decomposition.png/svg`：训练阶段奖励信号拆解图。
- `figure3_training_result_route_evidence.png/svg`：训练后路线行为证据图。
- `figure4_gp_ablation_route_card.png/svg`：G/P 消融曲线、统计与路线图。
- `figure5_defense_talk_card.png/svg`：答辩讲解卡片。

## 推荐汇报顺序

1. 先放 `figure1`，直接建立“本文方法最好”的正式证据。
2. 再放 `figure2`，解释为什么这是训练阶段机制。
3. 接着放 `figure4`，回答为什么 G 和 PBRS 要组合。
4. 最后放 `figure3`，用路线图让不懂算法的人直观看到策略差异。
"""
    (REPORTS / "defense_reward_final_visual_analysis_zh.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    setup_style()
    metrics, components, _routes = read_training_logs()
    metrics.to_csv(SUMMARY_TABLES / "raw_training_metrics_indexed.csv", index=False)
    draw_main_evidence(metrics)
    draw_training_signal(metrics, components)
    draw_route_evidence()
    draw_gp_ablation_card(metrics)
    draw_talk_card()
    copy_reference_images()
    write_report(metrics)


if __name__ == "__main__":
    main()
