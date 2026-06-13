#!/usr/bin/env python
"""Build defense-oriented training reward and route visualizations.

The figures in this package separate two claims:
1. Training-stage reward designs change the PPO learning curves.
2. The learned checkpoints then produce different route behavior at evaluation.

The evaluation figures must not be described as reward being injected at test
time. Formal evaluation uses the learned policy only.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures" / "defense_reward_route"
REPORTS = RESULTS / "reports"
TRAIN_LOG_ROOT = RESULTS / "training_logs" / "reward_route_defense"

GEO_ANALYSIS = Path(r"F:\bishe\GeoExplorer\analysis")
LINEAR_LONG_TABLE = (
    GEO_ANALYSIS
    / "pipeline_20260520_reward_gate_linear_main_primary_eval"
    / "linear_gate_main_primary_long_table.csv"
)

INK = "#111827"
MUTED = "#5F6B7A"
GRID = "#D7DEE8"
PAPER = "#F8FAFC"
CARD = "#FFFFFF"
BLUE = "#1764AB"
ORANGE = "#D9822B"
GREEN = "#1B9E77"
PURPLE = "#7C5CC4"
RED = "#B33A3A"
GRAY = "#7A8699"


REWARD_RUNS = {
    "reward_external_only": {
        "label": "纯外部奖励",
        "short": "外部",
        "color": ORANGE,
        "group": "reward_control",
        "table_run": "reward_external_only_seed321_t480k",
    },
    "reward_intrinsic_only": {
        "label": "纯内在奖励",
        "short": "内在",
        "color": PURPLE,
        "group": "reward_control",
        "table_run": "reward_intrinsic_only_seed321_t480k",
    },
    "reward_no_decay_mixed": {
        "label": "外部+内在（无衰减）",
        "short": "无衰减混合",
        "color": GREEN,
        "group": "reward_control",
        "table_run": "reward_intrinsic_no_decay_seed321_t480k",
    },
    "gp_full_linear_pbrs": {
        "label": "本文混合奖励（线性门控+PBRS）",
        "short": "本文方法",
        "color": BLUE,
        "group": "anchor",
        "table_run": "linear_0.405_pb",
    },
}

GP_RUNS = {
    "gp_off_off": {
        "label": "G/P 全关",
        "short": "全关",
        "color": GRAY,
        "branch": "g0_p0_e1_v1",
    },
    "gp_g_only": {
        "label": "只开 G",
        "short": "G",
        "color": ORANGE,
        "branch": "g1_p0_e1_v1",
    },
    "gp_p_only": {
        "label": "只开 P",
        "short": "P",
        "color": GREEN,
        "branch": "g0_p1_e1_v1",
    },
    "gp_full_linear_pbrs": {
        "label": "G+P 全开",
        "short": "G+P",
        "color": BLUE,
        "branch": "g1_p1_e1_v1",
    },
}


def ensure_dirs() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (TABLES / "defense_reward_route").mkdir(parents=True, exist_ok=True)


def setup_style() -> None:
    for font in [
        "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/timesbd.ttf",
        "C:/Windows/Fonts/simsun.ttc",
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
            "axes.edgecolor": "#BFC7D2",
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


def save_figure(fig: plt.Figure, stem: str, dpi: int = 300) -> None:
    png = FIGURES / f"{stem}.png"
    svg = FIGURES / f"{stem}.svg"
    fig.savefig(png, dpi=dpi, bbox_inches="tight", pad_inches=0.16)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


def clean_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#C9D2DE")
    ax.spines["bottom"].set_color("#C9D2DE")
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.65)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.10,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
    )


def add_header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.035, 0.965, title, fontsize=24, fontweight="bold", ha="left", va="top")
    fig.text(0.035, 0.925, subtitle, fontsize=11.5, color=MUTED, ha="left", va="top")
    fig.lines.append(
        plt.Line2D([0.035, 0.965], [0.895, 0.895], transform=fig.transFigure, color="#CCD6E2", lw=1.2)
    )


def moving_average(values: pd.Series, window: int = 17) -> pd.Series:
    return values.astype(float).rolling(window=window, min_periods=1, center=True).mean()


def normalize_curve(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    smooth = moving_average(values, 17)
    lo = float(smooth.quantile(0.03))
    hi = float(smooth.quantile(0.97))
    if math.isclose(lo, hi):
        return smooth * 0.0
    return ((smooth - lo) / (hi - lo)).clip(0, 1)


def read_metrics(run_key: str) -> pd.DataFrame:
    path = TRAIN_LOG_ROOT / run_key / "training_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["time_step_k"] = df["time_step"].astype(float) / 1000.0
    df["reward_norm"] = normalize_curve(df["rolling_avg_reward"])
    df["success_smooth"] = moving_average(df["rolling_success_ratio"], 17)
    df["val_success_smooth"] = moving_average(df["val_success"], 13)
    df["best_val_success_smooth"] = moving_average(df["best_val_success"], 13)
    return df


def load_all_metrics(run_map: dict[str, dict]) -> dict[str, pd.DataFrame]:
    return {run_key: read_metrics(run_key) for run_key in run_map}


def reward_control_eval_summary() -> pd.DataFrame:
    long_df = pd.read_csv(TABLES / "ablation" / "reward_control_long_table.csv")
    gate_df = pd.read_csv(TABLES / "ablation" / "reward_gate_type_mmgag_only_table_with_linear.csv")
    rows: list[dict] = []

    for run_key, meta in REWARD_RUNS.items():
        if run_key == "gp_full_linear_pbrs":
            gate_row = gate_df[gate_df["value"] == "linear_0.405_pb"].iloc[0]
            rows.append(
                {
                    "run_key": run_key,
                    "label": meta["label"],
                    "short": meta["short"],
                    "mmgag_mean_sr": float(gate_row["mmgag_mean_sr"]),
                    "d6_d8_mmgag_mean": np.nan,
                    "source": "reward_gate_type_mmgag_only_table_with_linear.csv",
                }
            )
            continue

        sub = long_df[long_df["run"] == meta["table_run"]]
        mm = sub[sub["benchmark"].isin(["mmgag_aerial", "mmgag_ground", "mmgag_text"])]
        rows.append(
            {
                "run_key": run_key,
                "label": meta["label"],
                "short": meta["short"],
                "mmgag_mean_sr": float(mm["sr"].mean()),
                "d6_d8_mmgag_mean": float(mm[["d6", "d7", "d8"]].mean().mean()),
                "source": "reward_control_long_table.csv",
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "defense_reward_route" / "reward_control_eval_summary.csv", index=False)
    return out


def gp_eval_summary() -> pd.DataFrame:
    ab = pd.read_csv(TABLES / "ablation" / "anchor0624_generalization_table.csv")
    rows = []
    for run_key, meta in GP_RUNS.items():
        row = ab[ab["branch"] == meta["branch"]].iloc[0]
        rows.append(
            {
                "run_key": run_key,
                "branch": meta["branch"],
                "label": meta["label"],
                "short": meta["short"],
                "primary_generalization_mean": float(row["primary_generalization_mean"]),
                "mmgag_mean": float(row[["mmgag_aerial", "mmgag_ground", "mmgag_text"]].mean()),
                "masa_d6_d8_mean": float(row[["masa_d6", "masa_d7", "masa_d8"]].mean()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "defense_reward_route" / "gp_training_eval_summary.csv", index=False)
    return out


def route_behavior_summary() -> pd.DataFrame:
    df = pd.read_csv(TABLES / "trajectory_analysis" / "trajectory_behavior_by_distance.csv")
    sub = df[df["distance"].isin([6, 8])].copy()
    sub.to_csv(TABLES / "defense_reward_route" / "route_behavior_medium_long.csv", index=False)
    return sub


def draw_training_curves() -> None:
    metrics = load_all_metrics(REWARD_RUNS)
    eval_df = reward_control_eval_summary()

    fig = plt.figure(figsize=(14.4, 8.1))
    add_header(
        fig,
        "训练阶段奖励曲线：混合奖励不是测试规则，而是训练信号",
        "同为 480k、seed=321 的训练日志；曲线展示奖励设计如何影响 PPO 学习过程，右下角为训练后同协议评测结果。",
    )
    gs = fig.add_gridspec(2, 2, left=0.06, right=0.97, top=0.84, bottom=0.10, hspace=0.36, wspace=0.26)
    ax_success = fig.add_subplot(gs[0, 0])
    ax_val = fig.add_subplot(gs[0, 1])
    ax_reward = fig.add_subplot(gs[1, 0])
    ax_bar = fig.add_subplot(gs[1, 1])

    for run_key, meta in REWARD_RUNS.items():
        df = metrics[run_key]
        color = meta["color"]
        label = meta["short"]
        ax_success.plot(df["time_step_k"], df["success_smooth"] * 100, color=color, lw=2.4, label=label)
        ax_val.plot(df["time_step_k"], df["val_success_smooth"], color=color, lw=2.4, label=label)
        ax_reward.plot(df["time_step_k"], df["reward_norm"], color=color, lw=2.4, label=label)

    ax_success.set_title("训练成功趋势")
    ax_success.set_xlabel("训练步数（千步）")
    ax_success.set_ylabel("近 20 轮成功趋势（%）")
    ax_success.set_ylim(0, 38)
    clean_axes(ax_success)
    ax_success.legend(frameon=False, ncol=2, loc="lower right")

    ax_val.set_title("验证集成功数")
    ax_val.set_xlabel("训练步数（千步）")
    ax_val.set_ylabel("验证成功数（0-20）")
    ax_val.set_ylim(-0.5, 21)
    clean_axes(ax_val)
    ax_val.axhline(20, color="#8EC3A7", lw=1.2, ls="--")
    ax_val.text(8, 19.2, "验证上限", color="#357A4F", fontsize=9)

    ax_reward.set_title("训练回报趋势（按各自曲线归一化）")
    ax_reward.set_xlabel("训练步数（千步）")
    ax_reward.set_ylabel("归一化趋势")
    ax_reward.set_ylim(-0.05, 1.05)
    clean_axes(ax_reward)
    ax_reward.text(
        0.02,
        0.05,
        "说明：不同奖励的数值尺度不同，\n这里只比较上升/稳定趋势，不比较绝对大小。",
        transform=ax_reward.transAxes,
        fontsize=9,
        color=MUTED,
        va="bottom",
    )

    eval_order = list(REWARD_RUNS.keys())
    colors = [REWARD_RUNS[k]["color"] for k in eval_order]
    bars = eval_df.set_index("run_key").loc[eval_order]
    y = np.arange(len(bars))
    ax_bar.barh(y, bars["mmgag_mean_sr"] * 100, color=colors, alpha=0.92)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(bars["short"])
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("MM-GAG 平均 SR（%）")
    ax_bar.set_title("训练后同协议评测")
    ax_bar.set_xlim(0, 68)
    clean_axes(ax_bar, "x")
    for idx, value in enumerate(bars["mmgag_mean_sr"] * 100):
        ax_bar.text(value + 1.0, idx, f"{value:.1f}%", va="center", fontsize=10, fontweight="bold")
    ax_bar.text(
        0.02,
        -0.18,
        "评测阶段只加载 checkpoint，不再调用训练奖励函数。",
        transform=ax_bar.transAxes,
        fontsize=9,
        color=MUTED,
    )

    for label, ax in zip(["A", "B", "C", "D"], [ax_success, ax_val, ax_reward, ax_bar]):
        panel_label(ax, label)

    save_figure(fig, "defense_training_reward_curves")


def draw_gp_ablation_curves() -> None:
    metrics = load_all_metrics(GP_RUNS)
    eval_df = gp_eval_summary().set_index("run_key").loc[list(GP_RUNS.keys())]

    fig = plt.figure(figsize=(14.4, 8.1))
    add_header(
        fig,
        "训练消融曲线：G 门控与 PBRS 的作用要合在一起读",
        "固定 E=1、V=1 的 G/P 切片；左侧为训练过程，右侧为训练后泛化与中远距离表现。",
    )
    gs = fig.add_gridspec(2, 2, left=0.06, right=0.97, top=0.84, bottom=0.10, hspace=0.36, wspace=0.26)
    ax_val = fig.add_subplot(gs[0, 0])
    ax_reward = fig.add_subplot(gs[1, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_dist = fig.add_subplot(gs[1, 1])

    for run_key, meta in GP_RUNS.items():
        df = metrics[run_key]
        color = meta["color"]
        ax_val.plot(df["time_step_k"], df["val_success_smooth"], color=color, lw=2.3, label=meta["short"])
        ax_reward.plot(df["time_step_k"], df["reward_norm"], color=color, lw=2.3, label=meta["short"])

    ax_val.set_title("训练中的验证成功数")
    ax_val.set_xlabel("训练步数（千步）")
    ax_val.set_ylabel("验证成功数（0-20）")
    ax_val.set_ylim(-0.5, 21)
    clean_axes(ax_val)
    ax_val.legend(frameon=False, ncol=4, loc="lower right")

    ax_reward.set_title("训练回报趋势（归一化）")
    ax_reward.set_xlabel("训练步数（千步）")
    ax_reward.set_ylabel("归一化趋势")
    ax_reward.set_ylim(-0.05, 1.05)
    clean_axes(ax_reward)

    colors = [GP_RUNS[k]["color"] for k in eval_df.index]
    x = np.arange(len(eval_df))
    ax_bar.bar(x, eval_df["primary_generalization_mean"] * 100, color=colors, alpha=0.92)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(eval_df["short"])
    ax_bar.set_ylabel("Primary mean SR（%）")
    ax_bar.set_title("训练后泛化排名")
    ax_bar.set_ylim(0, 68)
    clean_axes(ax_bar)
    for idx, value in enumerate(eval_df["primary_generalization_mean"] * 100):
        ax_bar.text(idx, value + 1.2, f"{value:.1f}%", ha="center", fontsize=10, fontweight="bold")

    ax_dist.plot(
        x,
        eval_df["masa_d6_d8_mean"] * 100,
        color=BLUE,
        lw=2.4,
        marker="o",
        ms=7,
        label="MASA C=6-8",
    )
    ax_dist.plot(
        x,
        eval_df["mmgag_mean"] * 100,
        color=ORANGE,
        lw=2.4,
        marker="s",
        ms=6,
        label="MM-GAG 平均",
    )
    ax_dist.set_xticks(x)
    ax_dist.set_xticklabels(eval_df["short"])
    ax_dist.set_ylabel("SR（%）")
    ax_dist.set_title("中远距离与跨模态表现")
    ax_dist.set_ylim(20, 92)
    clean_axes(ax_dist)
    ax_dist.legend(frameon=False, loc="lower right")

    for label, ax in zip(["A", "B", "C", "D"], [ax_val, ax_reward, ax_bar, ax_dist]):
        panel_label(ax, label)

    save_figure(fig, "defense_gp_training_ablation_curves")


def draw_mechanism_flow() -> None:
    fig = plt.figure(figsize=(14.4, 8.1))
    add_header(
        fig,
        "一句话讲清楚：奖励只在训练阶段指导策略形成",
        "训练时比较不同奖励设计；测试时不再注入奖励，只观察学到的路线行为。",
    )
    ax = fig.add_axes([0.04, 0.10, 0.92, 0.74])
    ax.axis("off")

    stages = [
        ("训练输入", "当前位置、目标特征、历史动作\n进入 PPO 策略网络", "#EEF4FF", BLUE),
        ("奖励设计", "$r_t = r_{ext} + G(d_t)r_{int} + r_{PBRS}$\n远距离保留探索，接近目标后降低内在噪声", "#F0FDF4", GREEN),
        ("参数更新", "奖励序列进入 PPO 更新\n改变 actor/critic 权重", "#FFF7ED", ORANGE),
        ("训练后策略", "保存 checkpoint\n奖励函数不再参与测试", "#F5F3FF", PURPLE),
        ("路线表现", "greedy 策略输出动作\n观察是否稳定接近目标", "#F8FAFC", GRAY),
    ]

    x0s = np.linspace(0.03, 0.81, len(stages))
    y0 = 0.58
    w = 0.155
    h = 0.25
    for i, (head, body, face, color) in enumerate(stages):
        box = FancyBboxPatch(
            (x0s[i], y0),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.018",
            linewidth=1.8,
            edgecolor=color,
            facecolor=face,
            transform=ax.transAxes,
        )
        ax.add_patch(box)
        ax.text(x0s[i] + w / 2, y0 + h - 0.055, head, ha="center", va="center", fontsize=15, fontweight="bold", color=color)
        ax.text(x0s[i] + w / 2, y0 + 0.105, body, ha="center", va="center", fontsize=10.5, linespacing=1.45)
        if i < len(stages) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x0s[i] + w + 0.006, y0 + h / 2),
                    (x0s[i + 1] - 0.008, y0 + h / 2),
                    transform=ax.transAxes,
                    arrowstyle="-|>",
                    mutation_scale=18,
                    linewidth=1.8,
                    color="#7B8794",
                )
            )

    lower = [
        ("纯外部奖励", "早期缺少探索信号，容易依赖局部进展。", ORANGE),
        ("纯内在奖励", "只追求新奇或预测误差，难以稳定收敛到目标。", PURPLE),
        ("无衰减混合", "能学到一部分路线，但内在奖励在近目标阶段仍可能干扰。", GREEN),
        ("本文混合奖励", "用距离门控分阶段调节内在奖励，并用 PBRS 提供方向塑形。", BLUE),
    ]
    for i, (head, body, color) in enumerate(lower):
        x = 0.06 + i * 0.225
        box = FancyBboxPatch(
            (x, 0.18),
            0.19,
            0.19,
            boxstyle="round,pad=0.016,rounding_size=0.016",
            linewidth=1.2,
            edgecolor=color,
            facecolor="white",
            transform=ax.transAxes,
        )
        ax.add_patch(box)
        ax.text(x + 0.095, 0.31, head, ha="center", va="center", fontsize=13, fontweight="bold", color=color)
        ax.text(x + 0.095, 0.235, body, ha="center", va="center", fontsize=10.0, linespacing=1.35)

    ax.text(
        0.50,
        0.045,
        "答辩口径：本图说明训练信号如何塑造策略；路线图展示的是训练后 checkpoint 的行为结果，不表示测试阶段继续使用奖励函数。",
        ha="center",
        va="center",
        fontsize=11,
        color=MUTED,
    )
    save_figure(fig, "defense_reward_training_mechanism_flow")


def draw_route_joint_panel() -> None:
    reward_metrics = load_all_metrics(REWARD_RUNS)
    route_df = route_behavior_summary()

    fig = plt.figure(figsize=(14.4, 8.1))
    add_header(
        fig,
        "从训练曲线到路线行为：奖励塑造策略，策略决定路线",
        "左侧是训练阶段证据；右侧是训练后固定 greedy 评测下的中远距离路线表现。",
    )
    gs = fig.add_gridspec(
        2,
        3,
        left=0.05,
        right=0.97,
        top=0.84,
        bottom=0.08,
        width_ratios=[1.05, 1.05, 1.45],
        height_ratios=[1.0, 1.0],
        hspace=0.34,
        wspace=0.26,
    )
    ax_train = fig.add_subplot(gs[0, 0])
    ax_eval = fig.add_subplot(gs[1, 0])
    ax_route_stat = fig.add_subplot(gs[:, 1])
    ax_img = fig.add_subplot(gs[:, 2])

    for run_key, meta in REWARD_RUNS.items():
        df = reward_metrics[run_key]
        ax_train.plot(df["time_step_k"], df["val_success_smooth"], color=meta["color"], lw=2.2, label=meta["short"])
    ax_train.set_title("训练阶段：验证成功数")
    ax_train.set_xlabel("训练步数（千步）")
    ax_train.set_ylabel("成功数（0-20）")
    ax_train.set_ylim(-0.5, 21)
    clean_axes(ax_train)

    eval_df = reward_control_eval_summary().set_index("run_key").loc[list(REWARD_RUNS.keys())]
    y = np.arange(len(eval_df))
    ax_eval.barh(y, eval_df["mmgag_mean_sr"] * 100, color=[REWARD_RUNS[k]["color"] for k in eval_df.index])
    ax_eval.set_yticks(y)
    ax_eval.set_yticklabels(eval_df["short"])
    ax_eval.invert_yaxis()
    ax_eval.set_xlim(0, 68)
    ax_eval.set_xlabel("MM-GAG 平均 SR（%）")
    ax_eval.set_title("训练后：跨模态结果")
    clean_axes(ax_eval, "x")
    for idx, val in enumerate(eval_df["mmgag_mean_sr"] * 100):
        ax_eval.text(val + 1.0, idx, f"{val:.1f}%", va="center", fontsize=9.5, fontweight="bold")

    method_order = ["GeoExplorer-pristine", "GOMAA-Geo", "GeoExplorer-anchor0624"]
    method_label = {
        "GeoExplorer-pristine": "GeoExplorer",
        "GOMAA-Geo": "GOMAA-Geo",
        "GeoExplorer-anchor0624": "本文方法",
    }
    method_color = {
        "GeoExplorer-pristine": GRAY,
        "GOMAA-Geo": ORANGE,
        "GeoExplorer-anchor0624": BLUE,
    }
    x = np.arange(2)
    width = 0.23
    for i, method in enumerate(method_order):
        sub = route_df[route_df["method"] == method].set_index("distance").loc[[6, 8]]
        ax_route_stat.bar(
            x + (i - 1) * width,
            sub["success_rate"] * 100,
            width=width,
            color=method_color[method],
            label=method_label[method],
            alpha=0.92,
        )
    ax_route_stat.set_xticks(x)
    ax_route_stat.set_xticklabels(["C=6", "C=8"])
    ax_route_stat.set_ylabel("路线成功率（%）")
    ax_route_stat.set_title("训练后路线表现：中远距离更明显")
    ax_route_stat.set_ylim(0, 100)
    clean_axes(ax_route_stat)
    ax_route_stat.legend(frameon=False, loc="upper left")
    ax_route_stat.text(
        0.02,
        0.03,
        "路线统计来自固定 SwissViewMonuments 可视化任务；\n用于解释行为差异，不替代正式主表。",
        transform=ax_route_stat.transAxes,
        fontsize=9.2,
        color=MUTED,
        va="bottom",
    )

    image_path = RESULTS / "figures" / "chapter4_trajectories" / "figure4_y_three_method_hardcase_revised.png"
    img = Image.open(image_path).convert("RGB")
    ax_img.imshow(img)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_title("训练后典型路线：同一 C=6 任务", pad=10)
    for spine in ax_img.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color("#CCD6E2")

    for label, ax in zip(["A", "B", "C", "D"], [ax_train, ax_eval, ax_route_stat, ax_img]):
        panel_label(ax, label)

    save_figure(fig, "defense_training_to_route_joint_panel", dpi=260)


def draw_gp_route_consequence() -> None:
    fig = plt.figure(figsize=(14.4, 8.1))
    add_header(
        fig,
        "G/P 消融的可视化解释：训练曲线看过程，路线图看结果",
        "同一困难样例固定 E=1、V=1，仅改变 G/P 训练配置；路线为训练后 checkpoint 的行为。",
    )
    gs = fig.add_gridspec(2, 2, left=0.05, right=0.97, top=0.84, bottom=0.08, width_ratios=[1.0, 1.28], hspace=0.32, wspace=0.28)
    ax_curve = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[1, 0])
    ax_img = fig.add_subplot(gs[:, 1])

    metrics = load_all_metrics(GP_RUNS)
    for run_key, meta in GP_RUNS.items():
        df = metrics[run_key]
        ax_curve.plot(df["time_step_k"], df["val_success_smooth"], color=meta["color"], lw=2.3, label=meta["short"])
    ax_curve.set_title("训练阶段：验证成功数")
    ax_curve.set_xlabel("训练步数（千步）")
    ax_curve.set_ylabel("成功数（0-20）")
    ax_curve.set_ylim(-0.5, 21)
    clean_axes(ax_curve)
    ax_curve.legend(frameon=False, ncol=4, loc="lower right")

    eval_df = gp_eval_summary().set_index("run_key").loc[list(GP_RUNS.keys())]
    y = np.arange(len(eval_df))
    ax_bar.barh(y, eval_df["primary_generalization_mean"] * 100, color=[GP_RUNS[k]["color"] for k in eval_df.index])
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(eval_df["short"])
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, 68)
    ax_bar.set_xlabel("Primary mean SR（%）")
    ax_bar.set_title("训练后：完整 G+P 排名最高")
    clean_axes(ax_bar, "x")
    for idx, val in enumerate(eval_df["primary_generalization_mean"] * 100):
        ax_bar.text(val + 1.0, idx, f"{val:.1f}%", va="center", fontsize=9.5, fontweight="bold")

    img_path = RESULTS / "figures" / "showcase" / "reward_story" / "figure_b_gp_2x2_paths_img189.png"
    img = Image.open(img_path).convert("RGB")
    ax_img.imshow(img)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_title("训练后路线：只有 G+P 组合在该样例到达目标", pad=10)
    for spine in ax_img.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color("#CCD6E2")

    for label, ax in zip(["A", "B", "C"], [ax_curve, ax_bar, ax_img]):
        panel_label(ax, label)

    save_figure(fig, "defense_gp_training_route_consequence", dpi=260)


def copy_reference_assets() -> None:
    assets = {
        "route_three_method_hardcase_revised.png": RESULTS
        / "figures"
        / "chapter4_trajectories"
        / "figure4_y_three_method_hardcase_revised.png",
        "gp_2x2_route_case.png": RESULTS
        / "figures"
        / "showcase"
        / "reward_story"
        / "figure_b_gp_2x2_paths_img189.png",
        "gp_reward_matrix_case.png": RESULTS
        / "figures"
        / "showcase"
        / "reward_story"
        / "figure_c_gp_reward_matrices_img189.png",
    }
    for name, src in assets.items():
        if src.exists():
            shutil.copy2(src, FIGURES / name)


def write_report() -> None:
    reward_eval = reward_control_eval_summary()
    gp_eval = gp_eval_summary()
    route = route_behavior_summary()
    report = REPORTS / "defense_reward_route_training_analysis_zh.md"

    intrinsic = reward_eval[reward_eval["run_key"] == "reward_intrinsic_only"].iloc[0]
    external = reward_eval[reward_eval["run_key"] == "reward_external_only"].iloc[0]
    no_decay = reward_eval[reward_eval["run_key"] == "reward_no_decay_mixed"].iloc[0]
    ours = reward_eval[reward_eval["run_key"] == "gp_full_linear_pbrs"].iloc[0]
    gp_full = gp_eval[gp_eval["run_key"] == "gp_full_linear_pbrs"].iloc[0]
    gp_off = gp_eval[gp_eval["run_key"] == "gp_off_off"].iloc[0]
    route_ours_d8 = route[(route["method"] == "GeoExplorer-anchor0624") & (route["distance"] == 8)].iloc[0]
    route_gomaa_d8 = route[(route["method"] == "GOMAA-Geo") & (route["distance"] == 8)].iloc[0]

    text = f"""# 结题汇报：训练阶段混合奖励与路线行为可视化分析

## 是否需要重新训练

本次不建议重新训练。服务器上已经保留了不同奖励机制的完整训练日志，包括 `training_metrics.csv`、`config.json`、`heartbeat.json` 和训练输出日志；这些日志覆盖纯外部奖励、纯内在奖励、无衰减混合奖励、线性门控+PBRS，以及 G/P 消融切片。因此当前更应该做的是“训练日志可视化 + 训练后路线行为解释”，而不是重复 480k 训练。

## 核心口径

混合奖励机制属于训练阶段设计。训练时，奖励信号进入 PPO 更新并改变策略权重；正式测试时只加载训练后的 policy checkpoint，使用 greedy action 输出路线，不再调用 `gate_weight()`、`pbrs_bonus()` 或外部/内在奖励组合函数。因此，训练曲线用于说明奖励如何指导学习，路线图用于说明学习后的策略行为。

训练阶段可用如下简化表达讲解：

```text
r_t = r_ext + G(d_t) * r_int + r_PBRS
```

其中，`r_ext` 提供目标接近和到达信号，`r_int` 提供探索信号，`G(d_t)` 按距离调节内在奖励强度，`r_PBRS` 提供朝向目标的连续塑形反馈。该公式是讲解主项；代码中还保留了 finish bonus 等辅助项，但本次图主要解释门控内在奖励和 PBRS。

## 训练曲线结论

- 纯内在奖励训练后 MM-GAG 平均 SR 仅为 `{intrinsic['mmgag_mean_sr']:.4f}`，说明只依赖新奇性或预测误差不能稳定完成目标定位。
- 纯外部奖励达到 `{external['mmgag_mean_sr']:.4f}`，说明目标导向信号有效，但早期探索和中远距离引导仍有限。
- 外部+内在但不做距离衰减的混合奖励为 `{no_decay['mmgag_mean_sr']:.4f}`，说明简单相加不等于最优；内在奖励在近目标阶段仍可能干扰收敛。
- 线性门控+PBRS 的本文混合奖励达到 `{ours['mmgag_mean_sr']:.4f}`，在 reward-gate/PBRS 补充表中最高，适合作为结题汇报的主结论。

## G/P 消融结论

固定 `E=1,V=1` 的 G/P 切片显示，完整 `G+P` 分支的 primary generalization mean 为 `{gp_full['primary_generalization_mean']:.4f}`，高于 G/P 全关切片 `{gp_off['primary_generalization_mean']:.4f}`。这说明本文方法的优势不是“奖励数值变大”，而是训练阶段把探索、目标接近和方向塑形分配到不同搜索阶段。

## 路线行为解释

路线图不能作为训练阶段证据本身，但适合作为训练结果的行为解释。在 SwissViewMonuments 可视化任务中，本文方法在 `C=8` 距离上的路线成功率为 `{route_ours_d8['success_rate']:.4f}`，GOMAA-Geo 为 `{route_gomaa_d8['success_rate']:.4f}`。这与训练奖励设计的解释一致：中远距离任务更需要先保持探索方向，再在接近目标后收敛。

## 输出图件

- `defense_reward_training_mechanism_flow.png/svg`：给非专业老师看的训练阶段机制流程图。
- `defense_training_reward_curves.png/svg`：不同奖励机制的训练曲线和训练后 MM-GAG 结果。
- `defense_gp_training_ablation_curves.png/svg`：G/P 消融训练曲线与训练后统计。
- `defense_training_to_route_joint_panel.png/svg`：训练曲线与中远距离路线行为联合图。
- `defense_gp_training_route_consequence.png/svg`：G/P 消融曲线与同一样例路线结果。

## 建议汇报顺序

1. 先放机制流程图，强调奖励只在训练阶段使用。
2. 再放训练曲线，回答“不同奖励机制如何指导训练”。
3. 接着放 G/P 消融图，回答“为什么不是简单加内在奖励，而是需要距离门控和 PBRS”。
4. 最后放路线联合图，说明训练后策略在中远距离任务上的行为变化。
"""
    report.write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    setup_style()
    draw_mechanism_flow()
    draw_training_curves()
    draw_gp_ablation_curves()
    draw_route_joint_panel()
    draw_gp_route_consequence()
    copy_reference_assets()
    write_report()


if __name__ == "__main__":
    main()
