#!/usr/bin/env python
"""Build defense-ready training-stage reward and route visualizations.

This script reads the detailed logs produced by GeoExplorer/train.py:
- training_metrics.csv
- training_reward_components.csv
- training_route_samples.csv

The figures are explicitly training-stage evidence. Evaluation-time route
behavior should be described as the learned policy behavior, not as reward
functions being called during testing.
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
from matplotlib.patches import Rectangle


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
FIGURES = RESULTS / "figures" / "defense_reward_training_stage"
TABLES = RESULTS / "tables" / "defense_reward_training_stage"
REPORTS = RESULTS / "reports"

GEO_PIPELINE = Path(r"F:\bishe\GeoExplorer\analysis\pipeline_20260603_defense_reward_training_curves")
LOG_ROOT = GEO_PIPELINE / "training_logs"
STATUS_PATH = GEO_PIPELINE / "defense_reward_training_status_latest.json"
MANIFEST_PATH = Path(
    r"F:\bishe\GeoExplorer\ab_experiments\defense_reward_training_20260603"
    r"\reward_route_training_curves_3seed_480k\comparison_manifest.json"
)

INK = "#111827"
MUTED = "#5F6B7A"
GRID = "#D8DEE8"
PAPER = "#F8FAFC"
CARD = "#FFFFFF"
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
RED = "#C44E52"
CYAN = "#56B4E9"
GRAY = "#7A8699"

METHOD_STYLE = {
    "external_only": {"label": "仅外部奖励", "short": "外部", "color": ORANGE},
    "intrinsic_only": {"label": "仅内在奖励", "short": "内在", "color": PURPLE},
    "mixed_no_gate_no_pbrs": {"label": "外部+内在直接相加", "short": "直接相加", "color": GREEN},
    "mixed_gate_only": {"label": "外部+门控内在", "short": "门控内在", "color": CYAN},
    "mixed_pbrs_only": {"label": "外部+内在+PBRS", "short": "PBRS", "color": RED},
    "proposed_linear_gate_pbrs": {"label": "本文方法：门控内在+PBRS", "short": "本文方法", "color": BLUE},
}
METHOD_ORDER = list(METHOD_STYLE)
KEY_METHOD = "proposed_linear_gate_pbrs"


def ensure_dirs() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
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
            "axes.edgecolor": "#C7D0DC",
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
        }
    )


def save_figure(fig: plt.Figure, stem: str, dpi: int = 300) -> None:
    fig.savefig(FIGURES / f"{stem}.png", dpi=dpi, bbox_inches="tight", pad_inches=0.16)
    fig.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


def clean_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#C9D2DE")
    ax.spines["bottom"].set_color("#C9D2DE")
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)


def add_header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.035, 0.965, title, fontsize=23, fontweight="bold", ha="left", va="top")
    fig.text(0.035, 0.925, subtitle, fontsize=11.5, color=MUTED, ha="left", va="top")
    fig.lines.append(plt.Line2D([0.035, 0.965], [0.895, 0.895], transform=fig.transFigure, color="#CCD6E2", lw=1.2))


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.10, 1.07, label, transform=ax.transAxes, fontsize=14, fontweight="bold", ha="left", va="top")


def method_from_run(run_name: str) -> str | None:
    for key in METHOD_ORDER:
        if run_name.startswith(key + "_seed"):
            return key
    return None


def seed_from_run(run_name: str) -> int | None:
    marker = "_seed"
    if marker not in run_name:
        return None
    after = run_name.split(marker, 1)[1]
    seed_text = after.split("_", 1)[0]
    try:
        return int(seed_text)
    except ValueError:
        return None


def load_status() -> dict:
    if STATUS_PATH.exists():
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    return {}


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {"target_steps": 480000, "methods": [], "seeds": []}


def load_logs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_rows = []
    component_rows = []
    route_rows = []
    if not LOG_ROOT.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for run_dir in sorted(LOG_ROOT.iterdir()):
        if not run_dir.is_dir():
            continue
        method = method_from_run(run_dir.name)
        seed = seed_from_run(run_dir.name)
        if method is None or seed is None:
            continue
        for name, sink in [
            ("training_metrics.csv", metrics_rows),
            ("training_reward_components.csv", component_rows),
            ("training_route_samples.csv", route_rows),
        ]:
            path = run_dir / name
            if not path.exists() or path.stat().st_size == 0:
                continue
            try:
                df = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                continue
            if df.empty:
                continue
            df["run_name"] = run_dir.name
            df["method"] = method
            df["seed"] = seed
            df["method_label"] = METHOD_STYLE[method]["label"]
            sink.append(df)
    metrics = pd.concat(metrics_rows, ignore_index=True) if metrics_rows else pd.DataFrame()
    components = pd.concat(component_rows, ignore_index=True) if component_rows else pd.DataFrame()
    routes = pd.concat(route_rows, ignore_index=True) if route_rows else pd.DataFrame()
    return metrics, components, routes


def moving_average(series: pd.Series, window: int = 13) -> pd.Series:
    return series.astype(float).rolling(window=window, min_periods=1, center=True).mean()


def add_long_distance_fields(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    counts = sum(df.get(f"C{dist}_trajectory_count", 0).astype(float) for dist in (6, 7, 8))
    successes = sum(df.get(f"C{dist}_success_count", 0).astype(float) for dist in (6, 7, 8))
    df["c6_c8_success_rate"] = successes / counts.replace(0, np.nan)
    final_num = sum(
        df.get(f"C{dist}_mean_final_dist", np.nan).astype(float) * df.get(f"C{dist}_trajectory_count", 0).astype(float)
        for dist in (6, 7, 8)
    )
    df["c6_c8_final_dist"] = final_num / counts.replace(0, np.nan)
    df["c6_c8_trajectory_count"] = counts
    return df


def interpolate_metric(df: pd.DataFrame, y_col: str, max_progress: float | None = None, points: int = 121) -> pd.DataFrame:
    rows = []
    if df.empty or y_col not in df:
        return pd.DataFrame()
    if max_progress is None:
        max_progress = min(1.0, float(df["run_progress"].max()))
    grid = np.linspace(0.0, max_progress, points)
    for (method, seed, run_name), sub in df.groupby(["method", "seed", "run_name"]):
        sub = sub.sort_values("run_progress")
        x = sub["run_progress"].astype(float).to_numpy()
        y = moving_average(sub[y_col], 13).astype(float).to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if len(x) < 2:
            continue
        valid_grid = grid[(grid >= x.min()) & (grid <= x.max())]
        if len(valid_grid) == 0:
            continue
        interp = np.interp(valid_grid, x, y)
        rows.extend(
            {
                "method": method,
                "seed": seed,
                "run_name": run_name,
                "run_progress": gx,
                y_col: gy,
            }
            for gx, gy in zip(valid_grid, interp)
        )
    return pd.DataFrame(rows)


def plot_mean_band(ax: plt.Axes, source: pd.DataFrame, y_col: str, ylabel: str, title: str, higher_better: bool = True) -> None:
    for method in METHOD_ORDER:
        sub = source[source["method"] == method]
        if sub.empty:
            continue
        grouped = sub.groupby("run_progress")[y_col].agg(["mean", "std", "count"]).reset_index()
        grouped["std"] = grouped["std"].fillna(0.0)
        color = METHOD_STYLE[method]["color"]
        label = METHOD_STYLE[method]["label"]
        x = grouped["run_progress"].to_numpy() * 100
        mean = grouped["mean"].to_numpy()
        std = grouped["std"].to_numpy()
        is_key = method == KEY_METHOD
        ax.plot(
            x,
            mean,
            color=color,
            label=label,
            linewidth=3.0 if is_key else 1.8,
            alpha=0.98 if is_key else 0.72,
            zorder=5 if is_key else 3,
        )
        if grouped["count"].max() > 1:
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.16 if is_key else 0.08, linewidth=0)
    ax.set_title(title)
    ax.set_xlabel("训练进度 (%)")
    ax.set_ylabel(ylabel)
    clean_axes(ax)
    tag = "越高越好" if higher_better else "越低越好"
    ax.text(0.98, 0.05, tag, transform=ax.transAxes, ha="right", va="bottom", color=MUTED, fontsize=9)


def annotate_bars(ax: plt.Axes, values: pd.Series, fmt: str = "{:.2f}") -> None:
    finite_values = [float(v) for v in values if np.isfinite(float(v))]
    if not finite_values:
        return
    y_span = max(finite_values) - min(finite_values)
    offset = max(y_span * 0.025, max(abs(v) for v in finite_values) * 0.015, 0.015)
    for patch, value in zip(ax.patches, values):
        value = float(value)
        if not np.isfinite(value):
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() + offset,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=INK,
        )


def build_run_summary(metrics: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    rows = []
    comp = add_long_distance_fields(components)
    for method in METHOD_ORDER:
        runs = sorted(set(metrics[metrics["method"] == method]["run_name"]).union(set(comp[comp["method"] == method]["run_name"])))
        for run_name in runs:
            m = metrics[metrics["run_name"] == run_name].sort_values("run_progress") if not metrics.empty else pd.DataFrame()
            c = comp[comp["run_name"] == run_name].sort_values("run_progress") if not comp.empty else pd.DataFrame()
            seed = seed_from_run(run_name)
            tail_m = m.tail(max(5, int(len(m) * 0.2))) if not m.empty else pd.DataFrame()
            tail_c = c.tail(max(5, int(len(c) * 0.2))) if not c.empty else pd.DataFrame()
            rows.append(
                {
                    "method": method,
                    "method_label": METHOD_STYLE[method]["label"],
                    "run_name": run_name,
                    "seed": seed,
                    "max_progress": float(max(m["run_progress"].max() if not m.empty else 0, c["run_progress"].max() if not c.empty else 0)),
                    "max_time_step": int(max(m["time_step"].max() if not m.empty else 0, c["time_step"].max() if not c.empty else 0)),
                    "last_rolling_success": float(tail_m["rolling_success_ratio"].mean()) if "rolling_success_ratio" in tail_m else math.nan,
                    "last_best_val_success": float(tail_m["best_val_success"].max()) if "best_val_success" in tail_m else math.nan,
                    "last_c6_c8_success": float(tail_c["c6_c8_success_rate"].mean()) if "c6_c8_success_rate" in tail_c else math.nan,
                    "last_c6_c8_final_dist": float(tail_c["c6_c8_final_dist"].mean()) if "c6_c8_final_dist" in tail_c else math.nan,
                    "last_effective_external": float(tail_c["reward_ex_mean"].mean()) if "reward_ex_mean" in tail_c else math.nan,
                    "last_effective_intrinsic": float(tail_c["reward_in_gated_mean"].mean()) if "reward_in_gated_mean" in tail_c else math.nan,
                    "last_pbrs": float(tail_c["pbrs_bonus_mean"].mean()) if "pbrs_bonus_mean" in tail_c else math.nan,
                }
            )
    return pd.DataFrame(rows)


def build_method_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    agg = (
        summary.groupby(["method", "method_label"], as_index=False)
        .agg(
            run_count=("run_name", "nunique"),
            rolling_success_mean=("last_rolling_success", "mean"),
            rolling_success_std=("last_rolling_success", "std"),
            val_success_mean=("last_best_val_success", "mean"),
            c6_c8_success_mean=("last_c6_c8_success", "mean"),
            c6_c8_success_std=("last_c6_c8_success", "std"),
            c6_c8_final_dist_mean=("last_c6_c8_final_dist", "mean"),
            c6_c8_final_dist_std=("last_c6_c8_final_dist", "std"),
            effective_external_mean=("last_effective_external", "mean"),
            effective_intrinsic_mean=("last_effective_intrinsic", "mean"),
            pbrs_mean=("last_pbrs", "mean"),
        )
        .sort_values("method", key=lambda s: s.map({key: i for i, key in enumerate(METHOD_ORDER)}))
    )
    return agg


def figure_training_overview(metrics: pd.DataFrame, components: pd.DataFrame) -> None:
    components = add_long_distance_fields(components)
    max_progress = min(1.0, max(float(metrics["run_progress"].max()) if not metrics.empty else 0, float(components["run_progress"].max()) if not components.empty else 0))
    success_curve = interpolate_metric(metrics, "rolling_success_ratio", max_progress=max_progress)
    val_curve = interpolate_metric(metrics, "best_val_success", max_progress=max_progress)
    long_success = interpolate_metric(components, "c6_c8_success_rate", max_progress=max_progress)
    final_dist = interpolate_metric(components, "c6_c8_final_dist", max_progress=max_progress)

    fig = plt.figure(figsize=(13.8, 8.4))
    add_header(
        fig,
        "训练阶段奖励对照：混合奖励如何形成可学习的中长距离行为",
        "所有曲线均来自训练日志，阴影表示 3 个随机种子的波动；正式测试阶段不调用奖励函数，只加载训练后的策略。",
    )
    gs = fig.add_gridspec(2, 2, left=0.065, right=0.965, top=0.84, bottom=0.11, wspace=0.22, hspace=0.33)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    plot_mean_band(axes[0], success_curve, "rolling_success_ratio", "滚动成功率", "训练成功率随时间变化", True)
    plot_mean_band(axes[1], val_curve, "best_val_success", "最佳验证成功数", "中长距离验证任务累计提升", True)
    plot_mean_band(axes[2], long_success, "c6_c8_success_rate", "C6-C8 成功率", "中长距离训练采样成功率", True)
    plot_mean_band(axes[3], final_dist, "c6_c8_final_dist", "C6-C8 平均终止距离", "失败样本到目标的剩余距离", False)
    for label, ax in zip("ABCD", axes):
        panel_label(ax, label)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.015))
    save_figure(fig, "figure_training_stage_overview")


def aggregate_proposed_components(components: pd.DataFrame) -> pd.DataFrame:
    sub = components[components["method"] == "proposed_linear_gate_pbrs"].copy()
    if sub.empty:
        return pd.DataFrame()
    cols = ["reward_ex_mean", "reward_in_gated_mean", "pbrs_bonus_mean", "gate_weight_mean", "abs_reward_ex_share", "abs_reward_in_gated_share", "abs_pbrs_bonus_share"]
    rows = []
    grid = np.linspace(0.0, min(1.0, float(sub["run_progress"].max())), 121)
    for col in cols:
        cur = interpolate_metric(sub, col, max_progress=float(grid.max()), points=len(grid))
        if cur.empty:
            continue
        mean = cur.groupby("run_progress")[col].mean().reset_index()
        mean["metric"] = col
        rows.append(mean.rename(columns={col: "value"}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def figure_reward_components(components: pd.DataFrame, summary: pd.DataFrame) -> None:
    components = add_long_distance_fields(components)
    proposed = aggregate_proposed_components(components)
    fig = plt.figure(figsize=(13.8, 8.2))
    add_header(
        fig,
        "奖励成分动态：门控内在奖励负责探索，PBRS 负责方向塑形",
        "图中统计的是实际进入 PPO buffer 的有效奖励贡献；原始外部/内在值只用于诊断，不等同于训练信号。",
    )
    gs = fig.add_gridspec(2, 2, left=0.065, right=0.965, top=0.84, bottom=0.11, wspace=0.25, hspace=0.36)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    if not proposed.empty:
        component_labels = {
            "reward_ex_mean": ("外部目标反馈", ORANGE),
            "reward_in_gated_mean": ("门控内在探索", GREEN),
            "pbrs_bonus_mean": ("PBRS 方向塑形", BLUE),
        }
        for metric, (label, color) in component_labels.items():
            cur = proposed[proposed["metric"] == metric]
            if cur.empty:
                continue
            ax1.plot(cur["run_progress"] * 100, cur["value"], color=color, label=label)
        ax1.axhline(0, color="#9AA5B1", linewidth=1)
    else:
        ax1.text(0.5, 0.52, "等待本文方法训练日志", transform=ax1.transAxes, ha="center", va="center", color=MUTED, fontsize=13)
    ax1.set_title("本文方法的有效奖励来源")
    ax1.set_xlabel("训练进度 (%)")
    ax1.set_ylabel("每步平均奖励贡献")
    clean_axes(ax1)
    if ax1.get_legend_handles_labels()[0]:
        ax1.legend(frameon=False, loc="best")

    if not proposed.empty:
        shares = {
            "外部目标反馈": proposed[proposed["metric"] == "abs_reward_ex_share"].set_index("run_progress")["value"],
            "门控内在探索": proposed[proposed["metric"] == "abs_reward_in_gated_share"].set_index("run_progress")["value"],
            "PBRS 方向塑形": proposed[proposed["metric"] == "abs_pbrs_bonus_share"].set_index("run_progress")["value"],
        }
        share_df = pd.DataFrame(shares).sort_index().ffill().fillna(0)
        x = share_df.index.to_numpy() * 100
        ax2.stackplot(x, share_df.T.to_numpy(), labels=list(share_df.columns), colors=[ORANGE, GREEN, BLUE], alpha=0.82)
    else:
        ax2.text(0.5, 0.52, "等待本文方法训练日志", transform=ax2.transAxes, ha="center", va="center", color=MUTED, fontsize=13)
    ax2.set_title("有效奖励占比随训练进度变化")
    ax2.set_xlabel("训练进度 (%)")
    ax2.set_ylabel("绝对贡献占比")
    ax2.set_ylim(0, 1.02)
    clean_axes(ax2)
    if ax2.get_legend_handles_labels()[0]:
        ax2.legend(frameon=False, loc="upper right")

    final = summary.dropna(subset=["last_c6_c8_success"]).copy()
    final["order"] = final["method"].map({key: i for i, key in enumerate(METHOD_ORDER)})
    final = final.sort_values(["order", "seed"])
    group = final.groupby("method")["last_c6_c8_success"].agg(["mean", "std", "count"]).reindex(METHOD_ORDER).dropna(subset=["mean"])
    x = np.arange(len(group))
    colors = [METHOD_STYLE[m]["color"] for m in group.index]
    ax3.bar(x, group["mean"], yerr=group["std"].fillna(0), color=colors, alpha=0.88, capsize=3)
    ax3.set_xticks(x, [METHOD_STYLE[m]["short"] for m in group.index])
    ax3.set_title("训练末段 C6-C8 成功率")
    ax3.set_ylabel("成功率")
    clean_axes(ax3)
    annotate_bars(ax3, group["mean"], "{:.2f}")

    final_dist = summary.dropna(subset=["last_c6_c8_final_dist"]).copy()
    group2 = final_dist.groupby("method")["last_c6_c8_final_dist"].agg(["mean", "std", "count"]).reindex(METHOD_ORDER).dropna(subset=["mean"])
    x2 = np.arange(len(group2))
    colors2 = [METHOD_STYLE[m]["color"] for m in group2.index]
    ax4.bar(x2, group2["mean"], yerr=group2["std"].fillna(0), color=colors2, alpha=0.88, capsize=3)
    ax4.set_xticks(x2, [METHOD_STYLE[m]["short"] for m in group2.index])
    ax4.set_title("训练末段 C6-C8 终止距离")
    ax4.set_ylabel("距离目标的格数")
    clean_axes(ax4)
    ax4.text(0.98, 0.95, "越低越好", transform=ax4.transAxes, ha="right", va="top", color=MUTED, fontsize=9)
    annotate_bars(ax4, group2["mean"], "{:.2f}")

    for label, ax in zip("ABCD", [ax1, ax2, ax3, ax4]):
        panel_label(ax, label)
    save_figure(fig, "figure_reward_component_dynamics")


def parse_json_list(text: str) -> list:
    if isinstance(text, list):
        return text
    if not isinstance(text, str) or not text:
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return []


def select_route_sample(routes: pd.DataFrame, method: str, target_progress: float) -> pd.Series | None:
    sub = routes[routes["method"] == method].copy()
    if sub.empty:
        return None
    sub = sub[sub["optimal_steps"].astype(float) >= 6]
    if sub.empty:
        return None
    sub["distance_to_target_progress"] = (sub["run_progress"].astype(float) - target_progress).abs()
    window = sub[sub["distance_to_target_progress"] <= 0.12].copy()
    if window.empty:
        window = sub.nsmallest(80, "distance_to_target_progress").copy()
    prefer_success = target_progress >= 0.35
    window["success_rank"] = -window["success"].astype(float) if prefer_success else window["success"].astype(float) * 0.4
    window["route_rank"] = (
        window["success_rank"]
        + window["final_dist"].astype(float) * 0.08
        + window["revisit_count"].astype(float) * 0.025
        + window["deviation_from_opt"].astype(float).clip(lower=0) * 0.015
        + window["distance_to_target_progress"].astype(float)
    )
    return window.sort_values(["route_rank", "distance_to_target_progress"]).iloc[0]


def patch_xy(patch: int, patch_size: int = 5) -> tuple[int, int]:
    row, col = divmod(int(patch), patch_size)
    return col, row


def draw_route(ax: plt.Axes, row: pd.Series | None, method: str, phase_label: str) -> None:
    patch_size = 5
    ax.set_xlim(-0.5, patch_size - 0.5)
    ax.set_ylim(patch_size - 0.5, -0.5)
    ax.set_aspect("equal")
    for i in range(patch_size):
        for j in range(patch_size):
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#F4F7FB", edgecolor="#CBD5E1", linewidth=0.8))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    label = METHOD_STYLE[method]["short"]
    if row is None:
        ax.set_title(f"{label} | {phase_label}\n暂无样本", fontsize=10.5)
        return
    seq = parse_json_list(row["patch_sequence"])
    path = [int(x) for x in seq[1:]] if len(seq) > 1 else []
    goal = int(seq[0]) if seq else int(row["goal_patch"])
    if path:
        xs, ys = zip(*(patch_xy(patch, patch_size) for patch in path))
        ax.plot(xs, ys, color=METHOD_STYLE[method]["color"], linewidth=2.2, alpha=0.88)
        ax.scatter(xs[0], ys[0], s=70, color=CARD, edgecolor=INK, linewidth=1.5, zorder=4)
        ax.text(xs[0], ys[0], "起", ha="center", va="center", fontsize=9, fontweight="bold", zorder=5)
        ax.scatter(xs[-1], ys[-1], marker="x", s=85, color=RED if int(row["success"]) == 0 else GREEN, linewidth=2.2, zorder=5)
    gx, gy = patch_xy(goal, patch_size)
    ax.scatter(gx, gy, marker="s", s=85, color="#FDE68A", edgecolor=INK, linewidth=1.2, zorder=5)
    ax.text(gx, gy, "目", ha="center", va="center", fontsize=9, fontweight="bold", zorder=6)
    state = "成功" if int(row["success"]) else "未到达"
    title = (
        f"{label} | {phase_label}\n"
        f"{row['run_progress'] * 100:.1f}%  C{int(row['optimal_steps'])}  终距 {int(row['final_dist'])}  {state}"
    )
    ax.set_title(title, fontsize=10.5)


def figure_route_samples(routes: pd.DataFrame) -> None:
    methods = ["external_only", "mixed_no_gate_no_pbrs", "proposed_linear_gate_pbrs"]
    phases = [("早期", 0.08), ("中期", 0.50), ("后期", 0.90)]
    fig = plt.figure(figsize=(12.2, 10.0))
    add_header(
        fig,
        "训练时路线样本：曲线旁边的真实行为证据",
        "每个小图均为训练阶段真实采样到的 C6-C8 轨迹；“起”为起点，“目”为目标，叉号为终止位置。",
    )
    gs = fig.add_gridspec(3, 3, left=0.055, right=0.965, top=0.84, bottom=0.06, wspace=0.20, hspace=0.42)
    for r, method in enumerate(methods):
        for c, (phase_label, progress) in enumerate(phases):
            ax = fig.add_subplot(gs[r, c])
            row = select_route_sample(routes, method, progress) if not routes.empty else None
            draw_route(ax, row, method, phase_label)
    save_figure(fig, "figure_training_route_samples")


def write_report(status: dict, manifest: dict, summary: pd.DataFrame, metrics: pd.DataFrame, components: pd.DataFrame, routes: pd.DataFrame) -> None:
    expected_runs = len(manifest.get("methods", [])) * len(manifest.get("seeds", []))
    available_runs = len(set(summary["run_name"])) if not summary.empty else 0
    complete_runs = int(status.get("completed") or 0)
    phase = status.get("phase", "unknown")
    summary_path = TABLES / "training_stage_run_summary.csv"
    method_summary_path = TABLES / "training_stage_method_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    method_summary = build_method_summary(summary)
    if not method_summary.empty:
        method_summary.to_csv(method_summary_path, index=False, encoding="utf-8-sig")
    method_lines = []
    if not method_summary.empty:
        for _, row in method_summary.iterrows():
            method_lines.append(
                "- {label}：训练末段整体成功率 {overall:.3f}，C6-C8 成功率 {long_sr:.3f}，失败样本平均终止距离 {dist:.2f} 格。".format(
                    label=row["method_label"],
                    overall=float(row["rolling_success_mean"]),
                    long_sr=float(row["c6_c8_success_mean"]),
                    dist=float(row["c6_c8_final_dist_mean"]),
                )
            )
    method_block = "\n".join(method_lines) if method_lines else "- 暂无可汇总结果。"
    report = f"""# 训练阶段混合奖励可视化说明

## 当前状态
- 远端阶段：`{phase}`
- 计划训练 run：`{expected_runs}`
- 已进入本地日志汇总的 run：`{available_runs}`
- 远端标记完成 run：`{complete_runs}`
- 奖励机制：6 种
- 随机种子：3 个

## 图像用途
- `figure_training_stage_overview`：答辩主图，说明不同奖励机制在训练阶段的整体成功率、验证表现和 C6-C8 中长距离训练行为。
- `figure_reward_component_dynamics`：机制解释图，拆解本文方法中外部目标反馈、门控内在探索、PBRS 方向塑形的有效贡献。
- `figure_training_route_samples`：路线旁证图，展示训练过程中真实采样到的 C6-C8 中长距离轨迹，可放在训练曲线旁边解释行为变化。

## 训练末段汇总
{method_block}

## 可直接用于答辩的解释
1. 仅内在奖励在训练后期仍几乎不能完成 C6-C8 任务，说明“只鼓励探索”会偏离到达目标的任务目标。
2. 仅外部奖励和直接相加能够学到一部分中长距离行为，但它们没有显式区分“远距离需要探索”和“近目标需要收敛”的阶段差异。
3. 本文方法把内在奖励通过距离门控后再进入 PPO，同时加入 PBRS 的方向塑形，因此训练信号不是简单相加，而是在训练过程中同时提供探索压力和朝目标推进的形状约束。
4. 本组图是训练阶段证据，主要用于解释奖励机制如何指导学习过程；最终测试性能仍应引用正式评估表格。

## 关键表述
奖励、距离门控和 PBRS 都是训练阶段信号。正式测试或论文表格评估时，不再调用奖励函数，而是加载训练好的 checkpoint 并使用策略选择动作。

## 输出表
- `{summary_path}`
- `{method_summary_path}`
"""
    (REPORTS / "defense_reward_training_stage_analysis_zh.md").write_text(report, encoding="utf-8")


def copy_to_geo_pipeline() -> None:
    target = GEO_PIPELINE / "figures"
    target.mkdir(parents=True, exist_ok=True)
    for path in FIGURES.glob("*"):
        if path.is_file():
            shutil.copy2(path, target / path.name)


def main() -> int:
    ensure_dirs()
    setup_style()
    status = load_status()
    manifest = load_manifest()
    metrics, components, routes = load_logs()
    if not metrics.empty and "run_progress" not in metrics:
        target = float(manifest.get("target_steps", 480000))
        metrics["run_progress"] = metrics["time_step"].astype(float) / max(target, 1.0)
    components = add_long_distance_fields(components)
    summary = build_run_summary(metrics, components)
    if not summary.empty:
        summary.to_csv(TABLES / "training_stage_run_summary.csv", index=False, encoding="utf-8-sig")
        build_method_summary(summary).to_csv(TABLES / "training_stage_method_summary.csv", index=False, encoding="utf-8-sig")
    if not metrics.empty and not components.empty:
        figure_training_overview(metrics, components)
        figure_reward_components(components, summary)
    if not routes.empty:
        figure_route_samples(routes)
    write_report(status, manifest, summary, metrics, components, routes)
    copy_to_geo_pipeline()
    print(
        json.dumps(
            {
                "figures": [str(path) for path in sorted(FIGURES.glob("*.png"))],
                "summary": str(TABLES / "training_stage_run_summary.csv"),
                "report": str(REPORTS / "defense_reward_training_stage_analysis_zh.md"),
                "available_runs": int(summary["run_name"].nunique()) if not summary.empty else 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
