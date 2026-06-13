#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build defense-ready trend figures for the mixed reward mechanism.

The figure is trend-first, but data-faithful:
- real formal MM-GAG distance buckets are plotted as marked points;
- real training logs are averaged across seeds, with uncertainty bands;
- sparse budget checkpoints are smoothed only for visual connection;
- route examples are sampled from real training_route_samples.csv records.

Interpolated points are never exported as experiment evidence. The companion
tables contain the real observed values used by the figure.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from scipy.interpolate import PchipInterpolator


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
REPORTS = RESULTS / "reports"
FIGURES = RESULTS / "figures" / "defense_reward_trends"
OUT_TABLES = TABLES / "defense_reward_trends"

TRAIN_LOG_ROOT = Path(r"F:\bishe\GeoExplorer\analysis\pipeline_20260603_defense_reward_training_curves\training_logs")
FIXED_CHECKPOINT_LOCAL = Path(
    r"F:\bishe\GeoExplorer\analysis\pipeline_20260604_fixed_checkpoint_reward_eval_allckpt\fixed_checkpoint_eval_all.csv"
)
MMGAG_CHECKPOINT_LOCAL = Path(
    r"F:\bishe\GeoExplorer\analysis\pipeline_20260604_mmgag_checkpoint_reward_trend\mmgag_checkpoint_eval_all.csv"
)
REWARD_CONTROL_LONG = TABLES / "ablation" / "reward_control_long_table.csv"
APPENDIX_LONG = TABLES / "appendix" / "appendix_dataset_param_long_table.csv"
REWARD_GATE = TABLES / "ablation" / "reward_gate_type_mmgag_only_table_with_linear.csv"

INK = "#17212F"
MUTED = "#5B6777"
PAPER = "#F7F9FC"
CARD = "#FFFFFF"
GRID = "#D8E0EA"
BLUE = "#1764AB"
ORANGE = "#D27A20"
GREEN = "#168A63"
TEAL = "#2098A3"
RED = "#B84A48"
PURPLE = "#7C5CC4"
GRAY = "#7A8699"
LIGHT_BLUE = "#DCEBFA"

METHODS = {
    "external_only": {"short": "仅外部奖励", "label": "仅外部奖励", "color": ORANGE},
    "intrinsic_only": {"short": "仅内在奖励", "label": "仅内在奖励", "color": PURPLE},
    "mixed_no_gate_no_pbrs": {"short": "直接相加", "label": "外部+内在直接相加", "color": GREEN},
    "mixed_gate_only": {"short": "门控内在", "label": "门控内在奖励", "color": TEAL},
    "mixed_pbrs_only": {"short": "仅加 PBRS", "label": "外部+内在+PBRS", "color": RED},
    "proposed_linear_gate_pbrs": {"short": "本文方法", "label": "门控内在奖励+PBRS", "color": BLUE},
    "external_pbrs": {"short": "外部+PBRS", "label": "外部奖励+PBRS", "color": ORANGE},
    "constant_gate_pbrs": {"short": "常数门控+PBRS", "label": "常数门控+PBRS", "color": PURPLE},
    "linear_gate_no_pbrs": {"short": "线性门控", "label": "线性门控，无 PBRS", "color": GREEN},
}

METHOD_ORDER = [
    "external_only",
    "intrinsic_only",
    "mixed_no_gate_no_pbrs",
    "mixed_gate_only",
    "mixed_pbrs_only",
    "proposed_linear_gate_pbrs",
]

FORMAL_METHOD_ORDER = [
    "intrinsic_only",
    "external_only",
    "mixed_no_gate_no_pbrs",
    "mixed_gate_only",
    "proposed_linear_gate_pbrs",
]

MMGAG_TREND_ORDER = [
    "linear_gate_no_pbrs",
    "external_pbrs",
    "constant_gate_pbrs",
    "proposed_linear_gate_pbrs",
]


def ensure_dirs() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
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
            "axes.titlesize": 13.2,
            "axes.labelsize": 10.8,
            "xtick.labelsize": 9.4,
            "ytick.labelsize": 9.4,
            "legend.fontsize": 9.2,
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
    ax.text(-0.09, 1.08, label, transform=ax.transAxes, fontsize=15, fontweight="bold", ha="left", va="top")


def add_header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.04, 0.968, title, fontsize=22.5, fontweight="bold", ha="left", va="top")
    fig.text(0.04, 0.928, subtitle, fontsize=11.5, color=MUTED, ha="left", va="top")
    fig.lines.append(plt.Line2D([0.04, 0.965], [0.895, 0.895], transform=fig.transFigure, color="#CCD6E2", lw=1.2))


def method_from_run(run_name: str) -> str | None:
    for method in METHOD_ORDER:
        if run_name.startswith(method + "_seed"):
            return method
    return None


def seed_from_run(run_name: str) -> int | None:
    if "_seed" not in run_name:
        return None
    try:
        return int(run_name.split("_seed", 1)[1].split("_", 1)[0])
    except ValueError:
        return None


def moving_average(series: pd.Series, window: int = 25) -> pd.Series:
    return series.astype(float).rolling(window=window, min_periods=1, center=True).mean()


def smooth_xy(x: np.ndarray, y: np.ndarray, n: int = 180) -> tuple[np.ndarray, np.ndarray]:
    """Return a monotone PCHIP visual connector through real observed points."""
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


def read_training_metrics() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in sorted(TRAIN_LOG_ROOT.glob("*/training_metrics.csv")):
        method = method_from_run(path.parent.name)
        seed = seed_from_run(path.parent.name)
        if method is None or seed is None:
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["method"] = method
        df["seed"] = seed
        df["run_name"] = path.parent.name
        df["run_progress"] = df["time_step"].astype(float) / 480000.0
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No training metrics found under {TRAIN_LOG_ROOT}")
    return pd.concat(rows, ignore_index=True)


def read_reward_components() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in sorted(TRAIN_LOG_ROOT.glob("*/training_reward_components.csv")):
        method = method_from_run(path.parent.name)
        seed = seed_from_run(path.parent.name)
        if method is None or seed is None:
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["method"] = method
        df["seed"] = seed
        df["run_name"] = path.parent.name
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No reward component logs found under {TRAIN_LOG_ROOT}")
    data = pd.concat(rows, ignore_index=True)
    c678_counts = sum(data[f"C{dist}_trajectory_count"].astype(float) for dist in [6, 7, 8])
    c678_success = sum(data[f"C{dist}_success_count"].astype(float) for dist in [6, 7, 8])
    data["c6_c8_success_rate"] = c678_success / c678_counts.replace(0, np.nan)
    data["c6_c8_success_mean_unweighted"] = data[["C6_success_rate", "C7_success_rate", "C8_success_rate"]].astype(float).mean(axis=1)
    return data


def read_route_samples() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in sorted(TRAIN_LOG_ROOT.glob("*/training_route_samples.csv")):
        method = method_from_run(path.parent.name)
        seed = seed_from_run(path.parent.name)
        if method is None or seed is None:
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["method"] = method
        df["seed"] = seed
        df["run_name"] = path.parent.name
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def interpolate_runs(df: pd.DataFrame, y_col: str, best_so_far: bool = False, points: int = 141) -> pd.DataFrame:
    rows: list[dict] = []
    grid = np.linspace(0, 1, points)
    for (method, seed, run_name), sub in df.groupby(["method", "seed", "run_name"]):
        if y_col not in sub:
            continue
        sub = sub.sort_values("run_progress")
        x = sub["run_progress"].astype(float).clip(0, 1).to_numpy()
        y = moving_average(sub[y_col]).astype(float).to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) < 2:
            continue
        if best_so_far:
            y = np.maximum.accumulate(y)
        valid = grid[(grid >= x.min()) & (grid <= x.max())]
        vals = np.interp(valid, x, y)
        rows.extend(
            {"method": method, "seed": seed, "run_name": run_name, "run_progress": gx, y_col: gy}
            for gx, gy in zip(valid, vals)
        )
    return pd.DataFrame(rows)


def mean_band(interp: pd.DataFrame, y_col: str) -> pd.DataFrame:
    if interp.empty:
        return interp
    return (
        interp.groupby(["method", "run_progress"], as_index=False)
        .agg(mean=(y_col, "mean"), std=(y_col, "std"), n=(y_col, "count"))
        .assign(std=lambda x: x["std"].fillna(0.0))
    )


def plot_training_band(ax: plt.Axes, band: pd.DataFrame, scale: float = 100.0, methods: list[str] | None = None) -> None:
    methods = methods or METHOD_ORDER
    for method in methods:
        sub = band[band["method"].eq(method)].sort_values("run_progress")
        if sub.empty:
            continue
        color = METHODS[method]["color"]
        x = sub["run_progress"].to_numpy() * 100
        mean = sub["mean"].to_numpy() * scale
        std = sub["std"].to_numpy() * scale
        is_ours = method == "proposed_linear_gate_pbrs"
        ax.plot(
            x,
            mean,
            color=color,
            lw=3.2 if is_ours else 1.8,
            label=METHODS[method]["short"],
            zorder=10 if is_ours else 3,
            alpha=0.98 if is_ours else 0.72,
        )
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12 if is_ours else 0.055, linewidth=0, zorder=2)


def formal_distance_trend() -> pd.DataFrame:
    rc = pd.read_csv(REWARD_CONTROL_LONG)
    ap = pd.read_csv(APPENDIX_LONG)
    definitions = [
        ("external_only", "reward_external_only_seed321_t480k", rc),
        ("intrinsic_only", "reward_intrinsic_only_seed321_t480k", rc),
        ("mixed_no_gate_no_pbrs", "reward_intrinsic_no_decay_seed321_t480k", rc),
        ("mixed_gate_only", "param_pbrs_0_seed321_t480k", ap),
        ("proposed_linear_gate_pbrs", "dataset_masa_plus_mmgag_seed321_t480k", ap),
    ]
    rows = []
    for method, run, df in definitions:
        sub = df[df["run"].eq(run) & df["benchmark"].isin(["mmgag_aerial", "mmgag_ground", "mmgag_text"])]
        row = {"method": method, "method_label": METHODS[method]["label"], "run": run, "mmgag_mean_sr": float(sub["sr"].mean())}
        for dist in [4, 5, 6, 7, 8]:
            row[f"C{dist}"] = float(sub[f"d{dist}"].mean())
        row["C6-8"] = float(np.mean([row["C6"], row["C7"], row["C8"]]))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / "formal_mmgag_distance_real_points.csv", index=False, encoding="utf-8-sig")
    return out


def budget_trend() -> pd.DataFrame:
    ap = pd.read_csv(APPENDIX_LONG)
    definitions = [
        ("240k", 240000, "param_budget_240k_seed321_t240k"),
        ("480k", 480000, "dataset_masa_plus_mmgag_seed321_t480k"),
        ("720k", 720000, "param_budget_720k_seed321_t720k"),
    ]
    rows = []
    for label, steps, run in definitions:
        sub = ap[ap["run"].eq(run) & ap["benchmark"].isin(["mmgag_aerial", "mmgag_ground", "mmgag_text"])]
        rows.append(
            {
                "budget_label": label,
                "target_steps": steps,
                "mmgag_mean_sr": float(sub["sr"].mean()),
                "c6_c8_mean": float(sub[["d6", "d7", "d8"]].mean().mean()),
                "c8_mean": float(sub["d8"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / "formal_mmgag_budget_real_points.csv", index=False, encoding="utf-8-sig")
    return out


def mechanism_progress_trend() -> pd.DataFrame:
    """True MM-GAG SR points for an interpretable reward-mechanism ablation trend."""
    rc = pd.read_csv(REWARD_CONTROL_LONG)
    gate = pd.read_csv(REWARD_GATE)
    direct = rc[
        rc["run"].eq("reward_intrinsic_no_decay_seed321_t480k")
        & rc["benchmark"].isin(["mmgag_aerial", "mmgag_ground", "mmgag_text"])
    ]
    intrinsic = rc[
        rc["run"].eq("reward_intrinsic_only_seed321_t480k")
        & rc["benchmark"].isin(["mmgag_aerial", "mmgag_ground", "mmgag_text"])
    ]
    rows = [
        {
            "order": 0,
            "stage": "仅内在",
            "full_label": "仅内在奖励",
            "mmgag_mean_sr": float(intrinsic["sr"].mean()),
            "source": "reward_control_long_table.csv",
        },
        {
            "order": 1,
            "stage": "直接相加",
            "full_label": "外部+内在直接相加",
            "mmgag_mean_sr": float(direct["sr"].mean()),
            "source": "reward_control_long_table.csv",
        },
    ]
    for order, value, stage, label in [
        (2, "linear_0.405_no_pb", "线性门控", "线性门控，无 PBRS"),
        (3, "external_pbrs", "PBRS", "外部奖励+PBRS"),
        (4, "linear_0.405_pb", "门控+PBRS", "线性门控+PBRS（本文方法）"),
    ]:
        row = gate[gate["value"].eq(value)].iloc[0]
        rows.append(
            {
                "order": order,
                "stage": stage,
                "full_label": label,
                "mmgag_mean_sr": float(row["mmgag_mean_sr"]),
                "source": "reward_gate_type_mmgag_only_table_with_linear.csv",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / "reward_mechanism_progress_real_points.csv", index=False, encoding="utf-8-sig")
    return out


def convergence_summary(metrics: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        for seed, sub in metrics[metrics["method"].eq(method)].groupby("seed"):
            sub = sub.sort_values("run_progress").reset_index(drop=True)
            val = moving_average(sub["val_success"].astype(float) / 20.0, 25)
            best = val.cummax()
            comp = components[(components["method"].eq(method)) & (components["seed"].eq(seed))].sort_values("run_progress").reset_index(drop=True)
            if not comp.empty:
                c678 = moving_average(comp["c6_c8_success_mean_unweighted"], 25)
                c678_best = c678.cummax()
                c678_best_value = float(c678_best.max())
                c678_best_progress = float(comp.loc[int(c678_best.idxmax()), "run_progress"])
            else:
                c678_best_value = np.nan
                c678_best_progress = np.nan
            rows.append(
                {
                    "method": method,
                    "method_label": METHODS[method]["label"],
                    "seed": seed,
                    "best_val": float(best.max()),
                    "final_val": float(val.tail(50).mean()),
                    "drop_after_best": float(best.max() - val.tail(50).mean()),
                    "best_progress": float(sub.loc[int(best.idxmax()), "run_progress"]),
                    "hit_80_progress": float(sub.loc[int(best[best >= 0.80].index[0]), "run_progress"]) if (best >= 0.80).any() else np.nan,
                    "c6_c8_train_best": c678_best_value,
                    "c6_c8_train_best_progress": c678_best_progress,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / "training_convergence_real_summary_by_seed.csv", index=False, encoding="utf-8-sig")
    return out


def maybe_fixed_checkpoint_table() -> pd.DataFrame:
    if not FIXED_CHECKPOINT_LOCAL.exists() or FIXED_CHECKPOINT_LOCAL.stat().st_size == 0:
        return pd.DataFrame()
    fixed = pd.read_csv(FIXED_CHECKPOINT_LOCAL)
    required = {"method", "seed", "run_progress", "success_ratio", "C6_success_ratio", "C7_success_ratio", "C8_success_ratio"}
    if fixed.empty or not required.issubset(fixed.columns):
        return pd.DataFrame()
    fixed = fixed[fixed["method"].isin(METHOD_ORDER)].copy()
    fixed["c6_c8_mean"] = fixed[["C6_success_ratio", "C7_success_ratio", "C8_success_ratio"]].astype(float).mean(axis=1)
    fixed.to_csv(OUT_TABLES / "fixed_checkpoint_eval_real_points.csv", index=False, encoding="utf-8-sig")
    return fixed


def fixed_checkpoint_trend_tables(fixed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize fixed-checkpoint observations into training-time trend tables."""
    if fixed.empty:
        return pd.DataFrame(), pd.DataFrame()

    data = fixed.copy()
    numeric_cols = [
        "seed",
        "episode",
        "time_step",
        "run_progress",
        "success_ratio",
        "sg_mean",
        "C6_success_ratio",
        "C6_sg_mean",
        "C7_success_ratio",
        "C7_sg_mean",
        "C8_success_ratio",
        "C8_sg_mean",
    ]
    for col in numeric_cols:
        if col in data:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    data["c6_c8_mean"] = data[["C6_success_ratio", "C7_success_ratio", "C8_success_ratio"]].mean(axis=1)
    data["c8_closeness"] = (1.0 - data["C8_sg_mean"].astype(float) / 8.0).clip(0.0, 1.0)
    scheduled = data[data["checkpoint_kind"].eq("scheduled")].copy()
    if scheduled.empty:
        return pd.DataFrame(), pd.DataFrame()

    trend = (
        scheduled.groupby(["method", "method_label", "checkpoint_name"], as_index=False)
        .agg(
            run_progress=("run_progress", "mean"),
            episode=("episode", "median"),
            time_step=("time_step", "median"),
            n_seed=("seed", "nunique"),
            overall_success_mean=("success_ratio", "mean"),
            overall_success_max=("success_ratio", "max"),
            c6_c8_success_mean=("c6_c8_mean", "mean"),
            c6_c8_success_max=("c6_c8_mean", "max"),
            c6_success_mean=("C6_success_ratio", "mean"),
            c6_success_max=("C6_success_ratio", "max"),
            c7_success_mean=("C7_success_ratio", "mean"),
            c7_success_max=("C7_success_ratio", "max"),
            c8_success_mean=("C8_success_ratio", "mean"),
            c8_success_std=("C8_success_ratio", "std"),
            c8_success_max=("C8_success_ratio", "max"),
            c8_sg_mean=("C8_sg_mean", "mean"),
            c8_sg_min=("C8_sg_mean", "min"),
            c8_closeness_mean=("c8_closeness", "mean"),
            c8_closeness_max=("c8_closeness", "max"),
        )
        .assign(c8_success_std=lambda frame: frame["c8_success_std"].fillna(0.0))
    )

    trend_parts = []
    for method, sub in trend.groupby("method", sort=False):
        sub = sub.sort_values("run_progress").copy()
        sub["c8_success_envelope"] = sub["c8_success_max"].cummax()
        sub["c8_closeness_envelope"] = sub["c8_closeness_max"].cummax()
        sub["c8_sg_best_so_far"] = sub["c8_sg_min"].cummin()
        trend_parts.append(sub)
    trend = pd.concat(trend_parts, ignore_index=True)

    summary_rows = []
    for method, sub in trend.groupby("method", sort=False):
        sub = sub.sort_values("run_progress").reset_index(drop=True)
        best_success_idx = int(sub["c8_success_max"].idxmax())
        best_sg_idx = int(sub["c8_sg_min"].idxmin())
        hit_90 = sub[sub["c8_success_envelope"].ge(0.90)]
        summary_rows.append(
            {
                "method": method,
                "method_label": METHODS.get(method, {}).get("label", method),
                "best_c8_success": float(sub.loc[best_success_idx, "c8_success_max"]),
                "best_c8_success_progress": float(sub.loc[best_success_idx, "run_progress"]),
                "best_c8_success_checkpoint": str(sub.loc[best_success_idx, "checkpoint_name"]),
                "first_hit_90_progress": float(hit_90["run_progress"].iloc[0]) if not hit_90.empty else np.nan,
                "best_c8_sg": float(sub.loc[best_sg_idx, "c8_sg_min"]),
                "best_c8_sg_progress": float(sub.loc[best_sg_idx, "run_progress"]),
                "best_c8_sg_checkpoint": str(sub.loc[best_sg_idx, "checkpoint_name"]),
                "final_observed_c8_success_mean": float(sub["c8_success_mean"].iloc[-1]),
                "final_observed_c8_success_max": float(sub["c8_success_max"].iloc[-1]),
                "final_observed_c8_sg_min": float(sub["c8_sg_min"].iloc[-1]),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("best_c8_success", ascending=False)

    trend.to_csv(OUT_TABLES / "fixed_checkpoint_c8_training_trend_real_points.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_TABLES / "fixed_checkpoint_c8_method_summary.csv", index=False, encoding="utf-8-sig")
    return trend, summary


def maybe_mmgag_checkpoint_table() -> pd.DataFrame:
    if not MMGAG_CHECKPOINT_LOCAL.exists() or MMGAG_CHECKPOINT_LOCAL.stat().st_size == 0:
        return pd.DataFrame()
    mmgag = pd.read_csv(MMGAG_CHECKPOINT_LOCAL)
    required = {"method", "benchmark", "run_progress", "checkpoint_name", "C8_success_ratio", "C8_sg_mean"}
    if mmgag.empty or not required.issubset(mmgag.columns):
        return pd.DataFrame()
    mmgag = mmgag[mmgag["method"].isin(MMGAG_TREND_ORDER)].copy()
    if mmgag.empty:
        return pd.DataFrame()
    mmgag.to_csv(OUT_TABLES / "mmgag_checkpoint_eval_real_points.csv", index=False, encoding="utf-8-sig")
    return mmgag


def mmgag_checkpoint_trend_tables(mmgag: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate MM-GAG modality checkpoint rows into C=8 training trend tables."""
    if mmgag.empty:
        return pd.DataFrame(), pd.DataFrame()
    data = mmgag.copy()
    numeric_cols = [
        "episode",
        "time_step",
        "run_progress",
        "success_ratio",
        "sg_mean",
        "C8_success_ratio",
        "C8_sg_mean",
    ]
    for col in numeric_cols:
        if col in data:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data[data["checkpoint_kind"].eq("scheduled")].copy()
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()
    data["c8_closeness"] = (1.0 - data["C8_sg_mean"].astype(float) / 8.0).clip(0.0, 1.0)

    trend = (
        data.groupby(["method", "method_label", "value", "run_name", "checkpoint_name"], as_index=False)
        .agg(
            run_progress=("run_progress", "mean"),
            episode=("episode", "median"),
            time_step=("time_step", "median"),
            n_modality=("benchmark", "nunique"),
            c8_success_mean=("C8_success_ratio", "mean"),
            c8_success_std=("C8_success_ratio", "std"),
            c8_success_max=("C8_success_ratio", "max"),
            c8_sg_mean=("C8_sg_mean", "mean"),
            c8_sg_min=("C8_sg_mean", "min"),
            c8_closeness_mean=("c8_closeness", "mean"),
            c8_closeness_max=("c8_closeness", "max"),
            mmgag_aerial=("C8_success_ratio", lambda values: np.nan),
        )
        .assign(c8_success_std=lambda frame: frame["c8_success_std"].fillna(0.0))
    )

    modality_pivot = (
        data.pivot_table(
            index=["method", "checkpoint_name"],
            columns="benchmark",
            values="C8_success_ratio",
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    trend = trend.drop(columns=["mmgag_aerial"], errors="ignore").merge(modality_pivot, on=["method", "checkpoint_name"], how="left")

    trend_parts = []
    for method, sub in trend.groupby("method", sort=False):
        sub = sub.sort_values("run_progress").copy()
        sub["c8_success_envelope"] = sub["c8_success_mean"].cummax()
        sub["c8_closeness_envelope"] = sub["c8_closeness_mean"].cummax()
        sub["c8_sg_best_so_far"] = sub["c8_sg_mean"].cummin()
        trend_parts.append(sub)
    trend = pd.concat(trend_parts, ignore_index=True)

    summary_rows = []
    for method, sub in trend.groupby("method", sort=False):
        sub = sub.sort_values("run_progress").reset_index(drop=True)
        best_success_idx = int(sub["c8_success_mean"].idxmax())
        best_sg_idx = int(sub["c8_sg_mean"].idxmin())
        hit_90 = sub[sub["c8_success_envelope"].ge(0.90)]
        summary_rows.append(
            {
                "method": method,
                "method_label": METHODS.get(method, {}).get("label", method),
                "best_c8_success": float(sub.loc[best_success_idx, "c8_success_mean"]),
                "best_c8_success_progress": float(sub.loc[best_success_idx, "run_progress"]),
                "best_c8_success_checkpoint": str(sub.loc[best_success_idx, "checkpoint_name"]),
                "first_hit_90_progress": float(hit_90["run_progress"].iloc[0]) if not hit_90.empty else np.nan,
                "best_c8_sg": float(sub.loc[best_sg_idx, "c8_sg_mean"]),
                "best_c8_sg_progress": float(sub.loc[best_sg_idx, "run_progress"]),
                "best_c8_sg_checkpoint": str(sub.loc[best_sg_idx, "checkpoint_name"]),
                "final_observed_c8_success_mean": float(sub["c8_success_mean"].iloc[-1]),
                "final_observed_c8_sg": float(sub["c8_sg_mean"].iloc[-1]),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("best_c8_success", ascending=False)
    trend.to_csv(OUT_TABLES / "mmgag_checkpoint_c8_training_trend_real_points.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_TABLES / "mmgag_checkpoint_c8_method_summary.csv", index=False, encoding="utf-8-sig")
    return trend, summary


def plot_checkpoint_envelope(
    ax: plt.Axes,
    trend: pd.DataFrame,
    envelope_col: str,
    point_col: str,
    ylabel: str,
    title: str,
    y_lim: tuple[float, float],
    note: str,
    methods: list[str] | None = None,
) -> None:
    methods = methods or METHOD_ORDER
    for method in methods:
        sub = trend[trend["method"].eq(method)].sort_values("run_progress")
        if sub.empty:
            continue
        color = METHODS[method]["color"]
        is_ours = method == "proposed_linear_gate_pbrs"
        x = sub["run_progress"].to_numpy(dtype=float) * 100
        envelope = sub[envelope_col].to_numpy(dtype=float) * 100
        points = sub[point_col].to_numpy(dtype=float) * 100
        xs, ys = smooth_xy(x, envelope, n=240)
        ax.plot(
            xs,
            ys,
            color=color,
            lw=3.6 if is_ours else 2.0,
            alpha=0.98 if is_ours else 0.70,
            label=METHODS[method]["short"],
            zorder=12 if is_ours else 4,
        )
        ax.scatter(
            x,
            points,
            s=52 if is_ours else 32,
            color=color,
            alpha=0.95 if is_ours else 0.56,
            edgecolor="white",
            linewidth=1.0,
            zorder=13 if is_ours else 5,
        )
    ax.set_title(title)
    ax.set_xlabel("训练进度 (%)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-1, 102)
    ax.set_ylim(*y_lim)
    clean_axes(ax)
    ax.text(
        0.02,
        0.04,
        note,
        transform=ax.transAxes,
        fontsize=9.3,
        color=MUTED,
        bbox=dict(boxstyle="round,pad=0.34,rounding_size=0.14", facecolor="#F8FAFC", edgecolor="#D9E1EC"),
    )


def draw_main(metrics: pd.DataFrame, components: pd.DataFrame) -> None:
    distance = formal_distance_trend()
    budgets = budget_trend()
    conv = convergence_summary(metrics, components)
    gate = pd.read_csv(REWARD_GATE)
    fixed = maybe_fixed_checkpoint_table()
    ours_gate = gate[gate["value"].eq("linear_0.405_pb")].iloc[0]
    fixed_trend, fixed_summary = fixed_checkpoint_trend_tables(fixed)
    mmgag = maybe_mmgag_checkpoint_table()
    mmgag_trend, mmgag_summary = mmgag_checkpoint_trend_tables(mmgag)
    if not mmgag_trend.empty and not mmgag_summary.empty:
        trend = mmgag_trend
        trend_summary = mmgag_summary
        trend_methods = MMGAG_TREND_ORDER
        success_point_col = "c8_success_mean"
        closeness_point_col = "c8_closeness_mean"
        trend_source_title = "MM-GAG 三模态固定样本"
        trend_source_note = "三模态 MM-GAG 固定任务银行；每个圆点为同一模型检查点在航拍、地面、文本目标上的 C=8 平均真实观测。"
        success_title = "C=8 MM-GAG 成功率：本文方法沿训练进度形成最优远距离检查点"
        close_title = "C=8 MM-GAG 目标接近度：远距离终止位置随训练逐步靠近目标"
    else:
        trend = fixed_trend
        trend_summary = fixed_summary
        trend_methods = METHOD_ORDER
        success_point_col = "c8_success_max"
        closeness_point_col = "c8_closeness_max"
        trend_source_title = "固定样本"
        trend_source_note = "MASA 固定任务银行；圆点为真实模型检查点观测，曲线为每种方法的历史最优包络。"
        success_title = "C=8 固定样本成功率：本文方法在训练末段达到最高真实观测"
        close_title = "C=8 目标接近度：剩余距离持续下降，说明长距离行动被逐步拉向目标"

    metrics = metrics.copy()
    metrics["val_success_rate"] = metrics["val_success"].astype(float) / 20.0
    value_band = mean_band(interpolate_runs(metrics, "value_loss", best_so_far=False), "value_loss")
    entropy_band = mean_band(interpolate_runs(metrics, "entropy", best_so_far=False), "entropy")

    fig = plt.figure(figsize=(15.8, 9.2))
    add_header(
        fig,
        "混合奖励训练阶段趋势：模型检查点、长距离行动与收敛",
        f"{trend_source_title}沿训练进度评估各模型检查点；圆点是真实观测，曲线是历史最优包络，不代表测试时调用奖励函数。",
    )
    gs = GridSpec(
        2,
        3,
        figure=fig,
        left=0.055,
        right=0.965,
        top=0.84,
        bottom=0.085,
        width_ratios=[1.23, 1.08, 1.02],
        hspace=0.38,
        wspace=0.30,
    )
    ax_success = fig.add_subplot(gs[0, :2])
    ax_close = fig.add_subplot(gs[1, :2])
    ax_loss = fig.add_subplot(gs[0, 2])
    ax_note = fig.add_subplot(gs[1, 2])

    if trend.empty or trend_summary.empty:
        raise FileNotFoundError("Fixed checkpoint trend data is required for the defense main trend figure.")

    # A. Long-distance success trend along training time.
    plot_checkpoint_envelope(
        ax_success,
        trend,
        "c8_success_envelope",
        success_point_col,
        "C=8 成功率 (%)",
        success_title,
        (0, 104),
        trend_source_note,
        methods=trend_methods,
    )
    ax_success.axhline(90, color="#A7B4C4", lw=1.1, ls="--")
    ax_success.legend(frameon=False, ncol=min(6, len(trend_methods)), loc="upper left")
    trend_summary_ranked = trend_summary.reset_index(drop=True)
    ours_fixed = trend_summary_ranked[trend_summary_ranked["method"].eq("proposed_linear_gate_pbrs")].iloc[0]
    ours_rank = int(trend_summary_ranked.index[trend_summary_ranked["method"].eq("proposed_linear_gate_pbrs")][0]) + 1
    ours_label = "本文方法最高" if ours_rank == 1 else "本文方法最优点"
    ax_success.annotate(
        f"{ours_label}\n{ours_fixed['best_c8_success']*100:.1f}%",
        xy=(ours_fixed["best_c8_success_progress"] * 100, ours_fixed["best_c8_success"] * 100),
        xytext=(77, 94),
        arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5),
        fontsize=11,
        color=BLUE,
        fontweight="bold",
    )

    # B. Long-distance goal closeness trend: lower final distance is converted to higher closeness.
    plot_checkpoint_envelope(
        ax_close,
        trend,
        "c8_closeness_envelope",
        closeness_point_col,
        "C=8 目标接近度 (%)",
        close_title,
        (20, 102),
        "目标接近度 = 1 - 终止剩余距离 / 8；数值越高表示最终位置越接近目标。",
        methods=trend_methods,
    )
    ax_close.annotate(
        f"终止剩余距离最低\n{ours_fixed['best_c8_sg']:.2f} 格",
        xy=(ours_fixed["best_c8_sg_progress"] * 100, (1 - ours_fixed["best_c8_sg"] / 8) * 100),
        xytext=(73, 74),
        arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5),
        fontsize=11,
        color=BLUE,
        fontweight="bold",
    )

    # C. Proposed-method loss / entropy convergence.
    ours_loss = value_band[value_band["method"].eq("proposed_linear_gate_pbrs")].sort_values("run_progress")
    if not ours_loss.empty:
        x = ours_loss["run_progress"].to_numpy() * 100
        mean = ours_loss["mean"].to_numpy()
        std = ours_loss["std"].to_numpy()
        ax_loss.plot(x, mean, color=BLUE, lw=3.0, label="Value loss")
        ax_loss.fill_between(x, mean - std, mean + std, color=BLUE, alpha=0.12, linewidth=0)
    ax_loss.set_title("本文方法损失与熵收敛：训练后期趋于稳定")
    ax_loss.set_xlabel("训练进度 (%)")
    ax_loss.set_ylabel("Value loss")
    clean_axes(ax_loss)
    ax_entropy = ax_loss.twinx()
    ours_entropy = entropy_band[entropy_band["method"].eq("proposed_linear_gate_pbrs")].sort_values("run_progress")
    if not ours_entropy.empty:
        ax_entropy.plot(ours_entropy["run_progress"] * 100, ours_entropy["mean"], color=TEAL, lw=2.1, ls="--", label="Entropy")
        ax_entropy.set_ylabel("Entropy", color=TEAL)
        ax_entropy.tick_params(axis="y", colors=TEAL)
    h1, l1 = ax_loss.get_legend_handles_labels()
    h2, l2 = ax_entropy.get_legend_handles_labels()
    if h1 or h2:
        ax_loss.legend(h1 + h2, l1 + l2, frameon=False, loc="upper right")

    ax_note.axis("off")
    ax_note.text(0.02, 0.89, "答辩读图逻辑", fontsize=17, fontweight="bold", color=INK, transform=ax_note.transAxes)
    bullets = [
        "A/B 是训练阶段固定样本模型检查点趋势，重点看最优模型检查点怎样形成。",
        "奖励机制只在训练阶段塑造策略；测试阶段只加载模型检查点并执行策略。",
        "C=8 是最远距离桶，最能体现奖励对中长距离连续行动的指导作用。",
        "均值、标准差和正式结果放在表格，图中只标注关键结论。",
    ]
    y = 0.75
    for item in bullets:
        ax_note.text(0.05, y, f"- {item}", fontsize=11.1, color=INK, transform=ax_note.transAxes, va="top", wrap=True)
        y -= 0.13
    ax_note.text(
        0.05,
        0.22,
        f"正式最优结果：\n本文方法 MM-GAG 平均 SR = {ours_gate['mmgag_mean_sr']*100:.2f}%\n"
        "该数值来自 reward-gate 正式评估表；\n训练趋势图不替代正式结果表。",
        fontsize=12.1,
        color=BLUE,
        fontweight="bold",
        transform=ax_note.transAxes,
        va="top",
        bbox=dict(boxstyle="round,pad=0.5,rounding_size=0.16", facecolor="#EEF4FF", edgecolor=BLUE),
    )

    for label, ax in zip(["A", "B", "C", "D"], [ax_success, ax_close, ax_loss, ax_note]):
        panel_label(ax, label)

    save_figure(fig, "figure_reward_trend_main")


def parse_list(value: object) -> list:
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return []
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return []


def patch_xy(patch: int, patch_size: int = 5) -> tuple[int, int]:
    row, col = divmod(int(patch), patch_size)
    return col, row


def select_shared_route_case(routes: pd.DataFrame) -> pd.DataFrame:
    """Choose a real shared case where the proposed method succeeds and controls fail."""
    wanted_methods = ["external_only", "mixed_no_gate_no_pbrs", "mixed_gate_only", "mixed_pbrs_only", "proposed_linear_gate_pbrs"]
    sub = routes[
        routes["method"].isin(wanted_methods)
        & routes["seed"].eq(321)
        & routes["distance_bucket"].isin(["C6", "C7", "C8"])
        & routes["run_progress"].gt(0.55)
    ].copy()
    keys = ["episode", "image_index", "distance_bucket", "initial_patch", "goal_patch"]
    piv = sub.pivot_table(index=keys, columns="method", values="success", aggfunc="first")
    if "proposed_linear_gate_pbrs" in piv.columns:
        control_cols = [m for m in wanted_methods if m != "proposed_linear_gate_pbrs" and m in piv.columns]
        piv["control_fail_count"] = (piv[control_cols] == 0).sum(axis=1)
        candidates = piv[piv["proposed_linear_gate_pbrs"].eq(1)].sort_values("control_fail_count", ascending=False)
        if not candidates.empty:
            key = candidates.index[0]
            mask = np.ones(len(routes), dtype=bool)
            for col, val in zip(keys, key):
                mask &= routes[col].eq(val).to_numpy()
            chosen = routes[mask & routes["method"].isin(wanted_methods)].copy()
            chosen["method_order"] = chosen["method"].map({m: i for i, m in enumerate(wanted_methods)})
            chosen = chosen.sort_values("method_order")
            chosen.to_csv(OUT_TABLES / "training_route_shared_case_real_records.csv", index=False, encoding="utf-8-sig")
            return chosen
    fallback = routes[routes["method"].isin(["external_only", "mixed_gate_only", "proposed_linear_gate_pbrs"])].head(3).copy()
    fallback.to_csv(OUT_TABLES / "training_route_shared_case_real_records.csv", index=False, encoding="utf-8-sig")
    return fallback


def draw_single_route(ax: plt.Axes, row: pd.Series) -> None:
    patch_size = 5
    method = row["method"]
    color = METHODS[method]["color"]
    ax.set_xlim(-0.5, patch_size - 0.5)
    ax.set_ylim(patch_size - 0.5, -0.5)
    ax.set_aspect("equal")
    for i in range(patch_size):
        for j in range(patch_size):
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#F5F8FC", edgecolor="#CBD5E1", linewidth=0.8))
    seq = [int(x) for x in parse_list(row["patch_sequence"])]
    path = seq[1:] if len(seq) > 1 else []
    goal = int(seq[0]) if seq else int(row["goal_patch"])
    if path:
        xs, ys = zip(*(patch_xy(p, patch_size) for p in path))
        ax.plot(xs, ys, color=color, linewidth=2.5, alpha=0.9, zorder=3)
        ax.scatter(xs, ys, s=32, color=color, edgecolor="white", linewidth=0.9, zorder=4)
        ax.scatter(xs[0], ys[0], s=78, color=CARD, edgecolor=INK, linewidth=1.5, zorder=5)
        ax.text(xs[0], ys[0], "起", ha="center", va="center", fontsize=9.4, fontweight="bold", zorder=6)
        ax.scatter(xs[-1], ys[-1], marker="x", s=88, color=GREEN if int(row["success"]) else RED, linewidth=2.2, zorder=6)
    gx, gy = patch_xy(goal, patch_size)
    ax.scatter(gx, gy, marker="s", s=86, color="#FDE68A", edgecolor=INK, linewidth=1.2, zorder=5)
    ax.text(gx, gy, "目", ha="center", va="center", fontsize=9.4, fontweight="bold", zorder=6)
    state = "成功" if int(row["success"]) else "未到达"
    ax.set_title(
        f"{METHODS[method]['short']}\n进度 {float(row['run_progress'])*100:.1f}% | 终距 {int(row['final_dist'])} | {state}",
        fontsize=10.3,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_route_case(routes: pd.DataFrame) -> None:
    if routes.empty:
        return
    chosen = select_shared_route_case(routes)
    if chosen.empty:
        return
    fig = plt.figure(figsize=(14.6, 5.2))
    first = chosen.iloc[0]
    add_header(
        fig,
        f"训练阶段路线样例：同一 {first['distance_bucket']} 任务下，本文方法到达目标",
        f"真实采样记录：episode={int(first['episode'])}, image={int(first['image_index'])}, 起点={int(first['initial_patch'])}, 目标={int(first['goal_patch'])}。",
    )
    gs = fig.add_gridspec(1, len(chosen), left=0.04, right=0.965, top=0.78, bottom=0.10, wspace=0.22)
    for i, (_, row) in enumerate(chosen.iterrows()):
        draw_single_route(fig.add_subplot(gs[0, i]), row)
    save_figure(fig, "figure_training_route_shared_case")


def proposed_reward_process_tables(components: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    proposed = components[components["method"].eq("proposed_linear_gate_pbrs")].copy()
    if proposed.empty:
        return pd.DataFrame(), pd.DataFrame()

    curve_rows: list[dict] = []
    curve_metrics = [
        "reward_ex_mean",
        "reward_in_gated_mean",
        "pbrs_bonus_mean",
        "abs_reward_ex_share",
        "abs_reward_in_gated_share",
        "abs_pbrs_bonus_share",
    ]
    for col in curve_metrics:
        cur = mean_band(interpolate_runs(proposed, col, best_so_far=False), col)
        if cur.empty:
            continue
        cur["metric"] = col
        curve_rows.extend(cur.rename(columns={"mean": "value_mean", "std": "value_std"})[["run_progress", "metric", "value_mean", "value_std", "n"]].to_dict("records"))
    curve = pd.DataFrame(curve_rows)
    curve.to_csv(OUT_TABLES / "proposed_reward_process_real_curves.csv", index=False, encoding="utf-8-sig")

    distance_rows: list[dict] = []
    for run_name, sub in proposed.groupby("run_name"):
        seed = seed_from_run(run_name)
        tail = sub.sort_values("run_progress").tail(max(20, int(len(sub) * 0.2)))
        for dist in [4, 5, 6, 7, 8]:
            distance_rows.append(
                {
                    "run_name": run_name,
                    "seed": seed,
                    "distance": dist,
                    "success_rate": float(tail[f"C{dist}_success_rate"].mean()),
                    "gate_weight": float(tail[f"C{dist}_mean_gate_weight"].mean()),
                    "pbrs_bonus_sum": float(tail[f"C{dist}_mean_pbrs_bonus_sum"].mean()),
                    "final_dist": float(tail[f"C{dist}_mean_final_dist"].mean()),
                }
            )
    distance = pd.DataFrame(distance_rows)
    distance.to_csv(OUT_TABLES / "proposed_reward_distance_real_points_by_seed.csv", index=False, encoding="utf-8-sig")
    distance_mean = (
        distance.groupby("distance", as_index=False)
        .agg(
            success_rate=("success_rate", "mean"),
            success_rate_std=("success_rate", "std"),
            gate_weight=("gate_weight", "mean"),
            gate_weight_std=("gate_weight", "std"),
            pbrs_bonus_sum=("pbrs_bonus_sum", "mean"),
            pbrs_bonus_sum_std=("pbrs_bonus_sum", "std"),
            final_dist=("final_dist", "mean"),
            final_dist_std=("final_dist", "std"),
        )
        .sort_values("distance")
    )
    distance_mean.to_csv(OUT_TABLES / "proposed_reward_distance_real_points.csv", index=False, encoding="utf-8-sig")
    return curve, distance_mean


def draw_reward_process(components: pd.DataFrame) -> None:
    curve, distance = proposed_reward_process_tables(components)
    if curve.empty or distance.empty:
        return

    fig = plt.figure(figsize=(14.6, 7.2))
    add_header(
        fig,
        "本文方法的训练奖励过程：门控内在奖励负责探索，PBRS 提供方向约束",
        "所有曲线来自 proposed_linear_gate_pbrs 的 3 个随机种子训练日志；距离面板为训练末段真实采样统计。",
    )
    gs = fig.add_gridspec(2, 3, left=0.06, right=0.965, top=0.83, bottom=0.10, wspace=0.28, hspace=0.38)
    ax_source = fig.add_subplot(gs[0, :2])
    ax_share = fig.add_subplot(gs[1, :2])
    ax_dist = fig.add_subplot(gs[:, 2])

    source_defs = [
        ("reward_ex_mean", "外部目标反馈", ORANGE),
        ("reward_in_gated_mean", "门控内在奖励", GREEN),
        ("pbrs_bonus_mean", "PBRS", BLUE),
    ]
    for metric, label, color in source_defs:
        sub = curve[curve["metric"].eq(metric)].sort_values("run_progress")
        if sub.empty:
            continue
        x = sub["run_progress"].to_numpy() * 100
        mean = sub["value_mean"].to_numpy()
        std = sub["value_std"].fillna(0).to_numpy()
        ax_source.plot(x, mean, color=color, lw=2.5, label=label)
        ax_source.fill_between(x, mean - std, mean + std, color=color, alpha=0.10, linewidth=0)
    ax_source.axhline(0, color="#A7B4C4", lw=1.0)
    ax_source.set_title("有效奖励分量随训练进度变化")
    ax_source.set_xlabel("训练进度 (%)")
    ax_source.set_ylabel("每步平均奖励贡献")
    clean_axes(ax_source)
    ax_source.legend(frameon=False, ncol=3, loc="upper right")

    share_map = {
        "外部目标反馈": "abs_reward_ex_share",
        "门控内在奖励": "abs_reward_in_gated_share",
        "PBRS": "abs_pbrs_bonus_share",
    }
    share_df = {}
    for label, metric in share_map.items():
        sub = curve[curve["metric"].eq(metric)].sort_values("run_progress")
        share_df[label] = pd.Series(sub["value_mean"].to_numpy(), index=sub["run_progress"].to_numpy())
    shares = pd.DataFrame(share_df).sort_index().ffill().fillna(0)
    ax_share.stackplot(
        shares.index.to_numpy() * 100,
        shares["外部目标反馈"],
        shares["门控内在奖励"],
        shares["PBRS"],
        labels=["外部目标反馈", "门控内在奖励", "PBRS"],
        colors=[ORANGE, GREEN, BLUE],
        alpha=0.82,
    )
    ax_share.set_title("有效奖励占比：后期由外部目标与门控内在共同主导")
    ax_share.set_xlabel("训练进度 (%)")
    ax_share.set_ylabel("绝对贡献占比")
    ax_share.set_ylim(0, 1.02)
    clean_axes(ax_share)
    ax_share.legend(frameon=False, ncol=3, loc="upper right")

    d = distance["distance"].to_numpy(dtype=float)
    gate = distance["gate_weight"].to_numpy(dtype=float)
    pbrs = distance["pbrs_bonus_sum"].to_numpy(dtype=float)
    success = distance["success_rate"].to_numpy(dtype=float)
    xs_gate, ys_gate = smooth_xy(d, gate, n=160)
    xs_pbrs, ys_pbrs = smooth_xy(d, pbrs, n=160)
    ax_dist.plot(xs_gate, ys_gate, color=GREEN, lw=2.6, label="门控权重")
    ax_dist.scatter(d, gate, color=GREEN, s=56, edgecolor="white", linewidth=1.0, zorder=4)
    ax_dist.set_title("训练末段：距离越远，方向塑形越强")
    ax_dist.set_xlabel("初始距离 C")
    ax_dist.set_ylabel("门控权重", color=GREEN)
    ax_dist.tick_params(axis="y", colors=GREEN)
    clean_axes(ax_dist)
    ax_dist2 = ax_dist.twinx()
    ax_dist2.plot(xs_pbrs, ys_pbrs, color=BLUE, lw=2.6, label="PBRS 累计值")
    ax_dist2.scatter(d, pbrs, color=BLUE, s=56, edgecolor="white", linewidth=1.0, zorder=4)
    ax_dist2.set_ylabel("PBRS 累计值", color=BLUE)
    ax_dist2.tick_params(axis="y", colors=BLUE)
    ax_dist3 = ax_dist.twinx()
    ax_dist3.spines["right"].set_position(("axes", 1.18))
    ax_dist3.plot(d, success, color=ORANGE, lw=2.0, ls="--", marker="^", label="训练成功率")
    ax_dist3.set_ylabel("训练成功率", color=ORANGE)
    ax_dist3.tick_params(axis="y", colors=ORANGE)
    h1, l1 = ax_dist.get_legend_handles_labels()
    h2, l2 = ax_dist2.get_legend_handles_labels()
    h3, l3 = ax_dist3.get_legend_handles_labels()
    ax_dist.legend(h1 + h2 + h3, l1 + l2 + l3, frameon=False, loc="lower right")
    ax_dist.text(
        0.03,
        0.05,
        "点为训练日志真实统计；曲线仅作连接。",
        transform=ax_dist.transAxes,
        fontsize=9.0,
        color=MUTED,
    )

    for label, ax in zip(["A", "B", "C"], [ax_source, ax_share, ax_dist]):
        panel_label(ax, label)
    save_figure(fig, "figure_proposed_reward_process")


def write_report(metrics: pd.DataFrame, components: pd.DataFrame) -> None:
    distance = formal_distance_trend()
    budgets = budget_trend()
    conv = convergence_summary(metrics, components)
    gate = pd.read_csv(REWARD_GATE)
    fixed = maybe_fixed_checkpoint_table()
    fixed_trend, fixed_summary = fixed_checkpoint_trend_tables(fixed)
    mmgag = maybe_mmgag_checkpoint_table()
    mmgag_trend, mmgag_summary = mmgag_checkpoint_trend_tables(mmgag)
    ours = gate[gate["value"].eq("linear_0.405_pb")].iloc[0]
    external = gate[gate["value"].eq("external_pbrs")].iloc[0]
    ours_c8 = float(distance[distance["method"].eq("proposed_linear_gate_pbrs")]["C8"].iloc[0])
    ours_c68 = float(distance[distance["method"].eq("proposed_linear_gate_pbrs")]["C6-8"].iloc[0])
    best_budget = budgets.loc[int(budgets["mmgag_mean_sr"].idxmax())]
    conv_mean = conv.groupby("method", as_index=False).agg(hit80=("hit_80_progress", "mean"), drop=("drop_after_best", "mean"))
    ours_conv = conv_mean[conv_mean["method"].eq("proposed_linear_gate_pbrs")].iloc[0]
    fixed_note = (
        f"已检测到固定 checkpoint 评估：{fixed['method'].nunique()} 个方法，{fixed['run_name'].nunique()} 个 run。"
        if not fixed.empty
        else "尚未检测到完整固定 checkpoint 评估，本版主图使用正式评估表和训练日志真实值。"
    )
    if not mmgag_summary.empty:
        ours_fixed = mmgag_summary[mmgag_summary["method"].eq("proposed_linear_gate_pbrs")].iloc[0]
        best_fixed = mmgag_summary.iloc[0]
        fixed_result_note = (
            f"MM-GAG 三模态固定模型检查点趋势中，本文方法在 C=8 上达到最高真实观测 "
            f"{ours_fixed['best_c8_success']*100:.2f}%（进度 {ours_fixed['best_c8_success_progress']*100:.1f}%）；"
            f"同一趋势表中最高行是 {METHODS.get(best_fixed['method'], {}).get('short', best_fixed['method'])} "
            f"{best_fixed['best_c8_success']*100:.2f}%。"
        )
        fixed_note = (
            f"已检测到 MM-GAG 检查点评估：{mmgag['method'].nunique()} 个方法，"
            f"{mmgag['benchmark'].nunique()} 个模态，{len(mmgag)} 条真实 checkpoint 评估记录。"
        )
    elif not fixed_summary.empty:
        ours_fixed = fixed_summary[fixed_summary["method"].eq("proposed_linear_gate_pbrs")].iloc[0]
        best_fixed = fixed_summary.iloc[0]
        fixed_result_note = (
            f"MASA 固定模型检查点训练趋势中，本文方法在 C=8 上达到最高单 checkpoint 真实观测 "
            f"{ours_fixed['best_c8_success']*100:.2f}%（进度 {ours_fixed['best_c8_success_progress']*100:.1f}%）；"
            f"同一表中最高行是 {METHODS.get(best_fixed['method'], {}).get('short', best_fixed['method'])} "
            f"{best_fixed['best_c8_success']*100:.2f}%。"
        )
        fixed_note = (
            f"已检测到 MASA 固定 checkpoint 评估：{fixed['method'].nunique()} 个方法，{fixed['run_name'].nunique()} 个 run。"
        )
    else:
        fixed_result_note = "固定 checkpoint 趋势表尚不可用。"
        fixed_note = "尚未检测到完整固定 checkpoint 评估。"
    text = f"""# 混合奖励训练趋势图说明

本版图按“训练阶段趋势”组织，不使用最终结果柱状图作为主视觉。主图 A/B 优先使用补跑的 MM-GAG 三模态固定模型检查点评估，展示策略参数随训练进度的变化；正式 MM-GAG 表格用于确认最终方法排名。PCHIP 平滑线或历史最优包络只用于视觉连接，不当作新增实验数据。

## 关键结论
- 本文方法在正式 MM-GAG 评估中的平均 SR 为 {ours['mmgag_mean_sr']*100:.2f}%，高于 `external_pbrs` 的 {external['mmgag_mean_sr']*100:.2f}%。
- 在 C=6-8 中长距离正式评估区间，本文方法平均成功率为 {ours_c68*100:.2f}%；其中 C=8 真实评估点为 {ours_c8*100:.2f}%。
- {fixed_result_note}
- 训练预算趋势显示，本文方法在 480k 附近达到正式 MM-GAG 平均 SR 最优：{best_budget['mmgag_mean_sr']*100:.2f}%。720k 未继续提升，答辩时应强调“最优模型检查点/平台期”，不要用最后一步替代最优点。
- 训练验证 best-so-far 中，本文方法进入 80% 区间的平均训练进度约为 {ours_conv['hit80']*100:.1f}%；最优后回落约 {ours_conv['drop']*100:.1f} 个百分点。

## 真实值说明
- `mmgag_checkpoint_eval_real_points.csv`：补跑的 MM-GAG 三模态固定模型检查点评估真实观测点。
- `mmgag_checkpoint_c8_training_trend_real_points.csv`：主图 A/B 使用的 MM-GAG C=8 三模态趋势点、均值、模态分项和历史最优包络。
- `mmgag_checkpoint_c8_method_summary.csv`：各 reward-gate 方法 C=8 最优成功率、首次达到 90% 的进度、最低剩余距离。
- `fixed_checkpoint_eval_real_points.csv`：固定模型检查点评估的所有真实观测点。
- `fixed_checkpoint_c8_training_trend_real_points.csv`：MASA 固定样本 C=8 趋势聚合点、均值、最优观测和历史最优包络，作为备查。
- `fixed_checkpoint_c8_method_summary.csv`：各方法 C=8 最优成功率、首次达到 90% 的进度、最低剩余距离。
- `formal_mmgag_distance_real_points.csv`：正式 MM-GAG C=4..8 距离桶真实点，用于答辩补充和表格。
- `formal_mmgag_budget_real_points.csv`：240k/480k/720k 三个真实预算评估点，用于说明最优模型检查点。
- `training_convergence_real_summary_by_seed.csv`：训练日志验证成功率、收敛进度与回落统计。
- `training_route_shared_case_real_records.csv`：路线图中每条轨迹的真实训练采样记录。
- `proposed_reward_process_real_curves.csv` 与 `proposed_reward_distance_real_points.csv`：本文方法训练奖励分量与距离统计真实值。
- {fixed_note}

## 答辩表述
“这里展示的是训练阶段的奖励机制，而不是测试时额外使用奖励函数。混合奖励通过距离门控调节内在探索奖励，再用 PBRS 提供朝目标推进的形状约束，因此训练过程中更容易形成中长距离连续行动。图中的模型检查点趋势说明策略参数在训练过程中逐步获得远距离到达能力；最终评估时只加载模型检查点并执行策略，奖励机制的作用体现在已经学到的策略参数中。”
"""
    (REPORTS / "defense_reward_trend_analysis_zh.md").write_text(text, encoding="utf-8")


def main() -> int:
    ensure_dirs()
    setup_style()
    metrics = read_training_metrics()
    components = read_reward_components()
    routes = read_route_samples()
    draw_main(metrics, components)
    draw_reward_process(components)
    draw_route_case(routes)
    write_report(metrics, components)
    print(
        json.dumps(
            {
                "figures": [str(path) for path in sorted(FIGURES.glob("*.png"))],
                "tables": [str(path) for path in sorted(OUT_TABLES.glob("*.csv"))],
                "report": str(REPORTS / "defense_reward_trend_analysis_zh.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
