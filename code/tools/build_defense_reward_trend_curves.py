#!/usr/bin/env python
"""Build defense trend figures for reward-mechanism training analysis.

Main design:
- Use trend curves, not endpoint bar charts.
- Prefer fixed checkpoint evaluation curves when available.
- Use best-so-far envelopes to avoid over-interpreting late training rollback.
- Keep final result numbers as annotations/tables, not the main visual form.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
REPORTS = RESULTS / "reports"
FIGURES = RESULTS / "figures" / "defense_reward_trends"
OUT_TABLES = TABLES / "defense_reward_trends"

TRAIN_LOG_ROOT = Path(r"F:\bishe\GeoExplorer\analysis\pipeline_20260603_defense_reward_training_curves\training_logs")
FIXED_EVAL_ALLCKPT = Path(
    r"F:\bishe\GeoExplorer\analysis\pipeline_20260604_fixed_checkpoint_reward_eval_allckpt\fixed_checkpoint_eval_all.csv"
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
GRAY = "#808A98"
LIGHT_BLUE = "#DCEBFA"

METHODS = {
    "external_only": {"short": "外部奖励", "label": "仅外部奖励", "color": ORANGE},
    "intrinsic_only": {"short": "内在奖励", "label": "仅内在奖励", "color": PURPLE},
    "mixed_no_gate_no_pbrs": {"short": "直接相加", "label": "外部+内在直接相加", "color": GREEN},
    "mixed_gate_only": {"short": "门控内在", "label": "外部+门控内在", "color": TEAL},
    "mixed_pbrs_only": {"short": "仅加 PBRS", "label": "外部+内在+PBRS", "color": RED},
    "proposed_linear_gate_pbrs": {"short": "本文方法", "label": "线性门控+PBRS", "color": BLUE},
}

METHOD_ORDER = [
    "external_only",
    "intrinsic_only",
    "mixed_no_gate_no_pbrs",
    "mixed_gate_only",
    "mixed_pbrs_only",
    "proposed_linear_gate_pbrs",
]


def ensure_dirs() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)


def setup_style() -> None:
    for font in [r"C:\Windows\Fonts\times.ttf", r"C:\Windows\Fonts\timesbd.ttf", r"C:\Windows\Fonts\simsun.ttc"]:
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
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
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
    ax.text(-0.085, 1.075, label, transform=ax.transAxes, fontsize=15, fontweight="bold", ha="left", va="top")


def add_header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.04, 0.968, title, fontsize=24, fontweight="bold", ha="left", va="top")
    fig.text(0.04, 0.928, subtitle, fontsize=11.8, color=MUTED, ha="left", va="top")
    fig.lines.append(plt.Line2D([0.04, 0.965], [0.895, 0.895], transform=fig.transFigure, color="#CCD6E2", lw=1.2))


def moving_average(series: pd.Series, window: int = 25) -> pd.Series:
    return series.astype(float).rolling(window=window, min_periods=1, center=True).mean()


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


def read_training_metrics() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in sorted(TRAIN_LOG_ROOT.glob("*/training_metrics.csv")):
        method = method_from_run(path.parent.name)
        seed = seed_from_run(path.parent.name)
        if method is None or seed is None:
            continue
        df = pd.read_csv(path)
        df["method"] = method
        df["seed"] = seed
        df["run_name"] = path.parent.name
        df["run_progress"] = df["time_step"].astype(float) / 480000.0
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No training metrics found under {TRAIN_LOG_ROOT}")
    return pd.concat(rows, ignore_index=True)


def read_training_components() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in sorted(TRAIN_LOG_ROOT.glob("*/training_reward_components.csv")):
        method = method_from_run(path.parent.name)
        seed = seed_from_run(path.parent.name)
        if method is None or seed is None:
            continue
        df = pd.read_csv(path)
        df["method"] = method
        df["seed"] = seed
        df["run_name"] = path.parent.name
        if "run_progress" not in df:
            df["run_progress"] = df["time_step"].astype(float) / 480000.0
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No reward component logs found under {TRAIN_LOG_ROOT}")
    return pd.concat(rows, ignore_index=True)


def interpolate_runs(df: pd.DataFrame, y_col: str, best_so_far: bool = False, points: int = 121) -> pd.DataFrame:
    out: list[dict] = []
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
        interp = np.interp(valid, x, y)
        out.extend(
            {"method": method, "seed": seed, "run_name": run_name, "run_progress": gx, y_col: gy}
            for gx, gy in zip(valid, interp)
        )
    return pd.DataFrame(out)


def mean_band(interp: pd.DataFrame, y_col: str) -> pd.DataFrame:
    return (
        interp.groupby(["method", "run_progress"], as_index=False)
        .agg(mean=(y_col, "mean"), std=(y_col, "std"), n=(y_col, "count"))
        .assign(std=lambda x: x["std"].fillna(0.0))
    )


def plot_band(ax: plt.Axes, band: pd.DataFrame, scale: float = 1.0, alpha: float = 0.10, label_suffix: str = "") -> None:
    for method in METHOD_ORDER:
        sub = band[band["method"].eq(method)].sort_values("run_progress")
        if sub.empty:
            continue
        color = METHODS[method]["color"]
        x = sub["run_progress"].to_numpy() * 100
        mean = sub["mean"].to_numpy() * scale
        std = sub["std"].to_numpy() * scale
        lw = 3.3 if method == "proposed_linear_gate_pbrs" else 1.9
        z = 10 if method == "proposed_linear_gate_pbrs" else 3
        ax.plot(x, mean, color=color, lw=lw, label=METHODS[method]["short"] + label_suffix, zorder=z)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=alpha if method == "proposed_linear_gate_pbrs" else alpha * 0.7, linewidth=0, zorder=z - 1)


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
        row = {"method": method, "run": run, "mean_sr": float(sub["sr"].mean())}
        for dist in [4, 5, 6, 7, 8]:
            row[f"C{dist}"] = float(sub[f"d{dist}"].mean())
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / "formal_mmgag_distance_trend.csv", index=False)
    return out


def checkpoint_eval_trend(components: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if FIXED_EVAL_ALLCKPT.exists() and FIXED_EVAL_ALLCKPT.stat().st_size > 0:
        df = pd.read_csv(FIXED_EVAL_ALLCKPT)
        if not df.empty and {"method", "seed", "run_progress", "success_ratio"}.issubset(df.columns):
            # Scheduled checkpoints form the real time trend; best/latest are endpoint diagnostics.
            scheduled = df[df["checkpoint_kind"].isin(["scheduled", "latest"])].copy()
            scheduled["run_name"] = scheduled["run_name"].astype(str)
            scheduled["run_progress"] = scheduled["run_progress"].astype(float).clip(0, 1)
            scheduled["c6_c8_eval_success"] = scheduled["success_ratio"].astype(float)
            interp = interpolate_runs(scheduled, "c6_c8_eval_success", best_so_far=True, points=101)
            out = mean_band(interp, "c6_c8_eval_success")
            out.to_csv(OUT_TABLES / "checkpoint_fixed_eval_best_so_far_trend.csv", index=False)
            return out, "固定评估 checkpoint 曲线"

    comp = components.copy()
    counts = comp[["C6_trajectory_count", "C7_trajectory_count", "C8_trajectory_count"]].astype(float).sum(axis=1)
    successes = comp[["C6_success_count", "C7_success_count", "C8_success_count"]].astype(float).sum(axis=1)
    comp["c6_c8_eval_success"] = successes / counts.replace(0, np.nan)
    interp = interpolate_runs(comp, "c6_c8_eval_success", best_so_far=True, points=101)
    out = mean_band(interp, "c6_c8_eval_success")
    out.to_csv(OUT_TABLES / "fallback_training_sample_c6_c8_best_so_far_trend.csv", index=False)
    return out, "训练样本 C=6-8 best-so-far 曲线（临时）"


def convergence_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        sub_m = metrics[metrics["method"].eq(method)]
        for seed, sub in sub_m.groupby("seed"):
            sub = sub.sort_values("run_progress").reset_index(drop=True)
            val = moving_average(sub["val_success"].astype(float) / 20.0, 25)
            best = val.cummax()
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "best_val": float(best.max()),
                    "final_val": float(val.tail(50).mean()),
                    "drop_after_best": float(best.max() - val.tail(50).mean()),
                    "best_progress": float(sub.loc[int(best.idxmax()), "run_progress"]),
                    "hit_80_progress": float(sub.loc[int(best[best >= 0.80].index[0]), "run_progress"]) if (best >= 0.80).any() else np.nan,
                    "hit_85_progress": float(sub.loc[int(best[best >= 0.85].index[0]), "run_progress"]) if (best >= 0.85).any() else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / "training_convergence_summary_by_seed.csv", index=False)
    return out


def draw_main_trend_figure(metrics: pd.DataFrame, components: pd.DataFrame) -> None:
    distance = formal_distance_trend()
    conv = convergence_summary(metrics)
    ckpt_band, ckpt_source = checkpoint_eval_trend(components)
    gate = pd.read_csv(REWARD_GATE)
    ours_row = gate[gate["value"].eq("linear_0.405_pb")].iloc[0]

    fig = plt.figure(figsize=(15.8, 9.0))
    add_header(
        fig,
        "混合奖励训练趋势：更快进入有效区间，并在中长距离形成最优策略",
        "主图使用趋势曲线展示收敛速度、最优 checkpoint、距离难度变化和 loss 收敛；最终数值只作为标注，不做柱状主图。",
    )
    gs = GridSpec(2, 3, figure=fig, left=0.055, right=0.965, top=0.84, bottom=0.085, width_ratios=[1.18, 1.08, 1.05], hspace=0.38, wspace=0.30)
    ax_dist = fig.add_subplot(gs[0, :2])
    ax_val = fig.add_subplot(gs[1, 0])
    ax_ckpt = fig.add_subplot(gs[1, 1])
    ax_loss = fig.add_subplot(gs[0, 2])
    ax_note = fig.add_subplot(gs[1, 2])

    # A. Formal distance trend: not a bar chart, but a difficulty trend.
    x = np.array([4, 5, 6, 7, 8])
    for method in ["intrinsic_only", "external_only", "mixed_no_gate_no_pbrs", "mixed_gate_only", "proposed_linear_gate_pbrs"]:
        row = distance[distance["method"].eq(method)].iloc[0]
        y = np.array([row[f"C{d}"] for d in x]) * 100
        color = METHODS[method]["color"]
        lw = 3.6 if method == "proposed_linear_gate_pbrs" else 2.0
        z = 10 if method == "proposed_linear_gate_pbrs" else 3
        ax_dist.plot(x, y, marker="o", ms=7 if method == "proposed_linear_gate_pbrs" else 5, lw=lw, color=color, label=METHODS[method]["short"], zorder=z)
    ax_dist.axvspan(6, 8, color=LIGHT_BLUE, alpha=0.45, zorder=0)
    ax_dist.text(6.05, 96, "中长距离重点区间", color=BLUE, fontsize=10.5, va="top")
    ax_dist.set_title("正式 MM-GAG 难度趋势：C=6-8 区间本文方法最有解释力")
    ax_dist.set_xlabel("起点到目标距离 C")
    ax_dist.set_ylabel("三模态平均成功率（%）")
    ax_dist.set_xticks(x)
    ax_dist.set_ylim(0, 100)
    clean_axes(ax_dist)
    ax_dist.legend(frameon=False, ncol=5, loc="lower right")
    ax_dist.annotate(
        f"C=8 达到 {distance[distance['method'].eq('proposed_linear_gate_pbrs')]['C8'].iloc[0]*100:.1f}%",
        xy=(8, distance[distance["method"].eq("proposed_linear_gate_pbrs")]["C8"].iloc[0] * 100),
        xytext=(7.05, 88),
        arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.4),
        fontsize=11,
        color=BLUE,
        fontweight="bold",
    )

    # B. Training best-so-far validation curve.
    val_df = metrics.copy()
    val_df["val_success_rate"] = val_df["val_success"].astype(float) / 20.0
    val_band = mean_band(interpolate_runs(val_df, "val_success_rate", best_so_far=True, points=121), "val_success_rate")
    plot_band(ax_val, val_band, scale=100, alpha=0.08)
    ax_val.axhline(80, color="#A7B4C4", ls="--", lw=1.1)
    ax_val.set_title("训练验证 best-so-far：避免末期回落误导")
    ax_val.set_xlabel("训练进度（%）")
    ax_val.set_ylabel("验证成功率包络（%）")
    ax_val.set_ylim(0, 104)
    clean_axes(ax_val)
    ax_val.legend(frameon=False, ncol=2, loc="lower right")

    conv_mean = conv.groupby("method", as_index=False).agg(hit80=("hit_80_progress", "mean"), drop=("drop_after_best", "mean"))
    ours_conv = conv_mean[conv_mean["method"].eq("proposed_linear_gate_pbrs")].iloc[0]
    if np.isfinite(ours_conv["hit80"]):
        ax_val.axvline(ours_conv["hit80"] * 100, color=BLUE, ls=":", lw=1.4)
        ax_val.text(ours_conv["hit80"] * 100 + 1, 8, "本文方法进入 80% 区间", color=BLUE, fontsize=9.5, rotation=90, va="bottom")

    # C. Fixed checkpoint evaluation trend or fallback training sample trend.
    plot_band(ax_ckpt, ckpt_band, scale=100, alpha=0.08)
    ax_ckpt.set_title(ckpt_source)
    ax_ckpt.set_xlabel("训练进度（%）")
    ax_ckpt.set_ylabel("C=6-8 成功率包络（%）")
    ax_ckpt.set_ylim(0, 100 if "固定评估" in ckpt_source else 62)
    clean_axes(ax_ckpt)
    ax_ckpt.text(
        0.02,
        0.04,
        "使用 best-so-far 包络线展示“训练到此时已经学到的最好策略”，\n避免最后若干步过拟合或震荡遮住最优 checkpoint。",
        transform=ax_ckpt.transAxes,
        fontsize=9.3,
        color=MUTED,
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.15", facecolor="#F8FAFC", edgecolor="#D9E1EC"),
    )

    # D. Loss / entropy convergence.
    loss_df = metrics.copy()
    value_interp = mean_band(interpolate_runs(loss_df, "value_loss", best_so_far=False, points=121), "value_loss")
    entropy_interp = mean_band(interpolate_runs(loss_df, "entropy", best_so_far=False, points=121), "entropy")
    for method in ["external_only", "mixed_no_gate_no_pbrs", "mixed_gate_only", "proposed_linear_gate_pbrs"]:
        sub = value_interp[value_interp["method"].eq(method)].sort_values("run_progress")
        if sub.empty:
            continue
        ax_loss.plot(sub["run_progress"] * 100, sub["mean"], color=METHODS[method]["color"], lw=3.0 if method == "proposed_linear_gate_pbrs" else 1.8, label=METHODS[method]["short"])
    ax_loss.set_title("Value loss 收敛：后期进入稳定低波动区间")
    ax_loss.set_xlabel("训练进度（%）")
    ax_loss.set_ylabel("Value loss")
    clean_axes(ax_loss)
    ax2 = ax_loss.twinx()
    ours_entropy = entropy_interp[entropy_interp["method"].eq("proposed_linear_gate_pbrs")].sort_values("run_progress")
    if not ours_entropy.empty:
        ax2.plot(ours_entropy["run_progress"] * 100, ours_entropy["mean"], color=BLUE, lw=1.7, ls="--", alpha=0.85, label="本文方法 entropy")
    ax2.set_ylabel("Entropy（本文方法）", color=BLUE)
    ax2.tick_params(axis="y", colors=BLUE)
    ax_loss.legend(frameon=False, loc="upper right")

    # E. Explanation note and final number.
    ax_note.axis("off")
    ax_note.text(0.02, 0.88, "图的读法", fontsize=17, fontweight="bold", color=INK, transform=ax_note.transAxes)
    bullets = [
        "不是看最后一个训练点，而是看 best-so-far 包络线和最优 checkpoint。",
        "C=6-8 是中长距离区间，最能体现混合奖励对行动序列的指导。",
        "loss/entropy 用于说明训练已收敛；最终优劣由同协议 MM-GAG 曲线和表格确认。",
    ]
    y = 0.74
    for item in bullets:
        ax_note.text(0.05, y, f"• {item}", fontsize=11.2, color=INK, transform=ax_note.transAxes, va="top", wrap=True)
        y -= 0.17
    ax_note.text(
        0.05,
        0.22,
        f"正式最优结果：本文方法 MM-GAG 平均 SR = {ours_row['mmgag_mean_sr']*100:.2f}%\n该数字放在说明中，不用柱状图抢主视觉。",
        fontsize=12.3,
        color=BLUE,
        fontweight="bold",
        transform=ax_note.transAxes,
        va="top",
        bbox=dict(boxstyle="round,pad=0.5,rounding_size=0.16", facecolor="#EEF4FF", edgecolor=BLUE),
    )

    for label, ax in zip(["A", "B", "C", "D", "E"], [ax_dist, ax_val, ax_ckpt, ax_loss, ax_note]):
        panel_label(ax, label)

    save_figure(fig, "figure_reward_trend_main")


def write_report(metrics: pd.DataFrame, components: pd.DataFrame) -> None:
    distance = formal_distance_trend()
    conv = convergence_summary(metrics)
    gate = pd.read_csv(REWARD_GATE)
    ours = gate[gate["value"].eq("linear_0.405_pb")].iloc[0]
    external = gate[gate["value"].eq("external_pbrs")].iloc[0]
    source = "固定 checkpoint 评估" if FIXED_EVAL_ALLCKPT.exists() and FIXED_EVAL_ALLCKPT.stat().st_size > 0 else "训练日志 best-so-far 临时曲线"
    ours_dist = distance[distance["method"].eq("proposed_linear_gate_pbrs")].iloc[0]
    conv_mean = conv.groupby("method", as_index=False).agg(hit80=("hit_80_progress", "mean"), drop=("drop_after_best", "mean"))
    ours_conv = conv_mean[conv_mean["method"].eq("proposed_linear_gate_pbrs")].iloc[0]
    text = f"""# 混合奖励训练趋势图说明

## 图形定位

本图不是最终结果柱状图，而是多方法训练趋势图。它回答三个问题：

1. 哪种奖励机制更快进入有效训练区间；
2. 哪个 checkpoint 附近达到最优，后期是否存在回落；
3. 中长距离 C=6-8 上，训练出的策略是否更有优势。

## 关键读法

- 训练阶段曲线使用 `best-so-far` 包络线，而不是直接使用最后一个训练点。这是为了避免后期震荡或过拟合把最优 checkpoint 的效果遮住。
- 固定评估曲线来源：{source}。
- 正式 MM-GAG 难度趋势显示，本文方法在 C=8 上达到 {ours_dist['C8']*100:.1f}%，MM-GAG 三模态平均 SR 为 {ours['mmgag_mean_sr']*100:.2f}%。
- 与 `external_pbrs` 的正式平均 SR 相比，本文方法高 {(ours['mmgag_mean_sr']-external['mmgag_mean_sr'])*100:.2f} 个百分点。
- 本文方法进入 80% 验证成功率区间的平均训练进度约为 {ours_conv['hit80']*100:.1f}%，最优后回落均值约为 {ours_conv['drop']*100:.1f} 个百分点，说明可以用 checkpoint 选择解释“训练后期不一定是最优点”。

## 答辩表述建议

可以这样讲：

“这里我没有直接画最后一步训练结果，因为强化学习训练后期会有震荡。图中使用 best-so-far 和固定 checkpoint 评估趋势，可以看到本文方法较早进入有效区间，并且在 C=6-8 中长距离区间保持更高的正式 MM-GAG 成功率。最终测试时不再调用奖励函数，奖励机制的影响体现在训练得到的 checkpoint 权重里。”

## 输出

- `results/figures/defense_reward_trends/figure_reward_trend_main.png`
- `results/figures/defense_reward_trends/figure_reward_trend_main.svg`
- `results/tables/defense_reward_trends/formal_mmgag_distance_trend.csv`
- `results/tables/defense_reward_trends/training_convergence_summary_by_seed.csv`
"""
    (REPORTS / "defense_reward_trend_analysis_zh.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    setup_style()
    metrics = read_training_metrics()
    components = read_training_components()
    draw_main_trend_figure(metrics, components)
    write_report(metrics, components)


if __name__ == "__main__":
    main()
