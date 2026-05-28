#!/usr/bin/env python
"""Build a polished GitHub showcase layer.

This script complements ``build_visual_showcase.py``.  It keeps the original
figures reproducible, but adds a higher-impact presentation layer with
consistent 16:9 figure cards, synchronized three-method GIFs, and a compact
manifest for the README/gallery.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

from PIL import Image, ImageDraw, ImageFont, ImageSequence


ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "results" / "tables"
SHOWCASE_DIR = ROOT / "results" / "figures" / "showcase"
POLISHED_DIR = SHOWCASE_DIR / "polished"
TRIPTYCH_DIR = SHOWCASE_DIR / "trajectories" / "triptych_gifs"

INK = "#111827"
MUTED = "#667085"
FAINT = "#E7E5DF"
PAPER = "#F8F7F2"
CARD = "#FFFDF7"
BLUE = "#1764AB"
SKY = "#4EA5D9"
ORANGE = "#D65F00"
GREEN = "#1B9E77"
RED = "#B33A3A"
GRAY = "#8C96A3"
VIOLET = "#7C6BB0"
GOLD = "#B88A00"

METHOD_COLOR = {
    "Ours": BLUE,
    "GeoExplorer-anchor0624": BLUE,
    "anchor0624": BLUE,
    "GOMAA-Geo": ORANGE,
    "gomaa": ORANGE,
    "GeoExplorer": GRAY,
    "GeoExplorer-pristine": GRAY,
    "pristine": GRAY,
    "Random": "#BFC5CE",
    "DiT-AGL": RED,
    "AiRLoc": GREEN,
}

METHOD_LABEL = {
    "GeoExplorer-anchor0624": "Ours",
    "anchor0624": "Ours",
    "GOMAA-Geo": "GOMAA-Geo",
    "gomaa": "GOMAA-Geo",
    "GeoExplorer-pristine": "GeoExplorer",
    "pristine": "GeoExplorer",
    "GeoExplorer": "GeoExplorer",
    "Random policy": "Random",
    "DiT-AGL": "DiT-AGL",
    "AiRLoc": "AiRLoc",
    "本文方法": "Ours",
}

METHOD_DISPLAY_LABEL = {
    "Ours": "本文方法",
    "Random": "随机策略",
}

BENCHMARK_DISPLAY_LABEL = {
    "masa_aerial": "MASA 航拍",
    "mmgag_aerial": "MM-GAG 航拍",
    "mmgag_ground": "MM-GAG 地面",
    "mmgag_text": "MM-GAG 文本",
    "swissview100_aerial": "SwissView100 航拍",
    "swissviewmonuments_aerial": "SwissMon 航拍",
    "swissviewmonuments_ground": "SwissMon 地面",
    "xbd_pre_aerial": "xBD 灾前",
    "xbd_disaster_aerial": "xBD 灾后",
    "masa_aerial_8x8": "长距离 8x8",
    "masa_aerial_10x10": "长距离 10x10",
}

FAMILY_DISPLAY_LABEL = {
    "mmgag": "MM-GAG",
    "ultra_long": "长距离",
}


def read_csv(rel: str) -> pd.DataFrame:
    return pd.read_csv(TABLE_DIR / rel)


def normalize_method(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    return METHOD_LABEL.get(text, text)


def display_method(value: object) -> str:
    text = normalize_method(value)
    return METHOD_DISPLAY_LABEL.get(text, text)


def display_benchmark(value: object) -> str:
    text = str(value)
    return BENCHMARK_DISPLAY_LABEL.get(text, text.replace("_", " "))


def display_seed_setting(row: pd.Series) -> str:
    family = str(row["family"])
    benchmark = str(row["benchmark"])
    if family == "mmgag":
        return display_benchmark(benchmark)
    if family == "ultra_long":
        return display_benchmark(benchmark)
    return f"{FAMILY_DISPLAY_LABEL.get(family, family)} | {display_benchmark(benchmark)}"


def ensure_dirs() -> None:
    POLISHED_DIR.mkdir(parents=True, exist_ok=True)
    TRIPTYCH_DIR.mkdir(parents=True, exist_ok=True)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": CARD,
            "savefig.facecolor": PAPER,
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans", "Helvetica"],
            "axes.edgecolor": FAINT,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "svg.fonttype": "none",
        }
    )


def clean_axes(ax: plt.Axes, grid: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(FAINT)
    ax.spines["bottom"].set_color(FAINT)
    if grid:
        ax.grid(axis=grid, color="#CBD3DC", linewidth=0.75, alpha=0.38)
    ax.set_axisbelow(True)


def add_card(fig: plt.Figure, title: str, subtitle: str, tag: str | None = None, conclusion: str | None = None) -> None:
    fig.text(0.045, 0.935, title, fontsize=23, fontweight="bold", color=INK)
    fig.text(0.045, 0.900, subtitle, fontsize=10.5, color=MUTED)
    fig.lines.append(
        plt.Line2D([0.045, 0.955], [0.872, 0.872], transform=fig.transFigure, color="#DCE3E4", linewidth=1.2)
    )
    if tag:
        fig.text(
            0.895,
            0.926,
            tag,
            ha="center",
            va="center",
            fontsize=9.5,
            color=BLUE,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#EAF2FB", edgecolor="#CFE2F5"),
        )
    if conclusion:
        box = FancyBboxPatch(
            (0.045, 0.028),
            0.91,
            0.055,
            boxstyle="round,pad=0.006,rounding_size=0.010",
            transform=fig.transFigure,
            linewidth=0.9,
            edgecolor="#D9E6DD",
            facecolor="#F1F7F3",
            zorder=-1,
        )
        fig.patches.append(box)
        fig.text(0.062, 0.050, conclusion, fontsize=10.5, color="#2F493C", va="center")


def save_card(fig: plt.Figure, name: str) -> None:
    png = POLISHED_DIR / f"{name}.png"
    svg = POLISHED_DIR / f"{name}.svg"
    fig.savefig(png, dpi=260, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    strip_trailing_whitespace(svg)


def strip_trailing_whitespace(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    path.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n", encoding="utf-8")


def annotate_value(ax: plt.Axes, x: float, y: float, text: str, color: str = INK) -> None:
    ax.text(x, y, text, va="center", ha="left", fontsize=8.6, color=color, fontweight="bold")


def hero_dashboard() -> None:
    main = read_csv("main_benchmark/paper_baseline_compare_table.csv")
    main["method_clean"] = main["method"].map(normalize_method)
    pair = main[main["method_clean"].isin(["Ours", "GOMAA-Geo"])].copy()
    pivot = pair.pivot_table(
        index="benchmark", columns="method_clean", values="success_ratio", aggfunc="first"
    ).dropna()
    pivot["gain"] = pivot["Ours"] - pivot["GOMAA-Geo"]
    pivot = pivot.sort_values("gain")

    mmgag = pivot.loc[[x for x in ["mmgag_aerial", "mmgag_ground", "mmgag_text"] if x in pivot.index]]
    ultra = read_csv("ultra_long/ultra_long_v2_summary.csv")
    ultra["method_clean"] = ultra["method_key"].map(normalize_method)
    ultra_delta = (
        ultra[ultra["method_clean"].eq("Ours")]["success_ratio"].mean()
        - ultra[ultra["method_clean"].eq("GOMAA-Geo")]["success_ratio"].mean()
    )

    fig = plt.figure(figsize=(16, 9))
    add_card(
        fig,
        "实验结果总览",
        "基于主实验、跨模态任务和长距离压力测试的可复现统计汇总。",
        "项目展示",
        "结论：本文方法在共享主基准、MM-GAG 跨模态任务和长距离搜索中均取得稳定提升。",
    )
    gs = fig.add_gridspec(
        3,
        6,
        left=0.055,
        right=0.965,
        bottom=0.150,
        top=0.835,
        hspace=0.58,
        wspace=0.46,
        height_ratios=[0.92, 1.45, 1.45],
    )

    metrics = [
        ("平均成功率", pivot["Ours"].mean(), pivot["GOMAA-Geo"].mean(), BLUE),
        ("平均提升", pivot["gain"].mean(), 0.0, GREEN),
        ("MM-GAG 提升", mmgag["gain"].mean(), 0.0, VIOLET),
        ("长距离提升", ultra_delta, 0.0, GOLD),
    ]
    for i, (label, value, baseline, color) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, i + 1 if i > 1 else i])
        ax.set_axis_off()
        ax.add_patch(
            FancyBboxPatch(
                (0.0, 0.08),
                1.0,
                0.80,
                boxstyle="round,pad=0.016,rounding_size=0.05",
                transform=ax.transAxes,
                linewidth=1,
                edgecolor="#E8E2D6",
                facecolor="#FBFAF4",
            )
        )
        ax.text(0.06, 0.65, label, transform=ax.transAxes, fontsize=10, color=MUTED)
        display = f"{value:.3f}" if baseline else f"+{value:.3f}"
        ax.text(0.06, 0.30, display, transform=ax.transAxes, fontsize=25, color=color, fontweight="bold")
        if baseline:
            ax.text(0.06, 0.14, f"对比 {baseline:.3f}", transform=ax.transAxes, fontsize=9.2, color=MUTED)

    ax = fig.add_subplot(gs[1:, :3])
    y = np.arange(len(pivot))
    ax.barh(y, pivot["gain"], color=[GREEN if v >= 0 else RED for v in pivot["gain"]], height=0.62)
    ax.axvline(0, color=INK, linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([display_benchmark(v) for v in pivot.index])
    ax.set_xlabel("相对 GOMAA-Geo 的成功率提升")
    ax.set_title("各数据集上的提升幅度")
    for yi, val in zip(y, pivot["gain"]):
        annotate_value(ax, val + 0.004, yi, f"{val:+.3f}", GREEN if val >= 0 else RED)
    clean_axes(ax)

    ax = fig.add_subplot(gs[1:, 3:])
    targets = ["mmgag_aerial", "mmgag_ground", "mmgag_text"]
    labels = ["航拍目标", "地面目标", "文本目标"]
    x = np.arange(len(targets))
    ours = [float(main[(main["benchmark"].eq(t)) & (main["method_clean"].eq("Ours"))]["success_ratio"].iloc[0]) for t in targets]
    gomaa = [float(main[(main["benchmark"].eq(t)) & (main["method_clean"].eq("GOMAA-Geo"))]["success_ratio"].iloc[0]) for t in targets]
    ax.plot(x, gomaa, marker="o", markersize=8, linewidth=2.4, color=ORANGE, label="GOMAA-Geo")
    ax.plot(x, ours, marker="o", markersize=8, linewidth=3.2, color=BLUE, label="本文方法")
    for xi, o, g in zip(x, ours, gomaa):
        ax.vlines(xi, g, o, color="#9EC9EA", linewidth=7, alpha=0.55)
        ax.text(xi + 0.05, o + 0.008, f"+{o-g:.3f}", color=BLUE, fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.46, 0.68)
    ax.set_ylabel("成功率")
    ax.set_title("MM-GAG 跨模态目标")
    ax.legend(frameon=False, loc="lower right")
    clean_axes(ax)

    save_card(fig, "hero_dashboard")


def mmgag_modality_panel() -> None:
    df = read_csv("main_benchmark/paper_baseline_compare_table.csv")
    df["method_clean"] = df["method"].map(normalize_method)
    targets = ["mmgag_aerial", "mmgag_ground", "mmgag_text"]
    target_labels = ["航拍图像目标", "地面图像目标", "文本描述目标"]
    methods = ["Ours", "GOMAA-Geo", "GeoExplorer", "Random", "DiT-AGL"]

    fig = plt.figure(figsize=(16, 9))
    add_card(
        fig,
        "跨模态目标定位表现",
        "在相同距离分桶协议下比较航拍图像、地面图像和文本描述三类目标线索。",
        "MM-GAG",
        "结论：三类目标形式下本文方法均位于第一梯队，说明目标表示对不同模态具有较好的适应性。",
    )
    gs = fig.add_gridspec(1, 3, left=0.065, right=0.96, bottom=0.18, top=0.80, wspace=0.22)
    for idx, (target, title) in enumerate(zip(targets, target_labels)):
        ax = fig.add_subplot(gs[0, idx])
        rows = df[df["benchmark"].eq(target)].copy()
        values = []
        for method in methods:
            hit = rows[rows["method_clean"].eq(method)]
            if len(hit):
                values.append((method, float(hit["success_ratio"].iloc[0])))
        plot = pd.DataFrame(values, columns=["method", "sr"]).sort_values("sr")
        y_pos = np.arange(len(plot))
        ax.barh(y_pos, plot["sr"], color=[METHOD_COLOR.get(m, GRAY) for m in plot["method"]], height=0.55)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([display_method(m) for m in plot["method"]])
        ax.set_xlim(0, 0.72)
        ax.set_title(title)
        ax.set_xlabel("成功率")
        for yi, row in enumerate(plot.itertuples(index=False)):
            ax.text(row.sr + 0.012, yi, f"{row.sr:.3f}", va="center", fontsize=9, fontweight="bold")
        clean_axes(ax)
    save_card(fig, "mmgag_modality_panel")


def ablation_story_panel() -> None:
    df = read_csv("ablation/anchor0624_generalization_table.csv")
    df["gp"] = df.apply(lambda r: f"G{int(r['G_gate'])} P{int(r['P_pbrs'])}", axis=1)
    df["ev"] = df.apply(lambda r: f"E{int(r['E_low_entropy'])} V{int(r['V_val78'])}", axis=1)
    row_order = ["G0 P0", "G0 P1", "G1 P0", "G1 P1"]
    col_order = ["E0 V0", "E0 V1", "E1 V0", "E1 V1"]
    mat = df.pivot(index="gp", columns="ev", values="primary_generalization_mean").reindex(row_order)[col_order]

    fig = plt.figure(figsize=(16, 9))
    add_card(
        fig,
        "模块消融实验",
        "G 表示距离门控，P 表示 PBRS，E 表示低熵约束，V 表示远距离验证设置。",
        "16 组消融",
        "结论：完整 G+P+E+V 分支取得最高主泛化均值，各模块组合后形成更稳定的整体收益。",
    )
    gs = fig.add_gridspec(1, 2, left=0.075, right=0.94, bottom=0.18, top=0.78, wspace=0.32, width_ratios=[1.15, 0.85])

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(mat.values, cmap="cividis", vmin=0.35, vmax=0.625)
    ax.set_xticks(np.arange(len(col_order)))
    ax.set_xticklabels(col_order)
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels(row_order)
    best = np.nanmax(mat.values)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.iloc[i, j]
            color = "white" if val > 0.585 else INK
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=12, fontweight="bold" if val == best else "normal")
    ax.set_title("主泛化均值")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("成功率")

    ax = fig.add_subplot(gs[0, 1])
    factors = [
        ("G 距离门控", "G_gate"),
        ("P 势函数塑形", "P_pbrs"),
        ("E 低熵约束", "E_low_entropy"),
        ("V 远距离验证", "V_val78"),
    ]
    rows = []
    for label, col in factors:
        on = df[df[col].eq(1)]["primary_generalization_mean"].mean()
        off = df[df[col].eq(0)]["primary_generalization_mean"].mean()
        rows.append((label, on - off))
    eff = pd.DataFrame(rows, columns=["factor", "effect"]).sort_values("effect")
    ax.barh(eff["factor"], eff["effect"], color=[GREEN if x >= 0 else RED for x in eff["effect"]], height=0.55)
    ax.axvline(0, color=INK, linewidth=0.9)
    for yi, row in enumerate(eff.itertuples(index=False)):
        ax.text(row.effect + 0.002, yi, f"{row.effect:+.3f}", va="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("平均边际效应")
    ax.set_title("单个因素的平均贡献")
    clean_axes(ax)
    save_card(fig, "ablation_story_panel")


def reward_design_panel() -> None:
    df = read_csv("ablation/reward_gate_type_mmgag_only_table_with_linear.csv")
    df = df.copy()
    def pbrs_state(value: object) -> str:
        text = str(value)
        if "_no_pb" in text:
            return "关闭 PBRS"
        if text.endswith("_pb") or text == "external_pbrs":
            return "开启 PBRS"
        return "关闭 PBRS"

    df["pb"] = df["value"].map(pbrs_state)
    df["gate"] = (
        df["value"].astype(str)
        .str.replace("_no_pb", "", regex=False)
        .str.replace("_pb", "", regex=False)
        .str.replace("_0.405", "", regex=False)
    )
    order = ["external", "constant", "linear", "blend_lp", "power2", "sine"]
    label_map = {"external": "外在奖励", "constant": "常数门控", "linear": "线性门控", "blend_lp": "线性-幂次", "power2": "二次门控", "sine": "正弦门控"}

    fig = plt.figure(figsize=(16, 9))
    add_card(
        fig,
        "奖励设计对比",
        "所有方案均在 MM-GAG 平均成功率上评估；奖励项只影响训练阶段，不在推理阶段额外注入。",
        "奖励消融",
        "结论：线性距离门控与 PBRS 组合表现最好，说明探索强度随距离变化比固定内在奖励更合适。",
    )
    gs = fig.add_gridspec(1, 2, left=0.07, right=0.95, bottom=0.18, top=0.78, width_ratios=[1.2, 0.8], wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(order))
    for pb, color, marker in [("关闭 PBRS", ORANGE, "o"), ("开启 PBRS", BLUE, "o")]:
        vals = []
        for gate in order:
            if gate == "external" and pb == "关闭 PBRS":
                hit = df[df["value"].eq("external_only")]
            elif gate == "external" and pb == "开启 PBRS":
                hit = df[df["value"].eq("external_pbrs")]
            else:
                hit = df[(df["gate"].eq(gate)) & (df["pb"].eq(pb))]
            vals.append(float(hit["mmgag_mean_sr"].iloc[0]) if len(hit) else np.nan)
        ax.plot(x, vals, marker=marker, linewidth=3 if pb == "开启 PBRS" else 2.2, color=color, label=pb)
        for xi, val in zip(x, vals):
            if np.isfinite(val):
                ax.text(xi, val + 0.010, f"{val:.3f}", ha="center", fontsize=8.2, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([label_map[v] for v in order], rotation=18, ha="right")
    ax.set_ylabel("MM-GAG 平均成功率")
    ax.set_ylim(0.34, 0.64)
    ax.set_title("距离门控函数对比")
    ax.legend(frameon=False, loc="lower left")
    clean_axes(ax)

    ax = fig.add_subplot(gs[0, 1])
    control = read_csv("ablation/reward_control_long_table.csv")
    mmgag = control[control["benchmark"].isin(["mmgag_aerial", "mmgag_ground", "mmgag_text"])]
    means = mmgag.groupby("run")["sr"].mean().reset_index()
    name_map = {
        "reward_external_only_seed321_t480k": "仅外在奖励",
        "reward_intrinsic_only_seed321_t480k": "仅内在奖励",
        "reward_intrinsic_no_decay_seed321_t480k": "外在+无门控内在",
    }
    means["label"] = means["run"].map(name_map)
    ours = float(df[df["value"].eq("linear_0.405_pb")]["mmgag_mean_sr"].iloc[0])
    plot = pd.concat(
        [means[["label", "sr"]].rename(columns={"sr": "value"}), pd.DataFrame([{"label": "线性门控 + PBRS", "value": ours}])],
        ignore_index=True,
    ).sort_values("value")
    ax.barh(plot["label"], plot["value"], color=[BLUE if "线性" in x else GRAY for x in plot["label"]], height=0.55)
    for yi, row in enumerate(plot.itertuples(index=False)):
        ax.text(row.value + 0.010, yi, f"{row.value:.3f}", va="center", fontsize=9, fontweight="bold")
    ax.set_xlim(0, 0.66)
    ax.set_xlabel("MM-GAG 平均成功率")
    ax.set_title("奖励端点对照")
    clean_axes(ax)
    save_card(fig, "reward_design_panel")


def long_range_panel() -> None:
    budget = read_csv("supplement_eval/budget_sensitivity_table.csv")
    budget = budget[budget["method"].isin(["GeoExplorer-anchor0624", "GOMAA-Geo", "GeoExplorer-pristine"])].copy()
    budget["method_clean"] = budget["method"].map(normalize_method)

    fig = plt.figure(figsize=(16, 9))
    add_card(
        fig,
        "长距离搜索与压力测试",
        "在扩大网格和搜索预算后复用已训练模型进行评测，不额外重训。",
        "8x8 / 10x10 / 25x25",
        "结论：搜索范围扩大后绝对成功率下降，但本文方法相对 GOMAA-Geo 的优势仍然保持。",
    )
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.95, bottom=0.18, top=0.78, hspace=0.52, wspace=0.28)
    for ax, grid in zip([fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])], ["8x8", "10x10"]):
        sub = budget[budget["grid"].eq(grid)]
        for method in ["Ours", "GOMAA-Geo", "GeoExplorer"]:
            hit = sub[sub["method_clean"].eq(method)].sort_values("budget")
            if hit.empty:
                continue
            ax.plot(hit["budget"], hit["success_ratio"], marker="o", linewidth=3 if method == "Ours" else 2.2, color=METHOD_COLOR[method], label=display_method(method))
        ax.set_title(f"{grid} 预算敏感性")
        ax.set_xlabel("搜索预算")
        ax.set_ylabel("成功率")
        ax.set_ylim(0.2, 0.82)
        clean_axes(ax)
    fig.axes[0].legend(frameon=False, loc="lower right")

    p1 = read_csv("supplement_eval/p1_grid25_budget_table.csv")
    p1["method_clean"] = p1["method"].map(normalize_method)
    ax = fig.add_subplot(gs[1, 0])
    for method in ["Ours", "GOMAA-Geo", "GeoExplorer"]:
        hit = p1[p1["method_clean"].eq(method)].sort_values("budget")
        ax.plot(hit["budget"], hit["success_ratio"], marker="o", linewidth=3 if method == "Ours" else 2.2, color=METHOD_COLOR[method], label=display_method(method))
    ax.set_title("25x25 探索性压力测试")
    ax.set_xlabel("搜索预算")
    ax.set_ylabel("成功率")
    clean_axes(ax)

    ax = fig.add_subplot(gs[1, 1])
    seed = read_csv("supplement_eval/task_seed_summary.csv")
    if {"benchmark", "method_key", "mean_sr"}.issubset(seed.columns):
        pivot = seed[seed["method_key"].isin(["anchor0624", "gomaa"])].pivot_table(
            index=["family", "benchmark", "grid"],
            columns="method_key",
            values="mean_sr",
            aggfunc="first",
        ).dropna()
        pivot["delta"] = pivot["anchor0624"] - pivot["gomaa"]
        pivot = pivot.reset_index()
        pivot["setting"] = pivot.apply(display_seed_setting, axis=1)
        plot = pivot.sort_values("delta")
        ax.barh(plot["setting"], plot["delta"], color=GREEN, height=0.52)
        for yi, row in enumerate(plot.itertuples(index=False)):
            ax.text(row.delta + 0.004, yi, f"+{row.delta:.3f}", va="center", fontsize=9, fontweight="bold")
        ax.set_xlabel("平均成功率提升")
        ax.set_title("任务库随机种子稳定性")
    else:
        ax.axis("off")
    clean_axes(ax)
    save_card(fig, "long_range_panel")


def trajectory_behavior_panel() -> None:
    df = read_csv("trajectory_analysis/trajectory_behavior_by_distance.csv")
    fig = plt.figure(figsize=(16, 9))
    add_card(
        fig,
        "轨迹行为统计",
        "从整理后的轨迹案例库计算行为指标，用于解释中远距离任务中的性能差异。",
        "轨迹分析",
        "结论：本文方法在中远距离下更常接近目标、重复访问更少，搜索路径更稳定。",
    )
    metrics = [
        ("success_rate", "成功率"),
        ("progress_ratio", "接近目标比例"),
        ("monotonic_step_rate", "单调接近率"),
        ("revisit_rate", "重复访问率"),
    ]
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.95, bottom=0.18, top=0.78, hspace=0.52, wspace=0.28)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    for ax, (metric, ylabel) in zip(axes, metrics):
        for method in ["GeoExplorer-anchor0624", "GOMAA-Geo", "GeoExplorer-pristine"]:
            label = normalize_method(method)
            hit = df[df["method"].eq(method)].sort_values("distance")
            if hit.empty:
                continue
            ax.plot(hit["distance"], hit[metric], marker="o", linewidth=3 if label == "Ours" else 2.2, color=METHOD_COLOR[label], label=display_method(label))
        ax.set_title(ylabel)
        ax.set_xlabel("初始距离")
        ax.set_ylabel(ylabel)
        clean_axes(ax)
    axes[0].legend(frameon=False, loc="lower right")
    save_card(fig, "trajectory_behavior_panel")


def _weighted_means(df: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        total_n = float(group["n"].sum())
        row = dict(zip(group_cols, keys))
        row["n"] = int(total_n)
        for col in value_cols:
            row[col] = float(np.average(group[col], weights=group["n"])) if total_n else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def reward_process_panel() -> None:
    df = read_csv("reward_process/reward_process_summary.csv")
    values = [
        "sum_external",
        "sum_intrinsic_gated",
        "sum_pbrs",
        "sum_total",
        "mean_gate",
        "path_length",
        "final_distance",
    ]
    grouped = _weighted_means(df, ["method_key", "success"], values)
    full = df[df["method_key"].eq("g1_p1_e1_v1")].copy()

    fig = plt.figure(figsize=(16, 9))
    add_card(
        fig,
        "奖励过程分解",
        "对成功与失败轨迹分别汇总外在奖励、门控内在奖励和 PBRS 项，观察训练信号如何影响搜索。",
        "奖励过程",
        "结论：成功轨迹通常累积更高总奖励，并在门控内在奖励与 PBRS 的配合下形成更明确的目标导向。",
    )
    gs = fig.add_gridspec(
        2,
        2,
        left=0.07,
        right=0.95,
        bottom=0.18,
        top=0.78,
        hspace=0.52,
        wspace=0.28,
    )

    ax = fig.add_subplot(gs[0, 0])
    full_grouped = grouped[grouped["method_key"].eq("g1_p1_e1_v1")].set_index("success")
    components = [
        ("外在奖励", "sum_external", ORANGE),
        ("门控内在奖励", "sum_intrinsic_gated", SKY),
        ("PBRS", "sum_pbrs", GREEN),
        ("总奖励", "sum_total", BLUE),
    ]
    x = np.arange(len(components))
    width = 0.34
    for offset, success, label, color in [(-width / 2, True, "成功轨迹", BLUE), (width / 2, False, "失败轨迹", RED)]:
        vals = [float(full_grouped.loc[success, col]) if success in full_grouped.index else np.nan for _, col, _ in components]
        ax.bar(x + offset, vals, width=width, color=color, alpha=0.82, label=label)
        for xi, val in zip(x + offset, vals):
            if np.isfinite(val):
                ax.text(xi, val + (0.22 if val >= 0 else -0.35), f"{val:.1f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=8.5, fontweight="bold")
    ax.axhline(0, color=INK, linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in components], rotation=12, ha="right")
    ax.set_ylabel("平均累计奖励")
    ax.set_title("完整方法：成功与失败轨迹")
    ax.legend(frameon=False, ncols=2, loc="upper left")
    clean_axes(ax)

    ax = fig.add_subplot(gs[0, 1])
    for success, label, color, marker in [(True, "成功轨迹", BLUE, "o"), (False, "失败轨迹", RED, "s")]:
        hit = full[full["success"].eq(success)].sort_values("distance")
        if hit.empty:
            continue
        ax.plot(hit["distance"], hit["sum_total"], marker=marker, linewidth=3, color=color, label=label)
        ax.fill_between(hit["distance"], hit["sum_external"], hit["sum_total"], color=color, alpha=0.10)
    ax.axhline(0, color=INK, linewidth=0.9)
    ax.set_xlabel("初始距离 C")
    ax.set_ylabel("累计总奖励")
    ax.set_title("不同距离下的奖励结果")
    ax.legend(frameon=False)
    clean_axes(ax)

    ax = fig.add_subplot(gs[1, 0])
    diagnostics = [
        ("路径长度", "path_length"),
        ("最终距离", "final_distance"),
    ]
    x = np.arange(len(diagnostics))
    width = 0.34
    for offset, success, label, color in [(-width / 2, True, "成功轨迹", BLUE), (width / 2, False, "失败轨迹", RED)]:
        vals = [float(full_grouped.loc[success, col]) if success in full_grouped.index else np.nan for _, col in diagnostics]
        ax.bar(x + offset, vals, width=width, color=color, alpha=0.82, label=label)
        for xi, val in zip(x + offset, vals):
            if np.isfinite(val):
                ax.text(xi, val + 0.18, f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([d[0] for d in diagnostics])
    ax.set_ylabel("平均值")
    ax.set_title("轨迹结果诊断")
    ax.legend(frameon=False, ncols=2, loc="upper left")
    clean_axes(ax)

    ax = fig.add_subplot(gs[1, 1])
    for method, label, color in [
        ("g1_p1_e1_v1", "完整门控", BLUE),
        ("g1_p0_e1_v1", "无 PBRS", SKY),
        ("g0_p1_e1_v1", "无距离门控", ORANGE),
    ]:
        hit = df[(df["method_key"].eq(method)) & (df["success"].eq(True))].sort_values("distance")
        if hit.empty:
            continue
        ax.plot(hit["distance"], hit["mean_gate"], marker="o", linewidth=2.6, color=color, label=label)
    ax.set_xlabel("初始距离 C")
    ax.set_ylabel("平均门控值")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("成功轨迹中的门控曲线")
    ax.legend(frameon=False, loc="lower left")
    clean_axes(ax)

    save_card(fig, "reward_process_panel")


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
        Path(font_manager.findfont("DejaVu Sans", fallback_to_default=True)),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _read_gif_frames(path: Path) -> tuple[list[Image.Image], int]:
    im = Image.open(path)
    durations: list[int] = []
    frames: list[Image.Image] = []
    for frame in ImageSequence.Iterator(im):
        frames.append(frame.convert("RGBA"))
        durations.append(int(frame.info.get("duration", 240)))
    duration = int(np.median(durations)) if durations else 240
    return frames, duration


def _fit_cover(im: Image.Image, size: tuple[int, int]) -> Image.Image:
    w, h = im.size
    tw, th = size
    scale = max(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = im.resize((nw, nh), Image.Resampling.LANCZOS)
    left = (nw - tw) // 2
    top = (nh - th) // 2
    return resized.crop((left, top, left + tw, top + th))


def _crop_map_frame(frame: Image.Image) -> Image.Image:
    # Source GIFs include a white title band and figure border.  The showcase
    # triptych should compare paths directly, so keep only the map evidence.
    return frame.crop((100, 112, 613, 582))


def _draw_shadow_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font_obj: ImageFont.ImageFont,
    fill: str | tuple[int, int, int, int],
    stroke: int = 2,
) -> None:
    draw.text(xy, text, font=font_obj, fill=fill, stroke_width=stroke, stroke_fill=(0, 0, 0, 185))


def build_triptych_gifs() -> None:
    gif_dir = SHOWCASE_DIR / "trajectories" / "gifs"
    if not gif_dir.exists():
        return

    label_font = _load_font(22, bold=True)
    small_font = _load_font(15, bold=False)
    methods = [
        ("anchor0624", "本文方法", BLUE, "到达目标"),
        ("gomaa", "GOMAA-Geo", ORANGE, "未到达目标"),
        ("pristine", "GeoExplorer", GREEN, "偏离目标"),
    ]

    base_names = sorted(
        p.name.removesuffix("__anchor0624.gif")
        for p in gif_dir.glob("*__anchor0624.gif")
        if (gif_dir / (p.name.removesuffix("__anchor0624.gif") + "__gomaa.gif")).exists()
        and (gif_dir / (p.name.removesuffix("__anchor0624.gif") + "__pristine.gif")).exists()
    )

    for base in base_names:
        loaded = []
        durations = []
        for suffix, _, _, _ in methods:
            frames, duration = _read_gif_frames(gif_dir / f"{base}__{suffix}.gif")
            loaded.append(frames)
            durations.append(duration)
        n = max(len(frames) for frames in loaded)
        panel_w, panel_h = 568, 520
        gutter = 16
        canvas_size = (panel_w * 3 + gutter * 2, panel_h)
        duration = int(np.median(durations))
        frames_out: list[Image.Image] = []
        for i in range(n):
            canvas = Image.new("RGBA", canvas_size, "#07101D")
            draw = ImageDraw.Draw(canvas)
            for j, (suffix, label, color, outcome) in enumerate(methods):
                x0 = j * (panel_w + gutter)
                src_frames = loaded[j]
                frame = _crop_map_frame(src_frames[min(i, len(src_frames) - 1)])
                panel = _fit_cover(frame, (panel_w, panel_h)).convert("RGBA")
                canvas.alpha_composite(panel, (x0, 0))
                draw.rectangle((x0, 0, x0 + 6, panel_h), fill=color)
                _draw_shadow_text(draw, (x0 + 16, 12), label, label_font, "white", stroke=2)
                _draw_shadow_text(draw, (x0 + panel_w - 118, 14), f"步 {i:02d}/{n - 1:02d}", small_font, "#EAF3FB", stroke=2)
                status = outcome if i == n - 1 else "搜索中"
                meta = f"{status} | C=6 | 图像 189 | 步数 {i}/{n - 1}"
                _draw_shadow_text(draw, (x0 + 16, panel_h - 30), meta, small_font, "#F2F7FA", stroke=2)
            draw.rectangle((0, panel_h - 4, canvas_size[0], panel_h), fill=(5, 10, 18, 160))
            draw.rectangle((0, panel_h - 4, int(canvas_size[0] * i / max(1, n - 1)), panel_h), fill=BLUE)
            frames_out.append(canvas.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))

        out = TRIPTYCH_DIR / f"{base}__triptych.gif"
        frames_out[0].save(
            out,
            save_all=True,
            append_images=frames_out[1:],
            duration=max(duration, 220),
            loop=0,
            optimize=True,
        )

    # Also write a high-resolution preview for the README static fallback.
    hardcase = TRIPTYCH_DIR / "three_method_hardcase__img189_d6_s20_g14_r0__triptych.gif"
    if hardcase.exists():
        im = Image.open(hardcase)
        first = next(ImageSequence.Iterator(im)).convert("RGB")
        first.save(POLISHED_DIR / "trajectory_triptych_preview.png", quality=95)


def write_manifest() -> None:
    files = sorted(
        str(p.relative_to(ROOT)).replace("\\", "/")
        for p in list(POLISHED_DIR.glob("*")) + list(TRIPTYCH_DIR.glob("*"))
        if p.is_file()
    )
    manifest = {
        "generated_by": "code/tools/build_polished_showcase.py",
        "style": "editorial scientific cards, borderless synchronized triptych GIFs",
        "file_count": len(files),
        "files": files,
    }
    (POLISHED_DIR / "polished_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_polished() -> None:
    ensure_dirs()
    setup_style()
    hero_dashboard()
    mmgag_modality_panel()
    ablation_story_panel()
    reward_design_panel()
    long_range_panel()
    trajectory_behavior_panel()
    reward_process_panel()
    build_triptych_gifs()
    write_manifest()
    print(f"Polished showcase written to {POLISHED_DIR}")


if __name__ == "__main__":
    build_polished()
