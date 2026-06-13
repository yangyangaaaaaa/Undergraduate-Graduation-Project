#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build one refined PPT figure for the reward-mechanism explanation.

This is a presentation-only redrawing of a real training log case.  It keeps
the data source unchanged and fixes the layout issues noted during PPT review:

- route panel and distance curve share the same visual height;
- bottom reward ledger uses light cell colors;
- Chinese labels use SimSun and numbers use Times New Roman;
- all table values are centered;
- a compact driver row explains whether each step is target-driven,
  curiosity-driven, recovery, or convergence.
"""

from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import build_reward_guided_case_studies as base  # noqa: E402


CASE_ID = "case_12"

OUT_DIR = base.RESULTS / "figures" / "defense_reward_training_stage" / "single_reward_mechanism"
PPT_PACK_DIR = (
    base.RESULTS
    / "figures"
    / "ppt_candidate_pack_20260606"
    / "03_奖励机制_动作归因与GP故事"
)
SELECTED_TABLE = base.TABLES / "reward_guided_case_studies_selected.csv"

METHOD_ORDER = [
    "external_only",
    "intrinsic_only",
    "mixed_no_gate_no_pbrs",
    "mixed_gate_only",
    "mixed_pbrs_only",
    "proposed_linear_gate_pbrs",
]
PLOT_METHOD_ORDER = [method for method in METHOD_ORDER if method != "mixed_pbrs_only"]

METHOD_CN = {
    "external_only": "仅外在",
    "intrinsic_only": "仅好奇心",
    "mixed_no_gate_no_pbrs": "直接相加",
    "mixed_gate_only": "距离门控",
    "mixed_pbrs_only": "仅塑形",
    "proposed_linear_gate_pbrs": "本文方法",
}

METHOD_DIAGNOSIS = {
    "external_only": "缺少探索",
    "intrinsic_only": "目标约束弱",
    "mixed_no_gate_no_pbrs": "回访偏多",
    "mixed_gate_only": "后段远离",
    "mixed_pbrs_only": "接近未达",
}

ROUTE_OFFSETS = {
    "external_only": (-0.16, -0.12),
    "intrinsic_only": (-0.08, 0.12),
    "mixed_no_gate_no_pbrs": (0.07, -0.10),
    "mixed_gate_only": (0.15, 0.10),
    "mixed_pbrs_only": (-0.01, 0.18),
    "proposed_linear_gate_pbrs": (0.00, 0.00),
}

LINESTYLE = {
    "external_only": (0, (7, 3)),
    "intrinsic_only": (0, (1, 2)),
    "mixed_no_gate_no_pbrs": (0, (5, 2, 1, 2)),
    "mixed_gate_only": (0, (4, 2)),
    "mixed_pbrs_only": (0, (2, 2)),
    "proposed_linear_gate_pbrs": "solid",
}

ACTION_CN = {
    "up": "上",
    "down": "下",
    "left": "左",
    "right": "右",
}

INK = "#111827"
MUTED = "#566173"
PAPER = "#F7F9FC"
GRID = "#E5E7EB"
BLUE = base.BLUE
GREEN = "#168A63"
RED = "#B84A48"
ORANGE = base.ORANGE
PURPLE = base.PURPLE
TEAL = base.TEAL
YELLOW = base.YELLOW


def font_props() -> tuple[FontProperties, FontProperties, FontProperties]:
    simsun_path = Path(r"C:\Windows\Fonts\simsun.ttc")
    times_path = Path(r"C:\Windows\Fonts\times.ttf")
    times_bold_path = Path(r"C:\Windows\Fonts\timesbd.ttf")
    for font_path in [simsun_path, times_path, times_bold_path]:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
    cn_font = FontProperties(fname=str(simsun_path)) if simsun_path.exists() else FontProperties(family="SimSun")
    num_font = FontProperties(fname=str(times_path)) if times_path.exists() else FontProperties(family="Times New Roman")
    num_bold = FontProperties(fname=str(times_bold_path)) if times_bold_path.exists() else FontProperties(family="Times New Roman", weight="bold")
    return cn_font, num_font, num_bold


CN_FONT, NUM_FONT, NUM_BOLD = font_props()


def setup_style() -> None:
    base.setup_style()
    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "SimSun"],
            "axes.unicode_minus": False,
            "figure.facecolor": PAPER,
            "savefig.facecolor": PAPER,
        }
    )


def case_mask(routes: pd.DataFrame, case_row: pd.Series) -> pd.Series:
    mask = pd.Series(True, index=routes.index)
    for col in ["seed", "episode", "image_index", "distance_bucket", "initial_patch", "goal_patch"]:
        mask &= routes[col].astype(str).eq(str(case_row[col]))
    return mask


def load_case() -> tuple[pd.Series, dict[str, pd.Series], base.ImageAsset]:
    selected = pd.read_csv(SELECTED_TABLE)
    match = selected[selected["case_id"].astype(str).eq(CASE_ID)]
    if match.empty:
        raise RuntimeError(f"Cannot find {CASE_ID} in {SELECTED_TABLE}")
    case_row = match.iloc[0]

    routes = base.read_routes()
    group = routes[case_mask(routes, case_row)].copy()
    rows_by_method = {
        method: group[group["method"].eq(method)].iloc[0]
        for method in METHOD_ORDER
        if not group[group["method"].eq(method)].empty
    }
    missing = [method for method in METHOD_ORDER if method not in rows_by_method]
    if missing:
        raise RuntimeError(f"Missing methods for {CASE_ID}: {missing}")

    asset = base.ImageAsset(
        str(case_row.get("dataset", "")),
        str(case_row.get("image_id", "")),
        Path(str(case_row["image_path"])),
        f"{case_row.get('dataset', '')} overhead",
    )
    return case_row, rows_by_method, asset


def route_goal_and_path(row: pd.Series) -> tuple[int, list[int]]:
    seq = [int(x) for x in base.parse_list(row.get("patch_sequence", ""))]
    if len(seq) >= 2:
        return int(seq[0]), [int(x) for x in seq[1:]]
    return int(row["goal_patch"]), [int(row["initial_patch"]), int(row["final_patch"])]


def route_xy(path: list[int], method: str) -> list[tuple[float, float]]:
    dx, dy = ROUTE_OFFSETS.get(method, (0.0, 0.0))
    return [(x + dx, y + dy) for x, y in (base.patch_xy(p) for p in path)]


def draw_route_panel(ax: plt.Axes, image, rows_by_method: dict[str, pd.Series]) -> None:
    ax.imshow(image, extent=(-0.5, base.PATCH_SIZE - 0.5, base.PATCH_SIZE - 0.5, -0.5), zorder=0)
    ax.set_xlim(-0.5, base.PATCH_SIZE - 0.5)
    ax.set_ylim(base.PATCH_SIZE - 0.5, -0.5)
    ax.set_aspect("equal")
    base.draw_grid(ax)

    for method in PLOT_METHOD_ORDER:
        row = rows_by_method[method]
        _, path = route_goal_and_path(row)
        if not path:
            continue
        xy = route_xy(path, method)
        xs, ys = zip(*xy)
        key = method == "proposed_linear_gate_pbrs"
        color = base.METHOD_STYLE[method]["color"]
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=5.8 if key else 3.0,
            linestyle=LINESTYLE[method],
            alpha=1.0 if key else 0.72,
            solid_capstyle="round",
            zorder=16 if key else 9,
            path_effects=[
                pe.Stroke(linewidth=8.6 if key else 5.2, foreground="white", alpha=0.84),
                pe.Normal(),
            ],
        )
        final_x, final_y = xy[-1]
        success = int(row["success"]) == 1
        ax.scatter(
            [final_x],
            [final_y],
            s=150 if key else 95,
            marker="o" if success else "X",
            color="#10B981" if success else "#EF4444",
            edgecolor="white",
            linewidth=1.5,
            zorder=20 if key else 15,
        )

    proposed = rows_by_method["proposed_linear_gate_pbrs"]
    goal, path = route_goal_and_path(proposed)
    if path:
        sx, sy = base.patch_xy(path[0])
        ax.scatter([sx], [sy], s=220, marker="o", color="#10B981", edgecolor="white", linewidth=2.2, zorder=25)
        ax.text(sx, sy - 0.30, "起", ha="center", va="center", fontsize=12, fontproperties=CN_FONT, color=INK, zorder=26)
    gx, gy = base.patch_xy(goal)
    ax.scatter([gx], [gy], s=315, marker="*", color=YELLOW, edgecolor=INK, linewidth=1.2, zorder=25)
    ax.text(gx, gy + 0.34, "目标", ha="center", va="center", fontsize=12, fontproperties=CN_FONT, color=INK, zorder=26)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_distance_curve(ax: plt.Axes, rows_by_method: dict[str, pd.Series]) -> None:
    for method in PLOT_METHOD_ORDER:
        row = rows_by_method[method]
        dist = [float(x) for x in base.parse_list(row.get("dist_sequence", ""))]
        x = np.arange(0, len(dist))
        key = method == "proposed_linear_gate_pbrs"
        color = base.METHOD_STYLE[method]["color"]
        ax.plot(
            x,
            dist,
            color=color,
            linewidth=4.2 if key else 2.0,
            linestyle=LINESTYLE[method],
            marker="o" if key else None,
            markersize=5.8,
            alpha=1.0 if key else 0.64,
            label=METHOD_CN[method],
            zorder=12 if key else 6,
        )
        final_marker = "o" if int(row["success"]) == 1 else "X"
        ax.scatter([x[-1]], [dist[-1]], s=92 if key else 58, marker=final_marker, color=color, edgecolor="white", linewidth=1.1, zorder=14)

    ax.axhline(0, color="#94A3B8", linewidth=1.0)
    ax.set_xlim(-0.15, 9.25)
    ax.set_ylim(-0.35, 7.55)
    ax.set_yticks(range(0, 8))
    ax.set_xlabel("行动步", fontproperties=CN_FONT, fontsize=12)
    ax.set_ylabel("到目标距离", fontproperties=CN_FONT, fontsize=12)
    ax.grid(axis="y", color=GRID, linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(NUM_FONT)

    ax.annotate(
        "短暂回退",
        xy=(6, 3),
        xytext=(6.65, 4.05),
        fontproperties=CN_FONT,
        fontsize=13,
        color=RED,
        arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.7, shrinkA=2, shrinkB=2),
    )
    ax.annotate(
        "恢复并到达",
        xy=(9, 0),
        xytext=(6.95, 0.70),
        fontproperties=CN_FONT,
        fontsize=13,
        color=BLUE,
        arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.7, shrinkA=2, shrinkB=2),
    )


def draw_total_reward_curve(ax: plt.Axes, rows_by_method: dict[str, pd.Series]) -> None:
    ax.axhline(0, color="#94A3B8", linewidth=1.0)
    ax.axvspan(5.6, 7.4, color="#F97316", alpha=0.055, zorder=0)

    for method in PLOT_METHOD_ORDER:
        row = rows_by_method[method]
        total = [float(x) for x in base.parse_list(row.get("step_reward_total", ""))]
        if not total:
            continue
        x = np.arange(1, len(total) + 1)
        key = method == "proposed_linear_gate_pbrs"
        color = base.METHOD_STYLE[method]["color"]
        ax.plot(
            x,
            total,
            color=color,
            linewidth=3.8 if key else 1.9,
            linestyle=LINESTYLE[method],
            marker="o" if key else None,
            markersize=5.2,
            alpha=1.0 if key else 0.64,
            zorder=12 if key else 6,
        )

    ax.set_xlim(0.85, 9.25)
    ax.set_ylim(-1.35, 2.45)
    ax.set_yticks([-1, 0, 1, 2])
    ax.set_xlabel("行动步", fontproperties=CN_FONT, fontsize=12)
    ax.set_ylabel("总奖励", fontproperties=CN_FONT, fontsize=12)
    ax.grid(axis="y", color=GRID, linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(NUM_FONT)

    ax.annotate(
        "回退受惩罚",
        xy=(6, -0.74),
        xytext=(4.35, -1.05),
        fontproperties=CN_FONT,
        fontsize=12.5,
        color=RED,
        arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.5, shrinkA=2, shrinkB=2),
    )
    ax.annotate(
        "到达奖励最高",
        xy=(9, 2.17),
        xytext=(6.95, 1.62),
        fontproperties=CN_FONT,
        fontsize=12.5,
        color=BLUE,
        arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.5, shrinkA=2, shrinkB=2),
    )


def draw_legends(fig: plt.Figure) -> None:
    handles = []
    for method in PLOT_METHOD_ORDER:
        key = method == "proposed_linear_gate_pbrs"
        handles.append(
            Line2D(
                [0],
                [0],
                color=base.METHOD_STYLE[method]["color"],
                lw=4.6 if key else 2.2,
                linestyle=LINESTYLE[method],
                label=METHOD_CN[method],
            )
        )
    leg = fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.515, 0.895),
        ncol=5,
        frameon=False,
        fontsize=10.5,
        handlelength=2.2,
        columnspacing=1.1,
    )
    for text in leg.get_texts():
        text.set_fontproperties(CN_FONT)
        text.set_fontsize(10.5)


def draw_control_diagnosis(ax: plt.Axes, rows_by_method: dict[str, pd.Series]) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.00, 0.88, "对照方法为何未形成同样动作", ha="left", va="center", fontsize=14, fontproperties=CN_FONT, color=INK)
    x0 = 0.00
    width = 0.188
    for i, method in enumerate(METHOD_ORDER[:-1]):
        row = rows_by_method[method]
        x = x0 + i * (width + 0.010)
        color = base.METHOD_STYLE[method]["color"]
        ax.add_patch(Rectangle((x, 0.12), width, 0.54, facecolor="white", edgecolor="#D8DEE9", linewidth=1.0))
        ax.add_patch(Rectangle((x, 0.12), 0.010, 0.54, facecolor=color, edgecolor=color, linewidth=0))
        ax.text(x + 0.018, 0.53, METHOD_CN[method], ha="left", va="center", fontsize=10.8, fontproperties=CN_FONT, color=INK)
        ax.text(x + 0.018, 0.35, METHOD_DIAGNOSIS[method], ha="left", va="center", fontsize=10.5, fontproperties=CN_FONT, color=RED)
        ax.text(
            x + 0.018,
            0.19,
            f"终距 {int(row['final_dist'])}",
            ha="left",
            va="center",
            fontsize=10.2,
            fontproperties=CN_FONT,
            color=MUTED,
        )


def parse_arrays(row: pd.Series) -> dict[str, list[float] | list[str]]:
    return {
        "actions": [str(x) for x in base.parse_list(row.get("action_sequence", ""))],
        "dist": [float(x) for x in base.parse_list(row.get("dist_sequence", ""))],
        "ex": [float(x) for x in base.parse_list(row.get("step_reward_ex", ""))],
        "intrinsic": [float(x) for x in base.parse_list(row.get("step_reward_in_gated", ""))],
        "pbrs": [float(x) for x in base.parse_list(row.get("step_pbrs_bonus", ""))],
        "total": [float(x) for x in base.parse_list(row.get("step_reward_total", ""))],
    }


def blend(hex_color: str, strength: float) -> tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    rgb = np.array([int(hex_color[i : i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0
    strength = float(np.clip(strength, 0, 1))
    return tuple((1.0 - (1.0 - rgb) * strength).tolist())


def value_color(value: float, scale: float, pos: str = "#16A34A", neg: str = "#DC2626") -> tuple[float, float, float]:
    if not np.isfinite(value) or scale <= 0:
        return (1, 1, 1)
    # Keep colors intentionally pale so table numbers remain legible.
    strength = 0.10 + 0.28 * min(abs(value) / scale, 1.0)
    return blend(pos if value >= 0 else neg, strength)


def driver_labels(arrays: dict[str, list[float] | list[str]]) -> list[tuple[str, tuple[float, float, float]]]:
    dist = arrays["dist"]  # type: ignore[assignment]
    intrinsic = arrays["intrinsic"]  # type: ignore[assignment]
    pbrs = arrays["pbrs"]  # type: ignore[assignment]
    ex = arrays["ex"]  # type: ignore[assignment]
    labels: list[tuple[str, tuple[float, float, float]]] = []
    for i in range(len(ex)):
        before = float(dist[i])
        after = float(dist[i + 1])
        if after == 0:
            labels.append(("到达", value_color(1, 1, "#16A34A", "#DC2626")))
        elif after < before and float(ex[i]) > 0:
            labels.append(("目标驱动", value_color(1, 1, "#16A34A", "#DC2626")))
        elif after > before:
            labels.append(("探索回退", value_color(-1, 1, "#16A34A", "#DC2626")))
        elif float(intrinsic[i]) >= 0.30:
            labels.append(("好奇心", blend("#F97316", 0.28)))
        elif float(pbrs[i]) > 0:
            labels.append(("恢复", value_color(1, 1, "#16A34A", "#DC2626")))
        else:
            labels.append(("收敛", blend(BLUE, 0.20)))
    return labels


def draw_reward_ledger(ax: plt.Axes, proposed_row: pd.Series) -> None:
    arrays = parse_arrays(proposed_row)
    n = len(arrays["ex"])  # type: ignore[arg-type]
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    row_labels = ["动作", "距离", "外在", "好奇心×门控", "塑形", "总奖励", "驱动判断"]
    values: list[list[str]] = []
    colors: list[list[tuple[float, float, float]]] = []
    fonts: list[list[str]] = []

    actions = [ACTION_CN.get(a, a) for a in arrays["actions"]]  # type: ignore[index]
    values.append(actions)
    colors.append([(1, 1, 1) for _ in range(n)])
    fonts.append(["cn" for _ in range(n)])

    dist = arrays["dist"]  # type: ignore[assignment]
    dist_row = []
    dist_colors = []
    for i in range(n):
        before = float(dist[i])
        after = float(dist[i + 1])
        dist_row.append(f"{int(before)}→{int(after)}")
        if after < before:
            dist_colors.append(value_color(1, 1))
        elif after > before:
            dist_colors.append(value_color(-1, 1))
        else:
            dist_colors.append(blend("#64748B", 0.15))
    values.append(dist_row)
    colors.append(dist_colors)
    fonts.append(["num" for _ in range(n)])

    for key, fmt, scale, pos, neg in [
        ("ex", "{:+.0f}", 2.0, "#16A34A", "#DC2626"),
        ("intrinsic", "{:+.2f}", 0.50, "#F97316", "#DC2626"),
        ("pbrs", "{:+.3f}", 0.020, BLUE, "#DC2626"),
        ("total", "{:+.2f}", 2.20, "#16A34A", "#DC2626"),
    ]:
        arr = arrays[key]  # type: ignore[index]
        values.append([fmt.format(float(v)) for v in arr])
        colors.append([value_color(float(v), scale, pos, neg) for v in arr])
        fonts.append(["num" for _ in range(n)])

    driver = driver_labels(arrays)
    values.append([x[0] for x in driver])
    colors.append([x[1] for x in driver])
    fonts.append(["cn" for _ in range(n)])

    left = 0.010
    right = 0.990
    top = 0.885
    bottom = 0.055
    label_w = 0.112
    step_h = 0.050
    row_h = (top - bottom - step_h) / len(row_labels)
    cell_w = (right - left - label_w) / n

    for c in range(n):
        x = left + label_w + c * cell_w
        ax.text(
            x + cell_w / 2,
            top + 0.010,
            str(c + 1),
            ha="center",
            va="bottom",
            fontsize=10.8,
            fontproperties=NUM_FONT,
            color=MUTED,
        )
    ax.text(left + label_w - 0.012, top + 0.010, "步", ha="right", va="bottom", fontsize=10.8, fontproperties=CN_FONT, color=MUTED)

    for r, label in enumerate(row_labels):
        y = top - (r + 1) * row_h
        ax.add_patch(Rectangle((left, y), label_w, row_h - 0.004, facecolor="#F1F5F9", edgecolor="#D8DEE9", linewidth=0.9))
        ax.text(
            left + label_w * 0.50,
            y + row_h / 2,
            label,
            ha="center",
            va="center",
            fontsize=11.5,
            fontproperties=CN_FONT,
            color=INK,
        )
        for c in range(n):
            x = left + label_w + c * cell_w
            ax.add_patch(Rectangle((x, y), cell_w - 0.003, row_h - 0.004, facecolor=colors[r][c], edgecolor="#E2E8F0", linewidth=0.75))
            fp = CN_FONT if fonts[r][c] == "cn" else NUM_FONT
            size = 10.9 if r != len(row_labels) - 1 else 10.4
            ax.text(
                x + cell_w / 2,
                y + row_h / 2,
                values[r][c],
                ha="center",
                va="center",
                fontsize=size,
                fontproperties=fp,
                color=INK,
            )

    # Highlight the one exploratory regression and the subsequent recovery.
    for c, label in [(5, "探索后纠偏"), (6, "恢复方向")]:
        x = left + label_w + c * cell_w + cell_w / 2
        ax.annotate(
            label,
            xy=(x, bottom + row_h * 0.55),
            xytext=(x, 0.005),
            ha="center",
            va="bottom",
            fontsize=10.4,
            fontproperties=CN_FONT,
            color=BLUE if c == 6 else RED,
            arrowprops=dict(arrowstyle="-|>", lw=1.1, color=BLUE if c == 6 else RED, shrinkA=1, shrinkB=2),
        )


def add_summary_badges(fig: plt.Figure, case_row: pd.Series) -> None:
    badges = [
        ("训练阶段", "#EAF3FF", BLUE),
        ("本文到达", "#ECFDF5", "#047857"),
        (f"{int(case_row['fail_count'])}/5 对照未到达", "#FEF2F2", RED),
        ("回退后恢复", "#FFF7ED", ORANGE),
    ]
    x = 0.035
    for text, bg, fg in badges:
        w = 0.058 + len(text) * 0.010
        fig.patches.append(
            Rectangle((x, 0.900), w, 0.038, transform=fig.transFigure, facecolor=bg, edgecolor=fg, linewidth=1.0)
        )
        fig.text(x + w / 2, 0.919, text, ha="center", va="center", fontproperties=CN_FONT, fontsize=11.8, color=fg)
        x += w + 0.010


def build_figure() -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PPT_PACK_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    case_row, rows_by_method, asset = load_case()
    image = base.open_overhead_image(asset)

    fig = plt.figure(figsize=(16, 9), facecolor=PAPER)
    fig.text(
        0.035,
        0.970,
        "混合奖励机制如何指导训练动作",
        ha="left",
        va="top",
        fontsize=25,
        fontproperties=CN_FONT,
        color=INK,
    )
    fig.text(
        0.035,
        0.940,
        "同一行动步对齐观察：距离是否接近目标，奖励是否强化该动作",
        ha="left",
        va="top",
        fontsize=13.5,
        fontproperties=CN_FONT,
        color=MUTED,
    )
    fig.lines.append(Line2D([0.035, 0.965], [0.888, 0.888], transform=fig.transFigure, color="#CBD5E1", lw=1.1))

    # The route panel and the two stacked curves share the same vertical span.
    ax_map = fig.add_axes([0.045, 0.365, 0.340, 0.500])
    draw_route_panel(ax_map, image, rows_by_method)
    ax_dist = fig.add_axes([0.430, 0.625, 0.535, 0.240])
    draw_distance_curve(ax_dist, rows_by_method)
    ax_reward = fig.add_axes([0.430, 0.365, 0.535, 0.220])
    draw_total_reward_curve(ax_reward, rows_by_method)
    draw_legends(fig)

    ax_table = fig.add_axes([0.035, 0.040, 0.930, 0.280])
    draw_reward_ledger(ax_table, rows_by_method["proposed_linear_gate_pbrs"])

    stem = "13_奖励机制典型案例_case12_简化版"
    out_png = OUT_DIR / f"{stem}.png"
    pack_png = PPT_PACK_DIR / f"{stem}.png"
    fig.savefig(out_png, dpi=240, facecolor=PAPER)
    plt.close(fig)
    shutil.copy2(out_png, pack_png)

    note_path = OUT_DIR / f"{stem}_说明.txt"
    note_path.write_text(
        "\n".join(
            [
                "推荐 PPT 标题：混合奖励机制如何指导训练动作",
                "案例：case_12，C7，seed=123，episode=75，起点=4，目标=21。",
                "选择理由：本文方法有一次回退但能恢复到达；5 个对照均未到达且均出现回退/回访；底图白色区域比例很低，避免水域/相似区域干扰。",
                "讲法：先看左侧路线，再看右侧距离曲线和总奖励曲线。距离曲线说明对照方法多次回退或停在目标外；奖励曲线说明本文方法在回退时受到惩罚，在恢复并到达目标时获得更高回报。",
                "底部账本只展开本文方法，把动作、距离变化、外在奖励、好奇心门控项、距离塑形项和总奖励按行动步对齐。",
                "边界：该图解释训练阶段奖励信号如何影响策略学习；测试阶段只加载训练好的策略，不再调用奖励函数。",
            ]
        ),
        encoding="utf-8",
    )
    return out_png, pack_png


def main() -> int:
    out_png, pack_png = build_figure()
    print(f"saved: {out_png}")
    print(f"copied: {pack_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
