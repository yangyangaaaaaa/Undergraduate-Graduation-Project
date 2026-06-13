#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build stepwise reward attribution matrix for the smooth case."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import build_reward_guided_case_studies as base  # noqa: E402
import build_reward_mechanism_single_case_refined as refined  # noqa: E402


CASE_ID = "case_04"

OUT_DIR = base.RESULTS / "figures" / "defense_reward_training_stage" / "single_reward_mechanism"
PPT_PACK_DIR = (
    base.RESULTS
    / "figures"
    / "ppt_candidate_pack_20260606"
    / "03_奖励机制_动作归因与GP故事"
)

DIAGNOSIS = {
    "external_only": ("未达  终距 6", "全程负反馈\n有效接近也难被强化"),
    "intrinsic_only": ("未达  终距 6", "回退也为正\n探索压过目标"),
    "mixed_no_gate_no_pbrs": ("未达  终距 4", "好奇心抵消惩罚\n末端回退"),
    "mixed_gate_only": ("未达  终距 2", "能接近到 d=1\n最后一步偏离"),
    "proposed_linear_gate_pbrs": ("到达  终距 0", "关键进展与到达强正\n终点动作被强化"),
}


def load_case() -> dict[str, pd.Series]:
    selected = pd.read_csv(refined.SELECTED_TABLE)
    match = selected[selected["case_id"].astype(str).eq(CASE_ID)]
    if match.empty:
        raise RuntimeError(f"Cannot find {CASE_ID} in {refined.SELECTED_TABLE}")
    case_row = match.iloc[0]

    routes = base.read_routes()
    group = routes[refined.case_mask(routes, case_row)].copy()
    rows_by_method = {
        method: group[group["method"].eq(method)].iloc[0]
        for method in refined.PLOT_METHOD_ORDER
        if not group[group["method"].eq(method)].empty
    }
    missing = [method for method in refined.PLOT_METHOD_ORDER if method not in rows_by_method]
    if missing:
        raise RuntimeError(f"Missing methods for {CASE_ID}: {missing}")
    return rows_by_method


def add_text(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 12,
    color: str = refined.INK,
    ha: str = "center",
    va: str = "center",
    font=None,
    weight: str = "normal",
) -> None:
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=size,
        color=color,
        fontproperties=font or refined.CN_FONT,
        fontweight=weight,
    )


def draw_matrix(ax: plt.Axes, rows_by_method: dict[str, pd.Series]) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    methods = refined.PLOT_METHOD_ORDER
    n_steps = 8
    left = 0.030
    right = 0.975
    top = 0.805
    bottom = 0.250
    method_w = 0.135
    diag_w = 0.235
    heat_w = right - left - method_w - diag_w
    cell_w = heat_w / n_steps
    row_h = (top - bottom) / len(methods)

    header_y = top + 0.035
    add_text(ax, left + method_w * 0.5, header_y, "方法", size=12.6, color=refined.MUTED)
    for step in range(n_steps):
        x = left + method_w + step * cell_w
        add_text(ax, x + cell_w / 2, header_y, f"{step + 1}", size=12.5, color=refined.MUTED, font=refined.NUM_FONT)
    add_text(ax, left + method_w + heat_w + diag_w * 0.46, header_y, "结果与诊断", size=12.6, color=refined.MUTED)

    for r, method in enumerate(methods):
        row = rows_by_method[method]
        y = top - (r + 1) * row_h
        key = method == "proposed_linear_gate_pbrs"
        method_color = base.METHOD_STYLE[method]["color"]
        label_face = refined.blend(method_color, 0.10 if not key else 0.18)

        ax.add_patch(Rectangle((left, y), method_w - 0.004, row_h - 0.006, facecolor=label_face, edgecolor="#CBD5E1", linewidth=0.9))
        ax.add_patch(Rectangle((left, y), 0.008, row_h - 0.006, facecolor=method_color, edgecolor=method_color, linewidth=0))
        add_text(ax, left + method_w * 0.53, y + row_h / 2, refined.METHOD_CN[method], size=12.8, weight="bold" if key else "normal")

        dist = [int(float(x)) for x in base.parse_list(row.get("dist_sequence", ""))]
        totals = [float(x) for x in base.parse_list(row.get("step_reward_total", ""))]
        for c in range(n_steps):
            x = left + method_w + c * cell_w
            if c < len(totals):
                total = totals[c]
                face = refined.value_color(total, 2.20, "#16A34A", "#DC2626")
                dist_text = f"{dist[c]}→{dist[c + 1]}" if c + 1 < len(dist) else ""
                reward_text = f"{total:+.2f}"
            else:
                face = (1, 1, 1)
                dist_text = ""
                reward_text = ""
            ax.add_patch(Rectangle((x, y), cell_w - 0.004, row_h - 0.006, facecolor=face, edgecolor="#E2E8F0", linewidth=0.75))
            add_text(ax, x + cell_w / 2, y + row_h * 0.64, dist_text, size=11.4, font=refined.NUM_FONT, color=refined.INK)
            add_text(ax, x + cell_w / 2, y + row_h * 0.35, reward_text, size=12.1, font=refined.NUM_FONT, color=refined.INK, weight="bold" if key else "normal")

        diag_x = left + method_w + heat_w
        ax.add_patch(Rectangle((diag_x, y), diag_w - 0.004, row_h - 0.006, facecolor="#FFFFFF", edgecolor="#D8DEE9", linewidth=0.9))
        result, diagnosis = DIAGNOSIS[method]
        result_color = "#047857" if key else "#B91C1C"
        add_text(ax, diag_x + 0.018, y + row_h * 0.66, result, size=12.0, color=result_color, ha="left", font=refined.CN_FONT, weight="bold" if key else "normal")
        add_text(ax, diag_x + 0.018, y + row_h * 0.34, diagnosis, size=11.2, color=refined.INK, ha="left", font=refined.CN_FONT)

        if key:
            ax.add_patch(Rectangle((left - 0.004, y - 0.004), right - left + 0.004, row_h + 0.002, facecolor="none", edgecolor=refined.BLUE, linewidth=2.0))

    add_text(ax, left + method_w + heat_w / 2, bottom - 0.055, "单元格：距离变化 / 该步总奖励", size=12.0, color=refined.MUTED)


def draw_summary(ax: plt.Axes) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    items = [
        ("不是奖励总和最大", "仅好奇心累计回报较高，但回退也被正向奖励，目标约束不足。", refined.PURPLE),
        ("看正反馈落点", "有效奖励应集中在接近目标和到达目标的关键动作上。", refined.GREEN),
        ("本文方法的作用", "门控削弱末端无效探索，塑形补充方向信号，到达动作获得最高回报。", refined.BLUE),
    ]
    box_w = 0.305
    for i, (title, body, color) in enumerate(items):
        x = 0.020 + i * 0.325
        ax.add_patch(Rectangle((x, 0.12), box_w, 0.74, facecolor=refined.blend(color, 0.08), edgecolor="#D8DEE9", linewidth=1.0))
        ax.add_patch(Rectangle((x, 0.12), 0.010, 0.74, facecolor=color, edgecolor=color, linewidth=0))
        add_text(ax, x + 0.025, 0.66, title, size=12.8, color=refined.INK, ha="left", weight="bold")
        add_text(ax, x + 0.025, 0.40, body, size=11.4, color=refined.INK, ha="left")


def build_figure() -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PPT_PACK_DIR.mkdir(parents=True, exist_ok=True)
    refined.setup_style()
    rows_by_method = load_case()

    fig = plt.figure(figsize=(16, 9), facecolor=refined.PAPER)
    fig.text(
        0.040,
        0.955,
        "逐步奖励归因：为什么本文方法能到达目标",
        ha="left",
        va="top",
        fontsize=25,
        fontproperties=refined.CN_FONT,
        color=refined.INK,
    )
    fig.text(
        0.040,
        0.918,
        "同一任务下逐步对齐：比较每个方法的距离变化与该步总奖励反馈",
        ha="left",
        va="top",
        fontsize=13.5,
        fontproperties=refined.CN_FONT,
        color=refined.MUTED,
    )

    ax_matrix = fig.add_axes([0.020, 0.165, 0.960, 0.745])
    draw_matrix(ax_matrix, rows_by_method)

    ax_summary = fig.add_axes([0.045, 0.030, 0.910, 0.145])
    draw_summary(ax_summary)

    stem = "20_奖励机制_case04_逐步总奖励归因矩阵"
    out_png = OUT_DIR / f"{stem}.png"
    pack_png = PPT_PACK_DIR / f"{stem}.png"
    fig.savefig(out_png, dpi=240, facecolor=refined.PAPER)
    plt.close(fig)
    shutil.copy2(out_png, pack_png)
    return out_png, pack_png


def main() -> int:
    out_png, pack_png = build_figure()
    print(f"saved: {out_png}")
    print(f"copied: {pack_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
