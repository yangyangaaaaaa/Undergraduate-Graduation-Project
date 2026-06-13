#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build a same-task training evolution storyboard.

This figure is intentionally visual rather than tabular: it shows how routes on
the same real overhead image evolve across training progress for multiple
reward settings.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

import build_reward_guided_case_studies as base


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures" / "defense_reward_training_stage" / "evolution_cases"
REPORTS = RESULTS / "reports"
REPORT_PATH = REPORTS / "same_task_evolution_visual_zh.md"

TRAIN_LOG_ROOT = Path(r"F:\bishe\GeoExplorer\analysis\pipeline_20260603_defense_reward_training_curves\training_logs")

SEED = 123
IMAGE_INDEX = 20
DISTANCE_BUCKET = "C8"
INITIAL_PATCH = 0
GOAL_PATCH = 24

METHODS = [
    "external_only",
    "mixed_no_gate_no_pbrs",
    "mixed_gate_only",
    "proposed_linear_gate_pbrs",
]

METHOD_LABEL = {
    "external_only": "Ext",
    "mixed_no_gate_no_pbrs": "Ext+Int",
    "mixed_gate_only": "Gate",
    "proposed_linear_gate_pbrs": "Ours",
}

METHOD_LINESTYLE = {
    "external_only": (0, (7, 3)),
    "mixed_no_gate_no_pbrs": (0, (6, 2, 1, 2)),
    "mixed_gate_only": (0, (7, 2)),
    "proposed_linear_gate_pbrs": "solid",
}


def route_goal_and_path(row: pd.Series) -> tuple[int, list[int]]:
    seq = [int(x) for x in base.parse_list(row.get("patch_sequence", ""))]
    if len(seq) >= 2:
        return int(seq[0]), [int(x) for x in seq[1:]]
    return int(row["goal_patch"]), [int(row["initial_patch"]), int(row["final_patch"])]


def read_rows() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for method in METHODS:
        path = TRAIN_LOG_ROOT / f"{method}_seed{SEED}_t480k" / "training_route_samples.csv"
        df = pd.read_csv(path)
        mask = (
            df["image_index"].astype(int).eq(IMAGE_INDEX)
            & df["distance_bucket"].astype(str).eq(DISTANCE_BUCKET)
            & df["initial_patch"].astype(int).eq(INITIAL_PATCH)
            & df["goal_patch"].astype(int).eq(GOAL_PATCH)
        )
        rows = df[mask].sort_values("run_progress").copy()
        if rows.empty:
            raise RuntimeError(f"No matching rows for {method}")
        out[method] = rows
    return out


def draw_panel(ax: plt.Axes, image, row: pd.Series, method: str) -> None:
    ax.imshow(image, extent=(-0.5, base.PATCH_SIZE - 0.5, base.PATCH_SIZE - 0.5, -0.5), zorder=0)
    ax.set_xlim(-0.5, base.PATCH_SIZE - 0.5)
    ax.set_ylim(base.PATCH_SIZE - 0.5, -0.5)
    ax.set_aspect("equal")
    base.draw_grid(ax)

    goal, path = route_goal_and_path(row)
    xy = [base.patch_xy(p) for p in path]
    if xy:
        xs, ys = zip(*xy)
        key = method == "proposed_linear_gate_pbrs"
        color = base.METHOD_STYLE[method]["color"]
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=5.0 if key else 4.0,
            linestyle=METHOD_LINESTYLE[method],
            alpha=1.0 if key else 0.88,
            solid_capstyle="round",
            zorder=10,
            path_effects=[pe.Stroke(linewidth=8.0 if key else 6.6, foreground="white", alpha=0.78), pe.Normal()],
        )
        fx, fy = xy[-1]
        success = int(row["success"]) == 1
        ax.scatter(
            [fx],
            [fy],
            s=116,
            marker="o" if success else "X",
            color="#10B981" if success else "#EF4444",
            edgecolor="white",
            linewidth=1.6,
            zorder=14,
        )
        sx, sy = xy[0]
        ax.scatter([sx], [sy], s=130, marker="o", color="#10B981", edgecolor="white", linewidth=1.8, zorder=15)

    gx, gy = base.patch_xy(goal)
    ax.scatter([gx], [gy], s=175, marker="*", color=base.YELLOW, edgecolor=base.INK, linewidth=1.0, zorder=15)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def build_figure(rows_by_method: dict[str, pd.DataFrame]) -> Path:
    mix_index, masa_metadata, mmgag_index = base.load_image_mapping()
    asset = base.resolve_image_asset(IMAGE_INDEX, mix_index, masa_metadata, mmgag_index)
    if asset is None:
        raise RuntimeError(f"Cannot resolve image asset for image {IMAGE_INDEX}")
    image = base.open_overhead_image(asset)

    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    fig.patch.set_facecolor(base.PAPER)
    fig.text(0.030, 0.955, "Same Task Evolution", fontsize=23, fontweight="bold", ha="left", va="top")
    fig.text(0.030, 0.918, "C8 | seed123 | img20 | s0 -> g24", fontsize=11.8, color=base.MUTED, ha="left", va="top")
    fig.lines.append(Line2D([0.030, 0.985], [0.890, 0.890], transform=fig.transFigure, color="#CBD5E1", lw=1.1))

    for c, method in enumerate(METHODS):
        rows = rows_by_method[method]
        selected = [rows.iloc[0], rows.iloc[-1]]
        for r, row in enumerate(selected):
            ax = axes[r, c]
            draw_panel(ax, image, row, method)
            if c == 0:
                ax.text(
                    -0.11,
                    0.50,
                    "Early" if r == 0 else "Late",
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=15.0,
                    fontweight="bold",
                    rotation=90,
                    color=base.INK,
                )
            ax.set_title(
                f"{METHOD_LABEL[method]} | {float(row['run_progress']) * 100:.0f}%",
                fontsize=14.2,
                fontweight="bold",
                color=base.METHOD_STYLE[method]["color"],
                pad=6,
            )

    plt.subplots_adjust(left=0.055, right=0.985, top=0.835, bottom=0.045, wspace=0.035, hspace=0.155)
    out = FIGURES / "same_task_evolution_c8_seed123_img20_s0_g24.png"
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return out


def write_report(out: Path, rows_by_method: dict[str, pd.DataFrame]) -> None:
    manifest = {
        method: [
            {
                "episode": int(row["episode"]),
                "run_progress": float(row["run_progress"]),
                "success": int(row["success"]),
                "final_dist": int(row["final_dist"]),
                "progress_steps": int(row["progress_steps"]),
                "regress_steps": int(row["regress_steps"]),
                "patch_sequence": row["patch_sequence"],
                "dist_sequence": row["dist_sequence"],
            }
            for _, row in rows.iterrows()
        ]
        for method, rows in rows_by_method.items()
    }
    (FIGURES / "same_task_evolution_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 同一任务训练演化图说明",
        "",
        "这张图不展示表格数值，而是展示同一任务在训练过程中的路线行为变化。",
        "",
        f"- 输出图：`{out}`",
        f"- SVG：`{out.with_suffix('.svg')}`",
        f"- 清单：`{FIGURES / 'same_task_evolution_manifest.json'}`",
        "",
        "## 图面含义",
        "",
        "- 列表示奖励设置：`Ext`、`Ext+Int`、`Gate`、`Ours`。",
        "- 上排是同一任务的训练早期路线，下排是训练后期路线。",
        "- 绿色圆点为起点，黄色星标为目标；绿色终点表示到达，红色叉表示未到达。",
        "",
        "## 为什么这类图有价值",
        "",
        "表格只能说最后是否成功、最终距离是多少；这张图展示的是策略如何在同一空间任务上改变行动路径。为了减少空白，主图只保留早期和后期两个阶段。这个案例中，三个对照在后期仍停在目标附近但未到达，而本文方法在后期形成连续接近目标的路线并成功到达。",
        "",
        "## 建议使用",
        "",
        "这页适合作为动作归因图之后的补充页：先用动作归因说明奖励如何给动作反馈，再用这张同一任务演化图说明反馈最终怎样改变路线行为。",
        "",
        "## 表述边界",
        "",
        "该图来自训练日志中的真实采样片段，解释训练阶段学习过程；正式测试仍以固定 checkpoint 评估和论文结果表格为准。",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> None:
    base.setup_style()
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows_by_method = read_rows()
    out = build_figure(rows_by_method)
    write_report(out, rows_by_method)
    print({"figure": str(out), "svg": str(out.with_suffix(".svg")), "report": str(REPORT_PATH)})


if __name__ == "__main__":
    main()
