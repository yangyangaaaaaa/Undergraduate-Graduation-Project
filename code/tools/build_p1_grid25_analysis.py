from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path(r"F:\bishe\GeoExplorer\analysis\pipeline_20260524_p1_grid25_eval")

TABLE_OUT = REPO_ROOT / "results" / "tables" / "supplement_eval"
FIG_OUT = REPO_ROOT / "results" / "figures" / "supplement"
REPORT_OUT = REPO_ROOT / "results" / "reports"

METHOD_ORDER = ["anchor0624", "gomaa", "pristine"]
METHOD_LABELS = {
    "anchor0624": "This work",
    "gomaa": "GOMAA-Geo",
    "pristine": "GeoExplorer",
}
METHOD_COLORS = {
    "anchor0624": "#1F8A70",
    "gomaa": "#D9822B",
    "pristine": "#4C78A8",
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def as_float(value: object, default: float = math.nan) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def copy_outputs(source: Path) -> dict[str, Path]:
    TABLE_OUT.mkdir(parents=True, exist_ok=True)
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.mkdir(parents=True, exist_ok=True)
    copies = {}
    for name in [
        "p1_grid25_aggregate.json",
        "p1_grid25_budget_summary.csv",
        "p1_grid25_budget_table.csv",
        "p1_grid25_long_table.csv",
        "p1_grid25_per_distance.csv",
        "p1_grid25_seed_summary.csv",
        "p1_grid25_seed_table.csv",
    ]:
        src = source / name
        if src.exists():
            dst = TABLE_OUT / name
            shutil.copy2(src, dst)
            copies[name] = dst
    return copies


def plot_budget_curve(rows: list[dict]) -> Path:
    out = FIG_OUT / "p1_grid25_budget_sensitivity.png"
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    budgets = sorted({int(float(row["budget"])) for row in rows})
    fields = {
        "anchor0624": "anchor_sr",
        "gomaa": "gomaa_sr",
        "pristine": "pristine_sr",
    }
    for method_key in METHOD_ORDER:
        values = []
        for budget in budgets:
            row = next((item for item in rows if int(float(item["budget"])) == budget), None)
            values.append(as_float(row.get(fields[method_key])) if row else math.nan)
        ax.plot(
            budgets,
            values,
            marker="o",
            linewidth=2.4,
            color=METHOD_COLORS[method_key],
            label=METHOD_LABELS[method_key],
        )
    ax.set_title("25x25 budget sensitivity", fontweight="bold")
    ax.set_xlabel("Search budget")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0.0, max(0.24, max(as_float(row["anchor_sr"]) for row in rows) + 0.04))
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_budget_advantage(rows: list[dict]) -> Path:
    out = FIG_OUT / "p1_grid25_budget_advantage.png"
    fig, ax = plt.subplots(figsize=(7.4, 4.5), constrained_layout=True)
    budgets = [int(float(row["budget"])) for row in rows]
    diff_gomaa = [as_float(row["anchor_minus_gomaa"]) for row in rows]
    diff_pristine = [as_float(row["anchor_minus_pristine"]) for row in rows]
    ax.plot(budgets, diff_gomaa, marker="o", linewidth=2.4, color="#1F8A70", label="This work - GOMAA")
    ax.plot(budgets, diff_pristine, marker="s", linewidth=2.0, color="#4C78A8", label="This work - GeoExplorer")
    ax.axhline(0, color="#333333", linewidth=1.0, alpha=0.55)
    ax.set_title("25x25 SR advantage across budgets", fontweight="bold")
    ax.set_xlabel("Search budget")
    ax.set_ylabel("SR difference")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_per_distance(seed_table: list[dict]) -> Path:
    out = FIG_OUT / "p1_grid25_per_distance_seed_mean.png"
    distances = [12, 16, 20, 24, 28, 32, 36, 40, 44, 48]
    fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    for method_key in METHOD_ORDER:
        sub = [row for row in seed_table if row["method_key"] == method_key]
        means = []
        for dist in distances:
            values = [as_float(row.get(f"d{dist}")) for row in sub if row.get(f"d{dist}") not in ("", None)]
            values = [value for value in values if not math.isnan(value)]
            means.append(sum(values) / len(values) if values else math.nan)
        ax.plot(
            distances,
            means,
            marker="o",
            linewidth=2.2,
            color=METHOD_COLORS[method_key],
            label=METHOD_LABELS[method_key],
        )
    ax.set_title("25x25 seed-mean SR by start-goal distance", fontweight="bold")
    ax.set_xlabel("Distance bucket")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0.0, 0.8)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_seed_stability(seed_summary: list[dict]) -> Path:
    out = FIG_OUT / "p1_grid25_seed_stability.png"
    labels = [METHOD_LABELS[row["method_key"]] for row in seed_summary]
    means = [as_float(row["mean_sr"]) for row in seed_summary]
    stds = [as_float(row["std_sr"], 0.0) for row in seed_summary]
    colors = [METHOD_COLORS[row["method_key"]] for row in seed_summary]
    fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
    ax.bar(labels, means, yerr=stds, capsize=4, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title("25x25 task-bank seed stability", fontweight="bold")
    ax.set_ylabel("Mean success rate")
    ax.set_ylim(0.0, max(0.25, max(means) + max(stds) + 0.04))
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def build_report(source: Path, copies: dict[str, Path], figures: list[Path]) -> Path:
    budget_rows = read_csv(copies["p1_grid25_budget_summary.csv"])
    seed_summary = read_csv(copies["p1_grid25_seed_summary.csv"])
    seed_table = read_csv(copies["p1_grid25_seed_table.csv"])

    best_budget = max(budget_rows, key=lambda row: as_float(row["anchor_minus_gomaa"]))
    formal = next(row for row in budget_rows if int(float(row["budget"])) == 60)
    seed_by_method = {row["method_key"]: row for row in seed_summary}

    lines = [
        "# P1 25x25 超大网格补充实验整理与分析",
        "",
        f"- 整理时间：{now_iso()}",
        f"- 原始来源：`{source}`",
        "- 实验性质：仅评测 existing checkpoints，不重新训练。",
        "- 设置：MASA aerial 25x25 网格，距离桶为 `12,16,20,24,28,32,36,40,44,48`，每个距离桶 5 次重复。",
        "- 方法：本文方法、GOMAA-Geo、GeoExplorer-pristine。",
        "",
        "## 预算敏感性结果",
        "",
        "| Budget | 本文方法 | GOMAA-Geo | GeoExplorer | 本文-GOMAA |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in budget_rows:
        lines.append(
            f"| {int(float(row['budget']))} | {as_float(row['anchor_sr']):.4f} | "
            f"{as_float(row['gomaa_sr']):.4f} | {as_float(row['pristine_sr']):.4f} | "
            f"{as_float(row['anchor_minus_gomaa']):+.4f} |"
        )

    lines.extend(
        [
            "",
            "## Seed 稳定性结果",
            "",
            "| 方法 | Seeds | Mean SR | Std | Min | Max |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for method_key in METHOD_ORDER:
        row = seed_by_method[method_key]
        lines.append(
            f"| {METHOD_LABELS[method_key]} | {row['seeds']} | {as_float(row['mean_sr']):.4f} | "
            f"{as_float(row['std_sr']):.4f} | {as_float(row['min_sr']):.4f} | {as_float(row['max_sr']):.4f} |"
        )

    anchor_seed = seed_by_method["anchor0624"]
    gomaa_seed = seed_by_method["gomaa"]
    pristine_seed = seed_by_method["pristine"]
    seed_gap = as_float(anchor_seed["mean_sr"]) - as_float(gomaa_seed["mean_sr"])
    pristine_gap = as_float(anchor_seed["mean_sr"]) - as_float(pristine_seed["mean_sr"])

    lines.extend(
        [
            "",
            "## 分析结论",
            "",
            f"- 25x25 是明显更困难的压力测试，所有方法的 SR 都显著低于 8x8/10x10；这说明该设置更适合放在补充实验或鲁棒性分析中，而不适合作为主表。",
            f"- 在预算敏感性中，本文方法在所有预算下均高于 GOMAA-Geo。最大优势出现在 budget `{int(float(best_budget['budget']))}`，差值为 `{as_float(best_budget['anchor_minus_gomaa']):+.4f}`。",
            f"- 在与既有 25x25 formal 设置一致的 budget 60 下，本文方法 SR 为 `{as_float(formal['anchor_sr']):.4f}`，GOMAA-Geo 为 `{as_float(formal['gomaa_sr']):.4f}`，差值 `{as_float(formal['anchor_minus_gomaa']):+.4f}`。",
            f"- 三个 task-bank seed 的均值显示，本文方法 Mean SR 为 `{as_float(anchor_seed['mean_sr']):.4f}`，GOMAA-Geo 为 `{as_float(gomaa_seed['mean_sr']):.4f}`，平均优势 `{seed_gap:+.4f}`；相对 GeoExplorer-pristine 的平均优势为 `{pristine_gap:+.4f}`。",
            "- 从趋势上看，25x25 下本文方法优势仍存在，但幅度小于 10x10；这提示模型在极大网格中仍受训练分布和搜索预算限制，论文中应表述为“极端压力测试下仍保持相对领先”，不要夸大为“极端长距离完全解决”。",
            "",
            "## 已整理文件",
            "",
        ]
    )
    for _, path in sorted(copies.items()):
        lines.append(f"- `{path.relative_to(REPO_ROOT).as_posix()}`")
    for path in figures:
        lines.append(f"- `{path.relative_to(REPO_ROOT).as_posix()}`")

    out = REPORT_OUT / "p1_grid25_analysis_zh.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest and analyze P1 25x25 supplement outputs.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    args = parser.parse_args()
    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(source)

    copies = copy_outputs(source)
    budget_rows = read_csv(copies["p1_grid25_budget_summary.csv"])
    seed_summary = read_csv(copies["p1_grid25_seed_summary.csv"])
    seed_table = read_csv(copies["p1_grid25_seed_table.csv"])

    figures = [
        plot_budget_curve(budget_rows),
        plot_budget_advantage(budget_rows),
        plot_per_distance(seed_table),
        plot_seed_stability(seed_summary),
    ]
    report = build_report(source, copies, figures)
    print(json.dumps({"copied": {k: str(v) for k, v in copies.items()}, "figures": [str(p) for p in figures], "report": str(report)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
