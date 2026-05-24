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


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path(r"F:\bishe\GeoExplorer\analysis\pipeline_20260524_p0_supplement_eval")

TABLE_OUT = REPO_ROOT / "results" / "tables" / "supplement_eval"
FIG_OUT = REPO_ROOT / "results" / "figures" / "supplement"
REPORT_OUT = REPO_ROOT / "results" / "reports"

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


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


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
        "p0_supplement_long_table.csv",
        "p0_supplement_per_distance.csv",
        "budget_sensitivity_table.csv",
        "budget_sensitivity_summary.csv",
        "task_seed_mmgag_table.csv",
        "task_seed_ultra_table.csv",
        "task_seed_summary.csv",
        "p0_supplement_aggregate.json",
    ]:
        src = source / name
        if src.exists():
            dst = TABLE_OUT / name
            shutil.copy2(src, dst)
            copies[name] = dst

    summary_src = source / "p0_supplement_summary_zh.md"
    if summary_src.exists():
        dst = REPORT_OUT / "p0_supplement_eval_summary_zh.md"
        shutil.copy2(summary_src, dst)
        copies["p0_supplement_summary_zh.md"] = dst
    return copies


def plot_budget_sensitivity(rows: list[dict]) -> Path:
    out = FIG_OUT / "p0_budget_sensitivity.png"
    grids = sorted({row["grid"] for row in rows})
    fig, axes = plt.subplots(1, len(grids), figsize=(6.2 * len(grids), 4.6), constrained_layout=True)
    if len(grids) == 1:
        axes = [axes]

    for ax, grid in zip(axes, grids):
        sub = [row for row in rows if row["grid"] == grid]
        budgets = sorted({int(float(row["budget"])) for row in sub})
        series = {
            "anchor0624": "anchor_sr",
            "gomaa": "gomaa_sr",
            "pristine": "pristine_sr",
        }
        for method_key, field in series.items():
            values = []
            for budget in budgets:
                match = next((row for row in sub if int(float(row["budget"])) == budget), None)
                values.append(as_float(match.get(field)) if match else math.nan)
            ax.plot(
                budgets,
                values,
                marker="o",
                linewidth=2.2,
                color=METHOD_COLORS[method_key],
                label=METHOD_LABELS[method_key],
            )
        ax.set_title(f"{grid} budget sensitivity", fontweight="bold")
        ax.set_xlabel("Search budget")
        ax.set_ylabel("Success rate")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)

    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_budget_advantage(rows: list[dict]) -> Path:
    out = FIG_OUT / "p0_budget_advantage.png"
    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    for grid in sorted({row["grid"] for row in rows}):
        sub = sorted([row for row in rows if row["grid"] == grid], key=lambda item: int(float(item["budget"])))
        ax.plot(
            [int(float(row["budget"])) for row in sub],
            [as_float(row["anchor_minus_gomaa"]) for row in sub],
            marker="o",
            linewidth=2.4,
            label=grid,
        )
    ax.axhline(0, color="#333333", linewidth=1.0, alpha=0.6)
    ax.set_title("This work minus GOMAA under different budgets", fontweight="bold")
    ax.set_xlabel("Search budget")
    ax.set_ylabel("SR difference")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Grid", frameon=False)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_seed_stability(rows: list[dict]) -> Path:
    out = FIG_OUT / "p0_task_seed_stability.png"
    ordered = sorted(rows, key=lambda row: (row["family"], row["benchmark"], row["method_key"]))
    labels = [f"{row['benchmark']}\n{METHOD_LABELS.get(row['method_key'], row['method_key'])}" for row in ordered]
    means = [as_float(row["mean_sr"]) for row in ordered]
    errors = [as_float(row["std_sr"], 0.0) for row in ordered]
    colors = [METHOD_COLORS.get(row["method_key"], "#777777") for row in ordered]

    fig, ax = plt.subplots(figsize=(max(9.5, 0.42 * len(labels)), 5.4), constrained_layout=True)
    x = list(range(len(labels)))
    ax.bar(x, means, yerr=errors, capsize=3, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Mean success rate")
    ax.set_title("Task-bank seed stability", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def build_report(source: Path, copies: dict[str, Path], figures: list[Path]) -> Path:
    budget_rows = read_csv(copies["budget_sensitivity_summary.csv"]) if "budget_sensitivity_summary.csv" in copies else []
    seed_rows = read_csv(copies["task_seed_summary.csv"]) if "task_seed_summary.csv" in copies else []

    best_budget = None
    if budget_rows:
        best_budget = max(budget_rows, key=lambda row: as_float(row.get("anchor_minus_gomaa")))

    lines = [
        "# P0 补充评测整理说明",
        "",
        f"- 整理时间：{now_iso()}",
        f"- 原始来源：`{source}`",
        "- 实验性质：仅评测 existing checkpoints，不重新训练。",
        "- 主要用途：补强预算敏感性、任务库 seed 稳定性和超长距离结论的可信度。",
        "",
        "## 关键观察",
        "",
    ]
    if best_budget:
        lines.append(
            f"- 预算敏感性中，当前最大优势出现在 `{best_budget['grid']}`、budget `{best_budget['budget']}`："
            f"本文方法相对 GOMAA-Geo 的 SR 差值为 `{as_float(best_budget['anchor_minus_gomaa']):+.4f}`。"
        )
    if seed_rows:
        ultra = [row for row in seed_rows if row["family"] == "ultra_long" and row["method_key"] == "anchor0624"]
        mmgag = [row for row in seed_rows if row["family"] == "mmgag" and row["method_key"] == "anchor0624"]
        if ultra:
            mean_ultra = sum(as_float(row["mean_sr"]) for row in ultra) / len(ultra)
            lines.append(f"- Ultra-long seed 复评中，本文方法在可达高距离端的平均 SR 约为 `{mean_ultra:.4f}`。")
        if mmgag:
            mean_mmgag = sum(as_float(row["mean_sr"]) for row in mmgag) / len(mmgag)
            lines.append(f"- MM-GAG seed 复评用于检查任务库随机性的影响，本文方法三种目标模态平均 SR 约为 `{mean_mmgag:.4f}`。")

    lines.extend(["", "## 已整理文件", ""])
    for name, path in sorted(copies.items()):
        rel = path.relative_to(REPO_ROOT)
        lines.append(f"- `{rel.as_posix()}`")
    for path in figures:
        rel = path.relative_to(REPO_ROOT)
        lines.append(f"- `{rel.as_posix()}`")

    out = REPORT_OUT / "p0_supplement_eval_inventory_zh.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest P0 supplement evaluation outputs into the graduation-project repo.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Downloaded P0 supplement output directory.")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Source directory not found: {source}")

    copies = copy_outputs(source)
    figures = []
    budget_path = copies.get("budget_sensitivity_summary.csv")
    if budget_path:
        budget_rows = read_csv(budget_path)
        figures.append(plot_budget_sensitivity(budget_rows))
        figures.append(plot_budget_advantage(budget_rows))
    seed_path = copies.get("task_seed_summary.csv")
    if seed_path:
        figures.append(plot_seed_stability(read_csv(seed_path)))

    report = build_report(source, copies, figures)
    print(json.dumps({"copied": {k: str(v) for k, v in copies.items()}, "figures": [str(p) for p in figures], "report": str(report)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
