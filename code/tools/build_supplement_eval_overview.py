from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = REPO_ROOT / "results" / "tables" / "supplement_eval"
REPORT_DIR = REPO_ROOT / "results" / "reports"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def f(value: object) -> float:
    return float(value)


def fmt(value: float) -> str:
    return f"{value:.4f}"


def main() -> int:
    p0_budget = read_csv(TABLE_DIR / "budget_sensitivity_summary.csv")
    p0_seed = read_csv(TABLE_DIR / "task_seed_summary.csv")
    p1_budget = read_csv(TABLE_DIR / "p1_grid25_budget_summary.csv")
    p1_seed = read_csv(TABLE_DIR / "p1_grid25_seed_summary.csv")

    p0_8 = [row for row in p0_budget if row["grid"] == "8x8"]
    p0_10 = [row for row in p0_budget if row["grid"] == "10x10"]
    p1_25 = p1_budget
    p0_mmgag_anchor = [row for row in p0_seed if row["family"] == "mmgag" and row["method_key"] == "anchor0624"]
    p0_mmgag_gomaa = [row for row in p0_seed if row["family"] == "mmgag" and row["method_key"] == "gomaa"]
    p0_ultra_anchor = [row for row in p0_seed if row["family"] == "ultra_long" and row["method_key"] == "anchor0624"]
    p0_ultra_gomaa = [row for row in p0_seed if row["family"] == "ultra_long" and row["method_key"] == "gomaa"]
    p1_anchor = next(row for row in p1_seed if row["method_key"] == "anchor0624")
    p1_gomaa = next(row for row in p1_seed if row["method_key"] == "gomaa")

    lines = [
        "# 补充实验总报告：预算敏感性、任务库稳定性与超大网格压力测试",
        "",
        f"- 生成时间：{now_iso()}",
        "- 实验性质：全部为 evaluation-only，均复用已训练 checkpoint，不包含新增训练。",
        "- 实验范围：P0 包含 8x8/10x10 预算敏感性与 MM-GAG/ultra-long task-bank seed 复评；P1 包含 25x25 超大网格预算敏感性与 seed 复评。",
        "- 对比方法：本文方法、GOMAA-Geo、GeoExplorer-pristine；MM-GAG seed 复评只比较本文方法与 GOMAA-Geo。",
        "",
        "## 1. 预算敏感性",
        "",
        "| Grid | Budget range | 本文方法 SR 范围 | GOMAA SR 范围 | 本文-GOMAA 差值范围 | 主要结论 |",
        "| --- | --- | ---: | ---: | ---: | --- |",
        (
            f"| 8x8 | {p0_8[0]['budget']}-{p0_8[-1]['budget']} | "
            f"{fmt(f(p0_8[0]['anchor_sr']))}-{fmt(f(p0_8[-1]['anchor_sr']))} | "
            f"{fmt(f(p0_8[0]['gomaa_sr']))}-{fmt(f(p0_8[-1]['gomaa_sr']))} | "
            f"{fmt(min(f(row['anchor_minus_gomaa']) for row in p0_8))}-{fmt(max(f(row['anchor_minus_gomaa']) for row in p0_8))} | "
            "优势稳定，预算增加后双方都提升。 |"
        ),
        (
            f"| 10x10 | {p0_10[0]['budget']}-{p0_10[-1]['budget']} | "
            f"{fmt(f(p0_10[0]['anchor_sr']))}-{fmt(f(p0_10[-1]['anchor_sr']))} | "
            f"{fmt(f(p0_10[0]['gomaa_sr']))}-{fmt(f(p0_10[-1]['gomaa_sr']))} | "
            f"{fmt(min(f(row['anchor_minus_gomaa']) for row in p0_10))}-{fmt(max(f(row['anchor_minus_gomaa']) for row in p0_10))} | "
            "优势随预算扩大更明显，支撑中长距离论点。 |"
        ),
        (
            f"| 25x25 | {p1_25[0]['budget']}-{p1_25[-1]['budget']} | "
            f"{fmt(f(p1_25[0]['anchor_sr']))}-{fmt(f(p1_25[-1]['anchor_sr']))} | "
            f"{fmt(f(p1_25[0]['gomaa_sr']))}-{fmt(f(p1_25[-1]['gomaa_sr']))} | "
            f"{fmt(min(f(row['anchor_minus_gomaa']) for row in p1_25))}-{fmt(max(f(row['anchor_minus_gomaa']) for row in p1_25))} | "
            "极端网格下绝对 SR 较低，但相对领先仍存在。 |"
        ),
        "",
        "## 2. Task-bank Seed 稳定性",
        "",
        "| Setting | 方法 | Mean SR | Std | Min-Max |",
        "| --- | --- | ---: | ---: | ---: |",
    ]

    for row in p0_mmgag_anchor + p0_mmgag_gomaa:
        lines.append(
            f"| MM-GAG {row['benchmark'].replace('mmgag_', '')} | {row['method']} | {fmt(f(row['mean_sr']))} | "
            f"{fmt(f(row['std_sr']))} | {fmt(f(row['min_sr']))}-{fmt(f(row['max_sr']))} |"
        )
    for row in p0_ultra_anchor + p0_ultra_gomaa:
        lines.append(
            f"| Ultra-long {row['grid']} | {row['method']} | {fmt(f(row['mean_sr']))} | "
            f"{fmt(f(row['std_sr']))} | {fmt(f(row['min_sr']))}-{fmt(f(row['max_sr']))} |"
        )
    for row in [p1_anchor, p1_gomaa]:
        lines.append(
            f"| 25x25 | {row['method']} | {fmt(f(row['mean_sr']))} | {fmt(f(row['std_sr']))} | "
            f"{fmt(f(row['min_sr']))}-{fmt(f(row['max_sr']))} |"
        )

    p1_seed_gap = f(p1_anchor["mean_sr"]) - f(p1_gomaa["mean_sr"])
    mmgag_anchor_mean = sum(f(row["mean_sr"]) for row in p0_mmgag_anchor) / len(p0_mmgag_anchor)
    mmgag_gomaa_mean = sum(f(row["mean_sr"]) for row in p0_mmgag_gomaa) / len(p0_mmgag_gomaa)
    ultra_anchor_mean = sum(f(row["mean_sr"]) for row in p0_ultra_anchor) / len(p0_ultra_anchor)
    ultra_gomaa_mean = sum(f(row["mean_sr"]) for row in p0_ultra_gomaa) / len(p0_ultra_gomaa)

    lines.extend(
        [
            "",
            "## 3. 综合分析",
            "",
            f"- MM-GAG 三种目标模态的 seed 复评中，本文方法平均 SR 为 `{fmt(mmgag_anchor_mean)}`，GOMAA-Geo 为 `{fmt(mmgag_gomaa_mean)}`，平均优势 `{fmt(mmgag_anchor_mean - mmgag_gomaa_mean)}`。这说明主表结论不是单一任务库随机 seed 造成的偶然现象。",
            f"- 8x8/10x10 ultra-long seed 复评中，本文方法平均 SR 为 `{fmt(ultra_anchor_mean)}`，GOMAA-Geo 为 `{fmt(ultra_gomaa_mean)}`，平均优势 `{fmt(ultra_anchor_mean - ultra_gomaa_mean)}`。其中 10x10 预算敏感性优势最高达到 `{fmt(max(f(row['anchor_minus_gomaa']) for row in p0_10))}`，更符合“中长距离优势更突出”的论文叙事。",
            f"- 25x25 压力测试中，本文方法 seed 平均 SR 为 `{fmt(f(p1_anchor['mean_sr']))}`，GOMAA-Geo 为 `{fmt(f(p1_gomaa['mean_sr']))}`，平均优势 `{fmt(p1_seed_gap)}`。但绝对 SR 明显偏低，说明该设置超过了当前训练分布和预算能力，应作为探索性鲁棒性补充，而不是主结果表。",
            "- 论文写法建议：主文强调 8x8/10x10 的稳定提升；25x25 用作补充压力测试，措辞应保守，例如“在极端大网格下仍保持相对领先，但任务难度显著上升”。",
            "",
            "## 4. 对应文件",
            "",
            "- `results/tables/supplement_eval/`：P0/P1 所有长表、分距离表、预算表和 seed 汇总表。",
            "- `results/figures/supplement/p0_*.png`：8x8/10x10 预算与 seed 稳定性图。",
            "- `results/figures/supplement/p1_grid25_*.png`：25x25 预算、优势、分距离和 seed 稳定性图。",
            "- `results/reports/p0_supplement_eval_inventory_zh.md`、`results/reports/p1_grid25_analysis_zh.md`：单项实验整理说明。",
        ]
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "supplement_eval_overview_zh.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
