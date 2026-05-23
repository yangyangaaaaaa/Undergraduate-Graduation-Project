from __future__ import annotations

import ast
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(r"F:\bishe\GeoExplorer\analysis")

RESULTS = REPO_ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures" / "supplement"
REPORTS = RESULTS / "reports"

MAIN_RAW_ROOT = SOURCE_ROOT / "pipeline_20260516_paper_baseline_compare"
ULTRA_RAW_ROOTS = [
    SOURCE_ROOT / "pipeline_20260521_ultra_long_grid_stress_v2",
    SOURCE_ROOT / "pipeline_20260521_ultra_long_grid_stress_v3_grid25",
]
TRAJECTORY_CSV = TABLES / "main_benchmark" / "trajectory_records.csv"
REWARD_RECORDS = SOURCE_ROOT / "pipeline_20260518_reward_visualization" / "reward_vis_records.remote.json"


METHOD_ORDER = ["GeoExplorer-anchor0624", "GOMAA-Geo", "GeoExplorer-pristine"]
METHOD_COLORS = {
    "GeoExplorer-anchor0624": "#1F8A70",
    "GOMAA-Geo": "#D9822B",
    "GeoExplorer-pristine": "#4C78A8",
    "Random policy": "#8A8F98",
    "DiT-AGL": "#7B61FF",
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def ensure_dirs() -> None:
    for path in [
        TABLES / "statistical_analysis",
        TABLES / "trajectory_analysis",
        TABLES / "reward_process",
        TABLES / "experiment_plan",
        FIGURES,
        REPORTS,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        keys = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def fmt(value: float, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.{digits}f}"


def proportion_diff_ci(p1: float, n1: int, p0: float, n0: int, z: float = 1.96) -> tuple[float, float, float]:
    diff = p1 - p0
    se = math.sqrt(max(p1 * (1.0 - p1), 0.0) / max(n1, 1) + max(p0 * (1.0 - p0), 0.0) / max(n0, 1))
    return diff, diff - z * se, diff + z * se


def gather_main_raw() -> tuple[list[dict], dict[tuple[str, str], dict]]:
    rows = []
    by_key = {}
    for method_dir in sorted(MAIN_RAW_ROOT.iterdir()):
        if not method_dir.is_dir():
            continue
        for path in sorted(method_dir.glob("*.json")):
            payload = load_json(path)
            if "modes" not in payload or "benchmark" not in payload:
                continue
            mode = payload["modes"][0]
            method = payload.get("method", method_dir.name)
            benchmark = payload["benchmark"]
            row = {
                "method": method,
                "method_dir": method_dir.name,
                "benchmark": benchmark,
                "trials": int(mode.get("total_trials", payload.get("num_tasks", 0))),
                "success": int(mode.get("success", round(payload.get("success_ratio", 0) * payload.get("num_tasks", 0)))),
                "sr": float(mode.get("success_ratio", payload.get("success_ratio", math.nan))),
                "sr_ci_low": float(mode.get("success_ratio_ci_low", math.nan)),
                "sr_ci_high": float(mode.get("success_ratio_ci_high", math.nan)),
                "sg": float(mode.get("sg_mean", payload.get("sg_mean", math.nan))),
                "output_source": str(path),
            }
            for item in mode.get("per_dist", payload.get("per_distance", [])):
                distance = int(item["distance"])
                row[f"d{distance}_trials"] = int(item.get("trials", 0))
                row[f"d{distance}_success"] = int(item.get("success", 0))
                row[f"d{distance}_sr"] = float(item.get("success_ratio", math.nan))
                row[f"d{distance}_ci_low"] = float(item.get("success_ratio_ci_low", math.nan))
                row[f"d{distance}_ci_high"] = float(item.get("success_ratio_ci_high", math.nan))
            rows.append(row)
            by_key[(method, benchmark)] = row
    return rows, by_key


def build_main_ci_tables() -> tuple[list[dict], list[dict]]:
    rows, by_key = gather_main_raw()
    ci_path = TABLES / "statistical_analysis" / "main_benchmark_ci_table.csv"
    fieldnames = [
        "method",
        "benchmark",
        "trials",
        "success",
        "sr",
        "sr_ci_low",
        "sr_ci_high",
        "sg",
        "d4_sr",
        "d5_sr",
        "d6_sr",
        "d7_sr",
        "d8_sr",
        "output_source",
    ]
    write_csv(ci_path, rows, fieldnames=fieldnames)

    diff_rows = []
    anchor_method = "GeoExplorer-anchor0624"
    baselines = ["GOMAA-Geo", "Random policy", "DiT-AGL"]
    for benchmark in sorted({key[1] for key in by_key if key[0] == anchor_method}):
        anchor = by_key[(anchor_method, benchmark)]
        for baseline_method in baselines:
            baseline = by_key.get((baseline_method, benchmark))
            if not baseline:
                continue
            diff, low, high = proportion_diff_ci(anchor["sr"], anchor["trials"], baseline["sr"], baseline["trials"])
            row = {
                "benchmark": benchmark,
                "baseline": baseline_method,
                "ours_sr": anchor["sr"],
                "baseline_sr": baseline["sr"],
                "diff": diff,
                "diff_ci_low": low,
                "diff_ci_high": high,
                "ours_trials": anchor["trials"],
                "baseline_trials": baseline["trials"],
            }
            for distance in range(4, 9):
                key_trials = f"d{distance}_trials"
                key_sr = f"d{distance}_sr"
                if key_sr in anchor and key_sr in baseline:
                    d_diff, d_low, d_high = proportion_diff_ci(
                        anchor[key_sr], anchor.get(key_trials, 0), baseline[key_sr], baseline.get(key_trials, 0)
                    )
                    row[f"d{distance}_diff"] = d_diff
                    row[f"d{distance}_diff_ci_low"] = d_low
                    row[f"d{distance}_diff_ci_high"] = d_high
            diff_rows.append(row)

    write_csv(TABLES / "statistical_analysis" / "main_benchmark_diff_ci_table.csv", diff_rows)
    return rows, diff_rows


def build_ultra_ci_table() -> list[dict]:
    by_case = {}
    for root in ULTRA_RAW_ROOTS:
        if not root.exists():
            continue
        for path in sorted((root / "raw").glob("*/*/*/*.json")):
            payload = load_json(path)
            mode = payload["modes"][0]
            protocol = payload.get("protocol", {})
            method = payload.get("method", path.parent.name)
            raw_grid = protocol.get("grid", "")
            grid_text = raw_grid if isinstance(raw_grid, str) and "x" in raw_grid else f"{raw_grid}x{raw_grid}"
            key = (
                grid_text,
                int(protocol.get("budget", 0)),
                path.parent.name,
            )
            by_case[key] = {
                "grid": key[0],
                "budget": key[1],
                "method_key": path.parent.name,
                "method": method,
                "trials": int(mode["total_trials"]),
                "success": int(mode["success"]),
                "sr": float(mode["success_ratio"]),
                "sr_ci_low": float(mode["success_ratio_ci_low"]),
                "sr_ci_high": float(mode["success_ratio_ci_high"]),
                "sg": float(mode.get("sg_mean", math.nan)),
                "distances": ",".join(str(item) for item in protocol.get("distance_buckets", [])),
                "output_source": str(path),
            }

    rows = list(by_case.values())
    write_csv(TABLES / "statistical_analysis" / "ultra_long_ci_table.csv", rows)

    diff_rows = []
    for grid, budget in sorted({(row["grid"], row["budget"]) for row in rows}):
        anchors = [row for row in rows if row["grid"] == grid and row["budget"] == budget and row["method_key"] == "anchor0624"]
        if not anchors:
            continue
        anchor = anchors[0]
        for baseline in [row for row in rows if row["grid"] == grid and row["budget"] == budget and row["method_key"] != "anchor0624"]:
            diff, low, high = proportion_diff_ci(anchor["sr"], anchor["trials"], baseline["sr"], baseline["trials"])
            diff_rows.append(
                {
                    "grid": grid,
                    "budget": budget,
                    "baseline": baseline["method"],
                    "ours_sr": anchor["sr"],
                    "baseline_sr": baseline["sr"],
                    "diff": diff,
                    "diff_ci_low": low,
                    "diff_ci_high": high,
                    "ours_trials": anchor["trials"],
                    "baseline_trials": baseline["trials"],
                }
            )
    write_csv(TABLES / "statistical_analysis" / "ultra_long_diff_ci_table.csv", diff_rows)
    return diff_rows


def grid_distance(index_a: int, index_b: int, grid: int = 5) -> int:
    ra, ca = divmod(index_a, grid)
    rb, cb = divmod(index_b, grid)
    return abs(ra - rb) + abs(ca - cb)


def trajectory_metrics(row: dict) -> dict:
    traj = ast.literal_eval(row["traj"])
    goal = int(row["goal"])
    initial = int(row["distance"])
    path_length = max(int(row["path_length"]), 1)
    distances = [grid_distance(int(pos), goal) for pos in traj]
    decreasing = sum(1 for before, after in zip(distances, distances[1:]) if after < before)
    increasing = sum(1 for before, after in zip(distances, distances[1:]) if after > before)
    immediate_backtracks = sum(1 for i in range(2, len(traj)) if traj[i] == traj[i - 2])
    revisits = len(traj) - len(set(traj))
    return {
        "method": row["method"],
        "dataset": row["dataset"],
        "distance": int(row["distance"]),
        "case_id": row["case_id"],
        "success": row["success"].strip().lower() == "true",
        "final_distance": int(row["final_distance"]),
        "path_length": int(row["path_length"]),
        "optimal_steps": int(row["optimal_steps"]),
        "detour_steps": int(row["detour_steps"]),
        "progress_ratio": (initial - int(row["final_distance"])) / max(initial, 1),
        "monotonic_step_rate": decreasing / path_length,
        "regress_step_rate": increasing / path_length,
        "immediate_backtrack_rate": immediate_backtracks / max(path_length - 1, 1),
        "revisit_rate": revisits / max(len(traj), 1),
        "unique_coverage": len(set(traj)) / max(len(traj), 1),
    }


def aggregate_metrics(rows: list[dict], keys: list[str], metrics: list[str]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    out = []
    for group_key, items in sorted(grouped.items(), key=lambda item: item[0]):
        row = {key: value for key, value in zip(keys, group_key)}
        row["n"] = len(items)
        row["success_rate"] = sum(1 for item in items if item["success"]) / len(items)
        for metric in metrics:
            row[metric] = mean(float(item[metric]) for item in items)
        out.append(row)
    return out


def build_trajectory_tables() -> tuple[list[dict], list[dict], list[dict]]:
    raw_rows = read_csv(TRAJECTORY_CSV)
    metric_rows = [trajectory_metrics(row) for row in raw_rows]
    metrics = [
        "final_distance",
        "path_length",
        "detour_steps",
        "progress_ratio",
        "monotonic_step_rate",
        "regress_step_rate",
        "immediate_backtrack_rate",
        "revisit_rate",
        "unique_coverage",
    ]
    summary = aggregate_metrics(metric_rows, ["method"], metrics)
    by_distance = aggregate_metrics(metric_rows, ["method", "distance"], metrics)
    failures_c4 = [row for row in metric_rows if row["distance"] == 4 and not row["success"]]
    c4_failure = aggregate_metrics(
        failures_c4,
        ["method"],
        [
            "final_distance",
            "path_length",
            "detour_steps",
            "progress_ratio",
            "regress_step_rate",
            "immediate_backtrack_rate",
            "revisit_rate",
        ],
    )
    all_c4 = [row for row in metric_rows if row["distance"] == 4]
    c4_overall = aggregate_metrics(all_c4, ["method"], metrics)
    for row in c4_failure:
        total = next(item for item in c4_overall if item["method"] == row["method"])
        row["c4_total_n"] = total["n"]
        row["c4_success_rate"] = total["success_rate"]
        row["c4_failure_rate"] = 1.0 - total["success_rate"]

    write_csv(TABLES / "trajectory_analysis" / "trajectory_behavior_records.csv", metric_rows)
    write_csv(TABLES / "trajectory_analysis" / "trajectory_behavior_summary.csv", summary)
    write_csv(TABLES / "trajectory_analysis" / "trajectory_behavior_by_distance.csv", by_distance)
    write_csv(TABLES / "trajectory_analysis" / "c4_failure_profile.csv", c4_failure)
    return summary, by_distance, c4_failure


def sum_trace(record: dict, key: str) -> float:
    values = record.get(key, [])
    return float(sum(float(item) for item in values))


def mean_trace(record: dict, key: str) -> float:
    values = record.get(key, [])
    if not values:
        return math.nan
    return float(mean(float(item) for item in values))


def build_reward_tables() -> tuple[list[dict], list[dict]]:
    records = json.loads(REWARD_RECORDS.read_text(encoding="utf-8"))
    rows = []
    for record in records:
        rows.append(
            {
                "role": record.get("role", ""),
                "case_id": record["case_id"],
                "method_key": record["method_key"],
                "distance": int(record["distance"]),
                "success": bool(record["success"]),
                "path_length": int(record["path_length"]),
                "final_distance": int(record["final_distance"]),
                "sum_external": sum_trace(record, "r_ex_trace"),
                "sum_intrinsic_raw": sum_trace(record, "r_in_raw_trace"),
                "sum_intrinsic_gated": sum_trace(record, "r_in_gated_trace"),
                "sum_pbrs": sum_trace(record, "r_p_trace"),
                "sum_total": sum_trace(record, "r_total_trace"),
                "mean_gate": mean_trace(record, "gate_trace"),
                "first_gate": float(record.get("gate_trace", [math.nan])[0]),
                "last_gate": float(record.get("gate_trace", [math.nan])[-1]),
            }
        )
    write_csv(TABLES / "reward_process" / "reward_process_case_table.csv", rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method_key"], row["distance"], row["success"])].append(row)
    summary = []
    metrics = [
        "path_length",
        "final_distance",
        "sum_external",
        "sum_intrinsic_raw",
        "sum_intrinsic_gated",
        "sum_pbrs",
        "sum_total",
        "mean_gate",
        "first_gate",
        "last_gate",
    ]
    for (method_key, distance, success), items in sorted(grouped.items()):
        row = {"method_key": method_key, "distance": distance, "success": success, "n": len(items)}
        for metric in metrics:
            row[metric] = mean(float(item[metric]) for item in items)
        summary.append(row)
    write_csv(TABLES / "reward_process" / "reward_process_summary.csv", summary)
    return rows, summary


def plot_diff_ci(rows: list[dict], path: Path, title: str) -> None:
    if not rows:
        return
    labels = [f"{row.get('benchmark', row.get('grid'))}\nvs {row['baseline']}" for row in rows]
    diffs = np.array([float(row["diff"]) for row in rows])
    lows = np.array([float(row["diff_ci_low"]) for row in rows])
    highs = np.array([float(row["diff_ci_high"]) for row in rows])
    order = np.argsort(diffs)
    labels = [labels[i] for i in order]
    diffs = diffs[order]
    lows = lows[order]
    highs = highs[order]
    y = np.arange(len(labels))
    fig_h = max(4.0, len(labels) * 0.32)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    ax.barh(y, diffs, color="#1F8A70", alpha=0.82)
    ax.errorbar(diffs, y, xerr=[diffs - lows, highs - diffs], fmt="none", ecolor="#222222", elinewidth=1, capsize=2)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Success-rate difference (This work - baseline)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_trajectory(by_distance: list[dict], path: Path) -> None:
    metrics = [
        ("success_rate", "Success Rate"),
        ("progress_ratio", "Distance Reduction Ratio"),
        ("monotonic_step_rate", "Monotonic Step Rate"),
        ("revisit_rate", "Revisit Rate"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, (metric, label) in zip(axes.ravel(), metrics):
        for method in METHOD_ORDER:
            rows = sorted([row for row in by_distance if row["method"] == method], key=lambda item: int(item["distance"]))
            if not rows:
                continue
            ax.plot(
                [int(row["distance"]) for row in rows],
                [float(row[metric]) for row in rows],
                marker="o",
                linewidth=2,
                label=method.replace("GeoExplorer-", ""),
                color=METHOD_COLORS.get(method),
            )
        ax.set_title(label)
        ax.set_xlabel("Initial Distance C")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Trajectory Behavior on SwissViewMonuments Cases", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_c4_failure(c4_failure: list[dict], path: Path) -> None:
    methods = [row["method"] for row in c4_failure]
    metrics = [
        ("final_distance", "Final Distance"),
        ("revisit_rate", "Revisit Rate"),
        ("immediate_backtrack_rate", "Backtrack Rate"),
        ("regress_step_rate", "Regress Step Rate"),
    ]
    x = np.arange(len(methods))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, (metric, label) in enumerate(metrics):
        values = [float(row[metric]) for row in c4_failure]
        ax.bar(x + (idx - 1.5) * width, values, width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([method.replace("GeoExplorer-", "") for method in methods], rotation=12, ha="right")
    ax.set_title("C=4 Failure Profile")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def average_by_step(records: list[dict], trace_key: str, max_steps: int = 10) -> list[float]:
    values = []
    for step in range(max_steps):
        step_values = []
        for record in records:
            trace = record.get(trace_key, [])
            if step < len(trace):
                step_values.append(float(trace[step]))
        values.append(mean(step_values) if step_values else math.nan)
    return values


def plot_reward_process(path: Path) -> None:
    records = json.loads(REWARD_RECORDS.read_text(encoding="utf-8"))
    anchor = [record for record in records if record.get("method_key") == "g1_p1_e1_v1"]
    success_records = [record for record in anchor if record.get("success")]
    failure_records = [record for record in anchor if not record.get("success")]
    traces = [
        ("r_ex_trace", "External"),
        ("r_in_gated_trace", "Gated Intrinsic"),
        ("r_p_trace", "PBRS"),
        ("r_total_trace", "Total"),
        ("gate_trace", "Gate lambda"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    steps = np.arange(1, 11)
    for ax, (key, label) in zip(axes.ravel(), traces):
        ax.plot(steps, average_by_step(success_records, key), marker="o", label="success", color="#1F8A70")
        ax.plot(steps, average_by_step(failure_records, key), marker="s", label="failure", color="#C43C39")
        ax.set_title(label)
        ax.grid(alpha=0.25)
    axes.ravel()[-1].axis("off")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Reward Component Traces for This Work", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_figures(main_diff: list[dict], ultra_diff: list[dict], by_distance: list[dict], c4_failure: list[dict]) -> None:
    mmgag_rows = [
        row
        for row in main_diff
        if row["baseline"] == "GOMAA-Geo" and str(row["benchmark"]).startswith("mmgag_")
    ]
    plot_diff_ci(mmgag_rows, FIGURES / "mmgag_diff_ci.png", "MM-GAG Difference with 95% Approx. CI")
    plot_diff_ci(ultra_diff, FIGURES / "ultra_long_diff_ci.png", "Ultra-long Difference with 95% Approx. CI")
    plot_trajectory(by_distance, FIGURES / "trajectory_behavior_metrics.png")
    plot_c4_failure(c4_failure, FIGURES / "c4_failure_profile.png")
    plot_reward_process(FIGURES / "reward_component_traces.png")


def build_experiment_plan() -> list[dict]:
    rows = [
        {
            "priority": "P0",
            "experiment": "budget_sensitivity",
            "purpose": "Verify whether the long-distance advantage remains under tighter and looser search budgets.",
            "protocol": "Use existing checkpoints; MASA aerial; grids 8x8 and 10x10; distances follow the formal ultra-long setting; budgets 8x8={16,20,24,28,32}, 10x10={20,24,28,32,36,40}; 20 repeats per distance.",
            "methods": "GeoExplorer-anchor0624, GOMAA-Geo, GeoExplorer-pristine",
            "expected_output": "SR/SG curves by budget, difference CI table, budget-efficiency figure.",
            "cost": "Evaluation only; no training.",
            "thesis_use": "Chapter 4 long-distance robustness supplement.",
        },
        {
            "priority": "P0",
            "experiment": "task_bank_seed_rerun",
            "purpose": "Check whether main conclusions are stable to evaluation task sampling.",
            "protocol": "Use existing checkpoints; rerun main 5x5 MM-GAG A/G/T and ultra-long 8x8/10x10 with task_bank_seed={20260521,20260522,20260523}; keep task count fixed.",
            "methods": "GeoExplorer-anchor0624 and GOMAA-Geo first; add GeoExplorer-pristine if time allows.",
            "expected_output": "Mean SR, standard deviation across task-bank seeds, paired ranking stability.",
            "cost": "Evaluation only; moderate GPU time.",
            "thesis_use": "Reliability note or appendix table.",
        },
        {
            "priority": "P1",
            "experiment": "target_cue_robustness",
            "purpose": "Probe cross-modal robustness when target cues are degraded.",
            "protocol": "MM-GAG only; aerial target downsample/blur, ground target crop/blur, text target shortened or synonym-rewritten; distances C=4..8; same checkpoints.",
            "methods": "GeoExplorer-anchor0624 and GOMAA-Geo.",
            "expected_output": "Robustness SR drop table by cue type and severity.",
            "cost": "Requires writing cue-perturbation evaluator; no retraining.",
            "thesis_use": "Optional new angle for cross-modal target adaptation.",
        },
        {
            "priority": "P1",
            "experiment": "reward_trace_expansion",
            "purpose": "Make the reward mechanism explanation less dependent on a few showcase cases.",
            "protocol": "Sample at least 20 cases per C on SwissViewMonuments; log r_ex, raw intrinsic, gated intrinsic, PBRS, lambda, total reward for G/P conditions.",
            "methods": "Four G/P conditions with E=1,V=1 first.",
            "expected_output": "Average component curves and success/failure reward decomposition.",
            "cost": "Inference/logging only; light.",
            "thesis_use": "Mechanism visualization supplement.",
        },
        {
            "priority": "P2",
            "experiment": "larger_grid_middle_scale",
            "purpose": "Bridge formal 10x10 and noisy 25x25 observations.",
            "protocol": "Add one middle setting such as 12x12 or 15x15; choose valid long-distance buckets near 60%-90% of max Manhattan distance; budget around 1.6x-1.8x bucket minimum.",
            "methods": "GeoExplorer-anchor0624, GOMAA-Geo, GeoExplorer-pristine.",
            "expected_output": "A cleaner stress-test row between 10x10 and 25x25.",
            "cost": "Evaluation plus embedding generation if grid not cached.",
            "thesis_use": "Appendix only unless results are very clean.",
        },
    ]
    write_csv(TABLES / "experiment_plan" / "supplement_experiment_plan.csv", rows)
    return rows


def markdown_table(rows: list[dict], columns: list[tuple[str, str]], limit: int | None = None) -> list[str]:
    shown = rows[:limit] if limit else rows
    lines = ["| " + " | ".join(title for _, title in columns) + " |"]
    lines.append("| " + " | ".join("---" if not title.endswith("SR") and not title.endswith("CI") else "---:" for _, title in columns) + " |")
    for row in shown:
        values = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                values.append(fmt(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def build_report(
    main_rows: list[dict],
    main_diff: list[dict],
    ultra_diff: list[dict],
    traj_summary: list[dict],
    traj_by_distance: list[dict],
    c4_failure: list[dict],
    reward_summary: list[dict],
    plan_rows: list[dict],
) -> None:
    mmgag_diff = [row for row in main_diff if row["baseline"] == "GOMAA-Geo" and row["benchmark"].startswith("mmgag_")]
    mmgag_diff = sorted(mmgag_diff, key=lambda row: row["benchmark"])
    long_main = [row for row in ultra_diff if row["baseline"] == "GOMAA-Geo"]
    c4_sorted = sorted(c4_failure, key=lambda row: row["method"])
    anchor_traj = [row for row in traj_by_distance if row["method"] == "GeoExplorer-anchor0624"]
    anchor_reward = [row for row in reward_summary if row["method_key"] == "g1_p1_e1_v1"]

    lines = [
        "# 补充实验分析与后续实验方案",
        "",
        f"生成时间：{now_iso()}",
        "",
        "本文件汇总基于现有结果可直接完成的后处理分析，并列出下一批建议补跑的 evaluation-only 实验。这里没有重新训练模型，也没有改变服务器或系统配置。",
        "",
        "## 1. 统计置信度补充",
        "",
        "主表和长距离表已经包含任务级成功次数，因此可以补充成功率置信区间和方法差值的近似 95% CI。差值 CI 使用两个独立比例的正态近似，适合作为论文中的稳健性说明；如果后面要更严格，可以在保留逐任务结果后做 paired bootstrap。",
        "",
    ]
    lines.extend(
        markdown_table(
            mmgag_diff,
            [
                ("benchmark", "Benchmark"),
                ("ours_sr", "Ours SR"),
                ("baseline_sr", "GOMAA SR"),
                ("diff", "Diff"),
                ("diff_ci_low", "CI Low"),
                ("diff_ci_high", "CI High"),
            ],
        )
    )
    lines.extend(
        [
            "",
            "长距离扩展实验中，本文方法相对 GOMAA-Geo 的差值如下。8x8 和 10x10 的差值为正，且 10x10 的优势更明显；25x25 仍建议定位为探索性压力测试。",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            long_main,
            [
                ("grid", "Grid"),
                ("budget", "Budget"),
                ("ours_sr", "Ours SR"),
                ("baseline_sr", "GOMAA SR"),
                ("diff", "Diff"),
                ("diff_ci_low", "CI Low"),
                ("diff_ci_high", "CI High"),
            ],
        )
    )

    lines.extend(
        [
            "",
            "## 2. 轨迹行为补充",
            "",
            "轨迹行为分析不只看是否成功，还看搜索过程是否更接近目标方向。建议在论文中把这些指标作为定性可视化的量化支撑：成功率说明结果，单调接近比例和重复访问率说明过程。",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            traj_summary,
            [
                ("method", "Method"),
                ("success_rate", "SR"),
                ("progress_ratio", "Progress"),
                ("monotonic_step_rate", "Monotonic"),
                ("revisit_rate", "Revisit"),
                ("detour_steps", "Detour"),
            ],
        )
    )
    lines.extend(
        [
            "",
            "本文方法在 C=6 和 C=8 的成功率、目标距离缩短和单调接近趋势更有解释价值；C=4 不一定占优，应该在正文中诚实说明这是中远距离优化带来的取舍。",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            anchor_traj,
            [
                ("distance", "C"),
                ("n", "N"),
                ("success_rate", "SR"),
                ("progress_ratio", "Progress"),
                ("monotonic_step_rate", "Monotonic"),
                ("revisit_rate", "Revisit"),
            ],
        )
    )

    lines.extend(
        [
            "",
            "## 3. C=4 弱项分析",
            "",
            "C=4 是一个必须主动解释的点。短距离下最优路径很短，任何探索性绕行都会快速拉低成功率或 SG；而本文方法的奖励设计更偏向在中远距离维持探索方向和目标收敛。",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            c4_sorted,
            [
                ("method", "Method"),
                ("c4_success_rate", "C4 SR"),
                ("c4_failure_rate", "Fail Rate"),
                ("final_distance", "Fail FinalDist"),
                ("revisit_rate", "Fail Revisit"),
                ("immediate_backtrack_rate", "Fail Backtrack"),
            ],
        )
    )

    lines.extend(
        [
            "",
            "## 4. 奖励过程补充",
            "",
            "奖励过程记录显示，混合奖励只用于训练阶段的行为塑形分析；推理阶段仍然是策略网络根据状态特征选动作。这里的曲线和表格用于解释为什么门控内在奖励和 PBRS 能改善训练出的策略，而不是说测试时还在计算奖励后选动作。",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            anchor_reward,
            [
                ("distance", "C"),
                ("success", "Success"),
                ("n", "N"),
                ("sum_external", "Ext"),
                ("sum_intrinsic_gated", "Gated In"),
                ("sum_pbrs", "PBRS"),
                ("sum_total", "Total"),
                ("mean_gate", "Mean Lambda"),
            ],
        )
    )

    lines.extend(
        [
            "",
            "## 5. 下一批建议补跑实验",
            "",
            "优先级从 P0 到 P2。P0 是最推荐先跑的，因为它们不需要重新训练，只复用现有 checkpoint 做评测，能最快增强论文可信度。",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            plan_rows,
            [
                ("priority", "Priority"),
                ("experiment", "Experiment"),
                ("purpose", "Purpose"),
                ("cost", "Cost"),
                ("thesis_use", "Use"),
            ],
        )
    )

    lines.extend(
        [
            "",
            "## 6. 文件索引",
            "",
            "- `results/tables/statistical_analysis/main_benchmark_ci_table.csv`：主表 SR 置信区间。",
            "- `results/tables/statistical_analysis/main_benchmark_diff_ci_table.csv`：本文方法与基线的 SR 差值 CI。",
            "- `results/tables/statistical_analysis/ultra_long_diff_ci_table.csv`：长距离实验差值 CI。",
            "- `results/tables/trajectory_analysis/trajectory_behavior_summary.csv`：轨迹行为总体指标。",
            "- `results/tables/trajectory_analysis/c4_failure_profile.csv`：C=4 失败画像。",
            "- `results/tables/reward_process/reward_process_summary.csv`：奖励分量过程汇总。",
            "- `results/tables/experiment_plan/supplement_experiment_plan.csv`：后续补跑实验清单。",
            "- `results/figures/supplement/`：对应图件。",
            "",
        ]
    )

    (REPORTS / "supplement_experiment_analysis_zh.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ensure_dirs()
    main_rows, main_diff = build_main_ci_tables()
    ultra_diff = build_ultra_ci_table()
    traj_summary, traj_by_distance, c4_failure = build_trajectory_tables()
    _, reward_summary = build_reward_tables()
    plan_rows = build_experiment_plan()
    build_figures(main_diff, ultra_diff, traj_by_distance, c4_failure)
    build_report(main_rows, main_diff, ultra_diff, traj_summary, traj_by_distance, c4_failure, reward_summary, plan_rows)
    print(
        json.dumps(
            {
                "generated_at": now_iso(),
                "tables": [
                    str(TABLES / "statistical_analysis"),
                    str(TABLES / "trajectory_analysis"),
                    str(TABLES / "reward_process"),
                    str(TABLES / "experiment_plan"),
                ],
                "figures": str(FIGURES),
                "report": str(REPORTS / "supplement_experiment_analysis_zh.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
