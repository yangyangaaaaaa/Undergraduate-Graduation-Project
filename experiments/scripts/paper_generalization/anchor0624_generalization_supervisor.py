from __future__ import annotations

import csv
import itertools
import json
import math
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path


SERIES = "algo_paper_generalization_20260516"
EXPERIMENT = "anchor0624_factorial_generalization_seed321_480k"
SOURCE_SERIES = "algo_ablation_anchor0624_20260515"
SOURCE_TRAIN_EXPERIMENT = "masa_plus_mmgag_anchor0624_component_ablation_seed321_480k_gpu01"

REMOTE_ROOT = Path("/root/geoexplorer")
REPO_ROOT = REMOTE_ROOT / "GeoExplorer"
EXP_ROOT = REMOTE_ROOT / "ab_experiments" / SERIES / EXPERIMENT
STATUS_DIR = EXP_ROOT / "monitoring"
LOG_DIR = STATUS_DIR / "generalization_logs"
STATUS_PATH = STATUS_DIR / "anchor0624_generalization_status_latest.json"
PID_PATH = STATUS_DIR / "anchor0624_generalization_supervisor.pid"
OUTPUT_ROOT = REMOTE_ROOT / "analysis" / "pipeline_20260516_anchor0624_factorial_generalization"
CKPT_ROOT = REMOTE_ROOT / "results" / "checkpoint"
SOURCE_CKPT_ROOT = CKPT_ROOT / SOURCE_SERIES / SOURCE_TRAIN_EXPERIMENT
STRICT_SCRIPT = REMOTE_ROOT / "analysis" / "metric_audit_20260417" / "strict_fixed_eval.py"
MONUMENTS_SCRIPT = REPO_ROOT / "tuning" / "eval_swissviewmonuments.py"
LLM_CHECKPOINT = CKPT_ROOT / "env_modeling_fullrerun_20260407_111046" / "state_action.ckpt"

SEED = 321
TARGET_STEPS = 480000
TASK_BANK_SEED = 20260516
GPU_SLOTS = {0: 1, 1: 1}

PRIMARY_GENERALIZATION_BENCHMARKS = [
    "mmgag_aerial",
    "mmgag_ground",
    "mmgag_text",
    "swissviewmonuments_aerial_ground",
]
ALL_BENCHMARKS = [
    "masa_aerial",
    "mmgag_aerial",
    "mmgag_ground",
    "mmgag_text",
    "swissviewmonuments_aerial_ground",
]
FACTOR_KEYS = ["G_gate", "P_pbrs", "E_low_entropy", "V_val78"]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def make_branch(gate: int, pbrs: int, ent_low: int, val78: int) -> dict:
    branch = f"g{gate}_p{pbrs}_e{ent_low}_v{val78}"
    factors = {
        "G_gate": int(gate),
        "P_pbrs": int(pbrs),
        "E_low_entropy": int(ent_low),
        "V_val78": int(val78),
    }
    if gate and pbrs and ent_low and val78:
        role = "full_anchor0624"
    elif not gate and not pbrs and not ent_low and not val78:
        role = "same_data_no_added_mechanism_control"
    else:
        role = "factorial_ablation_cell"
    return {"branch": branch, "role": role, "factors": factors}


BRANCHES = [
    make_branch(g, p, e, v)
    for v in (0, 1)
    for e in (0, 1)
    for p in (0, 1)
    for g in (0, 1)
]
BRANCHES.sort(
    key=lambda item: (
        0 if item["branch"] == "g1_p1_e1_v1" else 1 if item["branch"] == "g0_p0_e0_v0" else 2,
        item["branch"],
    )
)


BENCHMARKS = [
    {
        "name": "masa_aerial",
        "kind": "strict",
        "dataset": "masa",
        "goal_mode": "aerial",
        "test_path_candidates": [
            "/root/geoexplorer/GeoExplorer/data/masa/sat_test_grid_5.npy",
            "/root/geoexplorer/data/masa/sat_test_grid_5.npy",
        ],
    },
    {
        "name": "mmgag_aerial",
        "kind": "strict",
        "dataset": "mmgag",
        "goal_mode": "aerial",
        "test_path_candidates": [
            "/root/geoexplorer/GeoExplorer/data/mm_gag/processed/mmgag_sat_grid_5.npy",
            "/root/geoexplorer/GeoExplorer/data/mm_gag/mmgag_sat_grid_5.npy",
        ],
    },
    {
        "name": "mmgag_ground",
        "kind": "strict",
        "dataset": "mmgag",
        "goal_mode": "ground",
        "test_path_candidates": [
            "/root/geoexplorer/GeoExplorer/data/mm_gag/processed/mmgag_sat_grid_5.npy",
            "/root/geoexplorer/GeoExplorer/data/mm_gag/mmgag_sat_grid_5.npy",
        ],
        "goal_embeds_candidates": [
            "/root/geoexplorer/GeoExplorer/data/mm_gag/processed/mmgag_ground_embeds.npy",
            "/root/geoexplorer/GeoExplorer/data/mm_gag/mmgag_ground_embeds.npy",
            "/root/geoexplorer/GeoExplorer/data/mm_gag/ground_embeds.npy",
        ],
    },
    {
        "name": "mmgag_text",
        "kind": "strict",
        "dataset": "mmgag",
        "goal_mode": "text",
        "test_path_candidates": [
            "/root/geoexplorer/GeoExplorer/data/mm_gag/processed/mmgag_sat_grid_5.npy",
            "/root/geoexplorer/GeoExplorer/data/mm_gag/mmgag_sat_grid_5.npy",
        ],
        "goal_embeds_candidates": [
            "/root/geoexplorer/GeoExplorer/data/mm_gag/processed/mmgag_text_embeds.npy",
            "/root/geoexplorer/GeoExplorer/data/mm_gag/mmgag_text_embeds.npy",
        ],
    },
    {
        "name": "swissviewmonuments_aerial_ground",
        "kind": "monuments",
        "dataset": "swissviewmonuments",
        "test_path_candidates": [
            "/root/geoexplorer/GeoExplorer/data/swissview/swissviewmonuments_sat_patches.npy",
            "/root/geoexplorer/GeoExplorer/data/swissview/swissviewmonuments_patches.npy",
            "/root/geoexplorer/GeoExplorer/data/swissview/processed/swissviewmonuments_sat_patches.npy",
        ],
        "goal_embeds_candidates": [
            "/root/geoexplorer/GeoExplorer/data/swissview/swissviewmonuments_grd.npy",
            "/root/geoexplorer/GeoExplorer/data/swissview/processed/swissviewmonuments_grd.npy",
        ],
    },
]


def resolve_existing(candidates: list[str]) -> str:
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError(f"missing all candidates: {candidates}")


def checkpoint_dir(branch: str) -> Path:
    return SOURCE_CKPT_ROOT / f"{branch}_seed{SEED}_t{TARGET_STEPS // 1000}k"


def build_specs() -> list[dict]:
    specs = []
    for item in BRANCHES:
        branch = item["branch"]
        specs.append(
            {
                "name": f"{branch}_seed{SEED}_best_val",
                "branch": branch,
                "role": item["role"],
                "factors": item["factors"],
                "seed": SEED,
                "target_steps": TARGET_STEPS,
                "checkpoint_dir": checkpoint_dir(branch),
                "checkpoint": checkpoint_dir(branch) / "geoexplorer.pt",
                "output_dir": OUTPUT_ROOT / f"{branch}_seed{SEED}_best_val",
                "status": "pending",
                "benchmarks": {benchmark["name"]: {"status": "pending"} for benchmark in BENCHMARKS},
            }
        )
    return specs


def clean_process_env() -> dict:
    return {key: value for key, value in os.environ.items() if not key.startswith("GEOEXPLORER_")}


def parse_metric_payload(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "success_ratio" in payload:
        return {"success_ratio": float(payload["success_ratio"]), "payload": payload}
    if "avg_combined_success_ratio" in payload:
        return {"success_ratio": float(payload["avg_combined_success_ratio"]), "payload": payload}
    if "combined_avg_success_ratio" in payload:
        return {"success_ratio": float(payload["combined_avg_success_ratio"]), "payload": payload}
    modes = payload.get("modes")
    if isinstance(modes, list):
        for row in modes:
            if isinstance(row, dict) and row.get("mode") == "greedy" and "success_ratio" in row:
                return {"success_ratio": float(row["success_ratio"]), "payload": payload}
    raise KeyError(f"unrecognized metric payload: {path}")


def parse_metric(path: Path) -> float:
    return parse_metric_payload(path)["success_ratio"]


def output_ready(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        parse_metric(path)
    except Exception:
        return False
    return True


def per_distance_from_strict_payload(payload: dict) -> dict[str, float]:
    modes = payload.get("modes")
    if not isinstance(modes, list):
        return {}
    greedy = next((row for row in modes if isinstance(row, dict) and row.get("mode") == "greedy"), None)
    if not greedy:
        return {}
    rows = greedy.get("per_dist")
    if not isinstance(rows, list):
        return {}
    return {f"d{int(row['distance'])}": float(row["success_ratio"]) for row in rows if "distance" in row}


def mark_existing_outputs(spec: dict) -> None:
    for benchmark in BENCHMARKS:
        status = spec["benchmarks"][benchmark["name"]]
        if status["status"] != "pending":
            continue
        output_path = spec["output_dir"] / f"{benchmark['name']}.json"
        if output_ready(output_path):
            status.update({"status": "completed", "output_path": str(output_path), "resume_reused": True})


def build_eval_command(spec: dict, benchmark: dict) -> list[str]:
    out_path = spec["output_dir"] / f"{benchmark['name']}.json"
    if benchmark["kind"] == "strict":
        cmd = [
            "/usr/bin/python3",
            str(STRICT_SCRIPT),
            "--run-dir",
            str(REPO_ROOT),
            "--checkpoint",
            str(spec["checkpoint"]),
            "--dataset",
            benchmark["dataset"],
            "--test-path",
            benchmark["resolved"]["test_path"],
            "--goal-mode",
            benchmark["goal_mode"],
            "--device",
            "cuda:0",
            "--distances",
            "4,5,6,7,8",
            "--repeats-per-dist",
            "5",
            "--seed",
            str(TASK_BANK_SEED),
            "--modes",
            "greedy",
            "--output-path",
            str(out_path),
        ]
        if "goal_embeds" in benchmark["resolved"]:
            cmd.extend(["--goal-embeds", benchmark["resolved"]["goal_embeds"]])
        return cmd
    return ["/usr/bin/python3", str(MONUMENTS_SCRIPT)]


def launch_eval(spec: dict, benchmark: dict, gpu: int) -> subprocess.Popen:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    spec["output_dir"].mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{spec['name']}__{benchmark['name']}.log"
    log_handle = log_path.open("ab", buffering=0)
    log_handle.write(f"\n[{now_iso()}] launching {spec['name']}::{benchmark['name']} on GPU{gpu}\n".encode("utf-8"))
    env = clean_process_env()
    env.update(
        {
            "PYTHONPATH": "/root/geoexplorer/env/geoexplorer_site:/root/geoexplorer:/root/geoexplorer/GeoExplorer",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "GEOEXPLORER_LLM_CHECKPOINT": str(LLM_CHECKPOINT),
        }
    )
    if benchmark["kind"] == "monuments":
        env.update(
            {
                "GEOEXPLORER_DEVICE": "cuda:0",
                "GEOEXPLORER_DATASET": "swissviewmonuments",
                "GEOEXPLORER_TEST_DATA": benchmark["resolved"]["test_path"],
                "GEOEXPLORER_GROUND_EMBEDS": benchmark["resolved"]["goal_embeds"],
                "GEOEXPLORER_TRAIN_CHECKPOINT_PATH": str(spec["checkpoint"]),
                "GEOEXPLORER_EVAL_OUTPUT": str(spec["output_dir"] / f"{benchmark['name']}.json"),
            }
        )
    return subprocess.Popen(
        build_eval_command(spec, benchmark),
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def metric_row(spec: dict) -> dict:
    metrics = {benchmark["name"]: parse_metric(spec["output_dir"] / f"{benchmark['name']}.json") for benchmark in BENCHMARKS}
    payloads = {
        benchmark["name"]: parse_metric_payload(spec["output_dir"] / f"{benchmark['name']}.json")["payload"]
        for benchmark in BENCHMARKS
    }
    primary_values = [metrics[name] for name in PRIMARY_GENERALIZATION_BENCHMARKS]
    all_values = [metrics[name] for name in ALL_BENCHMARKS]
    return {
        "name": spec["name"],
        "branch": spec["branch"],
        "role": spec["role"],
        "factors": spec["factors"],
        "seed": spec["seed"],
        "target_steps": spec["target_steps"],
        "checkpoint": str(spec["checkpoint"]),
        "metrics": metrics,
        "primary_generalization_mean": mean(primary_values),
        "all_benchmark_mean": mean(all_values),
        "masa_per_distance": per_distance_from_strict_payload(payloads["masa_aerial"]),
    }


def factorial_effects(rows: list[dict], metric_name: str) -> dict:
    effects = {}
    for factor in FACTOR_KEYS:
        on_values = [float(row[metric_name]) for row in rows if int(row["factors"].get(factor, 0)) == 1]
        off_values = [float(row[metric_name]) for row in rows if int(row["factors"].get(factor, 0)) == 0]
        effects[factor] = {"on_mean": mean(on_values), "off_mean": mean(off_values), "effect_on_minus_off": mean(on_values) - mean(off_values)}

    interactions = {}
    for left, right in itertools.combinations(FACTOR_KEYS, 2):
        def mean_for(a: int, b: int) -> float:
            vals = [
                float(row[metric_name])
                for row in rows
                if int(row["factors"].get(left, 0)) == a and int(row["factors"].get(right, 0)) == b
            ]
            return mean(vals)

        m11 = mean_for(1, 1)
        m10 = mean_for(1, 0)
        m01 = mean_for(0, 1)
        m00 = mean_for(0, 0)
        interactions[f"{left}__x__{right}"] = {
            "mean_11": m11,
            "mean_10": m10,
            "mean_01": m01,
            "mean_00": m00,
            "difference_of_differences": (m11 - m10) - (m01 - m00),
        }
    return {"main_effects": effects, "two_way_interactions": interactions}


def build_summary(specs: list[dict]) -> dict:
    rows = [metric_row(spec) for spec in specs]
    rows.sort(key=lambda row: row["primary_generalization_mean"], reverse=True)
    effects = {
        "primary_generalization_mean": factorial_effects(rows, "primary_generalization_mean"),
        "all_benchmark_mean": factorial_effects(rows, "all_benchmark_mean"),
    }
    for benchmark in ALL_BENCHMARKS:
        metric_rows = [{**row, f"metric__{benchmark}": row["metrics"][benchmark]} for row in rows]
        effects[benchmark] = factorial_effects(metric_rows, f"metric__{benchmark}")

    anchor = next((row for row in rows if row["branch"] == "g1_p1_e1_v1"), None)
    control = next((row for row in rows if row["branch"] == "g0_p0_e0_v0"), None)
    best = rows[0] if rows else None
    aggregate = {
        "generated_at": now_iso(),
        "series": SERIES,
        "experiment": EXPERIMENT,
        "source_series": SOURCE_SERIES,
        "source_train_experiment": SOURCE_TRAIN_EXPERIMENT,
        "task_bank_seed": TASK_BANK_SEED,
        "benchmarks": ALL_BENCHMARKS,
        "primary_generalization_benchmarks": PRIMARY_GENERALIZATION_BENCHMARKS,
        "rows": rows,
        "factorial_effects": effects,
        "key_comparisons": {
            "best_by_primary_generalization_mean": best,
            "full_anchor_g1_p1_e1_v1": anchor,
            "same_data_control_g0_p0_e0_v0": control,
            "anchor_minus_control_primary_generalization_mean": (
                anchor["primary_generalization_mean"] - control["primary_generalization_mean"]
                if anchor and control
                else math.nan
            ),
            "anchor_minus_control_all_benchmark_mean": (
                anchor["all_benchmark_mean"] - control["all_benchmark_mean"]
                if anchor and control
                else math.nan
            ),
        },
        "rigor_notes": [
            "This is evaluation-only; no additional training or checkpoint selection is performed.",
            "All 16 trained factorial cells are evaluated to avoid top-row selection bias.",
            "Primary generalization mean excludes masa_aerial so in-domain MASA does not dominate the transfer conclusion.",
            "Single seed 321 limits seed-stability claims; this round supports mechanism diagnosis and transfer evidence.",
        ],
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "anchor0624_generalization_aggregate.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(rows)
    write_markdown(rows, aggregate)
    return aggregate


def write_csv(rows: list[dict]) -> None:
    with (OUTPUT_ROOT / "anchor0624_generalization_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "branch",
                "role",
                "G_gate",
                "P_pbrs",
                "E_low_entropy",
                "V_val78",
                "primary_generalization_mean",
                "all_benchmark_mean",
                *ALL_BENCHMARKS,
                "masa_d4",
                "masa_d5",
                "masa_d6",
                "masa_d7",
                "masa_d8",
                "checkpoint",
            ]
        )
        for row in rows:
            factors = row["factors"]
            per_dist = row.get("masa_per_distance", {})
            writer.writerow(
                [
                    row["branch"],
                    row["role"],
                    factors.get("G_gate", 0),
                    factors.get("P_pbrs", 0),
                    factors.get("E_low_entropy", 0),
                    factors.get("V_val78", 0),
                    f"{row['primary_generalization_mean']:.6f}",
                    f"{row['all_benchmark_mean']:.6f}",
                    *[f"{row['metrics'][name]:.6f}" for name in ALL_BENCHMARKS],
                    *[f"{per_dist.get(f'd{d}', 0.0):.6f}" for d in range(4, 9)],
                    row["checkpoint"],
                ]
            )


def write_markdown(rows: list[dict], aggregate: dict) -> None:
    lines = [
        "# Anchor0624 Factorial Generalization Evaluation",
        "",
        "- design: all 16 anchor0624 factorial checkpoints, seed 321, 480k, best-val checkpoint.",
        "- primary transfer mean: average of `mmgag_aerial`, `mmgag_ground`, `mmgag_text`, and `swissviewmonuments_aerial_ground`.",
        "- protocol: greedy, `5x5`, `B=10`, `C={4,5,6,7,8}`, fixed generated task seed `20260516`; no additional training.",
        "",
        "| Rank | Branch | G | P | E | V | Role | Primary Gen Mean | All Mean | Masa | MM-GAG I | MM-GAG G | MM-GAG T | SwissMon |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for idx, row in enumerate(rows, start=1):
        factors = row["factors"]
        lines.append(
            "| "
            + str(idx)
            + " | "
            + row["branch"]
            + " | "
            + str(factors.get("G_gate", 0))
            + " | "
            + str(factors.get("P_pbrs", 0))
            + " | "
            + str(factors.get("E_low_entropy", 0))
            + " | "
            + str(factors.get("V_val78", 0))
            + " | "
            + row["role"]
            + " | "
            + f"{row['primary_generalization_mean']:.4f}"
            + " | "
            + f"{row['all_benchmark_mean']:.4f}"
            + " | "
            + " | ".join(f"{row['metrics'][name]:.4f}" for name in ALL_BENCHMARKS)
            + " |"
        )
    key = aggregate["key_comparisons"]
    lines.extend(
        [
            "",
            "## Key Comparisons",
            "",
            f"- best transfer row: `{key['best_by_primary_generalization_mean']['branch']}` with primary generalization mean `{key['best_by_primary_generalization_mean']['primary_generalization_mean']:.4f}`.",
            f"- full anchor `g1_p1_e1_v1` primary generalization mean: `{key['full_anchor_g1_p1_e1_v1']['primary_generalization_mean']:.4f}`.",
            f"- same-data control `g0_p0_e0_v0` primary generalization mean: `{key['same_data_control_g0_p0_e0_v0']['primary_generalization_mean']:.4f}`.",
            f"- anchor minus control primary generalization mean: `{key['anchor_minus_control_primary_generalization_mean']:.4f}`.",
            "",
            "## Rigor Notes",
            "",
        ]
    )
    for note in aggregate["rigor_notes"]:
        lines.append(f"- {note}")
    (OUTPUT_ROOT / "anchor0624_generalization_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def gpu_snapshot() -> list[dict]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        return [{"error": str(exc)}]
    rows = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 4:
            rows.append({"gpu": int(parts[0]), "name": parts[1], "used_mb": int(parts[2]), "util": int(parts[3])})
    return rows


def write_status(payload: dict) -> None:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def terminate_active_processes(active: dict[str, dict]) -> None:
    for item in active.values():
        proc = item["process"]
        if proc.poll() is not None:
            continue
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            proc.terminate()
    time.sleep(5)
    for item in active.values():
        proc = item["process"]
        if proc.poll() is not None:
            continue
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            proc.kill()


def main() -> int:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PID_PATH.write_text(str(os.getpid()) + "\n", encoding="utf-8")

    resolved_benchmarks = []
    try:
        for benchmark in BENCHMARKS:
            resolved = {"test_path": resolve_existing(benchmark["test_path_candidates"])}
            if "goal_embeds_candidates" in benchmark:
                resolved["goal_embeds"] = resolve_existing(benchmark["goal_embeds_candidates"])
            resolved_benchmarks.append({**benchmark, "resolved": resolved})
    except Exception as exc:
        write_status({"timestamp": now_iso(), "phase": "failed_dataset_resolution", "error": str(exc)})
        raise

    specs = build_specs()
    active: dict[str, dict] = {}
    summary = None
    try:
        while True:
            counts = {gpu: 0 for gpu in GPU_SLOTS}
            finished = []
            for key, item in active.items():
                counts[item["gpu"]] += 1
                code = item["process"].poll()
                if code is None:
                    continue
                spec = item["spec"]
                benchmark = item["benchmark"]
                output_path = spec["output_dir"] / f"{benchmark['name']}.json"
                ok = code == 0 and output_ready(output_path)
                spec["benchmarks"][benchmark["name"]].update(
                    {
                        "status": "completed" if ok else "failed",
                        "returncode": int(code),
                        "output_path": str(output_path),
                        "ended_at": now_iso(),
                    }
                )
                finished.append(key)
                counts[item["gpu"]] -= 1
            for key in finished:
                active.pop(key, None)

            for spec in specs:
                mark_existing_outputs(spec)
                if spec["status"] == "completed":
                    continue
                if not spec["checkpoint"].exists():
                    spec["status"] = "failed_missing_checkpoint"
                    continue
                pending = [benchmark for benchmark in BENCHMARKS if spec["benchmarks"][benchmark["name"]]["status"] == "pending"]
                running = [benchmark for benchmark in BENCHMARKS if spec["benchmarks"][benchmark["name"]]["status"] == "running"]
                failed = [benchmark for benchmark in BENCHMARKS if spec["benchmarks"][benchmark["name"]]["status"] == "failed"]
                if failed:
                    spec["status"] = "failed"
                    continue
                if not pending and not running:
                    spec["status"] = "completed"
                    continue
                if running:
                    spec["status"] = "running"
                    continue
                available = [gpu for gpu, cap in GPU_SLOTS.items() if counts[gpu] < cap]
                if not available:
                    continue
                benchmark = next(item for item in resolved_benchmarks if item["name"] == pending[0]["name"])
                gpu = available[0]
                process = launch_eval(spec, benchmark, gpu)
                spec["status"] = "running"
                spec["benchmarks"][benchmark["name"]].update(
                    {"status": "running", "gpu": gpu, "pid": int(process.pid), "started_at": now_iso()}
                )
                active[f"{spec['name']}::{benchmark['name']}"] = {
                    "gpu": gpu,
                    "process": process,
                    "spec": spec,
                    "benchmark": benchmark,
                }
                counts[gpu] += 1

            phase = "completed" if all(spec["status"] == "completed" for spec in specs) and not active else "running"
            if any(str(spec["status"]).startswith("failed") for spec in specs):
                phase = "failed"
            if phase == "completed" and summary is None:
                summary = build_summary(specs)

            payload = {
                "timestamp": now_iso(),
                "phase": phase,
                "series": SERIES,
                "experiment": EXPERIMENT,
                "source_train_experiment": f"{SOURCE_SERIES}/{SOURCE_TRAIN_EXPERIMENT}",
                "output_root": str(OUTPUT_ROOT),
                "summary_path": str(OUTPUT_ROOT / "anchor0624_generalization_summary.md") if summary else None,
                "table_path": str(OUTPUT_ROOT / "anchor0624_generalization_table.csv") if summary else None,
                "aggregate_path": str(OUTPUT_ROOT / "anchor0624_generalization_aggregate.json") if summary else None,
                "allowed_gpus": list(GPU_SLOTS),
                "gpu_snapshot": gpu_snapshot(),
                "active_eval_processes": len(active),
                "resolved_benchmarks": {
                    item["name"]: item["resolved"] for item in resolved_benchmarks
                },
                "runs": {
                    spec["name"]: {
                        "branch": spec["branch"],
                        "role": spec["role"],
                        "factors": spec["factors"],
                        "seed": spec["seed"],
                        "target_steps": spec["target_steps"],
                        "status": spec["status"],
                        "checkpoint": str(spec["checkpoint"]),
                        "output_dir": str(spec["output_dir"]),
                        "benchmarks": spec["benchmarks"],
                    }
                    for spec in specs
                },
            }
            if summary is not None:
                payload["summary"] = summary
            write_status(payload)

            if phase in {"completed", "failed"}:
                return 0 if phase == "completed" else 1
            time.sleep(30)
    except Exception as exc:
        terminate_active_processes(active)
        write_status({"timestamp": now_iso(), "phase": "failed", "error": str(exc), "active_eval_processes": len(active)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
