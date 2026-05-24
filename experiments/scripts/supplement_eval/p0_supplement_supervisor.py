from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path


SERIES = "supplement_eval_20260524"
EXPERIMENT = "p0_budget_seed_eval"

REMOTE_ROOT = Path("/root/geoexplorer")
REPO_ROOT = REMOTE_ROOT / "GeoExplorer"
EXP_ROOT = REMOTE_ROOT / "ab_experiments" / SERIES / EXPERIMENT
MONITORING = EXP_ROOT / "monitoring"
LOG_DIR = MONITORING / "logs"
STATUS_PATH = MONITORING / "p0_supplement_status_latest.json"
OUTPUT_ROOT = REMOTE_ROOT / "analysis" / "pipeline_20260524_p0_supplement_eval"
EVALUATOR = MONITORING / "paper_geo_evaluator.py"

COMPARE_ROOT = Path("/root/src/compare_baselines_bundle_20260505_v2/compare_baselines_bundle")
GOMAA_ROOT = COMPARE_ROOT / "gomaa_geo_official"
NVIDIA_COMPAT_LIB = REMOTE_ROOT / "env/nvidia_535_288/usr/lib/x86_64-linux-gnu"
NVIDIA_COMPAT_SMI = REMOTE_ROOT / "env/nvidia_535_288/usr/bin/nvidia-smi"

ANCHOR_CKPT = (
    REMOTE_ROOT
    / "results/checkpoint/algo_ablation_anchor0624_20260515/"
    / "masa_plus_mmgag_anchor0624_component_ablation_seed321_480k_gpu01/"
    / "g1_p1_e1_v1_seed321_t480k/geoexplorer.pt"
)
ANCHOR_LLM = REMOTE_ROOT / "results/checkpoint/env_modeling_fullrerun_20260407_111046/state_action.ckpt"
GOMAA_CKPT = GOMAA_ROOT / "gomaa_geo/checkpoint/formal_ppo_seed42_t480k/formal_ppo.pt"
GOMAA_LLM = GOMAA_ROOT / "gomaa_geo/checkpoint/formal_pretrain_seed42_e50/formal_falcon.ckpt"
PRISTINE_CKPT = (
    REMOTE_ROOT
    / "results/checkpoint/algo_dualseed480k_20260427/"
    / "masa_plus_mmgag_arena_pristine_a040_a0405_sine0405_dualseed480k/"
    / "wave2_seed321_4gpu/pristine_seed321_t480k/geoexplorer.pt"
)
PRISTINE_LLM = ANCHOR_LLM

FORMAL_ULTRA_SEED = 20260521
SEED_RERUNS = [20260521, 20260522, 20260523]
GPU_SLOTS = [0, 1, 2, 3]

DATASETS = {
    "masa_grid8": REMOTE_ROOT / "data/masa/sat_test_grid_8.npy",
    "masa_grid10": REMOTE_ROOT / "data/masa/sat_test_grid_10.npy",
    "mmgag_sat": REPO_ROOT / "data/mm_gag/processed/mmgag_sat_grid_5.npy",
    "mmgag_ground": REPO_ROOT / "data/mm_gag/processed/mmgag_ground_embeds.npy",
    "mmgag_text": REPO_ROOT / "data/mm_gag/processed/mmgag_text_embeds.npy",
}

METHODS = {
    "anchor0624": {
        "method": "geoexplorer",
        "label": "GeoExplorer-anchor0624",
        "display": "This work",
        "repo_dir": str(REPO_ROOT),
        "checkpoint": str(ANCHOR_CKPT),
        "llm_checkpoint": str(ANCHOR_LLM),
    },
    "gomaa": {
        "method": "gomaa",
        "label": "GOMAA-Geo",
        "display": "GOMAA-Geo",
        "repo_dir": str(GOMAA_ROOT),
        "checkpoint": str(GOMAA_CKPT),
        "llm_checkpoint": str(GOMAA_LLM),
    },
    "pristine": {
        "method": "geoexplorer",
        "label": "GeoExplorer-pristine",
        "display": "GeoExplorer",
        "repo_dir": str(REPO_ROOT),
        "checkpoint": str(PRISTINE_CKPT),
        "llm_checkpoint": str(PRISTINE_LLM),
    },
}

ULTRA_PROTOCOLS = {
    8: {"distances": [10, 11, 12, 13, 14], "formal_budget": 24, "budgets": [16, 20, 24, 28, 32]},
    10: {"distances": [14, 15, 16, 17, 18], "formal_budget": 32, "budgets": [20, 24, 28, 32, 36, 40]},
}

MMGAG_BENCHMARKS = {
    "mmgag_aerial": {"goal_mode": "aerial", "goal_embeds": None},
    "mmgag_ground": {"goal_mode": "ground", "goal_embeds": str(DATASETS["mmgag_ground"])},
    "mmgag_text": {"goal_mode": "text", "goal_embeds": str(DATASETS["mmgag_text"])},
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def ensure_inputs() -> None:
    required = [
        *DATASETS.values(),
        ANCHOR_CKPT,
        ANCHOR_LLM,
        GOMAA_CKPT,
        GOMAA_LLM,
        PRISTINE_CKPT,
        PRISTINE_LLM,
        EVALUATOR,
    ]
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


def output_ready(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return "success_ratio" in payload and "per_distance" in payload


def gpu_snapshot() -> list[dict]:
    smi = NVIDIA_COMPAT_SMI if NVIDIA_COMPAT_SMI.exists() else Path("nvidia-smi")
    env = os.environ.copy()
    if NVIDIA_COMPAT_LIB.exists():
        env["LD_LIBRARY_PATH"] = str(NVIDIA_COMPAT_LIB)
    try:
        completed = subprocess.run(
            [str(smi), "--query-gpu=index,name,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
    except Exception as exc:
        return [{"error": str(exc)}]
    rows = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 4:
            rows.append({"gpu": int(parts[0]), "name": parts[1], "used_mb": int(parts[2]), "util": int(parts[3])})
    return rows


def build_eval_command(job: dict) -> list[str]:
    method = METHODS[job["method_key"]]
    cmd = [
        "/usr/bin/python3",
        "-u",
        str(EVALUATOR),
        "--method",
        method["method"],
        "--method-label",
        method["label"],
        "--repo-dir",
        method["repo_dir"],
        "--checkpoint",
        method["checkpoint"],
        "--llm-checkpoint",
        method["llm_checkpoint"],
        "--benchmark",
        job["benchmark"],
        "--paper-table",
        job["paper_table"],
        "--dataset",
        job["dataset"],
        "--goal-mode",
        job["goal_mode"],
        "--fixed-goal-mode",
        "none",
        "--test-path",
        job["test_path"],
        "--device",
        "cuda:0",
        "--patch-size",
        str(job["patch_size"]),
        "--budget",
        str(job["budget"]),
        "--distances",
        ",".join(str(item) for item in job["distances"]),
        "--repeats-per-dist",
        str(job["repeats_per_dist"]),
        "--seed",
        str(job["seed"]),
        "--max-images",
        str(job["max_images"]),
        "--output-path",
        job["output_path"],
    ]
    if job.get("goal_embeds"):
        cmd.extend(["--goal-embeds", job["goal_embeds"]])
    return cmd


def job_output_path(*parts: str) -> str:
    return str(OUTPUT_ROOT.joinpath("raw", *parts).with_suffix(".json"))


def build_budget_jobs() -> list[dict]:
    jobs = []
    for grid, protocol in ULTRA_PROTOCOLS.items():
        for budget in protocol["budgets"]:
            for method_key in ["anchor0624", "gomaa", "pristine"]:
                jobs.append(
                    {
                        "job_type": "budget_sensitivity",
                        "key": f"budget_grid{grid}_b{budget}_{method_key}",
                        "benchmark": f"masa_aerial_grid{grid}_budget{budget}",
                        "paper_table": "Supplement budget sensitivity",
                        "dataset": "masa",
                        "goal_mode": "aerial",
                        "goal_embeds": None,
                        "test_path": str(DATASETS[f"masa_grid{grid}"]),
                        "patch_size": grid,
                        "grid": f"{grid}x{grid}",
                        "budget": budget,
                        "formal_budget": protocol["formal_budget"],
                        "distances": protocol["distances"],
                        "repeats_per_dist": 20,
                        "seed": FORMAL_ULTRA_SEED,
                        "max_images": 0,
                        "method_key": method_key,
                        "output_path": job_output_path("budget_sensitivity", f"grid{grid}", f"b{budget}", method_key),
                    }
                )
    return jobs


def build_mmgag_seed_jobs() -> list[dict]:
    jobs = []
    for seed in SEED_RERUNS:
        for benchmark, bench in MMGAG_BENCHMARKS.items():
            for method_key in ["anchor0624", "gomaa"]:
                jobs.append(
                    {
                        "job_type": "task_seed_mmgag",
                        "key": f"seed{seed}_{benchmark}_{method_key}",
                        "benchmark": benchmark,
                        "paper_table": "Supplement task-bank seed MM-GAG",
                        "dataset": "mmgag",
                        "goal_mode": bench["goal_mode"],
                        "goal_embeds": bench["goal_embeds"],
                        "test_path": str(DATASETS["mmgag_sat"]),
                        "patch_size": 5,
                        "grid": "5x5",
                        "budget": 10,
                        "formal_budget": 10,
                        "distances": [4, 5, 6, 7, 8],
                        "repeats_per_dist": 5,
                        "seed": seed,
                        "max_images": 0,
                        "method_key": method_key,
                        "output_path": job_output_path("task_seed_mmgag", f"seed{seed}", benchmark, method_key),
                    }
                )
    return jobs


def build_ultra_seed_jobs() -> list[dict]:
    jobs = []
    for seed in [item for item in SEED_RERUNS if item != FORMAL_ULTRA_SEED]:
        for grid, protocol in ULTRA_PROTOCOLS.items():
            for method_key in ["anchor0624", "gomaa", "pristine"]:
                jobs.append(
                    {
                        "job_type": "task_seed_ultra",
                        "key": f"seed{seed}_grid{grid}_{method_key}",
                        "benchmark": f"masa_aerial_grid{grid}_seed{seed}",
                        "paper_table": "Supplement task-bank seed ultra-long",
                        "dataset": "masa",
                        "goal_mode": "aerial",
                        "goal_embeds": None,
                        "test_path": str(DATASETS[f"masa_grid{grid}"]),
                        "patch_size": grid,
                        "grid": f"{grid}x{grid}",
                        "budget": protocol["formal_budget"],
                        "formal_budget": protocol["formal_budget"],
                        "distances": protocol["distances"],
                        "repeats_per_dist": 20,
                        "seed": seed,
                        "max_images": 0,
                        "method_key": method_key,
                        "output_path": job_output_path("task_seed_ultra", f"seed{seed}", f"grid{grid}", method_key),
                    }
                )
    return jobs


def build_jobs(mode: str) -> list[dict]:
    jobs = []
    if mode in {"all", "budget"}:
        jobs.extend(build_budget_jobs())
    if mode in {"all", "seed"}:
        jobs.extend(build_mmgag_seed_jobs())
        jobs.extend(build_ultra_seed_jobs())
    return jobs


def launch_job(job: dict, gpu: int) -> dict:
    out_path = Path(job["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{job['key']}.log"
    log_handle = log_path.open("ab", buffering=0)
    log_handle.write(f"\n[{now_iso()}] launching {job['key']} on GPU{gpu}\n".encode("utf-8"))
    env = {key: value for key, value in os.environ.items() if not key.startswith("GEOEXPLORER_")}
    env.update(
        {
            "PYTHONPATH": ":".join(
                [
                    "/root/geoexplorer/env/geoexplorer_site",
                    "/root/geoexplorer",
                    "/root/geoexplorer/GeoExplorer",
                    "/root/src/compare_baselines_bundle_20260505_v2/compare_baselines_bundle",
                ]
            ),
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    if NVIDIA_COMPAT_LIB.exists():
        env["LD_LIBRARY_PATH"] = str(NVIDIA_COMPAT_LIB)
    process = subprocess.Popen(
        build_eval_command(job),
        cwd=str(METHODS[job["method_key"]]["repo_dir"]),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return {"process": process, "log_handle": log_handle, "gpu": gpu, "log_path": str(log_path)}


def parse_metric(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mode = payload["modes"][0]
    per_dist = {int(row["distance"]): row for row in payload.get("per_distance", [])}
    return {
        "payload": payload,
        "trials": int(mode.get("total_trials", payload.get("num_tasks", 0))),
        "success": int(mode.get("success", round(payload["success_ratio"] * payload.get("num_tasks", 0)))),
        "sr": float(payload["success_ratio"]),
        "sr_ci_low": float(mode.get("success_ratio_ci_low", math.nan)),
        "sr_ci_high": float(mode.get("success_ratio_ci_high", math.nan)),
        "sg": float(payload.get("sg_mean", math.nan)),
        "per_distance": per_dist,
    }


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


def aggregate_outputs(jobs: list[dict]) -> dict:
    rows = []
    per_rows = []
    for job in jobs:
        metric = parse_metric(Path(job["output_path"]))
        row = {
            "job_type": job["job_type"],
            "seed": job["seed"],
            "grid": job["grid"],
            "budget": job["budget"],
            "formal_budget": job["formal_budget"],
            "benchmark": job["benchmark"],
            "dataset": job["dataset"],
            "goal_mode": job["goal_mode"],
            "method_key": job["method_key"],
            "method": METHODS[job["method_key"]]["label"],
            "success_ratio": metric["sr"],
            "success_ratio_ci_low": metric["sr_ci_low"],
            "success_ratio_ci_high": metric["sr_ci_high"],
            "sg_mean": metric["sg"],
            "trials": metric["trials"],
            "success": metric["success"],
            "distances": ",".join(str(item) for item in job["distances"]),
            "output_path": job["output_path"],
        }
        for dist in job["distances"]:
            dist_metric = metric["per_distance"].get(int(dist), {})
            row[f"d{dist}"] = dist_metric.get("success_ratio", math.nan)
            row[f"d{dist}_trials"] = dist_metric.get("trials", "")
            row[f"d{dist}_success"] = dist_metric.get("success", "")
            per_rows.append(
                {
                    "job_type": job["job_type"],
                    "seed": job["seed"],
                    "grid": job["grid"],
                    "budget": job["budget"],
                    "benchmark": job["benchmark"],
                    "method_key": job["method_key"],
                    "method": METHODS[job["method_key"]]["label"],
                    "distance": dist,
                    "success_ratio": dist_metric.get("success_ratio", math.nan),
                    "sg_mean": dist_metric.get("sg_mean", math.nan),
                    "trials": dist_metric.get("trials", ""),
                    "success": dist_metric.get("success", ""),
                    "output_path": job["output_path"],
                }
            )
        rows.append(row)

    rows.sort(key=lambda row: (row["job_type"], row["grid"], int(row["budget"]), int(row["seed"]), row["benchmark"], row["method_key"]))
    per_rows.sort(key=lambda row: (row["job_type"], row["grid"], int(row["budget"]), int(row["seed"]), row["benchmark"], row["method_key"], int(row["distance"])))

    write_csv(OUTPUT_ROOT / "p0_supplement_long_table.csv", rows)
    write_csv(OUTPUT_ROOT / "p0_supplement_per_distance.csv", per_rows)
    write_csv(OUTPUT_ROOT / "budget_sensitivity_table.csv", [row for row in rows if row["job_type"] == "budget_sensitivity"])
    write_csv(OUTPUT_ROOT / "task_seed_mmgag_table.csv", [row for row in rows if row["job_type"] == "task_seed_mmgag"])
    write_csv(OUTPUT_ROOT / "task_seed_ultra_table.csv", [row for row in rows if row["job_type"] in {"task_seed_ultra", "budget_sensitivity"} and int(row["budget"]) == int(row["formal_budget"])])

    seed_summary = summarize_seed_rows(rows)
    budget_summary = summarize_budget_rows(rows)
    write_csv(OUTPUT_ROOT / "task_seed_summary.csv", seed_summary)
    write_csv(OUTPUT_ROOT / "budget_sensitivity_summary.csv", budget_summary)

    aggregate = {
        "created_at": now_iso(),
        "series": SERIES,
        "experiment": EXPERIMENT,
        "rows": rows,
        "per_distance": per_rows,
        "seed_summary": seed_summary,
        "budget_summary": budget_summary,
        "notes": [
            "Evaluation-only supplement; no retraining.",
            "Budget sensitivity uses the fixed ultra-long task-bank seed 20260521 and varies only the search budget.",
            "Ultra-long task-seed summary combines formal-budget rows from budget_sensitivity seed 20260521 with task_seed_ultra seeds 20260522 and 20260523.",
            "MM-GAG task-seed reruns use existing checkpoints on aerial, ground, and text goals.",
        ],
    }
    (OUTPUT_ROOT / "p0_supplement_aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(aggregate)
    return aggregate


def summarize_seed_rows(rows: list[dict]) -> list[dict]:
    formal_ultra = [
        row
        for row in rows
        if row["job_type"] == "budget_sensitivity" and int(row["budget"]) == int(row["formal_budget"])
    ]
    seed_rows = [row for row in rows if row["job_type"] in {"task_seed_mmgag", "task_seed_ultra"}] + formal_ultra
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in seed_rows:
        if row["job_type"] == "task_seed_mmgag":
            key = ("mmgag", row["benchmark"], row["grid"], row["budget"], row["method_key"])
        else:
            key = ("ultra_long", f"masa_aerial_{row['grid']}", row["grid"], row["formal_budget"], row["method_key"])
        grouped[key].append(row)

    summary = []
    for (family, benchmark, grid, budget, method_key), items in sorted(grouped.items()):
        values = [float(item["success_ratio"]) for item in items]
        summary.append(
            {
                "family": family,
                "benchmark": benchmark,
                "grid": grid,
                "budget": budget,
                "method_key": method_key,
                "method": METHODS[method_key]["label"],
                "num_seeds": len(items),
                "seeds": ",".join(str(item["seed"]) for item in sorted(items, key=lambda row: int(row["seed"]))),
                "mean_sr": sum(values) / len(values),
                "std_sr": sample_std(values),
                "min_sr": min(values),
                "max_sr": max(values),
            }
        )
    return summary


def summarize_budget_rows(rows: list[dict]) -> list[dict]:
    budget_rows = [row for row in rows if row["job_type"] == "budget_sensitivity"]
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in budget_rows:
        grouped[(row["grid"], row["budget"])].append(row)
    summary = []
    for (grid, budget), items in sorted(grouped.items(), key=lambda item: (item[0][0], int(item[0][1]))):
        by_method = {item["method_key"]: item for item in items}
        anchor = by_method.get("anchor0624")
        gomaa = by_method.get("gomaa")
        pristine = by_method.get("pristine")
        summary.append(
            {
                "grid": grid,
                "budget": budget,
                "anchor_sr": float(anchor["success_ratio"]) if anchor else math.nan,
                "gomaa_sr": float(gomaa["success_ratio"]) if gomaa else math.nan,
                "pristine_sr": float(pristine["success_ratio"]) if pristine else math.nan,
                "anchor_minus_gomaa": float(anchor["success_ratio"]) - float(gomaa["success_ratio"]) if anchor and gomaa else math.nan,
                "anchor_minus_pristine": float(anchor["success_ratio"]) - float(pristine["success_ratio"]) if anchor and pristine else math.nan,
            }
        )
    return summary


def sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = sum(values) / len(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def write_markdown(aggregate: dict) -> None:
    lines = [
        "# P0 补充评测结果",
        "",
        f"- 生成时间：{aggregate['created_at']}",
        "- 类型：evaluation-only，不重新训练模型。",
        "- 目的：补充预算敏感性和 task-bank seed 稳定性证据。",
        "- 距离桶说明：8x8 与 10x10 使用当前网格内可达的高距离端；更大距离范围另由 25x25 探索性压力测试承担。",
        "",
        "## 预算敏感性",
        "",
        "| Grid | Budget | Ours SR | GOMAA SR | GeoExplorer SR | Ours-GOMAA |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate["budget_summary"]:
        lines.append(
            f"| {row['grid']} | {int(row['budget'])} | {float(row['anchor_sr']):.4f} | "
            f"{float(row['gomaa_sr']):.4f} | {float(row['pristine_sr']):.4f} | {float(row['anchor_minus_gomaa']):+.4f} |"
        )
    lines.extend(["", "## Task-bank Seed 稳定性", "", "| Family | Benchmark | Method | Seeds | Mean SR | Std | Min | Max |", "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |"])
    for row in aggregate["seed_summary"]:
        lines.append(
            f"| {row['family']} | {row['benchmark']} | {row['method']} | {row['seeds']} | "
            f"{float(row['mean_sr']):.4f} | {float(row['std_sr']):.4f} | {float(row['min_sr']):.4f} | {float(row['max_sr']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            "- `p0_supplement_long_table.csv`：所有 job 汇总。",
            "- `budget_sensitivity_table.csv` 与 `budget_sensitivity_summary.csv`：预算敏感性结果。",
            "- `task_seed_mmgag_table.csv`、`task_seed_ultra_table.csv` 与 `task_seed_summary.csv`：任务库 seed 复评结果。",
            "- `p0_supplement_per_distance.csv`：分距离结果。",
            "",
        ]
    )
    (OUTPUT_ROOT / "p0_supplement_summary_zh.md").write_text("\n".join(lines), encoding="utf-8")


def write_status(phase: str, jobs: list[dict], active: dict[str, dict] | None = None, aggregate: dict | None = None, **extra) -> None:
    active = active or {}
    counts = defaultdict(int)
    for job in jobs:
        counts[job.get("status", "pending")] += 1
    payload = {
        "timestamp": now_iso(),
        "phase": phase,
        "series": SERIES,
        "experiment": EXPERIMENT,
        "output_root": str(OUTPUT_ROOT),
        "total_jobs": len(jobs),
        "status_counts": dict(counts),
        "active_eval_processes": len(active),
        "gpu_snapshot": gpu_snapshot(),
        "jobs": {
            job["key"]: {
                "status": job.get("status", "pending"),
                "job_type": job["job_type"],
                "seed": job["seed"],
                "grid": job["grid"],
                "budget": job["budget"],
                "benchmark": job["benchmark"],
                "method_key": job["method_key"],
                "gpu": job.get("gpu"),
                "pid": job.get("pid"),
                "returncode": job.get("returncode"),
                "output_path": job["output_path"],
            }
            for job in jobs
        },
    }
    if aggregate is not None:
        payload["aggregate_path"] = str(OUTPUT_ROOT / "p0_supplement_aggregate.json")
        payload["summary_path"] = str(OUTPUT_ROOT / "p0_supplement_summary_zh.md")
    payload.update(extra)
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def terminate(active: dict[str, dict]) -> None:
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


def run_scheduler(jobs: list[dict], eval_gpus: list[int], workers_per_gpu: int) -> int:
    active: dict[str, dict] = {}
    aggregate = None
    try:
        while True:
            finished = []
            for key, item in active.items():
                code = item["process"].poll()
                if code is None:
                    continue
                try:
                    item["log_handle"].close()
                except Exception:
                    pass
                job = item["job"]
                ok = code == 0 and output_ready(Path(job["output_path"]))
                job.update({"status": "completed" if ok else "failed", "returncode": int(code), "ended_at": now_iso()})
                finished.append(key)
            for key in finished:
                active.pop(key, None)

            running_by_gpu = defaultdict(int)
            for item in active.values():
                running_by_gpu[int(item["gpu"])] += 1

            for job in jobs:
                if job.get("status", "pending") == "pending" and output_ready(Path(job["output_path"])):
                    job["status"] = "completed"
                    job["resume_reused"] = True
                if job.get("status", "pending") != "pending":
                    continue
                available = [gpu for gpu in eval_gpus if running_by_gpu[gpu] < workers_per_gpu]
                if not available:
                    continue
                gpu = available[0]
                launched = launch_job(job, gpu)
                job.update({"status": "running", "gpu": gpu, "pid": int(launched["process"].pid), "started_at": now_iso()})
                active[job["key"]] = {"job": job, "gpu": gpu, **launched}
                running_by_gpu[gpu] += 1

            if any(job.get("status") == "failed" for job in jobs):
                write_status("failed", jobs, active)
                terminate(active)
                return 1
            if all(job.get("status") == "completed" for job in jobs):
                aggregate = aggregate_outputs(jobs)
                write_status("completed", jobs, active, aggregate=aggregate)
                return 0
            write_status("running", jobs, active)
            time.sleep(20)
    except Exception as exc:
        terminate(active)
        write_status("failed", jobs, active, error=repr(exc))
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Run P0 supplement evaluation: budget sensitivity and task-bank seed reruns.")
    parser.add_argument("--mode", choices=["all", "budget", "seed"], default="all")
    parser.add_argument("--eval-gpus", default="0,1,2,3")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    args = parser.parse_args()
    eval_gpus = [int(item.strip()) for item in args.eval_gpus.split(",") if item.strip()]
    if not eval_gpus:
        eval_gpus = [0]
    workers_per_gpu = max(1, int(args.workers_per_gpu))
    ensure_inputs()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args.mode)
    write_status("starting", jobs, eval_gpus=eval_gpus, mode=args.mode, workers_per_gpu=workers_per_gpu)
    return run_scheduler(jobs, eval_gpus, workers_per_gpu)


if __name__ == "__main__":
    raise SystemExit(main())
