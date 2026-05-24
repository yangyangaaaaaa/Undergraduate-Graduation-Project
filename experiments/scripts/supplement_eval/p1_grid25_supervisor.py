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
EXPERIMENT = "p1_grid25_eval"

REMOTE_ROOT = Path("/root/geoexplorer")
REPO_ROOT = REMOTE_ROOT / "GeoExplorer"
EXP_ROOT = REMOTE_ROOT / "ab_experiments" / SERIES / EXPERIMENT
MONITORING = EXP_ROOT / "monitoring"
LOG_DIR = MONITORING / "logs"
STATUS_PATH = MONITORING / "p1_grid25_status_latest.json"
OUTPUT_ROOT = REMOTE_ROOT / "analysis" / "pipeline_20260524_p1_grid25_eval"
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
GRID25_PATH = REMOTE_ROOT / "data/masa/sat_test_grid_25.npy"

DISTANCES = [12, 16, 20, 24, 28, 32, 36, 40, 44, 48]
FORMAL_BUDGET = 60
BUDGETS = [40, 50, 60, 70]
BUDGET_SEED = 20260524
SEED_RERUNS = [20260524, 20260525, 20260526]
REPEATS_PER_DIST = 5

METHODS = {
    "anchor0624": {
        "method": "geoexplorer",
        "label": "GeoExplorer-anchor0624",
        "repo_dir": str(REPO_ROOT),
        "checkpoint": str(ANCHOR_CKPT),
        "llm_checkpoint": str(ANCHOR_LLM),
    },
    "gomaa": {
        "method": "gomaa",
        "label": "GOMAA-Geo",
        "repo_dir": str(GOMAA_ROOT),
        "checkpoint": str(GOMAA_CKPT),
        "llm_checkpoint": str(GOMAA_LLM),
    },
    "pristine": {
        "method": "geoexplorer",
        "label": "GeoExplorer-pristine",
        "repo_dir": str(REPO_ROOT),
        "checkpoint": str(PRISTINE_CKPT),
        "llm_checkpoint": str(ANCHOR_LLM),
    },
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def ensure_inputs() -> None:
    required = [
        GRID25_PATH,
        ANCHOR_CKPT,
        ANCHOR_LLM,
        GOMAA_CKPT,
        GOMAA_LLM,
        PRISTINE_CKPT,
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


def job_output_path(*parts: str) -> str:
    return str(OUTPUT_ROOT.joinpath("raw", *parts).with_suffix(".json"))


def build_jobs(mode: str) -> list[dict]:
    jobs = []
    if mode in {"all", "budget"}:
        for budget in BUDGETS:
            for method_key in METHODS:
                jobs.append(
                    {
                        "job_type": "grid25_budget_sensitivity",
                        "key": f"grid25_b{budget}_{method_key}",
                        "benchmark": f"masa_aerial_grid25_budget{budget}",
                        "budget": budget,
                        "seed": BUDGET_SEED,
                        "method_key": method_key,
                        "output_path": job_output_path("budget_sensitivity", f"b{budget}", method_key),
                    }
                )
    if mode in {"all", "seed"}:
        for seed in SEED_RERUNS:
            for method_key in METHODS:
                jobs.append(
                    {
                        "job_type": "grid25_task_seed",
                        "key": f"seed{seed}_grid25_{method_key}",
                        "benchmark": f"masa_aerial_grid25_seed{seed}",
                        "budget": FORMAL_BUDGET,
                        "seed": seed,
                        "method_key": method_key,
                        "output_path": job_output_path("task_seed", f"seed{seed}", method_key),
                    }
                )
    return jobs


def build_eval_command(job: dict) -> list[str]:
    method = METHODS[job["method_key"]]
    return [
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
        "Supplement grid25 stress",
        "--dataset",
        "masa",
        "--goal-mode",
        "aerial",
        "--fixed-goal-mode",
        "none",
        "--test-path",
        str(GRID25_PATH),
        "--device",
        "cuda:0",
        "--patch-size",
        "25",
        "--budget",
        str(job["budget"]),
        "--distances",
        ",".join(str(item) for item in DISTANCES),
        "--repeats-per-dist",
        str(REPEATS_PER_DIST),
        "--seed",
        str(job["seed"]),
        "--max-images",
        "0",
        "--output-path",
        job["output_path"],
    ]


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
        "trials": int(mode.get("total_trials", payload.get("num_tasks", 0))),
        "success": int(mode.get("success", round(payload["success_ratio"] * payload.get("num_tasks", 0)))),
        "sr": float(payload["success_ratio"]),
        "sg": float(payload.get("sg_mean", math.nan)),
        "per_distance": per_dist,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = sum(values) / len(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def aggregate_outputs(jobs: list[dict]) -> dict:
    rows = []
    per_rows = []
    for job in jobs:
        metric = parse_metric(Path(job["output_path"]))
        row = {
            "job_type": job["job_type"],
            "seed": job["seed"],
            "grid": "25x25",
            "budget": job["budget"],
            "benchmark": job["benchmark"],
            "method_key": job["method_key"],
            "method": METHODS[job["method_key"]]["label"],
            "success_ratio": metric["sr"],
            "sg_mean": metric["sg"],
            "trials": metric["trials"],
            "success": metric["success"],
            "distances": ",".join(str(item) for item in DISTANCES),
            "output_path": job["output_path"],
        }
        for dist in DISTANCES:
            dist_metric = metric["per_distance"].get(int(dist), {})
            row[f"d{dist}"] = dist_metric.get("success_ratio", math.nan)
            per_rows.append(
                {
                    "job_type": job["job_type"],
                    "seed": job["seed"],
                    "grid": "25x25",
                    "budget": job["budget"],
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

    write_csv(OUTPUT_ROOT / "p1_grid25_long_table.csv", rows)
    write_csv(OUTPUT_ROOT / "p1_grid25_per_distance.csv", per_rows)
    write_csv(OUTPUT_ROOT / "p1_grid25_budget_table.csv", [row for row in rows if row["job_type"] == "grid25_budget_sensitivity"])
    write_csv(OUTPUT_ROOT / "p1_grid25_seed_table.csv", [row for row in rows if row["job_type"] == "grid25_task_seed"])

    budget_summary = []
    for budget in BUDGETS:
        sub = [row for row in rows if row["job_type"] == "grid25_budget_sensitivity" and int(row["budget"]) == budget]
        by_method = {row["method_key"]: row for row in sub}
        anchor = by_method.get("anchor0624")
        gomaa = by_method.get("gomaa")
        pristine = by_method.get("pristine")
        budget_summary.append(
            {
                "grid": "25x25",
                "budget": budget,
                "anchor_sr": anchor["success_ratio"] if anchor else math.nan,
                "gomaa_sr": gomaa["success_ratio"] if gomaa else math.nan,
                "pristine_sr": pristine["success_ratio"] if pristine else math.nan,
                "anchor_minus_gomaa": anchor["success_ratio"] - gomaa["success_ratio"] if anchor and gomaa else math.nan,
                "anchor_minus_pristine": anchor["success_ratio"] - pristine["success_ratio"] if anchor and pristine else math.nan,
            }
        )
    write_csv(OUTPUT_ROOT / "p1_grid25_budget_summary.csv", budget_summary)

    seed_summary = []
    seed_rows = [row for row in rows if row["job_type"] == "grid25_task_seed"]
    for method_key in METHODS:
        values = [row["success_ratio"] for row in seed_rows if row["method_key"] == method_key]
        if not values:
            continue
        seed_summary.append(
            {
                "grid": "25x25",
                "budget": FORMAL_BUDGET,
                "method_key": method_key,
                "method": METHODS[method_key]["label"],
                "num_seeds": len(values),
                "seeds": ",".join(str(seed) for seed in SEED_RERUNS),
                "mean_sr": sum(values) / len(values),
                "std_sr": sample_std(values),
                "min_sr": min(values),
                "max_sr": max(values),
            }
        )
    write_csv(OUTPUT_ROOT / "p1_grid25_seed_summary.csv", seed_summary)

    aggregate = {
        "created_at": now_iso(),
        "series": SERIES,
        "experiment": EXPERIMENT,
        "rows": rows,
        "per_distance": per_rows,
        "budget_summary": budget_summary,
        "seed_summary": seed_summary,
        "notes": [
            "Evaluation-only 25x25 stress supplement; no retraining.",
            "Distance buckets follow the earlier 25x25 exploratory setting.",
            "Each job uses 10 distance buckets and 5 repeats per distance over all MASA test images.",
        ],
    }
    (OUTPUT_ROOT / "p1_grid25_aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return aggregate


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
                "grid": "25x25",
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
        payload["aggregate_path"] = str(OUTPUT_ROOT / "p1_grid25_aggregate.json")
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
    parser = argparse.ArgumentParser(description="Run 25x25 P1 supplement evaluation.")
    parser.add_argument("--mode", choices=["all", "budget", "seed"], default="all")
    parser.add_argument("--eval-gpus", default="0,1,2,3")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    args = parser.parse_args()
    eval_gpus = [int(item.strip()) for item in args.eval_gpus.split(",") if item.strip()]
    if not eval_gpus:
        eval_gpus = [0]
    ensure_inputs()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args.mode)
    write_status("starting", jobs, eval_gpus=eval_gpus, mode=args.mode, workers_per_gpu=max(1, int(args.workers_per_gpu)))
    return run_scheduler(jobs, eval_gpus, max(1, int(args.workers_per_gpu)))


if __name__ == "__main__":
    raise SystemExit(main())
