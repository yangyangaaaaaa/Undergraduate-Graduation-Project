from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path


SERIES = "algo_paper_generalization_20260516"
EXPERIMENT = "anchor0624_factorial_generalization_seed321_480k"
REMOTE_ROOT = Path("/root/geoexplorer")
REPO_ROOT = REMOTE_ROOT / "GeoExplorer"
EXP_ROOT = REMOTE_ROOT / "ab_experiments" / SERIES / EXPERIMENT
STATUS_DIR = EXP_ROOT / "monitoring"
LOG_DIR = STATUS_DIR / "paper_baseline_logs"
STATUS_PATH = STATUS_DIR / "paper_baseline_compare_status_latest.json"
PID_PATH = STATUS_DIR / "paper_baseline_compare_supervisor.pid"
OUTPUT_ROOT = REMOTE_ROOT / "analysis" / "pipeline_20260516_paper_baseline_compare"
EVALUATOR = STATUS_DIR / "paper_baseline_evaluator.py"

COMPARE_ROOT = Path("/root/src/compare_baselines_bundle_20260505_v2/compare_baselines_bundle")
GOMAA_ROOT = COMPARE_ROOT / "gomaa_geo_official"
DIT_ROOT = COMPARE_ROOT / "dit_agl_baseline"
ANCHOR_CKPT = (
    REMOTE_ROOT
    / "results/checkpoint/algo_ablation_anchor0624_20260515/"
    / "masa_plus_mmgag_anchor0624_component_ablation_seed321_480k_gpu01/"
    / "g1_p1_e1_v1_seed321_t480k/geoexplorer.pt"
)
ANCHOR_LLM = REMOTE_ROOT / "results/checkpoint/env_modeling_fullrerun_20260407_111046/state_action.ckpt"
GOMAA_CKPT = GOMAA_ROOT / "gomaa_geo/checkpoint/formal_ppo_seed42_t480k/formal_ppo.pt"
GOMAA_LLM = GOMAA_ROOT / "gomaa_geo/checkpoint/formal_pretrain_seed42_e50/formal_falcon.ckpt"
DIT_LLM = DIT_ROOT / "dit_agl/checkpoint/formal_pretrain_seed42_e50/formal_falcon.ckpt"

TASK_BANK_SEED = 20260516
GPU_SLOTS = {2: 1, 3: 1}


BENCHMARKS = {
    "masa_aerial": {
        "paper_table": "Table 1",
        "dataset": "masa",
        "goal_mode": "aerial",
        "fixed_goal_mode": "none",
        "repeats_per_dist": 5,
        "test_path_candidates": [
            "/root/geoexplorer/GeoExplorer/data/masa/sat_test_grid_5.npy",
            "/root/geoexplorer/data/masa/sat_test_grid_5.npy",
        ],
    },
    "mmgag_aerial": {
        "paper_table": "Table 2 Goal=I",
        "dataset": "mmgag",
        "goal_mode": "aerial",
        "fixed_goal_mode": "none",
        "repeats_per_dist": 5,
        "test_path_candidates": [
            "/root/geoexplorer/GeoExplorer/data/mm_gag/processed/mmgag_sat_grid_5.npy",
            "/root/geoexplorer/GeoExplorer/data/mm_gag/mmgag_sat_grid_5.npy",
        ],
    },
    "mmgag_ground": {
        "paper_table": "Table 2 Goal=G",
        "dataset": "mmgag",
        "goal_mode": "ground",
        "fixed_goal_mode": "none",
        "repeats_per_dist": 5,
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
    "mmgag_text": {
        "paper_table": "Table 2 Goal=T",
        "dataset": "mmgag",
        "goal_mode": "text",
        "fixed_goal_mode": "none",
        "repeats_per_dist": 5,
        "test_path_candidates": [
            "/root/geoexplorer/GeoExplorer/data/mm_gag/processed/mmgag_sat_grid_5.npy",
            "/root/geoexplorer/GeoExplorer/data/mm_gag/mmgag_sat_grid_5.npy",
        ],
        "goal_embeds_candidates": [
            "/root/geoexplorer/GeoExplorer/data/mm_gag/processed/mmgag_text_embeds.npy",
            "/root/geoexplorer/GeoExplorer/data/mm_gag/mmgag_text_embeds.npy",
        ],
    },
    "swissviewmonuments_aerial": {
        "paper_table": "Table 4 Goal=I",
        "dataset": "swissviewmonuments",
        "goal_mode": "aerial",
        "fixed_goal_mode": "monuments",
        "repeats_per_dist": 1,
        "test_path_candidates": [
            "/root/geoexplorer/GeoExplorer/data/swissview/swissviewmonuments_sat_patches.npy",
            "/root/geoexplorer/GeoExplorer/data/swissview/swissviewmonuments_patches.npy",
            "/root/geoexplorer/GeoExplorer/data/swissview/processed/swissviewmonuments_sat_patches.npy",
        ],
    },
    "swissviewmonuments_ground": {
        "paper_table": "Table 4 Goal=G",
        "dataset": "swissviewmonuments",
        "goal_mode": "ground",
        "fixed_goal_mode": "monuments",
        "repeats_per_dist": 1,
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
    "swissview100_aerial": {
        "paper_table": "Table S4",
        "dataset": "swissview",
        "goal_mode": "aerial",
        "fixed_goal_mode": "none",
        "repeats_per_dist": 5,
        "optional": True,
        "test_path_candidates": [
            "/root/geoexplorer/GeoExplorer/data/swissview/swissview100_sat_patches.npy",
            "/root/geoexplorer/GeoExplorer/data/swissview/processed/swissview100_sat_patches.npy",
        ],
    },
    "xbd_pre_aerial": {
        "paper_table": "Table S3 xBD-pre",
        "dataset": "xbd",
        "goal_mode": "aerial",
        "fixed_goal_mode": "none",
        "repeats_per_dist": 5,
        "test_path_candidates": [
            "/root/geoexplorer/GeoExplorer/data/xbd/processed/paper_test800/xbd_pre_grid_5.npy",
            "/root/geoexplorer/GeoExplorer/data/xbd/processed/xbd_pre_grid_5.npy",
        ],
    },
    "xbd_disaster_aerial": {
        "paper_table": "Table 3 / Table S3 xBD-disaster",
        "dataset": "xbd-post",
        "goal_mode": "aerial",
        "fixed_goal_mode": "none",
        "repeats_per_dist": 5,
        "test_path_candidates": [
            "/root/geoexplorer/GeoExplorer/data/xbd/processed/paper_test800/xbd_post_grid_5.npy",
            "/root/geoexplorer/GeoExplorer/data/xbd/processed/xbd_post_grid_5.npy",
        ],
        "pre_goal_path_candidates": [
            "/root/geoexplorer/GeoExplorer/data/xbd/processed/paper_test800/xbd_pre_grid_5.npy",
            "/root/geoexplorer/GeoExplorer/data/xbd/processed/xbd_pre_grid_5.npy",
        ],
    },
}


METHODS = {
    "random": {
        "method": "random",
        "label": "Random policy",
        "repo_dir": "",
        "checkpoint": "",
        "llm_checkpoint": "",
        "benchmarks": ["masa_aerial", "mmgag_aerial", "xbd_pre_aerial", "xbd_disaster_aerial"],
        "notes": "Original paper reports Random for Table 1, Table 2 Goal=I, and xBD; not used for ground/text.",
    },
    "gomaa": {
        "method": "gomaa",
        "label": "GOMAA-Geo",
        "repo_dir": str(GOMAA_ROOT),
        "checkpoint": str(GOMAA_CKPT),
        "llm_checkpoint": str(GOMAA_LLM),
        "benchmarks": [
            "masa_aerial",
            "mmgag_aerial",
            "mmgag_ground",
            "mmgag_text",
            "swissviewmonuments_aerial",
            "swissviewmonuments_ground",
            "swissview100_aerial",
            "xbd_pre_aerial",
            "xbd_disaster_aerial",
        ],
        "notes": "Primary multimodal external baseline from the paper.",
    },
    "dit": {
        "method": "dit",
        "label": "DiT-AGL",
        "repo_dir": str(DIT_ROOT),
        "checkpoint": "",
        "llm_checkpoint": str(DIT_LLM),
        "benchmarks": ["masa_aerial", "mmgag_aerial", "xbd_pre_aerial", "xbd_disaster_aerial"],
        "notes": "Action-only sequence baseline; limited here to aerial-goal settings.",
    },
    "anchor0624": {
        "method": "geoexplorer",
        "label": "GeoExplorer-anchor0624",
        "repo_dir": str(REPO_ROOT),
        "checkpoint": str(ANCHOR_CKPT),
        "llm_checkpoint": str(ANCHOR_LLM),
        "benchmarks": [
            "masa_aerial",
            "mmgag_aerial",
            "mmgag_ground",
            "mmgag_text",
            "swissviewmonuments_aerial",
            "swissviewmonuments_ground",
            "swissview100_aerial",
            "xbd_pre_aerial",
            "xbd_disaster_aerial",
        ],
        "notes": "Our current best shared-MASA anchor, g1_p1_e1_v1, SR=0.6240 on prior shared-MASA.",
    },
}

BLOCKED_METHODS = {
    "PPO policy": "No validated standalone PPO baseline checkpoint/evaluator is available in this local bundle for all requested paper settings.",
    "AiRLoc": "Existing AiRLoc bundle is unimodal aerial and MASA-layout specific; do not report it on ground/text without a validated adapter.",
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def resolve_existing(candidates: list[str], optional: bool = False) -> str | None:
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    if optional:
        return None
    raise FileNotFoundError(f"missing all candidates: {candidates}")


def resolved_benchmarks() -> dict[str, dict]:
    out = {}
    for name, bench in BENCHMARKS.items():
        optional = bool(bench.get("optional", False))
        test_path = resolve_existing(bench["test_path_candidates"], optional=optional)
        if test_path is None:
            out[name] = {**bench, "status": "skipped_missing_optional_dataset"}
            continue
        resolved = {"test_path": test_path}
        if "pre_goal_path_candidates" in bench:
            resolved_pre_goal = resolve_existing(bench["pre_goal_path_candidates"], optional=optional)
            if resolved_pre_goal is None:
                out[name] = {**bench, "status": "skipped_missing_pre_goal_path"}
                continue
            resolved["pre_goal_path"] = resolved_pre_goal
        if "goal_embeds_candidates" in bench:
            resolved_goal = resolve_existing(bench["goal_embeds_candidates"], optional=optional)
            if resolved_goal is None:
                out[name] = {**bench, "status": "skipped_missing_optional_goal_embeds"}
                continue
            resolved["goal_embeds"] = resolved_goal
        out[name] = {**bench, "status": "ready", "resolved": resolved}
    return out


def build_specs(benchmarks: dict[str, dict]) -> list[dict]:
    specs = []
    for method_key, method in METHODS.items():
        for bench_name in method["benchmarks"]:
            bench = benchmarks[bench_name]
            if bench.get("status") != "ready":
                continue
            name = f"{method_key}__{bench_name}"
            specs.append(
                {
                    "name": name,
                    "method_key": method_key,
                    "method": method["method"],
                    "method_label": method["label"],
                    "method_notes": method["notes"],
                    "benchmark": bench_name,
                    "paper_table": bench["paper_table"],
                    "dataset": bench["dataset"],
                    "goal_mode": bench["goal_mode"],
                    "fixed_goal_mode": bench["fixed_goal_mode"],
                    "repeats_per_dist": int(bench.get("repeats_per_dist", 5)),
                    "repo_dir": method["repo_dir"],
                    "checkpoint": method["checkpoint"],
                    "llm_checkpoint": method["llm_checkpoint"],
                    "resolved": bench["resolved"],
                    "output_path": OUTPUT_ROOT / method_key / f"{bench_name}.json",
                    "status": "pending",
                }
            )
    return specs


def output_ready(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return "success_ratio" in payload or isinstance(payload.get("modes"), list)


def parse_metric(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "success_ratio" in payload:
        sr = float(payload["success_ratio"])
    else:
        sr = float(payload["modes"][0]["success_ratio"])
    sg = float(payload.get("sg_mean", payload.get("modes", [{}])[0].get("sg_mean", float("nan"))))
    per_distance = payload.get("per_distance") or payload.get("modes", [{}])[0].get("per_dist", [])
    return {"success_ratio": sr, "sg_mean": sg, "per_distance": per_distance, "payload": payload}


def clean_env() -> dict:
    return {key: value for key, value in os.environ.items() if not key.startswith("GEOEXPLORER_")}


def build_command(spec: dict) -> list[str]:
    cmd = [
        "/usr/bin/python3",
        str(EVALUATOR),
        "--method",
        spec["method"],
        "--method-label",
        spec["method_label"],
        "--benchmark",
        spec["benchmark"],
        "--paper-table",
        spec["paper_table"],
        "--dataset",
        spec["dataset"],
        "--goal-mode",
        spec["goal_mode"],
        "--fixed-goal-mode",
        spec["fixed_goal_mode"],
        "--test-path",
        spec["resolved"]["test_path"],
        "--device",
        "cuda:0",
        "--patch-size",
        "5",
        "--budget",
        "10",
        "--distances",
        "4,5,6,7,8",
        "--repeats-per-dist",
        str(spec.get("repeats_per_dist", 5)),
        "--seed",
        str(TASK_BANK_SEED),
        "--output-path",
        str(spec["output_path"]),
    ]
    if spec["repo_dir"]:
        cmd.extend(["--repo-dir", spec["repo_dir"]])
    if spec["checkpoint"]:
        cmd.extend(["--checkpoint", spec["checkpoint"]])
    if spec["llm_checkpoint"]:
        cmd.extend(["--llm-checkpoint", spec["llm_checkpoint"]])
    if "goal_embeds" in spec["resolved"]:
        cmd.extend(["--goal-embeds", spec["resolved"]["goal_embeds"]])
    if "pre_goal_path" in spec["resolved"]:
        cmd.extend(["--pre-goal-path", spec["resolved"]["pre_goal_path"]])
    return cmd


def launch_eval(spec: dict, gpu: int) -> subprocess.Popen:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    spec["output_path"].parent.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{spec['name']}.log"
    log_handle = log_path.open("ab", buffering=0)
    log_handle.write(f"\n[{now_iso()}] launching {spec['name']} on GPU{gpu}\n".encode("utf-8"))
    env = clean_env()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONPATH": ":".join(
                [
                    "/root/geoexplorer/env/geoexplorer_site",
                    "/root/geoexplorer",
                    "/root/geoexplorer/GeoExplorer",
                    str(GOMAA_ROOT),
                    str(DIT_ROOT),
                ]
            ),
        }
    )
    return subprocess.Popen(
        build_command(spec),
        cwd=str(spec["repo_dir"] or REPO_ROOT),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def write_outputs(specs: list[dict], benchmarks: dict[str, dict]) -> dict:
    rows = []
    for spec in specs:
        metric = parse_metric(spec["output_path"])
        per_dist = {int(row["distance"]): float(row["success_ratio"]) for row in metric["per_distance"]}
        rows.append(
            {
                "method": spec["method_label"],
                "method_key": spec["method_key"],
                "benchmark": spec["benchmark"],
                "paper_table": spec["paper_table"],
                "dataset": spec["dataset"],
                "goal_mode": spec["goal_mode"],
                "pre_goal_path": spec["resolved"].get("pre_goal_path"),
                "repeats_per_dist": spec["repeats_per_dist"],
                "success_ratio": metric["success_ratio"],
                "sg_mean": metric["sg_mean"],
                "d4": per_dist.get(4, float("nan")),
                "d5": per_dist.get(5, float("nan")),
                "d6": per_dist.get(6, float("nan")),
                "d7": per_dist.get(7, float("nan")),
                "d8": per_dist.get(8, float("nan")),
                "output_path": str(spec["output_path"]),
                "checkpoint": spec["checkpoint"] or None,
                "llm_checkpoint": spec["llm_checkpoint"] or None,
            }
        )
    rows.sort(key=lambda row: (row["benchmark"], -row["success_ratio"], row["method"]))

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    aggregate = {
        "generated_at": now_iso(),
        "series": SERIES,
        "experiment": EXPERIMENT,
        "task_bank_seed": TASK_BANK_SEED,
        "rows": rows,
        "blocked_or_not_applicable": BLOCKED_METHODS,
        "benchmarks": benchmarks,
        "rigor_notes": [
            "This is paper-aligned method comparison, separate from the anchor0624 factorial mechanism ablation.",
            "Ground/text comparisons are limited to methods with validated multimodal goal interfaces.",
            "xBD rows use the deterministic paper-test800 subset when available; xBD-disaster uses post-disaster observations with pre-disaster aerial goals.",
            "All included rows use greedy/argmax evaluation, 5x5 grid, B=10, and C={4,5,6,7,8}.",
            "Masa, MM-GAG, and SwissView100 use 5 generated start-goal pairs per image and distance; SwissViewMonuments follows the repository unseen-target protocol with 1 fixed-goal configuration per image and distance.",
        ],
    }
    (OUTPUT_ROOT / "paper_baseline_compare_aggregate.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with (OUTPUT_ROOT / "paper_baseline_compare_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "benchmark",
                "paper_table",
                "method",
                "success_ratio",
                "sg_mean",
                "d4",
                "d5",
                "d6",
                "d7",
                "d8",
                "dataset",
                "goal_mode",
                "repeats_per_dist",
                "output_path",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in writer.fieldnames})

    lines = [
        "# Paper-Aligned Baseline Compare",
        "",
        "- protocol: greedy/argmax, `5x5`, `B=10`, `C={4,5,6,7,8}`, fixed task-bank seed `20260516`.",
        "- task counts: Masa/MM-GAG/SwissView100 use `5` start-goal samples per image and distance; SwissViewMonuments uses the paper/repository unseen-target setting with `1` fixed-goal sample per image and distance.",
        "- scope: evaluates only method-task combinations with validated local interfaces; blocked combinations are listed below.",
        "- separation: this is external method comparison, not the anchor0624 mechanism ablation.",
        "",
    ]
    for benchmark in sorted({row["benchmark"] for row in rows}):
        lines.extend(
            [
                f"## {benchmark}",
                "",
                "| Method | SR | SG | d4 | d5 | d6 | d7 | d8 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in [item for item in rows if item["benchmark"] == benchmark]:
            lines.append(
                f"| {row['method']} | {row['success_ratio']:.4f} | {row['sg_mean']:.4f} | "
                f"{row['d4']:.4f} | {row['d5']:.4f} | {row['d6']:.4f} | {row['d7']:.4f} | {row['d8']:.4f} |"
            )
        lines.append("")
    lines.extend(["## Blocked / Not Applicable", ""])
    for name, reason in BLOCKED_METHODS.items():
        lines.append(f"- `{name}`: {reason}")
    lines.extend(["", "## Rigor Notes", ""])
    for note in aggregate["rigor_notes"]:
        lines.append(f"- {note}")
    (OUTPUT_ROOT / "paper_baseline_compare_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return aggregate


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


def main() -> int:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PID_PATH.write_text(str(os.getpid()) + "\n", encoding="utf-8")

    benchmarks = resolved_benchmarks()
    specs = build_specs(benchmarks)
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
                ok = code == 0 and output_ready(spec["output_path"])
                spec.update({"status": "completed" if ok else "failed", "returncode": int(code), "ended_at": now_iso()})
                finished.append(key)
                counts[item["gpu"]] -= 1
            for key in finished:
                active.pop(key, None)

            for spec in specs:
                if spec["status"] != "pending":
                    continue
                if output_ready(spec["output_path"]):
                    spec.update({"status": "completed", "resume_reused": True})
                    continue
                if spec["checkpoint"] and not Path(spec["checkpoint"]).exists():
                    spec.update({"status": "failed_missing_checkpoint"})
                    continue
                if spec["llm_checkpoint"] and not Path(spec["llm_checkpoint"]).exists():
                    spec.update({"status": "failed_missing_llm_checkpoint"})
                    continue
                available = [gpu for gpu, cap in GPU_SLOTS.items() if counts[gpu] < cap]
                if not available:
                    continue
                gpu = available[0]
                process = launch_eval(spec, gpu)
                spec.update({"status": "running", "gpu": gpu, "pid": int(process.pid), "started_at": now_iso()})
                active[spec["name"]] = {"gpu": gpu, "process": process, "spec": spec}
                counts[gpu] += 1

            phase = "completed" if specs and all(spec["status"] == "completed" for spec in specs) and not active else "running"
            if any(str(spec["status"]).startswith("failed") for spec in specs):
                phase = "failed"
            if phase == "completed" and summary is None:
                summary = write_outputs(specs, benchmarks)

            payload = {
                "timestamp": now_iso(),
                "phase": phase,
                "series": SERIES,
                "experiment": EXPERIMENT,
                "output_root": str(OUTPUT_ROOT),
                "summary_path": str(OUTPUT_ROOT / "paper_baseline_compare_summary.md") if summary else None,
                "table_path": str(OUTPUT_ROOT / "paper_baseline_compare_table.csv") if summary else None,
                "aggregate_path": str(OUTPUT_ROOT / "paper_baseline_compare_aggregate.json") if summary else None,
                "allowed_gpus": list(GPU_SLOTS),
                "gpu_snapshot": gpu_snapshot(),
                "active_eval_processes": len(active),
                "blocked_or_not_applicable": BLOCKED_METHODS,
                "benchmarks": benchmarks,
                "runs": {
                    spec["name"]: {
                        "method": spec["method_label"],
                        "benchmark": spec["benchmark"],
                        "paper_table": spec["paper_table"],
                        "repeats_per_dist": spec["repeats_per_dist"],
                        "status": spec["status"],
                        "gpu": spec.get("gpu"),
                        "pid": spec.get("pid"),
                        "returncode": spec.get("returncode"),
                        "output_path": str(spec["output_path"]),
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
        terminate(active)
        write_status({"timestamp": now_iso(), "phase": "failed", "error": str(exc), "active_eval_processes": len(active)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
