#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the C=1,2,3 short-distance evaluation pack on the server.

This is evaluation-only. It reuses the paper-aligned evaluator and existing
checkpoints, then writes comparison and factorial-ablation summaries.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REMOTE_ROOT = Path("/root/geoexplorer")
REPO_ROOT = REMOTE_ROOT / "GeoExplorer"
OUTPUT_BASE = REMOTE_ROOT / "analysis"
LATEST_LINK = OUTPUT_BASE / "short_distance_c123_eval_latest"
LATEST_STATUS = OUTPUT_BASE / "short_distance_c123_eval_latest_status.json"

GENERALIZATION_EXP = (
    REMOTE_ROOT
    / "ab_experiments/algo_paper_generalization_20260516/"
    / "anchor0624_factorial_generalization_seed321_480k"
)
MONITORING_DIR = GENERALIZATION_EXP / "monitoring"
EVALUATOR = MONITORING_DIR / "paper_baseline_evaluator.py"

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
PRISTINE_CKPT = (
    REMOTE_ROOT
    / "results/checkpoint/algo_dualseed480k_20260427/"
    / "masa_plus_mmgag_arena_pristine_a040_a0405_sine0405_dualseed480k/"
    / "wave2_seed321_4gpu/pristine_seed321_t480k/geoexplorer.pt"
)
GOMAA_CKPT = GOMAA_ROOT / "gomaa_geo/checkpoint/formal_ppo_seed42_t480k/formal_ppo.pt"
GOMAA_LLM = GOMAA_ROOT / "gomaa_geo/checkpoint/formal_pretrain_seed42_e50/formal_falcon.ckpt"
DIT_LLM = DIT_ROOT / "dit_agl/checkpoint/formal_pretrain_seed42_e50/formal_falcon.ckpt"

SOURCE_CKPT_ROOT = (
    REMOTE_ROOT
    / "results/checkpoint/algo_ablation_anchor0624_20260515/"
    / "masa_plus_mmgag_anchor0624_component_ablation_seed321_480k_gpu01"
)

TASK_BANK_SEED = 20260516
DEFAULT_DISTANCES = [1, 2, 3]
FACTOR_KEYS = ["G_gate", "P_pbrs", "E_low_entropy", "V_val78"]


BENCHMARKS: dict[str, dict] = {
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


METHODS: dict[str, dict] = {
    "random": {
        "method": "random",
        "label": "Random policy",
        "repo_dir": "",
        "checkpoint": "",
        "llm_checkpoint": "",
        "benchmarks": ["masa_aerial", "mmgag_aerial", "xbd_pre_aerial", "xbd_disaster_aerial"],
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
    },
    "dit": {
        "method": "dit",
        "label": "DiT-AGL",
        "repo_dir": str(DIT_ROOT),
        "checkpoint": "",
        "llm_checkpoint": str(DIT_LLM),
        "benchmarks": ["masa_aerial", "mmgag_aerial", "xbd_pre_aerial", "xbd_disaster_aerial"],
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
    },
    "pristine": {
        "method": "geoexplorer",
        "label": "GeoExplorer-pristine",
        "repo_dir": str(REPO_ROOT),
        "checkpoint": str(PRISTINE_CKPT),
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
    },
}


ABLATION_BENCHMARKS = [
    "masa_aerial",
    "mmgag_aerial",
    "mmgag_ground",
    "mmgag_text",
    "swissviewmonuments_aerial",
    "swissviewmonuments_ground",
]
ABLATION_TRANSFER_BENCHMARKS = [
    "mmgag_aerial",
    "mmgag_ground",
    "mmgag_text",
    "swissviewmonuments_aerial",
    "swissviewmonuments_ground",
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def mean(values: list[float]) -> float:
    finite = [float(item) for item in values if not math.isnan(float(item))]
    return float(sum(finite) / len(finite)) if finite else math.nan


def resolve_existing(candidates: list[str], optional: bool = False) -> str | None:
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    if optional:
        return None
    raise FileNotFoundError("missing all candidates: " + ", ".join(candidates))


def resolved_benchmarks() -> dict[str, dict]:
    resolved: dict[str, dict] = {}
    for name, bench in BENCHMARKS.items():
        optional = bool(bench.get("optional", False))
        test_path = resolve_existing(bench["test_path_candidates"], optional=optional)
        if test_path is None:
            resolved[name] = {**bench, "status": "skipped_missing_optional_dataset"}
            continue
        row = {"test_path": test_path}
        if "goal_embeds_candidates" in bench:
            goal_path = resolve_existing(bench["goal_embeds_candidates"], optional=optional)
            if goal_path is None:
                resolved[name] = {**bench, "status": "skipped_missing_goal_embeds"}
                continue
            row["goal_embeds"] = goal_path
        if "pre_goal_path_candidates" in bench:
            pre_goal_path = resolve_existing(bench["pre_goal_path_candidates"], optional=optional)
            if pre_goal_path is None:
                resolved[name] = {**bench, "status": "skipped_missing_pre_goal_path"}
                continue
            row["pre_goal_path"] = pre_goal_path
        resolved[name] = {**bench, "status": "ready", "resolved": row}
    return resolved


def make_branch(gate: int, pbrs: int, ent_low: int, val78: int) -> dict:
    branch = f"g{gate}_p{pbrs}_e{ent_low}_v{val78}"
    role = "factorial_ablation_cell"
    if branch == "g1_p1_e1_v1":
        role = "full_anchor0624"
    elif branch == "g0_p0_e0_v0":
        role = "same_data_no_added_mechanism_control"
    return {
        "branch": branch,
        "role": role,
        "factors": {
            "G_gate": int(gate),
            "P_pbrs": int(pbrs),
            "E_low_entropy": int(ent_low),
            "V_val78": int(val78),
        },
    }


def all_branches() -> list[dict]:
    branches = [
        make_branch(g, p, e, v)
        for v in (0, 1)
        for e in (0, 1)
        for p in (0, 1)
        for g in (0, 1)
    ]
    branches.sort(
        key=lambda item: (
            0 if item["branch"] == "g1_p1_e1_v1" else 1 if item["branch"] == "g0_p0_e0_v0" else 2,
            item["branch"],
        )
    )
    return branches


def checkpoint_for_branch(branch: str) -> Path:
    return SOURCE_CKPT_ROOT / f"{branch}_seed321_t480k" / "geoexplorer.pt"


def build_specs(
    scope: str,
    output_root: Path,
    benchmarks: dict[str, dict],
) -> list[dict]:
    specs: list[dict] = []
    if scope in {"all", "compare"}:
        for method_key, method in METHODS.items():
            for bench_name in method["benchmarks"]:
                bench = benchmarks[bench_name]
                if bench.get("status") != "ready":
                    continue
                specs.append(
                    {
                        "group": "comparison",
                        "name": f"comparison__{method_key}__{bench_name}",
                        "method_key": method_key,
                        "method_type": method["method"],
                        "method_label": method["label"],
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
                        "output_path": output_root / "eval_json/comparison" / method_key / f"{bench_name}.json",
                        "status": "pending",
                    }
                )
    if scope in {"all", "ablation"}:
        for item in all_branches():
            branch = item["branch"]
            for bench_name in ABLATION_BENCHMARKS:
                bench = benchmarks[bench_name]
                if bench.get("status") != "ready":
                    continue
                specs.append(
                    {
                        "group": "ablation",
                        "name": f"ablation__{branch}__{bench_name}",
                        "branch": branch,
                        "role": item["role"],
                        "factors": item["factors"],
                        "method_key": branch,
                        "method_type": "geoexplorer",
                        "method_label": f"GeoExplorer-{branch}",
                        "benchmark": bench_name,
                        "paper_table": bench["paper_table"],
                        "dataset": bench["dataset"],
                        "goal_mode": bench["goal_mode"],
                        "fixed_goal_mode": bench["fixed_goal_mode"],
                        "repeats_per_dist": int(bench.get("repeats_per_dist", 5)),
                        "repo_dir": str(REPO_ROOT),
                        "checkpoint": str(checkpoint_for_branch(branch)),
                        "llm_checkpoint": str(ANCHOR_LLM),
                        "resolved": bench["resolved"],
                        "output_path": output_root / "eval_json/ablation" / branch / f"{bench_name}.json",
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
    return "success_ratio" in payload and isinstance(payload.get("per_distance"), list)


def parse_metric(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mode = payload.get("modes", [{}])[0]
    per_distance_rows = payload.get("per_distance") or mode.get("per_dist") or []
    return {
        "payload": payload,
        "num_images": int(payload.get("num_images", 0)),
        "num_tasks": int(payload.get("num_tasks", mode.get("total_trials", 0))),
        "success_ratio": float(payload.get("success_ratio", mode.get("success_ratio", math.nan))),
        "success_ratio_ci_low": float(mode.get("success_ratio_ci_low", math.nan)),
        "success_ratio_ci_high": float(mode.get("success_ratio_ci_high", math.nan)),
        "sg_mean": float(payload.get("sg_mean", mode.get("sg_mean", math.nan))),
        "avg_steps_on_success": float(mode.get("avg_steps_on_success", math.nan)),
        "avg_deviation_on_success": float(mode.get("avg_deviation_on_success", math.nan)),
        "per_distance": {
            int(row["distance"]): {
                "trials": int(row.get("trials", 0)),
                "success": int(row.get("success", 0)),
                "success_ratio": float(row.get("success_ratio", math.nan)),
                "success_ratio_ci_low": float(row.get("success_ratio_ci_low", math.nan)),
                "success_ratio_ci_high": float(row.get("success_ratio_ci_high", math.nan)),
                "sg_mean": float(row.get("sg_mean", math.nan)),
            }
            for row in per_distance_rows
            if "distance" in row
        },
    }


def clean_eval_env(gpu: int) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("GEOEXPLORER_")}
    env.update(
        {
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "LD_LIBRARY_PATH": "/usr/local/nvidia/lib64:/usr/local/cuda/lib64",
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
    return env


def build_command(spec: dict, distances: list[int], max_images: int) -> list[str]:
    cmd = [
        "/usr/bin/python3",
        str(EVALUATOR),
        "--method",
        spec["method_type"],
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
        ",".join(str(item) for item in distances),
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
    if max_images > 0:
        cmd.extend(["--max-images", str(max_images)])
    return cmd


def launch_eval(spec: dict, gpu: int, output_root: Path, distances: list[int], max_images: int) -> subprocess.Popen:
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    spec["output_path"].parent.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{spec['name']}.log"
    log_handle = log_path.open("ab", buffering=0)
    log_handle.write(f"\n[{now_iso()}] launching {spec['name']} on GPU{gpu}\n".encode("utf-8"))
    log_handle.write(("CMD " + json.dumps(build_command(spec, distances, max_images)) + "\n").encode("utf-8"))
    cwd = str(spec["repo_dir"] or REPO_ROOT)
    return subprocess.Popen(
        build_command(spec, distances, max_images),
        cwd=cwd,
        env=clean_eval_env(gpu),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def gpu_snapshot() -> list[dict]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:
        return [{"error": str(exc)}]
    rows = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 4:
            rows.append({"gpu": int(parts[0]), "name": parts[1], "used_mb": int(parts[2]), "util": int(parts[3])})
    return rows


def write_status(output_root: Path, payload: dict) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    (output_root / "status.json").write_text(text, encoding="utf-8")
    LATEST_STATUS.write_text(text, encoding="utf-8")


def safe_symlink_latest(output_root: Path) -> None:
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    try:
        if LATEST_LINK.is_symlink() or LATEST_LINK.exists():
            if LATEST_LINK.is_dir() and not LATEST_LINK.is_symlink():
                shutil.rmtree(LATEST_LINK)
            else:
                LATEST_LINK.unlink()
        LATEST_LINK.symlink_to(output_root, target_is_directory=True)
    except Exception:
        (OUTPUT_BASE / "short_distance_c123_eval_latest_path.txt").write_text(
            str(output_root) + "\n",
            encoding="utf-8",
        )


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


def run_rows(specs: list[dict], distances: list[int]) -> list[dict]:
    rows = []
    for spec in specs:
        if spec["status"] != "completed":
            continue
        metric = parse_metric(spec["output_path"])
        row = {
            "group": spec["group"],
            "name": spec["name"],
            "method": spec["method_label"],
            "method_key": spec["method_key"],
            "method_type": spec["method_type"],
            "branch": spec.get("branch", ""),
            "role": spec.get("role", ""),
            "benchmark": spec["benchmark"],
            "paper_table": spec["paper_table"],
            "dataset": spec["dataset"],
            "goal_mode": spec["goal_mode"],
            "fixed_goal_mode": spec["fixed_goal_mode"],
            "repeats_per_dist": spec["repeats_per_dist"],
            "num_images": metric["num_images"],
            "num_tasks": metric["num_tasks"],
            "success_ratio": metric["success_ratio"],
            "success_ratio_ci_low": metric["success_ratio_ci_low"],
            "success_ratio_ci_high": metric["success_ratio_ci_high"],
            "sg_mean": metric["sg_mean"],
            "avg_steps_on_success": metric["avg_steps_on_success"],
            "avg_deviation_on_success": metric["avg_deviation_on_success"],
            "checkpoint": spec["checkpoint"] or "",
            "llm_checkpoint": spec["llm_checkpoint"] or "",
            "output_path": str(spec["output_path"]),
        }
        for factor in FACTOR_KEYS:
            row[factor] = spec.get("factors", {}).get(factor, "")
        for dist in distances:
            per = metric["per_distance"].get(dist, {})
            row[f"d{dist}_trials"] = per.get("trials", "")
            row[f"d{dist}_success"] = per.get("success", "")
            row[f"d{dist}_sr"] = per.get("success_ratio", math.nan)
            row[f"d{dist}_ci_low"] = per.get("success_ratio_ci_low", math.nan)
            row[f"d{dist}_ci_high"] = per.get("success_ratio_ci_high", math.nan)
            row[f"d{dist}_sg_mean"] = per.get("sg_mean", math.nan)
        rows.append(row)
    return rows


def write_csv_dicts(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def factorial_effects(rows: list[dict], metric_name: str) -> dict:
    effects: dict[str, dict] = {}
    for factor in FACTOR_KEYS:
        on_values = [float(row[metric_name]) for row in rows if int(row.get(factor, 0)) == 1]
        off_values = [float(row[metric_name]) for row in rows if int(row.get(factor, 0)) == 0]
        effects[factor] = {
            "on_mean": mean(on_values),
            "off_mean": mean(off_values),
            "effect_on_minus_off": mean(on_values) - mean(off_values),
        }
    interactions = {}
    for left, right in itertools.combinations(FACTOR_KEYS, 2):
        def mean_for(a: int, b: int) -> float:
            vals = [
                float(row[metric_name])
                for row in rows
                if int(row.get(left, 0)) == a and int(row.get(right, 0)) == b
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


def build_aggregate(output_root: Path, specs: list[dict], benchmarks: dict[str, dict], distances: list[int], args) -> dict:
    rows = run_rows(specs, distances)
    table_dir = output_root / "tables"
    report_dir = output_root / "reports"
    comparison_rows = [row for row in rows if row["group"] == "comparison"]
    ablation_run_rows = [row for row in rows if row["group"] == "ablation"]

    base_fields = [
        "group",
        "name",
        "method",
        "method_key",
        "method_type",
        "branch",
        "role",
        *FACTOR_KEYS,
        "benchmark",
        "paper_table",
        "dataset",
        "goal_mode",
        "fixed_goal_mode",
        "repeats_per_dist",
        "num_images",
        "num_tasks",
        "success_ratio",
        "success_ratio_ci_low",
        "success_ratio_ci_high",
        "sg_mean",
        "avg_steps_on_success",
        "avg_deviation_on_success",
    ]
    dist_fields = []
    for dist in distances:
        dist_fields.extend(
            [
                f"d{dist}_trials",
                f"d{dist}_success",
                f"d{dist}_sr",
                f"d{dist}_ci_low",
                f"d{dist}_ci_high",
                f"d{dist}_sg_mean",
            ]
        )
    tail_fields = ["output_path", "checkpoint", "llm_checkpoint"]
    write_csv_dicts(table_dir / "all_run_metrics.csv", rows, base_fields + dist_fields + tail_fields)
    write_csv_dicts(table_dir / "comparison_method_metrics.csv", comparison_rows, base_fields + dist_fields + tail_fields)

    ablation_by_branch: dict[str, list[dict]] = {}
    for row in ablation_run_rows:
        ablation_by_branch.setdefault(row["branch"], []).append(row)
    ablation_rows = []
    for branch, branch_rows in ablation_by_branch.items():
        factors = {key: branch_rows[0].get(key, 0) for key in FACTOR_KEYS}
        by_benchmark = {row["benchmark"]: row for row in branch_rows}
        out = {
            "branch": branch,
            "role": branch_rows[0]["role"],
            **factors,
            "all_benchmark_mean": mean([float(row["success_ratio"]) for row in branch_rows]),
            "transfer_mean": mean(
                [
                    float(by_benchmark[name]["success_ratio"])
                    for name in ABLATION_TRANSFER_BENCHMARKS
                    if name in by_benchmark
                ]
            ),
            "checkpoint": branch_rows[0]["checkpoint"],
        }
        for name in ABLATION_BENCHMARKS:
            out[name] = float(by_benchmark[name]["success_ratio"]) if name in by_benchmark else math.nan
        if "masa_aerial" in by_benchmark:
            for dist in distances:
                out[f"masa_d{dist}"] = by_benchmark["masa_aerial"].get(f"d{dist}_sr", math.nan)
        ablation_rows.append(out)
    ablation_rows.sort(key=lambda row: row["transfer_mean"], reverse=True)
    write_csv_dicts(
        table_dir / "ablation_branch_summary.csv",
        ablation_rows,
        [
            "branch",
            "role",
            *FACTOR_KEYS,
            "transfer_mean",
            "all_benchmark_mean",
            *ABLATION_BENCHMARKS,
            *[f"masa_d{dist}" for dist in distances],
            "checkpoint",
        ],
    )

    method_summary = []
    for method in sorted({row["method"] for row in comparison_rows}):
        method_rows = [row for row in comparison_rows if row["method"] == method]
        method_summary.append(
            {
                "method": method,
                "num_benchmarks": len(method_rows),
                "supported_mean_success_ratio": mean([float(row["success_ratio"]) for row in method_rows]),
                "supported_mean_sg": mean([float(row["sg_mean"]) for row in method_rows]),
                "benchmarks": ",".join(sorted(row["benchmark"] for row in method_rows)),
            }
        )
    method_summary.sort(key=lambda row: row["supported_mean_success_ratio"], reverse=True)
    write_csv_dicts(
        table_dir / "comparison_method_summary.csv",
        method_summary,
        ["method", "num_benchmarks", "supported_mean_success_ratio", "supported_mean_sg", "benchmarks"],
    )

    effects = {
        "transfer_mean": factorial_effects(ablation_rows, "transfer_mean") if ablation_rows else {},
        "all_benchmark_mean": factorial_effects(ablation_rows, "all_benchmark_mean") if ablation_rows else {},
    }
    aggregate = {
        "generated_at": now_iso(),
        "output_root": str(output_root),
        "scope": args.scope,
        "distances": distances,
        "budget": 10,
        "task_bank_seed": TASK_BANK_SEED,
        "max_images": args.max_images,
        "comparison_method_summary": method_summary,
        "ablation_branch_summary": ablation_rows,
        "factorial_effects": effects,
        "benchmarks": benchmarks,
        "tables": {
            "all_run_metrics": str(table_dir / "all_run_metrics.csv"),
            "comparison_method_metrics": str(table_dir / "comparison_method_metrics.csv"),
            "comparison_method_summary": str(table_dir / "comparison_method_summary.csv"),
            "ablation_branch_summary": str(table_dir / "ablation_branch_summary.csv"),
        },
    }
    (output_root / "short_distance_c123_aggregate.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown_summary(report_dir / "short_distance_c123_summary_zh.md", aggregate)
    return aggregate


def fmt(value: float) -> str:
    try:
        if math.isnan(float(value)):
            return "NA"
        return f"{float(value):.4f}"
    except Exception:
        return "NA"


def write_markdown_summary(path: Path, aggregate: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    method_summary = aggregate.get("comparison_method_summary", [])
    ablation_rows = aggregate.get("ablation_branch_summary", [])
    effects = aggregate.get("factorial_effects", {}).get("transfer_mean", {}).get("main_effects", {})
    distances = ",".join(str(item) for item in aggregate.get("distances", []))

    lines = [
        "# C=1,2,3 短距离完整评测汇总",
        "",
        f"- 评测口径：只评测，不重训；`5x5` 网格，`B=10`，距离桶 `C={distances}`，任务种子 `{TASK_BANK_SEED}`。",
        "- 对比方法：Random、GOMAA-Geo、DiT-AGL、GeoExplorer-anchor0624，并额外保留验收常用的 GeoExplorer-pristine。",
        "- 消融实验：anchor0624 的 16 个 G/P/E/V 因子组合，全用现有 seed321 480k checkpoint。",
        "- xBD-disaster 口径：灾后图像中搜索，目标仍使用灾前图像嵌入。",
        "",
        "## 对比方法均值",
        "",
        "| 方法 | 覆盖 benchmark 数 | 支持范围内平均 SR | 支持范围内平均 SG |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in method_summary:
        lines.append(
            f"| {row['method']} | {row['num_benchmarks']} | "
            f"{fmt(row['supported_mean_success_ratio'])} | {fmt(row['supported_mean_sg'])} |"
        )

    lines.extend(
        [
            "",
            "## 消融 Top 分支",
            "",
            "| 排名 | 分支 | G | P | E | V | transfer mean | all mean |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for idx, row in enumerate(ablation_rows[:8], start=1):
        lines.append(
            f"| {idx} | {row['branch']} | {row['G_gate']} | {row['P_pbrs']} | "
            f"{row['E_low_entropy']} | {row['V_val78']} | {fmt(row['transfer_mean'])} | "
            f"{fmt(row['all_benchmark_mean'])} |"
        )

    anchor = next((row for row in ablation_rows if row.get("branch") == "g1_p1_e1_v1"), None)
    control = next((row for row in ablation_rows if row.get("branch") == "g0_p0_e0_v0"), None)
    if anchor and control:
        lines.extend(
            [
                "",
                "## 关键对照",
                "",
                f"- 完整方法 `g1_p1_e1_v1` transfer mean：`{fmt(anchor['transfer_mean'])}`。",
                f"- 同数据无新增机制控制 `g0_p0_e0_v0` transfer mean：`{fmt(control['transfer_mean'])}`。",
                f"- 完整方法 - 控制组：`{fmt(float(anchor['transfer_mean']) - float(control['transfer_mean']))}`。",
            ]
        )

    if effects:
        lines.extend(["", "## 主效应", ""])
        for factor, row in effects.items():
            lines.append(
                f"- `{factor}`：on `{fmt(row['on_mean'])}`，off `{fmt(row['off_mean'])}`，"
                f"差值 `{fmt(row['effect_on_minus_off'])}`。"
            )

    lines.extend(
        [
            "",
            "## 文件",
            "",
            f"- 总表：`{aggregate['tables']['all_run_metrics']}`",
            f"- 对比方法逐 benchmark 表：`{aggregate['tables']['comparison_method_metrics']}`",
            f"- 对比方法均值表：`{aggregate['tables']['comparison_method_summary']}`",
            f"- 消融分支汇总表：`{aggregate['tables']['ablation_branch_summary']}`",
            f"- JSON 汇总：`{aggregate['output_root']}/short_distance_c123_aggregate.json`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_status(path: Path = LATEST_STATUS) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def tail_file(path: Path, lines: int) -> str:
    if not path.exists():
        return ""
    data = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(data[-lines:])


def maybe_background(args) -> int:
    if not args.background:
        return -1
    output_root = Path(args.output_root) if args.output_root else OUTPUT_BASE / f"short_distance_c123_eval_{timestamp()}"
    output_root.mkdir(parents=True, exist_ok=True)
    safe_symlink_latest(output_root)
    log_path = output_root / "orchestrator.stdout.log"
    cmd = [
        "/usr/bin/python3",
        "-u",
        str(Path(__file__).resolve()),
        "--scope",
        args.scope,
        "--distances",
        args.distances,
        "--gpus",
        args.gpus,
        "--output-root",
        str(output_root),
        "--poll-seconds",
        str(args.poll_seconds),
    ]
    if args.max_images > 0:
        cmd.extend(["--max-images", str(args.max_images)])
    with log_path.open("ab", buffering=0) as handle:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=clean_eval_env(0),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    payload = {
        "timestamp": now_iso(),
        "phase": "background_launched",
        "pid": int(proc.pid),
        "output_root": str(output_root),
        "status_path": str(output_root / "status.json"),
        "latest_status_path": str(LATEST_STATUS),
        "log_path": str(log_path),
        "command": " ".join(cmd),
    }
    write_status(output_root, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def orchestrate(args) -> int:
    if not EVALUATOR.exists():
        raise FileNotFoundError(str(EVALUATOR))
    distances = parse_int_list(args.distances)
    output_root = Path(args.output_root) if args.output_root else OUTPUT_BASE / f"short_distance_c123_eval_{timestamp()}"
    output_root.mkdir(parents=True, exist_ok=True)
    safe_symlink_latest(output_root)

    benchmarks = resolved_benchmarks()
    specs = build_specs(args.scope, output_root, benchmarks)
    gpus = parse_int_list(args.gpus)
    gpu_slots = {gpu: args.workers_per_gpu for gpu in gpus}
    active: dict[str, dict] = {}
    summary = None

    config_payload = {
        "generated_at": now_iso(),
        "output_root": str(output_root),
        "scope": args.scope,
        "distances": distances,
        "gpus": gpus,
        "workers_per_gpu": args.workers_per_gpu,
        "max_images": args.max_images,
        "num_specs": len(specs),
        "benchmarks": benchmarks,
    }
    (output_root / "run_config.json").write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    try:
        while True:
            counts = {gpu: 0 for gpu in gpu_slots}
            finished = []
            for key, item in active.items():
                counts[item["gpu"]] += 1
                code = item["process"].poll()
                if code is None:
                    continue
                spec = item["spec"]
                ok = code == 0 and output_ready(spec["output_path"])
                spec.update(
                    {
                        "status": "completed" if ok else "failed",
                        "returncode": int(code),
                        "ended_at": now_iso(),
                    }
                )
                if not ok:
                    spec["log_tail"] = tail_file(output_root / "logs" / f"{spec['name']}.log", 80)
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
                available = [gpu for gpu, cap in gpu_slots.items() if counts[gpu] < cap]
                if not available:
                    continue
                gpu = available[0]
                process = launch_eval(spec, gpu, output_root, distances, args.max_images)
                spec.update({"status": "running", "gpu": gpu, "pid": int(process.pid), "started_at": now_iso()})
                active[spec["name"]] = {"gpu": gpu, "process": process, "spec": spec}
                counts[gpu] += 1

            statuses = [spec["status"] for spec in specs]
            phase = "completed" if specs and all(status == "completed" for status in statuses) and not active else "running"
            if any(str(status).startswith("failed") for status in statuses):
                phase = "failed"
            if phase == "completed" and summary is None:
                summary = build_aggregate(output_root, specs, benchmarks, distances, args)

            payload = {
                "timestamp": now_iso(),
                "phase": phase,
                "output_root": str(output_root),
                "scope": args.scope,
                "distances": distances,
                "max_images": args.max_images,
                "allowed_gpus": gpus,
                "active_eval_processes": len(active),
                "gpu_snapshot": gpu_snapshot(),
                "counts": {
                    "total": len(specs),
                    "completed": sum(1 for spec in specs if spec["status"] == "completed"),
                    "running": sum(1 for spec in specs if spec["status"] == "running"),
                    "pending": sum(1 for spec in specs if spec["status"] == "pending"),
                    "failed": sum(1 for spec in specs if str(spec["status"]).startswith("failed")),
                },
                "reports": {
                    "summary_zh": str(output_root / "reports/short_distance_c123_summary_zh.md") if summary else None,
                    "aggregate_json": str(output_root / "short_distance_c123_aggregate.json") if summary else None,
                    "all_run_metrics": str(output_root / "tables/all_run_metrics.csv") if summary else None,
                    "comparison_summary": str(output_root / "tables/comparison_method_summary.csv") if summary else None,
                    "ablation_summary": str(output_root / "tables/ablation_branch_summary.csv") if summary else None,
                },
                "runs": {
                    spec["name"]: {
                        "group": spec["group"],
                        "method": spec["method_label"],
                        "benchmark": spec["benchmark"],
                        "branch": spec.get("branch", ""),
                        "status": spec["status"],
                        "gpu": spec.get("gpu"),
                        "pid": spec.get("pid"),
                        "returncode": spec.get("returncode"),
                        "output_path": str(spec["output_path"]),
                        "log_tail": spec.get("log_tail"),
                    }
                    for spec in specs
                },
            }
            if summary is not None:
                payload["summary"] = {
                    "comparison_method_summary": summary["comparison_method_summary"],
                    "top_ablation_branches": summary["ablation_branch_summary"][:8],
                    "factorial_effects_transfer": summary["factorial_effects"].get("transfer_mean", {}),
                }
            write_status(output_root, payload)

            if phase == "failed":
                terminate(active)
                return 1
            if phase == "completed":
                return 0
            time.sleep(args.poll_seconds)
    except Exception as exc:
        terminate(active)
        write_status(
            output_root,
            {
                "timestamp": now_iso(),
                "phase": "failed_exception",
                "error": str(exc),
                "output_root": str(output_root),
                "active_eval_processes": len(active),
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Run C=1,2,3 comparison and ablation evaluation.")
    parser.add_argument("--scope", choices=["all", "compare", "ablation"], default="all")
    parser.add_argument("--distances", default="1,2,3")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--max-images", type=int, default=0, help="Smoke/debug only. 0 means full dataset.")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--background", action="store_true", help="Launch the orchestrator in the background.")
    parser.add_argument("--status", action="store_true", help="Print latest status JSON and exit.")
    parser.add_argument("--tail", type=int, default=0, help="Print latest orchestrator log tail and exit.")
    args = parser.parse_args()

    if args.status:
        status = load_status()
        print(json.dumps(status or {"phase": "not_started"}, ensure_ascii=False, indent=2))
        return 0
    if args.tail:
        status = load_status()
        if not status:
            print("not started")
            return 0
        log_path = Path(status.get("log_path") or Path(status["output_root"]) / "orchestrator.stdout.log")
        print(tail_file(log_path, args.tail))
        return 0

    bg_code = maybe_background(args)
    if bg_code >= 0:
        return bg_code
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
