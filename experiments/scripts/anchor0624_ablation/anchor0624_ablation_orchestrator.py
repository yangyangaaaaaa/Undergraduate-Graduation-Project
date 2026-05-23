from __future__ import annotations

import csv
import hashlib
import json
import os
import signal
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path


PIPELINE_SERIES = "algo_ablation_anchor0624_20260515"
PIPELINE_EXPERIMENT = "anchor_val78_component_ablation_seed321_480k_shared"
TRAIN_EXPERIMENT = "masa_plus_mmgag_anchor0624_component_ablation_seed321_480k_gpu01"

REMOTE_ROOT = Path("/root/geoexplorer")
REPO_ROOT = REMOTE_ROOT / "GeoExplorer"
CKPT_ROOT = REMOTE_ROOT / "results" / "checkpoint"
PIPELINE_ROOT = REMOTE_ROOT / "ab_experiments" / PIPELINE_SERIES / PIPELINE_EXPERIMENT
STATUS_DIR = PIPELINE_ROOT / "monitoring"
LOG_DIR = STATUS_DIR / "logs"
STATUS_PATH = STATUS_DIR / "anchor0624_ablation_status_latest.json"
PID_PATH = STATUS_DIR / "anchor0624_ablation_orchestrator.pid"

SHARED_EVAL_ROOT = REMOTE_ROOT / "analysis" / "pipeline_20260515_anchor0624_ablation_shared_compare"
LLM_CHECKPOINT = CKPT_ROOT / "env_modeling_fullrerun_20260407_111046" / "state_action.ckpt"
CANONICAL_TRAIN_GRID = (
    REPO_ROOT
    / "staging"
    / "algo_frontier8_20260426"
    / "masa_plus_mmgag_sourceclean_curveconfirm_dualseed240k"
    / "masa_plus_mmgag_train_grid_5.npy"
)

GPU_CAPACITY = {0: 1, 1: 1}
TARGET_STEPS = 480000
SEED = 321
SOURCE_SNAPSHOT_FILES = (
    "config.py",
    "train.py",
    "models/ppo.py",
    "models/pretrain_model.py",
    "models/model_falcon.py",
    "models/decision_transformer.py",
    "data_utils/__init__.py",
    "data_utils/sequence.py",
    "utils/__init__.py",
    "utils/get_test_config.py",
    "utils/random_seed.py",
)


def make_branch(gate: int, pbrs: int, ent_low: int, val78: int) -> dict:
    branch = f"g{gate}_p{pbrs}_e{ent_low}_v{val78}"
    factors = {
        "G_gate": int(gate),
        "P_pbrs": int(pbrs),
        "E_low_entropy": int(ent_low),
        "V_val78": int(val78),
    }
    env = {
        "GEOEXPLORER_GATE_MODE": "linear" if gate else "none",
        "GEOEXPLORER_GATE_FLOOR": "0.405" if gate else "1.0",
        "GEOEXPLORER_PBRS_COEF": "0.10" if pbrs else "0.0",
        "GEOEXPLORER_ENT_COEF": "0.005" if ent_low else "0.010",
        "GEOEXPLORER_VAL_DISTS": "7,8" if val78 else "4,5,6,7,8",
    }
    if gate and pbrs and ent_low and val78:
        role = "full_anchor0624"
    elif not gate and not pbrs and not ent_low and not val78:
        role = "same_data_no_added_mechanism_control"
    else:
        role = "factorial_ablation_cell"
    return {"branch": branch, "role": role, "factors": factors, "env": env}


BRANCHES = [
    make_branch(gate, pbrs, ent_low, val78)
    for val78 in (0, 1)
    for ent_low in (0, 1)
    for pbrs in (0, 1)
    for gate in (0, 1)
]
BRANCHES.sort(
    key=lambda item: (
        0 if item["branch"] == "g1_p1_e1_v1" else 1 if item["branch"] == "g0_p0_e0_v0" else 2,
        item["branch"],
    )
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def run_name(branch: str) -> str:
    return f"{branch}_seed{SEED}"


def ckpt_dir(branch: str) -> Path:
    return CKPT_ROOT / PIPELINE_SERIES / TRAIN_EXPERIMENT / f"{branch}_seed{SEED}_t{TARGET_STEPS // 1000}k"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_source_snapshot(snapshot_root: Path) -> list[dict]:
    snapshot_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for rel in SOURCE_SNAPSHOT_FILES:
        src = REPO_ROOT / rel
        if not src.exists():
            raise FileNotFoundError(f"missing source file for reproducibility snapshot: {src}")
        dst = snapshot_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        rows.append(
            {
                "path": rel,
                "sha256": sha256_file(src),
                "bytes": src.stat().st_size,
                "source": str(src),
                "snapshot": str(dst),
            }
        )
    return rows


def clean_process_env() -> dict:
    return {key: value for key, value in os.environ.items() if not key.startswith("GEOEXPLORER_")}


def train_env_overrides(branch_spec: dict, gpu: int | None, train_grid: Path) -> dict:
    branch = branch_spec["branch"]
    cuda_visible = str(gpu) if gpu is not None and gpu >= 0 else "unknown_recovered_complete_run"
    overrides = {
        "PYTHONPATH": "/root/geoexplorer/env/geoexplorer_site:/root/geoexplorer/GeoExplorer:/root/geoexplorer",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": cuda_visible,
        "GEOEXPLORER_DEVICE": "cuda:0",
        "GEOEXPLORER_DATASET": "masa",
        "GEOEXPLORER_PATCH_SIZE": "5",
        "GEOEXPLORER_MIN_BUDGET": "10",
        "GEOEXPLORER_MAX_BUDGET": "11",
        "GEOEXPLORER_BUDGET_STEP": "2",
        "GEOEXPLORER_MIN_C": "4",
        "GEOEXPLORER_MAX_C": "9",
        "GEOEXPLORER_NUM_CONFIG": "5",
        "GEOEXPLORER_REWARD": "in",
        "GEOEXPLORER_FACTOR": "1.0",
        "GEOEXPLORER_PROGRESS_METRIC": "l2sq",
        "GEOEXPLORER_TRAIN_DATA": str(train_grid),
        "GEOEXPLORER_VAL_DATA": "/root/geoexplorer/data/masa/sat_val_grid_5.npy",
        "GEOEXPLORER_TEST_DATA": "/root/geoexplorer/data/masa/sat_test_grid_5.npy",
        "GEOEXPLORER_PRETRAIN_CKPT_ROOT": str(CKPT_ROOT),
        "GEOEXPLORER_TRAIN_CKPT_ROOT": str(CKPT_ROOT),
        "GEOEXPLORER_LLM_CHECKPOINT": str(LLM_CHECKPOINT),
        "GEOEXPLORER_TRAIN_EXPT": f"{PIPELINE_SERIES}/{TRAIN_EXPERIMENT}/{branch}_seed{SEED}_t{TARGET_STEPS // 1000}k",
        "GEOEXPLORER_TRAIN_NAME": "geoexplorer.pt",
        "GEOEXPLORER_TRAIN_PREFIX": "geoexplorer_",
        "GEOEXPLORER_MAX_TRAINING_TIMESTEPS": str(TARGET_STEPS),
        "GEOEXPLORER_RANDOM_SEED": str(SEED),
        "GEOEXPLORER_GATE_POWER": "1.0",
        "GEOEXPLORER_GATE_BLEND_ALPHA": "0.0",
        "GEOEXPLORER_FINISH_BONUS_SCALE": "0.0",
        "GEOEXPLORER_FINISH_BONUS_RADIUS": "1",
        "GEOEXPLORER_ORACLE_BC_COEF": "0.0",
        "GEOEXPLORER_SIL_COEF": "0.0",
        "GEOEXPLORER_CURRICULUM_MODE": "none",
        "GEOEXPLORER_COMMIT_BEST_ON_PROGRESS": "0",
        "GEOEXPLORER_VAL_EVERY_EPISODES": "2",
        "GEOEXPLORER_TARGET_KL": "0.02",
        "GEOEXPLORER_EPS_CLIP": "0.20",
        "GEOEXPLORER_LR_ACTOR": "0.0001",
        "GEOEXPLORER_LR_CRITIC": "0.0001",
    }
    overrides.update(branch_spec["env"])
    return overrides


def write_pipeline_repro_bundle(train_grid: Path) -> None:
    repro_root = PIPELINE_ROOT / "reproducibility"
    source_rows = copy_source_snapshot(repro_root / "source_snapshot")
    (repro_root / "source_snapshot_manifest.json").write_text(
        json.dumps(
            {
                "generated_at": now_iso(),
                "pipeline_series": PIPELINE_SERIES,
                "pipeline_experiment": PIPELINE_EXPERIMENT,
                "train_experiment": TRAIN_EXPERIMENT,
                "target_steps": TARGET_STEPS,
                "seed": SEED,
                "train_grid": str(train_grid),
                "source_files": source_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def write_run_repro_bundle(branch_spec: dict, gpu: int | None, train_grid: Path) -> Path:
    out_dir = ckpt_dir(branch_spec["branch"])
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_path = out_dir / "ablation_spec.json"
    if (gpu is None or gpu < 0) and spec_path.exists() and (out_dir / "source_snapshot_manifest.json").exists():
        return spec_path
    recorded_gpu = gpu if gpu is not None and gpu >= 0 else None
    source_rows = copy_source_snapshot(out_dir / "source_snapshot")
    source_manifest_path = out_dir / "source_snapshot_manifest.json"
    source_manifest_path.write_text(
        json.dumps(
            {
                "generated_at": now_iso(),
                "pipeline_series": PIPELINE_SERIES,
                "pipeline_experiment": PIPELINE_EXPERIMENT,
                "train_experiment": TRAIN_EXPERIMENT,
                "branch": branch_spec["branch"],
                "run_name": run_name(branch_spec["branch"]),
                "source_files": source_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    spec_path.write_text(
        json.dumps(
            {
                "generated_at": now_iso(),
                "pipeline_series": PIPELINE_SERIES,
                "pipeline_experiment": PIPELINE_EXPERIMENT,
                "train_experiment": TRAIN_EXPERIMENT,
                "branch": branch_spec["branch"],
                "run_name": run_name(branch_spec["branch"]),
                "role": branch_spec["role"],
                "factors": branch_spec["factors"],
                "factor_env": branch_spec["env"],
                "effective_train_env_overrides": train_env_overrides(branch_spec, recorded_gpu, train_grid),
                "seed": SEED,
                "target_steps": TARGET_STEPS,
                "gpu": recorded_gpu,
                "checkpoint_dir": str(out_dir),
                "expected_best_checkpoint": str(out_dir / "geoexplorer.pt"),
                "expected_latest_checkpoint": str(out_dir / "geoexplorer_latest.pt"),
                "periodic_checkpoint_prefix": str(out_dir / "geoexplorer_"),
                "train_grid": str(train_grid),
                "source_snapshot": str(out_dir / "source_snapshot"),
                "source_snapshot_manifest": str(source_manifest_path),
                "reward_formula": "reward_total = reward_ex + factor * reward_in * gate_weight(dist) + pbrs_bonus(prev_dist, dist) + finish_bonus(dist)",
                "fixed_off_mechanisms": {
                    "finish_bonus": True,
                    "oracle_bc": True,
                    "self_imitation": True,
                    "curriculum": True,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return spec_path


def heartbeat(path: Path) -> dict | None:
    hb = path / "heartbeat.json"
    if not hb.exists():
        return None
    try:
        return json.loads(hb.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": str(exc)}


def checkpoint_complete(path: Path) -> bool:
    hb = heartbeat(path) or {}
    step = int(hb.get("time_step") or 0)
    return (
        step >= int(TARGET_STEPS * 0.98)
        and (path / "geoexplorer.pt").exists()
        and (path / "geoexplorer_latest.pt").exists()
    )


def gpu_snapshot() -> list[dict]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:
        return [{"error": str(exc)}]
    rows = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 4:
            rows.append({"gpu": int(parts[0]), "name": parts[1], "used_mb": int(parts[2]), "util": int(parts[3])})
    return rows


def write_status(phase: str, detail: dict | None = None) -> None:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": now_iso(),
        "phase": phase,
        "pipeline_series": PIPELINE_SERIES,
        "pipeline_experiment": PIPELINE_EXPERIMENT,
        "gpu_snapshot": gpu_snapshot(),
    }
    if detail:
        payload.update(detail)
    STATUS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def ensure_train_grid() -> Path:
    staging = REPO_ROOT / "staging" / PIPELINE_SERIES / TRAIN_EXPERIMENT
    train_grid = staging / "masa_plus_mmgag_train_grid_5.npy"
    staging.mkdir(parents=True, exist_ok=True)
    if not CANONICAL_TRAIN_GRID.exists():
        raise FileNotFoundError(f"missing canonical train grid: {CANONICAL_TRAIN_GRID}")
    if not train_grid.exists() or train_grid.stat().st_size != CANONICAL_TRAIN_GRID.stat().st_size:
        shutil.copy2(CANONICAL_TRAIN_GRID, train_grid)
    return train_grid


def base_train_env(branch_spec: dict, gpu: int, train_grid: Path) -> dict:
    env = clean_process_env()
    env.update(train_env_overrides(branch_spec, gpu, train_grid))
    return env


def launch_train(branch_spec: dict, gpu: int, train_grid: Path) -> subprocess.Popen:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    branch = branch_spec["branch"]
    out_dir = ckpt_dir(branch)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{run_name(branch)}.train.log"
    log = log_path.open("ab", buffering=0)
    log.write(f"\n[{now_iso()}] launching train {run_name(branch)} on GPU{gpu}\n".encode("utf-8"))
    return subprocess.Popen(
        ["/usr/bin/python3", "-u", "train.py"],
        cwd=str(REPO_ROOT),
        env=base_train_env(branch_spec, gpu, train_grid),
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def train_all() -> list[dict]:
    train_grid = ensure_train_grid()
    write_pipeline_repro_bundle(train_grid)
    states = {
        run_name(item["branch"]): {
            "branch": item["branch"],
            "seed": SEED,
            "role": item["role"],
            "factors": item["factors"],
            "status": "pending",
            "checkpoint_dir": str(ckpt_dir(item["branch"])),
            "env": item["env"],
        }
        for item in BRANCHES
    }
    active: dict[str, dict] = {}
    while True:
        finished = []
        for name, item in active.items():
            code = item["process"].poll()
            if code is None:
                continue
            branch = item["branch_spec"]["branch"]
            complete = checkpoint_complete(ckpt_dir(branch))
            states[name]["status"] = "completed" if code == 0 and complete else "failed"
            states[name]["returncode"] = int(code)
            states[name]["checkpoint_complete"] = bool(complete)
            states[name]["ended_at"] = now_iso()
            finished.append(name)
        for name in finished:
            active.pop(name, None)

        counts = {gpu: 0 for gpu in GPU_CAPACITY}
        for item in active.values():
            counts[int(item["gpu"])] += 1

        for branch_spec in BRANCHES:
            name = run_name(branch_spec["branch"])
            state = states[name]
            path = ckpt_dir(branch_spec["branch"])
            state["heartbeat"] = heartbeat(path)
            if state["status"] == "pending" and checkpoint_complete(path):
                state["repro_spec"] = str(write_run_repro_bundle(branch_spec, -1, train_grid))
                state["source_snapshot"] = str(ckpt_dir(branch_spec["branch"]) / "source_snapshot")
                state["status"] = "completed"
                continue
            if state["status"] != "pending":
                continue
            available = [gpu for gpu, cap in GPU_CAPACITY.items() if counts[gpu] < cap]
            if not available:
                continue
            gpu = available[0]
            repro_spec = write_run_repro_bundle(branch_spec, gpu, train_grid)
            proc = launch_train(branch_spec, gpu, train_grid)
            active[name] = {"process": proc, "gpu": gpu, "branch_spec": branch_spec}
            counts[gpu] += 1
            state.update(
                {
                    "status": "running",
                    "pid": int(proc.pid),
                    "gpu": gpu,
                    "started_at": now_iso(),
                    "repro_spec": str(repro_spec),
                    "source_snapshot": str(ckpt_dir(branch_spec["branch"]) / "source_snapshot"),
                }
            )

        write_status(
            "training_anchor0624_ablation_480k",
            {
                "active_train_processes": len(active),
                "training": {
                    "series": PIPELINE_SERIES,
                    "experiment": TRAIN_EXPERIMENT,
                    "target_steps": TARGET_STEPS,
                    "runs": states,
                },
            },
        )
        if any(state["status"] == "failed" for state in states.values()):
            terminate_active_processes(active)
            raise RuntimeError("anchor0624 ablation training failed")
        if all(state["status"] == "completed" for state in states.values()):
            return [
                {
                    "branch": item["branch"],
                    "seed": SEED,
                    "role": item["role"],
                    "factors": item["factors"],
                    "checkpoint": str(ckpt_dir(item["branch"]) / "geoexplorer.pt"),
                    "target_steps": TARGET_STEPS,
                    "env": item["env"],
                    "repro_spec": str(ckpt_dir(item["branch"]) / "ablation_spec.json"),
                    "source_snapshot": str(ckpt_dir(item["branch"]) / "source_snapshot"),
                }
                for item in BRANCHES
            ]
        time.sleep(60)


def parse_metric(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "overall" in payload and isinstance(payload["overall"], dict) and "success_ratio" in payload["overall"]:
        return float(payload["overall"]["success_ratio"])
    raise KeyError(f"unrecognized metric payload: {path}")


def output_ready(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        parse_metric(path)
    except Exception:
        return False
    return True


def factorial_effects(rows: list[dict]) -> dict:
    effects = {}
    factors = ["G_gate", "P_pbrs", "E_low_entropy", "V_val78"]
    for factor in factors:
        on = [row["overall"]["success_ratio"] for row in rows if int(row.get("factors", {}).get(factor, 0)) == 1]
        off = [row["overall"]["success_ratio"] for row in rows if int(row.get("factors", {}).get(factor, 0)) == 0]
        effects[factor] = {
            "mean_on": float(sum(on) / max(len(on), 1)),
            "mean_off": float(sum(off) / max(len(off), 1)),
            "effect_on_minus_off": float(sum(on) / max(len(on), 1) - sum(off) / max(len(off), 1)),
            "n_on": len(on),
            "n_off": len(off),
        }
    interactions = {}
    for left, right in (("G_gate", "P_pbrs"), ("G_gate", "V_val78"), ("P_pbrs", "V_val78"), ("E_low_entropy", "V_val78")):
        def mean_for(a: int, b: int) -> float:
            vals = [
                row["overall"]["success_ratio"]
                for row in rows
                if int(row.get("factors", {}).get(left, 0)) == a and int(row.get("factors", {}).get(right, 0)) == b
            ]
            return float(sum(vals) / max(len(vals), 1))

        m11 = mean_for(1, 1)
        m10 = mean_for(1, 0)
        m01 = mean_for(0, 1)
        m00 = mean_for(0, 0)
        interactions[f"{left}__x__{right}"] = {
            "mean_11": m11,
            "mean_10": m10,
            "mean_01": m01,
            "mean_00": m00,
            "difference_of_differences": float((m11 - m10) - (m01 - m00)),
        }
    return {"main_effects": effects, "two_way_interactions": interactions}


def write_shared_eval_script() -> Path:
    script_path = STATUS_DIR / "geo_shared_eval_one.py"
    script_path.write_text(
        r'''
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path("/root/geoexplorer/GeoExplorer")
sys.path.insert(0, str(ROOT))

from config import cfg
from models.ppo import PPO
from utils import seed_everything


BUNDLE_PATH = Path("/root/src/compare_baselines_bundle_20260505_v2/compare_baselines_bundle/shared_compare_round1/shared_task_bundle.json")
TEST_PATH_CANDIDATES = [
    Path("/root/geoexplorer/GeoExplorer/data/masa/sat_test_grid_5.npy"),
    Path("/root/geoexplorer/data/masa/sat_test_grid_5.npy"),
]
LLM_CHECKPOINT = "/root/geoexplorer/results/checkpoint/env_modeling_fullrerun_20260407_111046/state_action.ckpt"


def resolve_test_path() -> str:
    for path in TEST_PATH_CANDIDATES:
        if path.exists():
            return str(path)
    raise FileNotFoundError("missing MASA test path")


def to_config_dict(raw):
    return {key: [tuple(int(v) for v in pair) for pair in value] for key, value in raw.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--factors-json", required=True)
    parser.add_argument("--train-budget", type=int, required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    cfg.dataset = "masa"
    cfg.reward = "in"
    cfg.factor = 1.0
    cfg.data.patch_size = 5
    cfg.data.test_path = resolve_test_path()
    cfg.min_c = 4
    cfg.max_c = 9
    cfg.num_config_per_img = 5
    cfg.train.llm_checkpoint = LLM_CHECKPOINT
    cfg.train.checkpoint_path = args.checkpoint
    cfg.train.device = "cuda:0"
    cfg.train.hparams.random_seed = 42

    device = torch.device(cfg.train.device)
    seed_everything(cfg.train.hparams.random_seed)
    agent = PPO(
        cfg.train.hparams.lr_actor,
        cfg.train.hparams.lr_critic,
        cfg.train.hparams.lr_llm,
        cfg.train.hparams.gamma,
        cfg.train.hparams.K_epochs,
        cfg.train.hparams.eps_clip,
        cfg.train.hparams.lr_gamma,
    ).to(device)
    agent.load_state_dict(torch.load(cfg.train.checkpoint_path, map_location=device))
    agent.eval()

    rows = []
    for distance in bundle["protocol"]["distance_buckets"]:
        config = to_config_dict(bundle["by_distance"][str(distance)])
        cases = sum(len(value) for value in config.values())
        success, records = agent.validate(config, cfg.data.test_path, n_config_per_img=cfg.num_config_per_img)
        rows.append(
            {
                "distance": int(distance),
                "num_cases": int(cases),
                "success_count": int(success),
                "success_ratio": float(success / max(cases, 1)),
                "records_sample": records[:3],
            }
        )

    total = sum(row["num_cases"] for row in rows)
    successes = sum(row["success_count"] for row in rows)
    protocol = dict(bundle["protocol"])
    protocol.update(
        {
            "train_budget": {"rl_timesteps": int(args.train_budget)},
            "shared_tasks": True,
            "source_checkpoint_role": "anchor0624_component_ablation",
        }
    )
    payload = {
        "method": args.method,
        "variant": args.variant,
        "branch": args.branch,
        "role": args.role,
        "factors": json.loads(args.factors_json),
        "dataset": "masa",
        "benchmark": "masa_aerial",
        "protocol": protocol,
        "checkpoint_path": args.checkpoint,
        "llm_checkpoint": LLM_CHECKPOINT,
        "overall": {"num_cases": int(total), "success_ratio": float(successes / max(total, 1))},
        "per_distance": rows,
    }
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "overall": payload["overall"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
'''.lstrip(),
        encoding="utf-8",
    )
    return script_path


def launch_shared_eval(script: Path, spec: dict, gpu: int, out_path: Path, ordinal: int) -> subprocess.Popen:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"shared_{ordinal}_{spec['branch']}.log"
    log = log_path.open("ab", buffering=0)
    log.write(f"\n[{now_iso()}] launching shared eval {spec['branch']} on GPU{gpu}\n".encode("utf-8"))
    env = clean_process_env()
    env.update(
        {
            "PYTHONPATH": "/root/geoexplorer/env/geoexplorer_site:/root/geoexplorer:/root/geoexplorer/GeoExplorer",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": str(gpu),
        }
    )
    return subprocess.Popen(
        [
            "/usr/bin/python3",
            str(script),
            "--checkpoint",
            spec["checkpoint"],
            "--method",
            "GeoExplorer-anchor0624-ablation",
            "--variant",
            f"{spec['branch']}_seed{spec['seed']}_{spec['target_steps']}",
            "--branch",
            spec["branch"],
            "--role",
            spec["role"],
            "--factors-json",
            json.dumps(spec["factors"], ensure_ascii=False, sort_keys=True),
            "--train-budget",
            str(spec["target_steps"]),
            "--output-path",
            str(out_path),
        ],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def shared_eval(run_specs: list[dict]) -> dict:
    SHARED_EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    script = write_shared_eval_script()
    eval_snapshot = SHARED_EVAL_ROOT / "source_snapshot" / "geo_shared_eval_one.py"
    eval_snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(script, eval_snapshot)
    specs = []
    for ordinal, run_spec in enumerate(run_specs, start=1):
        specs.append(
            {
                "ordinal": ordinal,
                "run": run_spec,
                "output": SHARED_EVAL_ROOT / f"{run_spec['branch']}_seed{run_spec['seed']}_strict.json",
                "status": "pending",
            }
        )

    active: dict[int, dict] = {}
    while True:
        finished = []
        for ordinal, item in active.items():
            code = item["process"].poll()
            if code is None:
                continue
            spec = item["spec"]
            spec["status"] = "completed" if code == 0 and output_ready(spec["output"]) else "failed"
            spec["returncode"] = int(code)
            finished.append(ordinal)
        for ordinal in finished:
            active.pop(ordinal, None)

        counts = {gpu: 0 for gpu in GPU_CAPACITY}
        for item in active.values():
            counts[int(item["gpu"])] += 1
        for spec in specs:
            if spec["status"] != "pending":
                continue
            if output_ready(spec["output"]):
                spec["status"] = "completed"
                continue
            available = [gpu for gpu, cap in GPU_CAPACITY.items() if counts[gpu] < cap]
            if not available:
                continue
            gpu = available[0]
            proc = launch_shared_eval(script, spec["run"], gpu, spec["output"], spec["ordinal"])
            active[spec["ordinal"]] = {"process": proc, "gpu": gpu, "spec": spec}
            counts[gpu] += 1
            spec.update({"status": "running", "pid": int(proc.pid), "gpu": gpu})

        write_status(
            "shared_eval_anchor0624_ablation",
            {
                "shared_eval_output_root": str(SHARED_EVAL_ROOT),
                "active_shared_eval_processes": len(active),
                "shared_eval_runs": {
                    spec["run"]["branch"]: {
                        "status": spec["status"],
                        "role": spec["run"]["role"],
                        "factors": spec["run"]["factors"],
                        "checkpoint": spec["run"]["checkpoint"],
                        "output": str(spec["output"]),
                    }
                    for spec in specs
                },
            },
        )
        if any(spec["status"] == "failed" for spec in specs):
            terminate_active_processes(active)
            raise RuntimeError("anchor0624 ablation shared eval failed")
        if all(spec["status"] == "completed" for spec in specs):
            return build_shared_table(specs)
        time.sleep(30)


def build_shared_table(specs: list[dict]) -> dict:
    rows = [json.loads(spec["output"].read_text(encoding="utf-8")) for spec in specs]
    rows.sort(key=lambda item: item["overall"]["success_ratio"], reverse=True)
    aggregate = {"generated_at": now_iso(), "factorial_effects": factorial_effects(rows), "rows": rows}
    (SHARED_EVAL_ROOT / "anchor0624_ablation_shared_aggregate.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (SHARED_EVAL_ROOT / "anchor0624_ablation_shared_table.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["branch", "role", "G_gate", "P_pbrs", "E_low_entropy", "V_val78", "overall_sr", "d4", "d5", "d6", "d7", "d8", "train_budget", "variant", "checkpoint"])
        for payload in rows:
            by_dist = {item["distance"]: item["success_ratio"] for item in payload["per_distance"]}
            factors = payload.get("factors", {})
            writer.writerow(
                [
                    payload["branch"],
                    payload["role"],
                    factors.get("G_gate", 0),
                    factors.get("P_pbrs", 0),
                    factors.get("E_low_entropy", 0),
                    factors.get("V_val78", 0),
                    f"{payload['overall']['success_ratio']:.4f}",
                    f"{by_dist.get(4, 0.0):.4f}",
                    f"{by_dist.get(5, 0.0):.4f}",
                    f"{by_dist.get(6, 0.0):.4f}",
                    f"{by_dist.get(7, 0.0):.4f}",
                    f"{by_dist.get(8, 0.0):.4f}",
                    json.dumps(payload.get("protocol", {}).get("train_budget", {}), ensure_ascii=False),
                    payload.get("variant", ""),
                    payload.get("checkpoint_path", ""),
                ]
            )
    lines = [
        "# Anchor0624 Component Ablation Shared MASA Compare",
        "",
        "- protocol: same strict shared MASA tasks as the existing comparison table, `5x5`, `B=10`, `C={4,5,6,7,8}`, greedy.",
        "- control: all rows use seed 321, MASA+MM-GAG train grid, and 480k PPO budget; no top-k selection before shared evaluation.",
        "- factors: `G` distance gate, `P` PBRS, `E` low entropy coefficient 0.005, `V` val78 checkpoint selection.",
        "",
        "| Branch | G | P | E | V | Role | Overall SR | d4 | d5 | d6 | d7 | d8 | Variant |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for payload in rows:
        by_dist = {item["distance"]: item["success_ratio"] for item in payload["per_distance"]}
        factors = payload.get("factors", {})
        lines.append(
            "| "
            + payload["branch"]
            + " | "
            + str(factors.get("G_gate", 0))
            + " | "
            + str(factors.get("P_pbrs", 0))
            + " | "
            + str(factors.get("E_low_entropy", 0))
            + " | "
            + str(factors.get("V_val78", 0))
            + " | "
            + payload["role"]
            + " | "
            + f"{payload['overall']['success_ratio']:.4f}"
            + " | "
            + " | ".join(f"{by_dist.get(d, 0.0):.4f}" for d in range(4, 9))
            + " | "
            + payload.get("variant", "")
            + " |"
        )
    (SHARED_EVAL_ROOT / "anchor0624_ablation_shared_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return aggregate


def terminate_active_processes(active: dict) -> None:
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
    PID_PATH.write_text(str(os.getpid()) + "\n", encoding="utf-8")
    try:
        run_specs = train_all()
        shared_summary = shared_eval(run_specs)
        write_status(
            "completed",
            {
                "training_experiment": f"{PIPELINE_SERIES}/{TRAIN_EXPERIMENT}",
                "shared_eval_summary": str(SHARED_EVAL_ROOT / "anchor0624_ablation_shared_summary.md"),
                "shared_eval_table": str(SHARED_EVAL_ROOT / "anchor0624_ablation_shared_table.csv"),
                "shared_eval_aggregate": str(SHARED_EVAL_ROOT / "anchor0624_ablation_shared_aggregate.json"),
                "shared_eval": shared_summary,
            },
        )
        return 0
    except Exception as exc:
        write_status("failed", {"error": str(exc)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
