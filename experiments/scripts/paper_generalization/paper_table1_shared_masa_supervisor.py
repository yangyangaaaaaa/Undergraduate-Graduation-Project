from __future__ import annotations

import argparse
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
REMOTE_ROOT = Path("/root/geoexplorer")
REPO_ROOT = REMOTE_ROOT / "GeoExplorer"
EXP_ROOT = REMOTE_ROOT / "ab_experiments" / SERIES / EXPERIMENT
STATUS_DIR = EXP_ROOT / "monitoring"
LOG_DIR = STATUS_DIR / "paper_table1_shared_masa_logs"
STATUS_PATH = STATUS_DIR / "paper_table1_shared_masa_status_latest.json"
PID_PATH = STATUS_DIR / "paper_table1_shared_masa_supervisor.pid"
OUTPUT_ROOT = REMOTE_ROOT / "analysis" / "pipeline_20260516_paper_table1_shared_masa"
ANCHOR_EVAL_SCRIPT = STATUS_DIR / "paper_table1_anchor_shared_eval_one.py"

BUNDLE_CANDIDATES = [
    STATUS_DIR / "shared_task_bundle.json",
    Path("/root/src/compare_baselines_bundle_20260505_v2/compare_baselines_bundle/shared_compare_round1/shared_task_bundle.json"),
    Path("F:/bishe/GeoExplorer/analysis/pipeline_20260507/shared_masa_compare_round1/shared_task_bundle.json"),
]
ANCHOR_CKPT = (
    REMOTE_ROOT
    / "results/checkpoint/algo_ablation_anchor0624_20260515/"
    / "masa_plus_mmgag_anchor0624_component_ablation_seed321_480k_gpu01/"
    / "g1_p1_e1_v1_seed321_t480k/geoexplorer.pt"
)

GPU = 0


REFERENCE_ROWS = [
    {
        "method": "GOMAA-Geo",
        "variant": "formal_ppo_seed42_t480k",
        "success_ratio": 0.5080,
        "d4": 0.3400,
        "d5": 0.3200,
        "d6": 0.5400,
        "d7": 0.5800,
        "d8": 0.7600,
        "num_cases": 250,
        "source": "F:/bishe/GeoExplorer/analysis/pipeline_20260507/shared_masa_compare_round1/gomaa_geo_strict.json",
        "notes": "Formal GOMAA-Geo checkpoint evaluated on the same shared-MASA task bank.",
    },
    {
        "method": "AiRLoc",
        "variant": "formal_rl_200_epochs_seed42",
        "success_ratio": 0.4880,
        "d4": 0.5200,
        "d5": 0.4200,
        "d6": 0.5000,
        "d7": 0.6600,
        "d8": 0.3400,
        "num_cases": 250,
        "source": "F:/bishe/GeoExplorer/analysis/pipeline_20260507/shared_masa_compare_round1/airloc_strict.json",
        "notes": "AiRLoc is included only for Table 1 Masa/aerial; it is not a ground/text baseline.",
    },
    {
        "method": "DiT-AGL",
        "variant": "formal_pretrain_seed42_e50",
        "success_ratio": 0.0040,
        "d4": 0.0000,
        "d5": 0.0200,
        "d6": 0.0000,
        "d7": 0.0000,
        "d8": 0.0000,
        "num_cases": 250,
        "source": "F:/bishe/GeoExplorer/analysis/pipeline_20260507/shared_masa_compare_round1/dit_agl_strict.json",
        "notes": "Action-only sequence baseline evaluated on the same shared-MASA task bank.",
    },
    {
        "method": "GeoExplorer-main-current",
        "variant": "gate_a041_masa_plus_mmgag_480k_seed123",
        "success_ratio": 0.5800,
        "d4": 0.2600,
        "d5": 0.2600,
        "d6": 0.6400,
        "d7": 0.8000,
        "d8": 0.9400,
        "num_cases": 250,
        "source": "F:/bishe/GeoExplorer/analysis/pipeline_20260507/shared_masa_compare_round1/geoexplorer_main_current_strict.json",
        "notes": "Prior local main-current GeoExplorer row on the same shared-MASA task bank.",
    },
]


PAPER_REFERENCE_ROWS = [
    {"method": "Random policy", "d4": 0.1412, "d5": 0.0584, "d6": 0.0640, "d7": 0.0247, "d8": 0.0236},
    {"method": "PPO policy", "d4": 0.1427, "d5": 0.1775, "d6": 0.1921, "d7": 0.2269, "d8": 0.2595},
    {"method": "AiRLoc", "d4": 0.1786, "d5": 0.1561, "d6": 0.2134, "d7": 0.2415, "d8": 0.2393},
    {"method": "DiT", "d4": 0.2011, "d5": 0.2956, "d6": 0.3567, "d7": 0.4216, "d8": 0.4559},
    {"method": "GOMAA-Geo", "d4": 0.4090, "d5": 0.5056, "d6": 0.7168, "d7": 0.8034, "d8": 0.7854},
    {"method": "GeoExplorer", "d4": 0.4324, "d5": 0.5318, "d6": 0.8156, "d7": 0.9229, "d8": 0.9497},
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def resolve_bundle_path() -> Path:
    for path in BUNDLE_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("missing shared_task_bundle.json")


def load_bundle() -> dict:
    return json.loads(resolve_bundle_path().read_text(encoding="utf-8"))


def valid_actions(current_patch: int, patch_size: int = 5) -> list[int]:
    actions = []
    row, col = divmod(int(current_patch), patch_size)
    if row > 0:
        actions.append(0)
    if col < patch_size - 1:
        actions.append(1)
    if row < patch_size - 1:
        actions.append(2)
    if col > 0:
        actions.append(3)
    return actions


def step_patch(current_patch: int, action: int, patch_size: int = 5) -> int:
    row, col = divmod(int(current_patch), patch_size)
    if action == 0 and row > 0:
        return current_patch - patch_size
    if action == 1 and col < patch_size - 1:
        return current_patch + 1
    if action == 2 and row < patch_size - 1:
        return current_patch + patch_size
    if action == 3 and col > 0:
        return current_patch - 1
    return current_patch


def simulate_random_shared(bundle: dict, seed: int = 42, budget: int = 10, patch_size: int = 5) -> dict:
    import numpy as np

    rng = np.random.default_rng(seed)
    per_distance = []
    total_cases = 0
    total_success = 0
    for distance in bundle["protocol"]["distance_buckets"]:
        rows = bundle["by_distance"][str(distance)]
        cases = 0
        successes = 0
        for items in rows.values():
            for goal_patch, current_patch in items:
                cases += 1
                patch = int(current_patch)
                for _ in range(budget):
                    actions = valid_actions(patch, patch_size)
                    action = int(actions[int(rng.integers(len(actions)))])
                    patch = step_patch(patch, action, patch_size)
                    if patch == int(goal_patch):
                        successes += 1
                        break
        total_cases += cases
        total_success += successes
        per_distance.append(
            {
                "distance": int(distance),
                "num_cases": int(cases),
                "success_count": int(successes),
                "success_ratio": float(successes / max(cases, 1)),
            }
        )
    return {
        "method": "Random policy",
        "variant": f"shared_sim_seed{seed}",
        "dataset": "masa",
        "benchmark": "masa_aerial",
        "protocol": dict(bundle["protocol"]) | {"shared_tasks": True, "random_seed": int(seed)},
        "overall": {"num_cases": int(total_cases), "success_ratio": float(total_success / max(total_cases, 1))},
        "per_distance": per_distance,
    }


def write_anchor_eval_script() -> None:
    ANCHOR_EVAL_SCRIPT.write_text(
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


STATUS_DIR = Path(__file__).resolve().parent
BUNDLE_CANDIDATES = [
    STATUS_DIR / "shared_task_bundle.json",
    Path("/root/src/compare_baselines_bundle_20260505_v2/compare_baselines_bundle/shared_compare_round1/shared_task_bundle.json"),
]
TEST_PATH_CANDIDATES = [
    Path("/root/geoexplorer/GeoExplorer/data/masa/sat_test_grid_5.npy"),
    Path("/root/geoexplorer/data/masa/sat_test_grid_5.npy"),
]
LLM_CHECKPOINT = "/root/geoexplorer/results/checkpoint/env_modeling_fullrerun_20260407_111046/state_action.ckpt"


def resolve_existing(paths):
    for path in paths:
        if Path(path).exists():
            return Path(path)
    raise FileNotFoundError("missing required path")


def to_config_dict(raw):
    return {key: [tuple(int(v) for v in pair) for pair in value] for key, value in raw.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--method", default="GeoExplorer-anchor0624")
    parser.add_argument("--variant", default="g1_p1_e1_v1_seed321_t480k")
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    bundle = json.loads(resolve_existing(BUNDLE_CANDIDATES).read_text(encoding="utf-8"))
    cfg.dataset = "masa"
    cfg.reward = "in"
    cfg.factor = 1.0
    cfg.data.patch_size = 5
    cfg.data.test_path = str(resolve_existing(TEST_PATH_CANDIDATES))
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
        ent_coef=getattr(cfg.train.hparams, "ent_coef", 0.01),
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
    payload = {
        "method": args.method,
        "variant": args.variant,
        "dataset": "masa",
        "benchmark": "masa_aerial",
        "protocol": dict(bundle["protocol"]) | {
            "train_budget": {"rl_timesteps": 480000},
            "shared_tasks": True,
            "source_checkpoint_role": "anchor0624_g1_p1_e1_v1",
        },
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


def clean_env() -> dict:
    return {key: value for key, value in os.environ.items() if not key.startswith("GEOEXPLORER_")}


def output_ready(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return "overall" in payload and "per_distance" in payload


def launch_anchor_eval(output_path: Path) -> subprocess.Popen:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "anchor0624_shared_masa.log"
    log_handle = log_path.open("ab", buffering=0)
    log_handle.write(f"\n[{now_iso()}] launching anchor0624 shared-MASA Table 1 eval on GPU{GPU}\n".encode("utf-8"))
    env = clean_env()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(GPU),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONPATH": "/root/geoexplorer/env/geoexplorer_site:/root/geoexplorer:/root/geoexplorer/GeoExplorer",
        }
    )
    return subprocess.Popen(
        [
            "/usr/bin/python3",
            str(ANCHOR_EVAL_SCRIPT),
            "--checkpoint",
            str(ANCHOR_CKPT),
            "--output-path",
            str(output_path),
        ],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def row_from_payload(payload: dict, source: str) -> dict:
    per = {int(row["distance"]): float(row["success_ratio"]) for row in payload["per_distance"]}
    return {
        "method": payload["method"],
        "variant": payload.get("variant", ""),
        "success_ratio": float(payload["overall"]["success_ratio"]),
        "d4": per.get(4, math.nan),
        "d5": per.get(5, math.nan),
        "d6": per.get(6, math.nan),
        "d7": per.get(7, math.nan),
        "d8": per.get(8, math.nan),
        "num_cases": int(payload["overall"]["num_cases"]),
        "source": source,
        "notes": payload.get("protocol", {}).get("source_checkpoint_role", ""),
    }


def write_outputs(anchor_output: Path, random_payload: dict, bundle: dict) -> dict:
    rows = []
    rows.append(row_from_payload(random_payload, "computed_from_shared_task_bundle"))
    rows.extend(REFERENCE_ROWS)
    rows.append(row_from_payload(json.loads(anchor_output.read_text(encoding="utf-8")), str(anchor_output)))
    rows.sort(key=lambda row: row["success_ratio"], reverse=True)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    aggregate = {
        "generated_at": now_iso(),
        "paper_item": "Table 1",
        "protocol": bundle["protocol"],
        "rows": rows,
        "paper_reference_rows": PAPER_REFERENCE_ROWS,
        "blocked_or_not_applicable": {
            "PPO policy": "Not re-run locally because no validated standalone PPO checkpoint/evaluator is available in the current bundle.",
            "Full 895-case Table 1 with AiRLoc": "AiRLoc requires original-image/split layout, so the fair local comparison uses the fixed 250-case shared-MASA task bank from 20260507.",
        },
        "rigor_notes": [
            "This table is a shared-task Table 1 reproduction layer, not the full external generalization benchmark.",
            "All included measured rows use the same 250 Masa aerial tasks, 5x5 grid, B=10, C={4,5,6,7,8}, and greedy/argmax policy.",
            "Paper reference rows are included only as literature anchors and should not be merged with the measured shared-task rows.",
        ],
    }
    (OUTPUT_ROOT / "paper_table1_shared_masa_aggregate.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with (OUTPUT_ROOT / "paper_table1_shared_masa_table.csv").open("w", encoding="utf-8", newline="") as handle:
        import csv

        fieldnames = ["method", "success_ratio", "d4", "d5", "d6", "d7", "d8", "variant", "num_cases", "source"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    lines = [
        "# Paper Table 1 Shared-MASA Reproduction",
        "",
        "- protocol: fixed shared `Masa` aerial task bank, `5x5`, `B=10`, `C={4,5,6,7,8}`, `250` total cases, greedy/argmax.",
        "- role: fair local Table 1-style method comparison; paper reference values are listed only as literature anchors in the aggregate JSON.",
        "",
        "| Method | Overall SR | d4 | d5 | d6 | d7 | d8 | Variant |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['success_ratio']:.4f} | {row['d4']:.4f} | {row['d5']:.4f} | "
            f"{row['d6']:.4f} | {row['d7']:.4f} | {row['d8']:.4f} | {row['variant']} |"
        )
    lines.extend(
        [
            "",
            "## Rigor Notes",
            "",
            "- Do not mix these measured shared-task rows with the original paper's full Table 1 numbers without labeling the protocol difference.",
            "- AiRLoc is included only for Masa/aerial; it is not a valid ground/text baseline.",
            "- PPO remains blocked until a validated standalone checkpoint/evaluator is available.",
        ]
    )
    (OUTPUT_ROOT / "paper_table1_shared_masa_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
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


def terminate(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        proc.terminate()
    time.sleep(5)
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            proc.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a paper Table 1 shared-MASA reproduction for anchor0624.")
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PID_PATH.write_text(str(os.getpid()) + "\n", encoding="utf-8")

    anchor_output = OUTPUT_ROOT / "geoexplorer_anchor0624_shared_masa.json"
    proc: subprocess.Popen | None = None
    try:
        bundle = load_bundle()
        random_payload = simulate_random_shared(bundle, seed=args.random_seed)
        random_path = OUTPUT_ROOT / "random_policy_shared_masa.json"
        random_path.write_text(json.dumps(random_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        write_anchor_eval_script()
        if not output_ready(anchor_output):
            if not ANCHOR_CKPT.exists():
                raise FileNotFoundError(f"missing anchor checkpoint: {ANCHOR_CKPT}")
            proc = launch_anchor_eval(anchor_output)
            phase = "running"
        else:
            phase = "completed"

        summary = None
        while True:
            if proc is not None:
                code = proc.poll()
                if code is None:
                    phase = "running"
                elif code == 0 and output_ready(anchor_output):
                    phase = "completed"
                else:
                    phase = "failed"
            if phase == "completed" and summary is None:
                summary = write_outputs(anchor_output, random_payload, bundle)

            payload = {
                "timestamp": now_iso(),
                "phase": phase,
                "series": SERIES,
                "experiment": EXPERIMENT,
                "paper_item": "Table 1",
                "output_root": str(OUTPUT_ROOT),
                "summary_path": str(OUTPUT_ROOT / "paper_table1_shared_masa_summary.md") if summary else None,
                "table_path": str(OUTPUT_ROOT / "paper_table1_shared_masa_table.csv") if summary else None,
                "aggregate_path": str(OUTPUT_ROOT / "paper_table1_shared_masa_aggregate.json") if summary else None,
                "anchor_output": str(anchor_output),
                "random_output": str(random_path),
                "active_eval_processes": int(phase == "running"),
                "anchor_pid": proc.pid if proc is not None and proc.poll() is None else None,
                "gpu_snapshot": gpu_snapshot(),
            }
            if summary is not None:
                payload["summary"] = summary
            write_status(payload)

            if phase in {"completed", "failed"}:
                return 0 if phase == "completed" else 1
            time.sleep(20)
    except Exception as exc:
        terminate(proc)
        write_status({"timestamp": now_iso(), "phase": "failed", "error": str(exc), "active_eval_processes": 0})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
