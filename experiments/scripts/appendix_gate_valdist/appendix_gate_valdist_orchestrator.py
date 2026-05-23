from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import signal
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path


SERIES = "appendix_compare_20260519"
EXPERIMENT = "anchor0624_gate_valdist_dense_followup_seed321_480k"
TRAIN_EXPERIMENT = "anchor0624_gate_valdist_dense_followup"

REMOTE_ROOT = Path("/root/geoexplorer")
REPO_ROOT = REMOTE_ROOT / "GeoExplorer"
EXP_ROOT = REMOTE_ROOT / "ab_experiments" / SERIES / EXPERIMENT
STATUS_DIR = EXP_ROOT / "monitoring"
LOG_DIR = STATUS_DIR / "logs"
STATUS_PATH = STATUS_DIR / "appendix_gate_valdist_status_latest.json"
PID_PATH = STATUS_DIR / "appendix_gate_valdist_orchestrator.pid"

OUTPUT_ROOT = REMOTE_ROOT / "analysis" / "pipeline_20260519_appendix_gate_valdist_dense_followup"
CKPT_ROOT = REMOTE_ROOT / "results" / "checkpoint"
TRAIN_CKPT_ROOT = CKPT_ROOT / SERIES / TRAIN_EXPERIMENT
STAGING_DIR = REPO_ROOT / "staging" / SERIES / EXPERIMENT
EVALUATOR = STATUS_DIR / "paper_geo_evaluator.py"
NVIDIA_COMPAT_ROOT = REMOTE_ROOT / "env" / "nvidia_535_288"
NVIDIA_COMPAT_LIB = NVIDIA_COMPAT_ROOT / "usr" / "lib" / "x86_64-linux-gnu"
NVIDIA_COMPAT_SMI = NVIDIA_COMPAT_ROOT / "usr" / "bin" / "nvidia-smi"

LLM_CHECKPOINT = CKPT_ROOT / "env_modeling_fullrerun_20260407_111046" / "state_action.ckpt"
CANONICAL_TRAIN_GRID = (
    REPO_ROOT
    / "staging"
    / "algo_frontier8_20260426"
    / "masa_plus_mmgag_sourceclean_curveconfirm_dualseed240k"
    / "masa_plus_mmgag_train_grid_5.npy"
)

TASK_BANK_SEED = 20260519
DEFAULT_SEED = 321
DEFAULT_TARGET_STEPS = 480000
GPU_SLOTS = {0: 1, 1: 1, 2: 1, 3: 1}

BENCHMARK_ORDER = [
    "masa_aerial",
    "mmgag_aerial",
    "mmgag_ground",
    "mmgag_text",
    "swissview100_aerial",
    "swissviewmonuments_aerial",
    "swissviewmonuments_ground",
    "xbd_pre_aerial",
    "xbd_disaster_aerial",
]
PRIMARY_TRANSFER_BENCHMARKS = [
    "swissviewmonuments_aerial",
    "swissviewmonuments_ground",
    "xbd_pre_aerial",
    "xbd_disaster_aerial",
]

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
    "swissview100_aerial": {
        "paper_table": "Table S4",
        "dataset": "swissview",
        "goal_mode": "aerial",
        "fixed_goal_mode": "none",
        "repeats_per_dist": 5,
        "test_path_candidates": [
            "/root/geoexplorer/GeoExplorer/data/swissview/swissview100_sat_patches.npy",
            "/root/geoexplorer/GeoExplorer/data/swissview/processed/swissview100_sat_patches.npy",
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


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def slug_float(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def clean_process_env() -> dict:
    return {key: value for key, value in os.environ.items() if not key.startswith("GEOEXPLORER_")}


def cuda_compat_env() -> dict[str, str]:
    return {"LD_LIBRARY_PATH": str(NVIDIA_COMPAT_LIB)}


def mean(values: list[float]) -> float:
    vals = [float(value) for value in values if not math.isnan(float(value))]
    return float(sum(vals) / max(len(vals), 1)) if vals else math.nan


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
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
    if EVALUATOR.exists():
        dst = snapshot_root / "monitoring" / "paper_geo_evaluator.py"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(EVALUATOR, dst)
        rows.append(
            {
                "path": "monitoring/paper_geo_evaluator.py",
                "sha256": sha256_file(EVALUATOR),
                "bytes": EVALUATOR.stat().st_size,
                "source": str(EVALUATOR),
                "snapshot": str(dst),
            }
        )
    return rows


def base_anchor_overrides() -> dict[str, str]:
    return {
        "GEOEXPLORER_GATE_MODE": "linear",
        "GEOEXPLORER_GATE_FLOOR": "0.405",
        "GEOEXPLORER_GATE_POWER": "1.0",
        "GEOEXPLORER_GATE_BLEND_ALPHA": "0.0",
        "GEOEXPLORER_PBRS_COEF": "0.10",
        "GEOEXPLORER_ENT_COEF": "0.005",
        "GEOEXPLORER_VAL_DISTS": "7,8",
    }


def spec_name(prefix: str, seed: int, target_steps: int) -> str:
    return f"{prefix}_seed{seed}_t{target_steps // 1000}k"


def make_spec(
    name_prefix: str,
    group: str,
    trainset_key: str,
    seed: int = DEFAULT_SEED,
    target_steps: int = DEFAULT_TARGET_STEPS,
    env_overrides: dict[str, str] | None = None,
    factor_values: dict[str, str | float | int] | None = None,
    table_roles: list[dict] | None = None,
    note: str = "",
) -> dict:
    env = base_anchor_overrides()
    if env_overrides:
        env.update({str(key): str(value) for key, value in env_overrides.items()})
    return {
        "name": spec_name(name_prefix, seed, target_steps),
        "name_prefix": name_prefix,
        "group": group,
        "trainset_key": trainset_key,
        "seed": int(seed),
        "target_steps": int(target_steps),
        "env": env,
        "factor_values": factor_values or {},
        "table_roles": table_roles or [],
        "note": note,
    }


def build_run_specs() -> list[dict]:
    specs: list[dict] = []

    gate_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
    for value in gate_values:
        specs.append(
            make_spec(
                f"gate_floor_dense_{slug_float(value)}",
                group="gate_floor_dense",
                trainset_key="masa_plus_mmgag",
                env_overrides={"GEOEXPLORER_GATE_FLOOR": f"{value:.3f}"},
                factor_values={"gate_floor": value},
                table_roles=[{"table": "param", "family": "gate_floor_dense", "value": f"{value:.3f}"}],
                note="门控下限全范围补充；0/1 是门控端点，不是纯外部/纯内部奖励",
            )
        )

    val_dist_rows = [
        ("4", "near_single"),
        ("4,5", "near"),
        ("4,5,6", "near_mid"),
        ("5,6,7", "mid"),
        ("8", "far_single"),
    ]
    for value, label in val_dist_rows:
        suffix = value.replace(",", "")
        specs.append(
            make_spec(
                f"val_dists_{suffix}",
                group="val_dists_bias",
                trainset_key="masa_plus_mmgag",
                env_overrides={"GEOEXPLORER_VAL_DISTS": value},
                factor_values={"val_dists": value, "val_dists_bias": label},
                table_roles=[{"table": "param", "family": "val_dists_bias", "value": value}],
                note="验证距离选择偏置补充",
            )
        )

    reward_controls = [
        (
            "reward_external_only",
            {"GEOEXPLORER_REWARD": "ex", "GEOEXPLORER_PBRS_COEF": "0.00"},
            "external_only",
            "严格纯外部奖励：reward_ex，无 intrinsic，无 PBRS",
        ),
        (
            "reward_intrinsic_only",
            {
                "GEOEXPLORER_REWARD": "intrinsic_only",
                "GEOEXPLORER_GATE_FLOOR": "1.000",
                "GEOEXPLORER_PBRS_COEF": "0.00",
            },
            "intrinsic_only",
            "严格纯内部奖励：reward_in * factor，无 external，无 PBRS；需要本轮新增兼容分支",
        ),
        (
            "reward_intrinsic_no_decay",
            {"GEOEXPLORER_GATE_FLOOR": "1.000", "GEOEXPLORER_PBRS_COEF": "0.00"},
            "intrinsic_plus_external_no_decay_no_pbrs",
            "内部+外部但无距离衰减、无 PBRS；用于和纯内部/纯外部区分",
        ),
    ]
    for prefix, env_overrides, value, note in reward_controls:
        specs.append(
            make_spec(
                prefix,
                group="reward_control",
                trainset_key="masa_plus_mmgag",
                env_overrides=env_overrides,
                factor_values={"reward_control": value},
                table_roles=[{"table": "param", "family": "reward_control", "value": value}],
                note=note,
            )
        )

    gate_type_rows = [
        (
            "reward_external_pbrs",
            {"GEOEXPLORER_REWARD": "ex", "GEOEXPLORER_PBRS_COEF": "0.10"},
            "none",
            "pb",
            "external_pbrs",
            "外部奖励 + PBRS；无 intrinsic，用于隔离 PBRS 本身的贡献",
        ),
        (
            "gate_type_constant_0405_no_pbrs",
            {
                "GEOEXPLORER_GATE_MODE": "constant",
                "GEOEXPLORER_GATE_FLOOR": "0.405",
                "GEOEXPLORER_PBRS_COEF": "0.00",
            },
            "constant_0.405",
            "no_pb",
            "constant_0.405_no_pb",
            "外部奖励 + 固定 0.405 内在奖励；无 PBRS",
        ),
        (
            "gate_type_constant_0405_pbrs",
            {
                "GEOEXPLORER_GATE_MODE": "constant",
                "GEOEXPLORER_GATE_FLOOR": "0.405",
                "GEOEXPLORER_PBRS_COEF": "0.10",
            },
            "constant_0.405",
            "pb",
            "constant_0.405_pb",
            "外部奖励 + 固定 0.405 内在奖励 + PBRS",
        ),
        (
            "gate_type_sine_no_pbrs",
            {"GEOEXPLORER_GATE_MODE": "sine", "GEOEXPLORER_PBRS_COEF": "0.00"},
            "sine",
            "no_pb",
            "sine_no_pb",
            "外部奖励 + sine 距离门控内在奖励；无 PBRS",
        ),
        (
            "gate_type_sine_pbrs",
            {"GEOEXPLORER_GATE_MODE": "sine", "GEOEXPLORER_PBRS_COEF": "0.10"},
            "sine",
            "pb",
            "sine_pb",
            "外部奖励 + sine 距离门控内在奖励 + PBRS",
        ),
        (
            "gate_type_power2_no_pbrs",
            {
                "GEOEXPLORER_GATE_MODE": "power",
                "GEOEXPLORER_GATE_POWER": "2.0",
                "GEOEXPLORER_PBRS_COEF": "0.00",
            },
            "power2",
            "no_pb",
            "power2_no_pb",
            "外部奖励 + 二次幂距离门控内在奖励；无 PBRS",
        ),
        (
            "gate_type_power2_pbrs",
            {
                "GEOEXPLORER_GATE_MODE": "power",
                "GEOEXPLORER_GATE_POWER": "2.0",
                "GEOEXPLORER_PBRS_COEF": "0.10",
            },
            "power2",
            "pb",
            "power2_pb",
            "外部奖励 + 二次幂距离门控内在奖励 + PBRS",
        ),
        (
            "gate_type_blendlp_no_pbrs",
            {
                "GEOEXPLORER_GATE_MODE": "blend_lp",
                "GEOEXPLORER_GATE_POWER": "2.0",
                "GEOEXPLORER_GATE_BLEND_ALPHA": "0.5",
                "GEOEXPLORER_PBRS_COEF": "0.00",
            },
            "blend_lp",
            "no_pb",
            "blend_lp_no_pb",
            "外部奖励 + 线性/幂函数混合门控内在奖励；无 PBRS",
        ),
        (
            "gate_type_blendlp_pbrs",
            {
                "GEOEXPLORER_GATE_MODE": "blend_lp",
                "GEOEXPLORER_GATE_POWER": "2.0",
                "GEOEXPLORER_GATE_BLEND_ALPHA": "0.5",
                "GEOEXPLORER_PBRS_COEF": "0.10",
            },
            "blend_lp",
            "pb",
            "blend_lp_pb",
            "外部奖励 + 线性/幂函数混合门控内在奖励 + PBRS",
        ),
    ]
    for prefix, env_overrides, gate_type, pb_status, value, note in gate_type_rows:
        specs.append(
            make_spec(
                prefix,
                group="reward_gate_type",
                trainset_key="masa_plus_mmgag",
                env_overrides=env_overrides,
                factor_values={"gate_type": gate_type, "pbrs": pb_status},
                table_roles=[{"table": "param", "family": "reward_gate_type", "value": value}],
                note=note,
            )
        )

    priority = os.environ.get("GEOEXPLORER_PRIORITY_ABLATION_ONLY", "").strip().lower()
    if priority in {"reward_control", "reward_controls", "algorithm", "algorithm_ablation"}:
        return [spec for spec in specs if spec["group"] == "reward_control"]
    if priority in {"parameter", "parameters", "param", "parameter_fullrange", "gate_valdist_param", "gate_valdist"}:
        return [spec for spec in specs if spec["group"] in {"gate_floor_dense", "val_dists_bias"}]
    if priority in {"gate_floor_dense", "gate_floor"}:
        return [spec for spec in specs if spec["group"] == "gate_floor_dense"]
    if priority in {"val_dists_bias", "val_dists", "validation_distance"}:
        return [spec for spec in specs if spec["group"] == "val_dists_bias"]
    if priority in {"1", "true", "yes", "reward_gate_type"}:
        return [spec for spec in specs if spec["group"] == "reward_gate_type"]
    return specs


RUN_SPECS = build_run_specs()


def ckpt_dir(spec: dict) -> Path:
    return TRAIN_CKPT_ROOT / spec["name"]


def heartbeat(path: Path) -> dict | None:
    hb = path / "heartbeat.json"
    if not hb.exists():
        return None
    try:
        return json.loads(hb.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": str(exc)}


def checkpoint_complete(spec: dict) -> bool:
    path = ckpt_dir(spec)
    hb = heartbeat(path) or {}
    step = int(hb.get("time_step") or 0)
    return (
        step >= int(int(spec["target_steps"]) * 0.98)
        and (path / "geoexplorer.pt").exists()
        and (path / "geoexplorer_latest.pt").exists()
    )


def resolve_existing(candidates: list[str], optional: bool = False) -> str | None:
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    if optional:
        return None
    raise FileNotFoundError(f"missing all candidates: {candidates}")


def natural_key(key: str) -> tuple[int, str]:
    tail = str(key).split("_")[-1]
    return (int(tail), str(key)) if tail.isdigit() else (10**9, str(key))


def load_dict_npy(path: Path) -> dict:
    import numpy as np

    payload = np.load(path, allow_pickle=True)
    if getattr(payload, "shape", None) == ():
        data = payload.item()
    else:
        data = payload[()]
    if not isinstance(data, dict):
        raise TypeError(f"expected dict payload from {path}, got {type(data).__name__}")
    return data


def save_merged_grid(output_path: Path, sources: list[tuple[str, Path]], seed: int = DEFAULT_SEED) -> dict:
    import numpy as np

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged = {}
    rows = []
    index = 0
    for source_name, source_path in sources:
        data = load_dict_npy(source_path)
        for key in sorted(data.keys(), key=natural_key):
            arr = np.asarray(data[key])
            merged_key = f"img_{index}"
            merged[merged_key] = arr
            rows.append(
                {
                    "merged_key": merged_key,
                    "source_dataset": source_name,
                    "source_key": str(key),
                    "shape": list(arr.shape),
                    "mean_patch_norm": float(np.linalg.norm(arr, axis=-1).mean()) if arr.ndim >= 2 else math.nan,
                }
            )
            index += 1
    np.save(output_path, merged)
    index_csv = output_path.with_suffix(".index.csv")
    with index_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["merged_key", "source_dataset", "source_key", "shape", "mean_patch_norm"])
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "generated_at": now_iso(),
        "seed": seed,
        "output_npy": str(output_path),
        "output_index_csv": str(index_csv),
        "total_count": len(merged),
        "sources": [{"name": name, "path": str(path)} for name, path in sources],
        "source_counts": {
            name: sum(1 for row in rows if row["source_dataset"] == name)
            for name, _ in sources
        },
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def ensure_train_assets() -> dict[str, dict]:
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    source_paths = {
        "masa": Path(
            resolve_existing(
                [
                    "/root/geoexplorer/GeoExplorer/data/masa/sat_train_grid_5.npy",
                    "/root/geoexplorer/data/masa/sat_train_grid_5.npy",
                ]
            )
        ),
        "mmgag": Path(
            resolve_existing(
                [
                    "/root/geoexplorer/GeoExplorer/data/mm_gag/processed/mmgag_sat_grid_5.npy",
                    "/root/geoexplorer/GeoExplorer/data/mm_gag/mmgag_sat_grid_5.npy",
                ]
            )
        ),
        "swissview": Path(
            resolve_existing(
                [
                    "/root/geoexplorer/GeoExplorer/data/swissview/swissview100_sat_patches.npy",
                    "/root/geoexplorer/GeoExplorer/data/swissview/processed/swissview100_sat_patches.npy",
                ]
            )
        ),
    }
    definitions = {
        "masa_only": [("masa", source_paths["masa"])],
        "mmgag_only": [("mmgag", source_paths["mmgag"])],
        "swissview_only": [("swissview", source_paths["swissview"])],
        "masa_plus_swissview": [("masa", source_paths["masa"]), ("swissview", source_paths["swissview"])],
        "mmgag_plus_swissview": [("mmgag", source_paths["mmgag"]), ("swissview", source_paths["swissview"])],
        "all_three": [("masa", source_paths["masa"]), ("mmgag", source_paths["mmgag"]), ("swissview", source_paths["swissview"])],
    }
    assets: dict[str, dict] = {}
    for key, sources in definitions.items():
        out = STAGING_DIR / f"{key}_train_grid_5.npy"
        manifest = save_merged_grid(out, sources)
        assets[key] = {"path": str(out), "manifest": manifest}

    canonical_out = STAGING_DIR / "masa_plus_mmgag_train_grid_5.npy"
    if CANONICAL_TRAIN_GRID.exists():
        shutil.copy2(CANONICAL_TRAIN_GRID, canonical_out)
        try:
            count = len(load_dict_npy(canonical_out))
        except Exception:
            count = None
        manifest = {
            "generated_at": now_iso(),
            "output_npy": str(canonical_out),
            "copied_from": str(CANONICAL_TRAIN_GRID),
            "total_count": count,
            "note": "canonical current MASA+MM-GAG train grid used by anchor0624 line",
        }
        canonical_out.with_suffix(".manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        manifest = save_merged_grid(canonical_out, [("masa", source_paths["masa"]), ("mmgag", source_paths["mmgag"])])
    assets["masa_plus_mmgag"] = {"path": str(canonical_out), "manifest": manifest}
    (STAGING_DIR / "appendix_gate_valdist_train_assets_manifest.json").write_text(
        json.dumps({"generated_at": now_iso(), "assets": assets}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return assets


def train_env_overrides(spec: dict, gpu: int | None, train_grid: str) -> dict:
    cuda_visible = str(gpu) if gpu is not None and gpu >= 0 else "unknown_recovered_complete_run"
    overrides = {
        "PYTHONPATH": "/root/geoexplorer/env/geoexplorer_site:/root/geoexplorer/GeoExplorer:/root/geoexplorer",
        **cuda_compat_env(),
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
        "GEOEXPLORER_TRAIN_DATA": train_grid,
        "GEOEXPLORER_VAL_DATA": "/root/geoexplorer/data/masa/sat_val_grid_5.npy",
        "GEOEXPLORER_TEST_DATA": "/root/geoexplorer/data/masa/sat_test_grid_5.npy",
        "GEOEXPLORER_PRETRAIN_CKPT_ROOT": str(CKPT_ROOT),
        "GEOEXPLORER_TRAIN_CKPT_ROOT": str(CKPT_ROOT),
        "GEOEXPLORER_LLM_CHECKPOINT": str(LLM_CHECKPOINT),
        "GEOEXPLORER_TRAIN_EXPT": f"{SERIES}/{TRAIN_EXPERIMENT}/{spec['name']}",
        "GEOEXPLORER_TRAIN_NAME": "geoexplorer.pt",
        "GEOEXPLORER_TRAIN_PREFIX": "geoexplorer_",
        "GEOEXPLORER_MAX_TRAINING_TIMESTEPS": str(spec["target_steps"]),
        "GEOEXPLORER_RANDOM_SEED": str(spec["seed"]),
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
    overrides.update(spec["env"])
    return overrides


def write_run_repro_bundle(spec: dict, gpu: int | None, train_assets: dict) -> Path:
    out_dir = ckpt_dir(spec)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_path = out_dir / "appendix_gate_valdist_spec.json"
    if (gpu is None or gpu < 0) and spec_path.exists() and (out_dir / "source_snapshot_manifest.json").exists():
        return spec_path
    train_grid = train_assets[spec["trainset_key"]]["path"]
    source_rows = copy_source_snapshot(out_dir / "source_snapshot")
    source_manifest = {
        "generated_at": now_iso(),
        "series": SERIES,
        "experiment": EXPERIMENT,
        "train_experiment": TRAIN_EXPERIMENT,
        "run": spec["name"],
        "source_files": source_rows,
    }
    (out_dir / "source_snapshot_manifest.json").write_text(
        json.dumps(source_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    payload = {
        "generated_at": now_iso(),
        "series": SERIES,
        "experiment": EXPERIMENT,
        "run": spec,
        "seed": spec["seed"],
        "target_steps": spec["target_steps"],
        "gpu": gpu if gpu is not None and gpu >= 0 else None,
        "checkpoint_dir": str(out_dir),
        "expected_best_checkpoint": str(out_dir / "geoexplorer.pt"),
        "expected_latest_checkpoint": str(out_dir / "geoexplorer_latest.pt"),
        "train_asset": train_assets[spec["trainset_key"]],
        "effective_train_env_overrides": train_env_overrides(spec, gpu, train_grid),
        "source_snapshot": str(out_dir / "source_snapshot"),
        "source_snapshot_manifest": str(out_dir / "source_snapshot_manifest.json"),
    }
    spec_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return spec_path


def base_train_env(spec: dict, gpu: int, train_assets: dict) -> dict:
    env = clean_process_env()
    train_grid = train_assets[spec["trainset_key"]]["path"]
    env.update(train_env_overrides(spec, gpu, train_grid))
    return env


def launch_train(spec: dict, gpu: int, train_assets: dict) -> subprocess.Popen:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = ckpt_dir(spec)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{spec['name']}.train.log"
    log = log_path.open("ab", buffering=0)
    log.write(f"\n[{now_iso()}] launching train {spec['name']} on GPU{gpu}\n".encode("utf-8"))
    return subprocess.Popen(
        ["/usr/bin/python3", "-u", "train.py"],
        cwd=str(REPO_ROOT),
        env=base_train_env(spec, gpu, train_assets),
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def gpu_snapshot() -> list[dict]:
    try:
        env = clean_process_env()
        env.update(cuda_compat_env())
        smi = str(NVIDIA_COMPAT_SMI) if NVIDIA_COMPAT_SMI.exists() else "nvidia-smi"
        result = subprocess.run(
            [smi, "--query-gpu=index,name,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            env=env,
        )
    except Exception as exc:
        return [{"error": str(exc)}]
    if result.returncode != 0:
        text = (result.stderr or result.stdout or "").strip()
        return [{"error": text or f"nvidia-smi exited with {result.returncode}"}]
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
        "series": SERIES,
        "experiment": EXPERIMENT,
        "output_root": str(OUTPUT_ROOT),
        "gpu_snapshot": gpu_snapshot(),
    }
    if detail:
        payload.update(detail)
    STATUS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def train_all(train_assets: dict) -> list[dict]:
    states = {
        spec["name"]: {
            "status": "pending",
            "group": spec["group"],
            "trainset_key": spec["trainset_key"],
            "seed": spec["seed"],
            "target_steps": spec["target_steps"],
            "checkpoint_dir": str(ckpt_dir(spec)),
            "env": spec["env"],
            "table_roles": spec["table_roles"],
        }
        for spec in RUN_SPECS
    }
    active: dict[str, dict] = {}
    while True:
        finished = []
        for name, item in active.items():
            code = item["process"].poll()
            if code is None:
                continue
            spec = item["spec"]
            complete = checkpoint_complete(spec)
            states[name]["status"] = "completed" if code == 0 and complete else "failed"
            states[name]["returncode"] = int(code)
            states[name]["checkpoint_complete"] = bool(complete)
            states[name]["ended_at"] = now_iso()
            finished.append(name)
        for name in finished:
            active.pop(name, None)

        counts = {gpu: 0 for gpu in GPU_SLOTS}
        for item in active.values():
            counts[int(item["gpu"])] += 1

        for spec in RUN_SPECS:
            name = spec["name"]
            states[name]["heartbeat"] = heartbeat(ckpt_dir(spec))
            if states[name]["status"] == "pending" and checkpoint_complete(spec):
                states[name]["repro_spec"] = str(write_run_repro_bundle(spec, -1, train_assets))
                states[name]["status"] = "completed"
                continue
            if states[name]["status"] != "pending":
                continue
            available = [gpu for gpu, cap in GPU_SLOTS.items() if counts[gpu] < cap]
            if not available:
                continue
            gpu = available[0]
            repro_spec = write_run_repro_bundle(spec, gpu, train_assets)
            process = launch_train(spec, gpu, train_assets)
            active[name] = {"process": process, "gpu": gpu, "spec": spec}
            counts[gpu] += 1
            states[name].update(
                {
                    "status": "running",
                    "pid": int(process.pid),
                    "gpu": gpu,
                    "started_at": now_iso(),
                    "repro_spec": str(repro_spec),
                }
            )

        write_status(
            "training_appendix_gate_valdist",
            {
                "active_train_processes": len(active),
                "total_train_runs": len(RUN_SPECS),
                "train_assets": train_assets,
                "training_runs": states,
            },
        )
        if any(state["status"] == "failed" for state in states.values()):
            terminate_active_processes(active)
            raise RuntimeError("appendix training failed")
        if all(state["status"] == "completed" for state in states.values()):
            return [
                {
                    **spec,
                    "checkpoint": str(ckpt_dir(spec) / "geoexplorer.pt"),
                    "checkpoint_dir": str(ckpt_dir(spec)),
                    "repro_spec": str(ckpt_dir(spec) / "appendix_gate_valdist_spec.json"),
                }
                for spec in RUN_SPECS
            ]
        time.sleep(60)


def resolved_benchmarks() -> dict[str, dict]:
    out = {}
    for name in BENCHMARK_ORDER:
        bench = BENCHMARKS[name]
        resolved = {"test_path": resolve_existing(bench["test_path_candidates"])}
        if "pre_goal_path_candidates" in bench:
            resolved["pre_goal_path"] = resolve_existing(bench["pre_goal_path_candidates"])
        if "goal_embeds_candidates" in bench:
            resolved["goal_embeds"] = resolve_existing(bench["goal_embeds_candidates"])
        out[name] = {**bench, "resolved": resolved}
    return out


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
        success_ratio = float(payload["success_ratio"])
    else:
        success_ratio = float(payload["modes"][0]["success_ratio"])
    sg_mean = float(payload.get("sg_mean", payload.get("modes", [{}])[0].get("sg_mean", math.nan)))
    per_distance = payload.get("per_distance") or payload.get("modes", [{}])[0].get("per_dist", [])
    return {"success_ratio": success_ratio, "sg_mean": sg_mean, "per_distance": per_distance, "payload": payload}


def eval_output_path(spec: dict, benchmark_name: str) -> Path:
    return OUTPUT_ROOT / "raw" / spec["name"] / f"{benchmark_name}.json"


def build_eval_command(spec: dict, bench_name: str, bench: dict, out_path: Path) -> list[str]:
    cmd = [
        "/usr/bin/python3",
        str(EVALUATOR),
        "--method",
        "geoexplorer",
        "--method-label",
        "GeoExplorer-anchor0624-appendix",
        "--repo-dir",
        str(REPO_ROOT),
        "--checkpoint",
        spec["checkpoint"],
        "--llm-checkpoint",
        str(LLM_CHECKPOINT),
        "--benchmark",
        bench_name,
        "--paper-table",
        bench["paper_table"],
        "--dataset",
        bench["dataset"],
        "--goal-mode",
        bench["goal_mode"],
        "--fixed-goal-mode",
        bench["fixed_goal_mode"],
        "--test-path",
        bench["resolved"]["test_path"],
        "--device",
        "cuda:0",
        "--patch-size",
        "5",
        "--budget",
        "10",
        "--distances",
        "4,5,6,7,8",
        "--repeats-per-dist",
        str(bench.get("repeats_per_dist", 5)),
        "--seed",
        str(TASK_BANK_SEED),
        "--output-path",
        str(out_path),
    ]
    if "goal_embeds" in bench["resolved"]:
        cmd.extend(["--goal-embeds", bench["resolved"]["goal_embeds"]])
    if "pre_goal_path" in bench["resolved"]:
        cmd.extend(["--pre-goal-path", bench["resolved"]["pre_goal_path"]])
    return cmd


def launch_eval(spec: dict, bench_name: str, bench: dict, gpu: int) -> subprocess.Popen:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = eval_output_path(spec, bench_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"eval__{spec['name']}__{bench_name}.log"
    log = log_path.open("ab", buffering=0)
    log.write(f"\n[{now_iso()}] launching eval {spec['name']}::{bench_name} on GPU{gpu}\n".encode("utf-8"))
    env = clean_process_env()
    env.update(
        {
            "PYTHONPATH": "/root/geoexplorer/env/geoexplorer_site:/root/geoexplorer:/root/geoexplorer/GeoExplorer",
            **cuda_compat_env(),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": str(gpu),
        }
    )
    return subprocess.Popen(
        build_eval_command(spec, bench_name, bench, out_path),
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def evaluate_all(run_specs: list[dict]) -> dict:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    benchmarks = resolved_benchmarks()
    eval_states = {
        f"{spec['name']}::{bench_name}": {
            "status": "pending",
            "run": spec["name"],
            "benchmark": bench_name,
            "output_path": str(eval_output_path(spec, bench_name)),
        }
        for spec in run_specs
        for bench_name in BENCHMARK_ORDER
    }
    active: dict[str, dict] = {}
    while True:
        finished = []
        for key, item in active.items():
            code = item["process"].poll()
            if code is None:
                continue
            spec = item["spec"]
            bench_name = item["benchmark_name"]
            out_path = eval_output_path(spec, bench_name)
            ok = code == 0 and output_ready(out_path)
            eval_states[key].update({"status": "completed" if ok else "failed", "returncode": int(code), "ended_at": now_iso()})
            finished.append(key)
        for key in finished:
            active.pop(key, None)

        counts = {gpu: 0 for gpu in GPU_SLOTS}
        for item in active.values():
            counts[int(item["gpu"])] += 1

        for spec in run_specs:
            if not Path(spec["checkpoint"]).exists():
                raise FileNotFoundError(f"missing checkpoint for eval: {spec['checkpoint']}")
            for bench_name in BENCHMARK_ORDER:
                key = f"{spec['name']}::{bench_name}"
                state = eval_states[key]
                out_path = eval_output_path(spec, bench_name)
                if state["status"] == "pending" and output_ready(out_path):
                    state["status"] = "completed"
                    state["resume_reused"] = True
                    continue
                if state["status"] != "pending":
                    continue
                available = [gpu for gpu, cap in GPU_SLOTS.items() if counts[gpu] < cap]
                if not available:
                    continue
                gpu = available[0]
                process = launch_eval(spec, bench_name, benchmarks[bench_name], gpu)
                active[key] = {"process": process, "gpu": gpu, "spec": spec, "benchmark_name": bench_name}
                counts[gpu] += 1
                state.update({"status": "running", "gpu": gpu, "pid": int(process.pid), "started_at": now_iso()})

        write_status(
            "evaluating_appendix_gate_valdist",
            {
                "active_eval_processes": len(active),
                "total_eval_jobs": len(eval_states),
                "resolved_benchmarks": {name: item["resolved"] for name, item in benchmarks.items()},
                "eval_runs": eval_states,
            },
        )
        if any(state["status"] == "failed" for state in eval_states.values()):
            terminate_active_processes(active)
            raise RuntimeError("appendix evaluation failed")
        if all(state["status"] == "completed" for state in eval_states.values()):
            return build_outputs(run_specs, benchmarks)
        time.sleep(30)


def spec_metrics(spec: dict) -> dict:
    metrics = {}
    for bench_name in BENCHMARK_ORDER:
        metric = parse_metric(eval_output_path(spec, bench_name))
        per_dist = {int(row["distance"]): float(row["success_ratio"]) for row in metric["per_distance"]}
        metrics[bench_name] = {
            "sr": metric["success_ratio"],
            "sg": metric["sg_mean"],
            "d4": per_dist.get(4, math.nan),
            "d5": per_dist.get(5, math.nan),
            "d6": per_dist.get(6, math.nan),
            "d7": per_dist.get(7, math.nan),
            "d8": per_dist.get(8, math.nan),
        }
    return metrics


def row_base(spec: dict, metrics: dict) -> dict:
    sr_values = [metrics[name]["sr"] for name in BENCHMARK_ORDER]
    sg_values = [metrics[name]["sg"] for name in BENCHMARK_ORDER]
    transfer_sr = [metrics[name]["sr"] for name in PRIMARY_TRANSFER_BENCHMARKS]
    transfer_sg = [metrics[name]["sg"] for name in PRIMARY_TRANSFER_BENCHMARKS]
    return {
        "run": spec["name"],
        "group": spec["group"],
        "trainset_key": spec["trainset_key"],
        "seed": spec["seed"],
        "target_steps": spec["target_steps"],
        "checkpoint": spec["checkpoint"],
        "mean_all_sr": mean(sr_values),
        "mean_all_sg": mean(sg_values),
        "mean_transfer_sr": mean(transfer_sr),
        "mean_transfer_sg": mean(transfer_sg),
    }


def write_long_table(rows: list[dict]) -> None:
    fieldnames = [
        "run",
        "group",
        "trainset_key",
        "seed",
        "target_steps",
        "benchmark",
        "sr",
        "sg",
        "d4",
        "d5",
        "d6",
        "d7",
        "d8",
        "checkpoint",
    ]
    with (OUTPUT_ROOT / "appendix_long_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_wide_table(path: Path, rows: list[dict], metric: str, include_role: bool = True) -> None:
    fieldnames = [
        "family",
        "value",
        "run",
        "group",
        "trainset_key",
        "seed",
        "target_steps",
        f"mean_all_{metric}",
        f"mean_transfer_{metric}",
        *BENCHMARK_ORDER,
        "checkpoint",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key, "") for key in fieldnames}
            for bench_name in BENCHMARK_ORDER:
                out[bench_name] = f"{row['metrics'][bench_name][metric]:.6f}"
            out[f"mean_all_{metric}"] = f"{row[f'mean_all_{metric}']:.6f}"
            out[f"mean_transfer_{metric}"] = f"{row[f'mean_transfer_{metric}']:.6f}"
            writer.writerow(out)


def family_rows(summary_rows: list[dict], table: str) -> list[dict]:
    out = []
    for row in summary_rows:
        for role in row["table_roles"]:
            if role.get("table") != table:
                continue
            out.append({**row, "family": role["family"], "value": role["value"]})
    return out


def write_family_per_distance_table(rows: list[dict]) -> None:
    fieldnames = [
        "family",
        "value",
        "run",
        "benchmark",
        "d4",
        "d5",
        "d6",
        "d7",
        "d8",
        "checkpoint",
    ]
    with (OUTPUT_ROOT / "appendix_gate_valdist_per_distance.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in family_rows(rows, "param"):
            for bench_name in BENCHMARK_ORDER:
                metric = row["metrics"][bench_name]
                writer.writerow(
                    {
                        "family": row["family"],
                        "value": row["value"],
                        "run": row["run"],
                        "benchmark": bench_name,
                        "d4": f"{metric['d4']:.6f}",
                        "d5": f"{metric['d5']:.6f}",
                        "d6": f"{metric['d6']:.6f}",
                        "d7": f"{metric['d7']:.6f}",
                        "d8": f"{metric['d8']:.6f}",
                        "checkpoint": row["checkpoint"],
                    }
                )


def best_by(rows: list[dict], key: str, reverse: bool = True) -> dict | None:
    if not rows:
        return None
    return sorted(rows, key=lambda item: item[key], reverse=reverse)[0]


def markdown_table(rows: list[dict], columns: list[tuple[str, str]], limit: int | None = None) -> list[str]:
    selected = rows[:limit] if limit else rows
    lines = [
        "| " + " | ".join(title for title, _ in columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in selected:
        vals = []
        for _, key in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                vals.append(f"{value:.4f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def write_markdown_summary(rows: list[dict]) -> None:
    param_rows = sorted(family_rows(rows, "param"), key=lambda item: (item["family"], -item["mean_transfer_sr"]))
    best_transfer = best_by(param_rows, "mean_transfer_sr")
    best_all = best_by(param_rows, "mean_all_sr")
    lines = [
        "# 门控与验证距离补充实验汇总（自动生成）",
        "",
        f"- 生成时间：`{now_iso()}`",
        f"- 训练 run 数：`{len(rows)}`",
        "- 训练数据固定：`MASA+MM-GAG`；默认配置为 `gate_floor=0.405`, `PBRS=0.10`, `entropy=0.005`, `VAL_DISTS=7,8`。",
        f"- 评测协议：`5x5`, `B=10`, `C={{4,5,6,7,8}}`, greedy，任务种子 `{TASK_BANK_SEED}`。",
        "- `mean_transfer` 默认取 SwissViewMonuments aerial/ground 与 xBD-pre/xBD-disaster，用于减少训练域同域测试造成的偏差。",
        "- `gate_floor=0` 和 `gate_floor=1` 是门控端点，不是纯外部/纯内部奖励；纯奖励端点见 `reward_control`。",
        "",
        f"- transfer 最优补充行：`{best_transfer['family'] if best_transfer else 'NA'}={best_transfer['value'] if best_transfer else 'NA'}`。",
        f"- all-benchmark 最优补充行：`{best_all['family'] if best_all else 'NA'}={best_all['value'] if best_all else 'NA'}`。",
        "",
        "## 参数敏感性（按参数族分组）",
        "",
    ]
    for family in sorted({row["family"] for row in param_rows}):
        family_subset = sorted([row for row in param_rows if row["family"] == family], key=lambda item: item["mean_transfer_sr"], reverse=True)
        lines.extend([f"### {family}", ""])
        lines.extend(
            markdown_table(
                family_subset,
                [
                    ("Value", "value"),
                    ("Run", "run"),
                    ("Transfer SR", "mean_transfer_sr"),
                    ("All SR", "mean_all_sr"),
                    ("Transfer SG", "mean_transfer_sg"),
                    ("All SG", "mean_all_sg"),
                ],
            )
        )
        lines.append("")
    lines.extend(
        [
            "## 结果解读提醒",
            "",
            "- `gate_floor_dense` 用于回答 intrinsic 距离衰减强弱是否必要，尤其看 `0.0` 与 `1.0` 两个端点。",
            "- `val_dists_bias` 用于回答 checkpoint 选择偏远距离是否必要，应结合 `appendix_gate_valdist_per_distance.csv` 看 d4-d8 形态。",
            "- `reward_control` 中 `external_only` 和 `intrinsic_only` 才能对应纯外部/纯内部奖励；不要把 `gate_floor=0/1` 写成纯奖励端点。",
            "",
            "- `xBD` 行使用当前构造的 deterministic paper-test800 子集，应保持该 caveat。",
        ]
    )
    (OUTPUT_ROOT / "appendix_summary_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_outputs(run_specs: list[dict], benchmarks: dict[str, dict]) -> dict:
    summary_rows = []
    long_rows = []
    for spec in run_specs:
        metrics = spec_metrics(spec)
        base = row_base(spec, metrics)
        row = {**base, "metrics": metrics, "table_roles": spec["table_roles"], "factor_values": spec["factor_values"]}
        summary_rows.append(row)
        for bench_name in BENCHMARK_ORDER:
            long_rows.append(
                {
                    **base,
                    "benchmark": bench_name,
                    **metrics[bench_name],
                }
            )

    param_rows = sorted(family_rows(summary_rows, "param"), key=lambda item: (item["family"], str(item["value"])))
    write_long_table(long_rows)
    write_wide_table(OUTPUT_ROOT / "appendix_gate_valdist_sr_table.csv", param_rows, "sr")
    write_wide_table(OUTPUT_ROOT / "appendix_gate_valdist_sg_table.csv", param_rows, "sg")
    write_family_per_distance_table(summary_rows)
    write_markdown_summary(summary_rows)

    aggregate = {
        "generated_at": now_iso(),
        "series": SERIES,
        "experiment": EXPERIMENT,
        "task_bank_seed": TASK_BANK_SEED,
        "benchmarks": benchmarks,
        "benchmark_order": BENCHMARK_ORDER,
        "primary_transfer_benchmarks": PRIMARY_TRANSFER_BENCHMARKS,
        "rows": summary_rows,
        "long_rows": long_rows,
        "outputs": {
            "long_table": str(OUTPUT_ROOT / "appendix_long_table.csv"),
            "gate_valdist_sr": str(OUTPUT_ROOT / "appendix_gate_valdist_sr_table.csv"),
            "gate_valdist_sg": str(OUTPUT_ROOT / "appendix_gate_valdist_sg_table.csv"),
            "per_distance": str(OUTPUT_ROOT / "appendix_gate_valdist_per_distance.csv"),
            "summary_zh": str(OUTPUT_ROOT / "appendix_summary_zh.md"),
        },
        "rigor_notes": [
            "All rows are trained and evaluated by the same gate/validation-distance follow-up orchestrator.",
            "Checkpoint selection is validation-based; fixed test banks are not used for model selection.",
            "Gate-floor endpoints are not reward endpoints; reward_control rows isolate strict external-only and intrinsic-only cases.",
            "Validation-distance rows diagnose checkpoint-selection bias and should be interpreted with per-distance results.",
        ],
    }
    (OUTPUT_ROOT / "appendix_all_results.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
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


def write_pipeline_repro_bundle(train_assets: dict) -> None:
    repro_root = EXP_ROOT / "reproducibility"
    rows = copy_source_snapshot(repro_root / "source_snapshot")
    payload = {
        "generated_at": now_iso(),
        "series": SERIES,
        "experiment": EXPERIMENT,
        "train_experiment": TRAIN_EXPERIMENT,
        "task_bank_seed": TASK_BANK_SEED,
        "run_specs": RUN_SPECS,
        "train_assets": train_assets,
        "source_files": rows,
    }
    (repro_root / "appendix_gate_valdist_repro_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PID_PATH.write_text(str(os.getpid()) + "\n", encoding="utf-8")
    try:
        write_status("preparing_train_assets", {"total_train_runs": len(RUN_SPECS)})
        train_assets = ensure_train_assets()
        write_pipeline_repro_bundle(train_assets)
        run_specs = train_all(train_assets)
        summary = evaluate_all(run_specs)
        write_status(
            "completed",
            {
                "total_train_runs": len(run_specs),
                "output_summary": summary["outputs"],
                "aggregate_path": str(OUTPUT_ROOT / "appendix_all_results.json"),
            },
        )
        return 0
    except Exception as exc:
        write_status("failed", {"error": str(exc)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
