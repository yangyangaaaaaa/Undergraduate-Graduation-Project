from __future__ import annotations

import csv
import gc
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch, Rectangle
from PIL import Image


def parse_int_env_list(value: str, default: list[int]) -> list[int]:
    if not value:
        return default
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    return parsed or default


REMOTE_ROOT = Path(os.environ.get("GEO_ROOT", "/root/geoexplorer"))
REPO_ROOT = REMOTE_ROOT / "GeoExplorer"
SERIES = "visualization_20260518"
EXPERIMENT = "anchor0624_swissviewmonuments_qualitative"
EXP_ROOT = REMOTE_ROOT / "ab_experiments" / SERIES / EXPERIMENT
MONITORING = EXP_ROOT / "monitoring"
STATUS_PATH = MONITORING / "anchor0624_visualization_status_latest.json"
OUTPUT_ROOT = Path(
    os.environ.get("VIS_ROOT", str(REMOTE_ROOT / "analysis" / "pipeline_20260517_anchor0624_visualization"))
)

TEST_PATH = Path(
    os.environ.get("ACCEPTANCE_TEST_PATH", str(REPO_ROOT / "data" / "swissview" / "swissviewmonuments_patches.npy"))
)
BASELINE_HELPER_DIR = MONITORING
PATCH_SIZE = 5
BUDGET = 10
DISTANCES = parse_int_env_list(os.getenv("ACCEPTANCE_DISTANCES", "4,6,8"), [4, 6, 8])
TASK_BANK_SEED = 20260516
REPEATS_PER_DIST = 1
DEVICE = "cuda:0"
IMAGE_FILTER_ENV = os.getenv("ACCEPTANCE_INFER_IMAGE", os.getenv("ACCEPTANCE_IMAGE_IDX", "")).strip()
CUSTOM_IMAGE_PATH = os.getenv("ACCEPTANCE_CUSTOM_IMAGE", "").strip()
CUSTOM_START_ENV = os.getenv("ACCEPTANCE_CUSTOM_START", "").strip()
CUSTOM_GOAL_ENV = os.getenv("ACCEPTANCE_CUSTOM_GOAL", "").strip()
FIXED_GOAL_MODE = os.getenv("ACCEPTANCE_FIXED_GOAL_MODE", "none" if CUSTOM_IMAGE_PATH else "monuments").strip()

GOMAA_ROOT = Path("/root/src/compare_baselines_bundle_20260505_v2/compare_baselines_bundle/gomaa_geo_official")
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

METHODS = [
    {
        "key": "anchor0624",
        "label": "GeoExplorer-anchor0624",
        "method": "geoexplorer",
        "repo_dir": str(REPO_ROOT),
        "checkpoint": str(ANCHOR_CKPT),
        "llm_checkpoint": str(ANCHOR_LLM),
        "color": "#0072B2",
    },
    {
        "key": "gomaa",
        "label": "GOMAA-Geo",
        "method": "gomaa",
        "repo_dir": str(GOMAA_ROOT),
        "checkpoint": str(GOMAA_CKPT),
        "llm_checkpoint": str(GOMAA_LLM),
        "color": "#D55E00",
    },
    {
        "key": "pristine",
        "label": "GeoExplorer-pristine",
        "method": "geoexplorer",
        "repo_dir": str(REPO_ROOT),
        "checkpoint": str(PRISTINE_CKPT),
        "llm_checkpoint": str(PRISTINE_LLM),
        "color": "#009E73",
    },
]

OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "purple": "#CC79A7",
}


@dataclass
class Args:
    method: str
    method_label: str
    repo_dir: str
    checkpoint: str
    llm_checkpoint: str
    dataset: str = "swissviewmonuments"
    goal_mode: str = "aerial"
    test_path: str = str(TEST_PATH)
    device: str = DEVICE
    patch_size: int = PATCH_SIZE
    budget: int = BUDGET
    seed: int = TASK_BANK_SEED


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_status(phase: str, extra: dict | None = None) -> None:
    MONITORING.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": now_iso(),
        "phase": phase,
        "experiment_root": str(EXP_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "distances": DISTANCES,
        "budget": BUDGET,
        "task_bank_seed": TASK_BANK_SEED,
    }
    if extra:
        payload.update(extra)
    STATUS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def configure_imports() -> None:
    for path in [str(BASELINE_HELPER_DIR), str(REPO_ROOT), str(REMOTE_ROOT), str(GOMAA_ROOT)]:
        if path not in sys.path:
            sys.path.insert(0, path)


def parse_image_filter(value: str) -> set[int]:
    if not value:
        return set()
    result = set()
    for item in value.split(","):
        text = item.strip().lower()
        if not text:
            continue
        if text.startswith("img_"):
            text = text[4:]
        elif text.startswith("img"):
            text = text[3:]
        result.add(int(text))
    return result


def filter_tasks_by_image(tasks: list[dict]) -> tuple[list[dict], set[int]]:
    image_filter = parse_image_filter(IMAGE_FILTER_ENV)
    if not image_filter:
        return tasks, image_filter
    filtered = [task for task in tasks if int(task["img_order"]) in image_filter]
    if not filtered:
        available = sorted({int(task["img_order"]) for task in tasks})
        preview = ",".join(str(idx) for idx in available[:20])
        raise ValueError(f"No tasks matched ACCEPTANCE_INFER_IMAGE={IMAGE_FILTER_ENV!r}; first available img_idx values: {preview}")
    return filtered, image_filter


def patch_to_row_col(patch: int) -> tuple[int, int]:
    return divmod(int(patch), PATCH_SIZE)


def patch_distance(a: int, b: int) -> int:
    ar, ac = patch_to_row_col(a)
    br, bc = patch_to_row_col(b)
    return abs(ar - br) + abs(ac - bc)


def build_explicit_custom_task_bank(dataset_dict) -> list[dict]:
    keys = sorted(dataset_dict[()].keys())
    if not keys:
        raise ValueError(f"No image embeddings found in {TEST_PATH}")
    if not CUSTOM_START_ENV or not CUSTOM_GOAL_ENV:
        raise ValueError("Both ACCEPTANCE_CUSTOM_START and ACCEPTANCE_CUSTOM_GOAL are required for explicit custom tasks")
    start_patch = int(CUSTOM_START_ENV)
    goal_patch = int(CUSTOM_GOAL_ENV)
    max_patch = PATCH_SIZE * PATCH_SIZE
    if not (0 <= start_patch < max_patch and 0 <= goal_patch < max_patch):
        raise ValueError(f"Custom start/goal must be in [0,{max_patch - 1}], got start={start_patch}, goal={goal_patch}")
    return [
        {
            "img_order": 0,
            "img_key": str(keys[0]),
            "distance": int(patch_distance(start_patch, goal_patch)),
            "repeat_idx": 0,
            "goal_patch": goal_patch,
            "current_patch": start_patch,
        }
    ]


def patch_center(patch: int, image_size: tuple[int, int]) -> tuple[float, float]:
    width, height = image_size
    row, col = patch_to_row_col(patch)
    return (col + 0.5) * width / PATCH_SIZE, (row + 0.5) * height / PATCH_SIZE


def patch_bounds(patch: int, image_size: tuple[int, int]) -> tuple[float, float, float, float]:
    width, height = image_size
    row, col = patch_to_row_col(patch)
    patch_w = width / PATCH_SIZE
    patch_h = height / PATCH_SIZE
    return col * patch_w, row * patch_h, patch_w, patch_h


def load_metadata():
    from utils.swissviewmonuments_metadata import load_monuments_metadata

    metadata, metadata_path = load_monuments_metadata(REPO_ROOT)
    if not metadata:
        raise FileNotFoundError(f"Missing SwissViewMonuments metadata under {REPO_ROOT}")
    return metadata, metadata_path


def resolve_aerial_path(img_idx: int, metadata: list[dict]) -> Path:
    if CUSTOM_IMAGE_PATH:
        cached = OUTPUT_ROOT / "asset_cache" / "aerial_view" / f"img_{int(img_idx):03d}.png"
        if cached.exists():
            return cached
        candidate = Path(CUSTOM_IMAGE_PATH)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Custom image not found: {CUSTOM_IMAGE_PATH}")

    from utils.swissviewmonuments_metadata import resolve_monuments_asset

    entry = metadata[int(img_idx)]
    rel_path = entry["aerial_view"]
    candidate, _ = resolve_monuments_asset(REPO_ROOT, rel_path)
    if candidate.exists():
        return candidate
    fallback = REMOTE_ROOT / "data" / "swissview" / rel_path
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Could not resolve aerial asset for img_{img_idx}: {rel_path}")


def load_patch_or_crop(aerial: Image.Image, img_idx: int, patch: int) -> np.ndarray:
    patch_path = REPO_ROOT / "data/swissview/swissviewmonuments_patches/patches" / f"img_{img_idx}" / f"patch_{patch}.jpg"
    if patch_path.exists():
        return np.asarray(Image.open(patch_path).convert("RGB"))
    x0, y0, w, h = patch_bounds(patch, aerial.size)
    return np.asarray(aerial.crop((int(x0), int(y0), int(x0 + w), int(y0 + h))).convert("RGB"))


def case_id_from_task(task: dict) -> str:
    return (
        f"img{int(task['img_order']):03d}_"
        f"d{int(task['distance'])}_"
        f"s{int(task['current_patch']):02d}_"
        f"g{int(task['goal_patch']):02d}_"
        f"r{int(task.get('repeat_idx', 0))}"
    )


def run_method(method_spec: dict, tasks: list[dict]) -> list[dict]:
    from paper_baseline_evaluator import build_sequence, extract_image_item, get_dist, load_bundle, model_action

    args = Args(
        method=method_spec["method"],
        method_label=method_spec["label"],
        repo_dir=method_spec["repo_dir"],
        checkpoint=method_spec["checkpoint"],
        llm_checkpoint=method_spec["llm_checkpoint"],
    )
    dataset_dict = np.load(TEST_PATH, allow_pickle=True)
    bundle = load_bundle(args)
    records = []
    started = time.time()
    for idx, task in enumerate(tasks):
        if idx % 100 == 0:
            write_status(
                "running_inference",
                {
                    "method": method_spec["label"],
                    "method_index": METHODS.index(method_spec),
                    "task_index": idx,
                    "total_tasks": len(tasks),
                },
            )
        goal_patch = int(task["goal_patch"])
        start_patch = int(task["current_patch"])
        env_embeds = extract_image_item(dataset_dict, task["img_key"], int(task["img_order"]))
        seq = build_sequence(
            bundle,
            env_embeds,
            goal_patch,
            goal_mode="aerial",
            patch_size=PATCH_SIZE,
        )
        seq.update_sequence_with_satellite_image_token(start_patch)
        traj = [int(start_patch)]
        actions = []
        reward_trace = []
        distance_trace = [int(get_dist(start_patch, goal_patch, PATCH_SIZE))]
        success = False

        for _ in range(BUDGET):
            prev_patch = int(seq.patch_sequence[-1])
            prev_dist = int(get_dist(prev_patch, goal_patch, PATCH_SIZE))
            action = int(model_action(bundle, seq, args))
            action_name = str(bundle.action_list[action])
            seq.update_sequence_with_action(action_name)
            current_patch = int(seq.patch_sequence[-1])
            current_dist = int(get_dist(current_patch, goal_patch, PATCH_SIZE))
            actions.append(action_name)
            traj.append(current_patch)
            distance_trace.append(current_dist)
            reward_trace.append(float(prev_dist - current_dist))
            if current_patch == goal_patch:
                success = True
                break

        final_patch = int(traj[-1])
        final_distance = int(get_dist(final_patch, goal_patch, PATCH_SIZE))
        optimal_steps = int(get_dist(start_patch, goal_patch, PATCH_SIZE))
        records.append(
            {
                "case_id": case_id_from_task(task),
                "method": method_spec["label"],
                "method_key": method_spec["key"],
                "checkpoint": method_spec["checkpoint"],
                "llm_checkpoint": method_spec["llm_checkpoint"],
                "dataset": "swissviewmonuments",
                "img_idx": int(task["img_order"]),
                "img_key": str(task["img_key"]),
                "distance": int(task["distance"]),
                "repeat_idx": int(task.get("repeat_idx", 0)),
                "start": start_patch,
                "goal": goal_patch,
                "final": final_patch,
                "success": bool(success),
                "final_distance": final_distance,
                "optimal_steps": optimal_steps,
                "traj": traj,
                "actions": actions,
                "reward_trace": reward_trace,
                "distance_trace": distance_trace,
                "path_length": int(len(traj) - 1),
                "detour_steps": int(max(0, (len(traj) - 1) - optimal_steps)),
            }
        )

    del bundle
    del dataset_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    write_status(
        "method_completed",
        {"method": method_spec["label"], "records": len(records), "elapsed_sec": round(time.time() - started, 2)},
    )
    return records


def choose_cases(records_by_method: dict[str, dict[str, dict]]) -> list[dict]:
    anchor = records_by_method["anchor0624"]
    selected: list[dict] = []
    selected_ids: set[str] = set()

    def add_case(role: str, case_id: str) -> None:
        if case_id in selected_ids:
            return
        selected_ids.add(case_id)
        selected.append({"role": role, "case_id": case_id, "task": anchor[case_id]})

    for dist in DISTANCES:
        dist_records = [r for r in anchor.values() if int(r["distance"]) == dist]
        successes = [r for r in dist_records if r["success"]]
        successes.sort(key=lambda r: (r["detour_steps"], r["path_length"], r["img_idx"], r["start"]))
        if successes:
            add_case(f"c{dist}_anchor_success", successes[0]["case_id"])

        difficult = [r for r in dist_records if (not r["success"]) or r["detour_steps"] > 0 or r["final_distance"] > 0]
        difficult.sort(key=lambda r: (r["success"], -r["final_distance"], -r["detour_steps"], r["img_idx"]))
        if difficult:
            add_case(f"c{dist}_anchor_failure_or_detour", difficult[0]["case_id"])

    hard_candidates = []
    for case_id, anchor_record in anchor.items():
        if int(anchor_record["distance"]) not in {6, 8}:
            continue
        gomaa = records_by_method["gomaa"].get(case_id)
        pristine = records_by_method["pristine"].get(case_id)
        if not gomaa or not pristine:
            continue
        gap = int(gomaa["final_distance"]) + int(pristine["final_distance"]) - int(anchor_record["final_distance"])
        fail_count = int(not gomaa["success"]) + int(not pristine["success"])
        hard_candidates.append((fail_count, gap, anchor_record["success"], case_id))
    hard_candidates.sort(key=lambda x: (-x[0], -x[1], not x[2], x[3]))
    if hard_candidates:
        add_case("three_method_hardcase", hard_candidates[0][3])

    return selected


def setup_axes(ax, record: dict, title: str, method_color: str) -> None:
    ax.set_title(title, fontsize=10, fontweight="bold", color="#111")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(2.4)
        spine.set_edgecolor(method_color)


def draw_trajectory_panel(ax, aerial_np: np.ndarray, record: dict, method_color: str, title: str) -> None:
    height, width = aerial_np.shape[:2]
    ax.imshow(aerial_np)
    setup_axes(ax, record, title, method_color)
    patch_w = width / PATCH_SIZE
    patch_h = height / PATCH_SIZE

    for grid_idx in range(1, PATCH_SIZE):
        ax.axvline(grid_idx * patch_w, color="white", alpha=0.55, linewidth=1.1)
        ax.axhline(grid_idx * patch_h, color="white", alpha=0.55, linewidth=1.1)

    for patch, color, lw in [
        (record["start"], OKABE_ITO["green"], 2.8),
        (record["goal"], OKABE_ITO["yellow"], 3.2),
        (record["final"], OKABE_ITO["blue"] if record["success"] else OKABE_ITO["vermillion"], 2.8),
    ]:
        x0, y0, w, h = patch_bounds(int(patch), (width, height))
        ax.add_patch(Rectangle((x0, y0), w, h, fill=False, linewidth=lw, edgecolor=color, zorder=5))

    centers = [patch_center(int(patch), (width, height)) for patch in record["traj"]]
    if len(centers) > 1:
        xs, ys = zip(*centers)
        ax.plot(xs, ys, color=method_color, linewidth=3.2, alpha=0.96, zorder=4)
    for step_idx, (x, y) in enumerate(centers):
        ax.scatter(x, y, s=62, c=method_color, edgecolors="white", linewidths=1.1, zorder=6)
        ax.text(x, y, str(step_idx), color="white", ha="center", va="center", fontsize=7, fontweight="bold", zorder=7)

    info = (
        f"{'success' if record['success'] else 'fail'} | "
        f"C={record['distance']} | "
        f"start={record['start']} goal={record['goal']} final={record['final']} | "
        f"steps={record['path_length']} fd={record['final_distance']}"
    )
    ax.text(
        0.02,
        0.98,
        info,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7.5,
        color="#111",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 3},
        zorder=8,
    )


def render_static(record: dict, metadata: list[dict], method_color: str, out_path: Path) -> None:
    aerial_path = resolve_aerial_path(record["img_idx"], metadata)
    aerial_img = Image.open(aerial_path).convert("RGB")
    aerial_np = np.asarray(aerial_img)
    goal_patch = load_patch_or_crop(aerial_img, record["img_idx"], record["goal"])
    start_patch = load_patch_or_crop(aerial_img, record["img_idx"], record["start"])
    final_patch = load_patch_or_crop(aerial_img, record["img_idx"], record["final"])

    fig = plt.figure(figsize=(9.2, 5.6))
    gs = fig.add_gridspec(2, 4, width_ratios=[3.0, 1.0, 1.0, 1.0], height_ratios=[1.0, 1.0], wspace=0.18, hspace=0.18)
    ax_main = fig.add_subplot(gs[:, 0])
    draw_trajectory_panel(ax_main, aerial_np, record, method_color, record["method"])

    patch_axes = [
        (fig.add_subplot(gs[0, 1]), goal_patch, "Goal", OKABE_ITO["yellow"]),
        (fig.add_subplot(gs[0, 2]), start_patch, "Start", OKABE_ITO["green"]),
        (fig.add_subplot(gs[0, 3]), final_patch, "Final", OKABE_ITO["blue"] if record["success"] else OKABE_ITO["vermillion"]),
    ]
    for ax, image, title, color in patch_axes:
        ax.imshow(image)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(2.5)
            spine.set_edgecolor(color)

    ax_text = fig.add_subplot(gs[1, 1:])
    ax_text.axis("off")
    actions = " -> ".join(record["actions"]) if record["actions"] else "none"
    text = "\n".join(
        [
            f"case: {record['case_id']}",
            f"image: img_{record['img_idx']}  distance: C={record['distance']}",
            f"trajectory: {record['traj']}",
            f"actions: {actions}",
            "reward_trace: progress delta in Manhattan distance",
        ]
    )
    ax_text.text(0, 1, text, va="top", ha="left", fontsize=8.2, linespacing=1.45)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_comparison(case: dict, records_by_method: dict[str, dict[str, dict]], metadata: list[dict], out_path: Path) -> None:
    case_id = case["case_id"]
    first = records_by_method["anchor0624"][case_id]
    aerial_path = resolve_aerial_path(first["img_idx"], metadata)
    aerial_np = np.asarray(Image.open(aerial_path).convert("RGB"))
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.4), constrained_layout=True)
    for ax, method in zip(axes, METHODS):
        record = records_by_method[method["key"]][case_id]
        draw_trajectory_panel(ax, aerial_np, record, method["color"], method["label"])
    fig.suptitle(f"{case['role']} | {case_id}", fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def render_anchor_figure_x(selected_cases: list[dict], records_by_method: dict[str, dict[str, dict]], metadata: list[dict]) -> Path | None:
    success_cases = []
    for dist in DISTANCES:
        match = next((case for case in selected_cases if case["role"] == f"c{dist}_anchor_success"), None)
        if match:
            success_cases.append(match)
    if len(success_cases) < 3:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.3), constrained_layout=True)
    for ax, case in zip(axes, success_cases):
        record = records_by_method["anchor0624"][case["case_id"]]
        aerial_np = np.asarray(Image.open(resolve_aerial_path(record["img_idx"], metadata)).convert("RGB"))
        draw_trajectory_panel(ax, aerial_np, record, METHODS[0]["color"], f"C={record['distance']} typical trajectory")
    fig.suptitle("GeoExplorer-anchor0624 typical trajectories on SwissViewMonuments", fontsize=13, fontweight="bold")
    out = OUTPUT_ROOT / "figure4_x_anchor_typical_c4_c6_c8.png"
    fig.savefig(out, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return out


def render_figure_y(selected_cases: list[dict], records_by_method: dict[str, dict[str, dict]], metadata: list[dict]) -> Path | None:
    hard = next((case for case in selected_cases if case["role"] == "three_method_hardcase"), None)
    if not hard:
        hard = next((case for case in selected_cases if "failure_or_detour" in case["role"]), None)
    if not hard:
        return None
    out = OUTPUT_ROOT / "figure4_y_three_method_hardcase.png"
    render_comparison(hard, records_by_method, metadata, out)
    return out


def render_gif(record: dict, metadata: list[dict], method_color: str, out_path: Path) -> None:
    aerial_np = np.asarray(Image.open(resolve_aerial_path(record["img_idx"], metadata)).convert("RGB"))
    frames = []
    for end_idx in range(1, len(record["traj"]) + 1):
        partial = dict(record)
        partial["traj"] = record["traj"][:end_idx]
        partial["final"] = record["traj"][end_idx - 1]
        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        draw_trajectory_panel(ax, aerial_np, partial, method_color, f"{record['method']} step {end_idx - 1}")
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
        frames.append(Image.fromarray(frame).resize((720, 680)))
        plt.close(fig)
    if frames:
        frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=700, loop=0)


def write_records(records: list[dict], selected_cases: list[dict]) -> None:
    (OUTPUT_ROOT / "trajectory_records.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_ROOT / "selected_cases.json").write_text(
        json.dumps(selected_cases, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    fieldnames = [
        "method",
        "checkpoint",
        "dataset",
        "img_idx",
        "distance",
        "start",
        "goal",
        "final",
        "success",
        "final_distance",
        "traj",
        "actions",
        "reward_trace",
        "path_length",
        "case_id",
        "optimal_steps",
        "detour_steps",
    ]
    with (OUTPUT_ROOT / "trajectory_records.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            out = {key: row.get(key, "") for key in fieldnames}
            for key in ("traj", "actions", "reward_trace"):
                out[key] = json.dumps(out[key], ensure_ascii=False)
            writer.writerow(out)


def write_readme(selected_cases: list[dict], metadata_path: Path, figure_paths: list[Path | None]) -> None:
    lines = [
        "# Anchor0624 Qualitative Visualization",
        "",
        f"Generated: {now_iso()}",
        "",
        "## Protocol",
        "",
        "- Dataset: SwissViewMonuments aerial goal setting.",
        "- Grid: `5x5`; budget: `B=10`; distances: `C={4,6,8}`.",
        f"- Task-bank seed: `{TASK_BANK_SEED}`; policy: greedy/argmax.",
        "- This is inference and visualization only. No model training was run.",
        "- `reward_trace` is stored as per-step Manhattan-distance progress, not as the PPO training reward.",
        "",
        "## Models",
        "",
    ]
    for method in METHODS:
        lines.extend(
            [
                f"- {method['label']}",
                f"  checkpoint: `{method['checkpoint']}`",
                f"  llm checkpoint: `{method['llm_checkpoint']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Data",
            "",
            f"- Embedding bank: `{TEST_PATH}`",
            f"- Metadata: `{metadata_path}`",
            "- The checkpoints match the Chapter 4 main tables: anchor0624 from the 2026-05-16 paper-aligned comparison, GOMAA from the same comparison, and pristine from the 2026-05-17 historical fixed-eval baseline.",
            "",
            "## Selected Cases",
            "",
            "| Role | Case | Distance | Image | Start | Goal |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for case in selected_cases:
        task = case["task"]
        lines.append(
            f"| {case['role']} | `{case['case_id']}` | {task['distance']} | {task['img_idx']} | {task['start']} | {task['goal']} |"
        )
    lines.extend(["", "## Figure Files", ""])
    for path in figure_paths:
        if path:
            lines.append(f"- `{path}`")
    (OUTPUT_ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    configure_imports()
    from paper_baseline_evaluator import build_task_bank

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for subdir in ["static_png", "comparison_png", "gif"]:
        (OUTPUT_ROOT / subdir).mkdir(parents=True, exist_ok=True)
    write_status("starting")
    if CUSTOM_IMAGE_PATH:
        metadata, metadata_path = [], Path(CUSTOM_IMAGE_PATH)
    else:
        metadata, metadata_path = load_metadata()
    dataset_dict = np.load(TEST_PATH, allow_pickle=True)
    if CUSTOM_IMAGE_PATH and CUSTOM_START_ENV and CUSTOM_GOAL_ENV:
        tasks = build_explicit_custom_task_bank(dataset_dict)
        skipped = []
    else:
        tasks, skipped = build_task_bank(
            dataset_dict,
            PATCH_SIZE,
            DISTANCES,
            repeats=REPEATS_PER_DIST,
            seed=TASK_BANK_SEED,
            max_images=None,
            fixed_goal_mode=FIXED_GOAL_MODE,
        )
    tasks, image_filter = filter_tasks_by_image(tasks)
    del dataset_dict
    write_status(
        "task_bank_ready",
        {
            "tasks": len(tasks),
            "skipped": len(skipped),
            "metadata": str(metadata_path),
            "image_filter": sorted(image_filter),
            "test_path": str(TEST_PATH),
            "custom_image": CUSTOM_IMAGE_PATH or None,
            "fixed_goal_mode": FIXED_GOAL_MODE,
        },
    )

    all_records = []
    records_by_method: dict[str, dict[str, dict]] = {}
    for method in METHODS:
        records = run_method(method, tasks)
        all_records.extend(records)
        records_by_method[method["key"]] = {record["case_id"]: record for record in records}

    selected_cases = choose_cases(records_by_method)
    write_records(all_records, selected_cases)
    write_status("rendering", {"selected_cases": selected_cases})

    static_outputs = []
    comparison_outputs = []
    gif_outputs = []
    for case in selected_cases:
        case_id = case["case_id"]
        for method in METHODS:
            record = records_by_method[method["key"]][case_id]
            out = OUTPUT_ROOT / "static_png" / f"{case['role']}__{case_id}__{method['key']}.png"
            render_static(record, metadata, method["color"], out)
            static_outputs.append(str(out))
            gif_out = OUTPUT_ROOT / "gif" / f"{case['role']}__{case_id}__{method['key']}.gif"
            render_gif(record, metadata, method["color"], gif_out)
            gif_outputs.append(str(gif_out))
        comp_out = OUTPUT_ROOT / "comparison_png" / f"{case['role']}__{case_id}__comparison.png"
        render_comparison(case, records_by_method, metadata, comp_out)
        comparison_outputs.append(str(comp_out))

    fig_x = render_anchor_figure_x(selected_cases, records_by_method, metadata)
    fig_y = render_figure_y(selected_cases, records_by_method, metadata)
    write_readme(selected_cases, metadata_path, [fig_x, fig_y])
    write_status(
        "completed",
        {
            "records": len(all_records),
            "tasks": len(tasks),
            "selected_cases_count": len(selected_cases),
            "static_png_count": len(static_outputs),
            "comparison_png_count": len(comparison_outputs),
            "gif_count": len(gif_outputs),
            "figure4_x": str(fig_x) if fig_x else None,
            "figure4_y": str(fig_y) if fig_y else None,
            "output_root": str(OUTPUT_ROOT),
        },
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        write_status("failed", {"error": str(exc)})
        raise
