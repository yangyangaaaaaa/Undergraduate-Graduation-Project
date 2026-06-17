import io
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from PIL import Image

from config import cfg
from models.ppo import PPO
from utils.swissviewmonuments_metadata import load_monuments_metadata, resolve_monuments_asset
from utils import generate_config, generate_config_unseen, seed_everything


PROJECT_ROOT = Path(__file__).resolve().parent
DEVICE = torch.device(cfg.train.device)
PATCH_COUNT = cfg.data.patch_size ** 2
GOAL_PATCH_LIST = (list(range(PATCH_COUNT)) * ((cfg.sample_number + PATCH_COUNT - 1) // PATCH_COUNT))[: cfg.sample_number]
MAX_DEBUG_IMAGES = int(os.getenv("GEOEXPLORER_MAX_IMAGES", "0")) or None
MAX_VIS_PER_DIST = int(os.getenv("GEOEXPLORER_MAX_VIS", "4"))
GRID_SIZE = cfg.data.patch_size


def idx_to_row_col(idx, grid_size=GRID_SIZE):
    row, col = divmod(int(idx), grid_size)
    return row, col


def patch_bounds(patch_id, image_size, grid_size=GRID_SIZE):
    width, height = image_size
    patch_w = width / grid_size
    patch_h = height / grid_size
    row, col = idx_to_row_col(patch_id, grid_size)
    return col * patch_w, row * patch_h, patch_w, patch_h


def patch_center(patch_id, image_size, grid_size=GRID_SIZE):
    x0, y0, patch_w, patch_h = patch_bounds(patch_id, image_size, grid_size)
    return x0 + patch_w / 2, y0 + patch_h / 2


def load_metadata():
    if cfg.dataset == "swissview":
        metadata_path = PROJECT_ROOT / "data" / "swissview" / "SwissView100.json"
        if not metadata_path.exists():
            return []
        with metadata_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    elif cfg.dataset == "swissviewmonuments":
        metadata, _ = load_monuments_metadata(PROJECT_ROOT, override=os.getenv("GEOEXPLORER_MONUMENTS_METADATA", ""))
        return metadata
    else:
        return []


def resolve_metadata_asset(rel_path):
    candidate, _ = resolve_monuments_asset(PROJECT_ROOT, rel_path)
    return candidate


def resolve_sample_assets(img_idx):
    if cfg.dataset == "swissview":
        aerial_path = PROJECT_ROOT / "data" / "swissview" / "SwissView100" / f"swisstopo_{img_idx:02d}.jpg"
        patch_dir = PROJECT_ROOT / "data" / "swissview" / "swissview100_patches" / "patches" / f"img_{img_idx}"
    elif cfg.dataset == "swissviewmonuments":
        metadata = load_metadata()
        entry = metadata[img_idx]
        aerial_path = resolve_metadata_asset(entry["aerial_view"])
        patch_dir = PROJECT_ROOT / "data" / "swissview" / "swissviewmonuments_patches" / "patches" / f"img_{img_idx}"
    else:
        raise ValueError(f"Unsupported dataset for visualization: {cfg.dataset}")

    return aerial_path, patch_dir


def load_patch_image(aerial_image, patch_dir, patch_id):
    patch_path = patch_dir / f"patch_{patch_id}.jpg"
    if patch_path.exists():
        return np.asarray(Image.open(patch_path).convert("RGB"))

    x0, y0, patch_w, patch_h = patch_bounds(patch_id, aerial_image.size)
    crop = aerial_image.crop((int(x0), int(y0), int(x0 + patch_w), int(y0 + patch_h)))
    return np.asarray(crop)


def compute_visit_matrix(traj, grid_size=GRID_SIZE):
    matrix = np.zeros((grid_size, grid_size), dtype=int)
    for patch_id in traj:
        row, col = idx_to_row_col(patch_id, grid_size)
        matrix[row, col] += 1
    return matrix


def pick_visualization_records(records, max_vis=MAX_VIS_PER_DIST):
    successes = [record for record in records if record["success"]]
    failures = [record for record in records if not record["success"]]

    selected = successes[: max(1, max_vis - 1)]
    if len(selected) < max_vis:
        selected.extend(failures[: max_vis - len(selected)])

    if not selected:
        selected = records[:max_vis]

    return selected


def draw_global_map(ax, metadata, sample_idx):
    coords = np.array([entry.get("LV95_coordinates", [0, 0]) for entry in metadata], dtype=float)
    if coords.size == 0:
        ax.axis("off")
        ax.set_title("Global Map Unavailable")
        return

    ax.scatter(coords[:, 0], coords[:, 1], s=35, c="#c7d9c1", alpha=0.85, edgecolors="none")
    ax.scatter(
        coords[sample_idx, 0],
        coords[sample_idx, 1],
        s=180,
        c="#ff9f1c",
        marker="*",
        edgecolors="black",
        linewidths=1.0,
        zorder=3,
    )
    ax.annotate(
        f"img_{sample_idx}",
        xy=(coords[sample_idx, 0], coords[sample_idx, 1]),
        xytext=(10, 8),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
    )
    ax.set_title("SwissView Global Coordinates")
    ax.set_xlabel("LV95 X")
    ax.set_ylabel("LV95 Y")
    ax.grid(alpha=0.2)


def draw_visit_heatmap(ax, visit_matrix):
    im = ax.imshow(visit_matrix, cmap="Greens")
    for row in range(visit_matrix.shape[0]):
        for col in range(visit_matrix.shape[1]):
            ax.text(
                col,
                row,
                str(int(visit_matrix[row, col])),
                ha="center",
                va="center",
                color="white" if visit_matrix[row, col] > visit_matrix.max() / 2 else "black",
                fontsize=11,
                fontweight="bold",
            )
    ax.set_title("Visited Patches")
    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))
    ax.set_xlabel("Col")
    ax.set_ylabel("Row")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def draw_stats_panel(ax, record):
    actions = " -> ".join(record["actions"]) if record["actions"] else "No action"
    text = "\n".join(
        [
            f"Result: {'SUCCESS' if record['success'] else 'FAILED'}",
            f"Distance C: {record['distance']}",
            f"Optimal steps: {record['optimal_steps']}",
            f"Taken steps: {record['path_length'] - 1}",
            f"Start patch: {record['start']}",
            f"Goal patch: {record['goal']}",
            f"Final patch: {record['final']}",
            f"Final dist to goal: {record['final_distance']}",
            f"Actions: {actions}",
        ]
    )
    ax.axis("off")
    ax.text(
        0.0,
        1.0,
        text,
        va="top",
        ha="left",
        fontsize=11,
        linespacing=1.5,
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#f8f5ef", "edgecolor": "#d8c7a7"},
    )
    ax.set_title("Inference Summary")


def draw_patch_panel(ax, patch_image, title, border_color):
    ax.imshow(patch_image)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(3)
        spine.set_edgecolor(border_color)


def draw_aerial_panel(ax, aerial_np, record):
    aerial_height, aerial_width = aerial_np.shape[:2]
    ax.imshow(aerial_np)
    ax.set_title(f"Search Area With Trajectory | img_{record['img_idx']} | C={record['distance']}")
    ax.set_xticks([])
    ax.set_yticks([])

    patch_w = aerial_width / GRID_SIZE
    patch_h = aerial_height / GRID_SIZE

    for grid_idx in range(1, GRID_SIZE):
        ax.axvline(grid_idx * patch_w, color="white", alpha=0.35, linewidth=1.5)
        ax.axhline(grid_idx * patch_h, color="white", alpha=0.35, linewidth=1.5)

    centers_x = []
    centers_y = []
    for step_idx, patch_id in enumerate(record["traj"]):
        center_x, center_y = patch_center(patch_id, (aerial_width, aerial_height))
        centers_x.append(center_x)
        centers_y.append(center_y)
        ax.scatter(center_x, center_y, s=60, c="#2f6fed", edgecolors="white", linewidths=1.2, zorder=4)
        ax.text(
            center_x,
            center_y,
            str(step_idx),
            color="white",
            fontsize=8,
            ha="center",
            va="center",
            zorder=5,
            fontweight="bold",
        )

    if len(centers_x) > 1:
        ax.plot(centers_x, centers_y, color="#ff9f1c", linewidth=4, alpha=0.92, zorder=3)

    start_x, start_y = patch_center(record["start"], (aerial_width, aerial_height))
    goal_x0, goal_y0, goal_w, goal_h = patch_bounds(record["goal"], (aerial_width, aerial_height))
    final_x0, final_y0, final_w, final_h = patch_bounds(record["final"], (aerial_width, aerial_height))
    start_x0, start_y0, start_w, start_h = patch_bounds(record["start"], (aerial_width, aerial_height))

    ax.add_patch(Rectangle((start_x0, start_y0), start_w, start_h, fill=False, linewidth=3, edgecolor="#2fb344"))
    ax.add_patch(Rectangle((goal_x0, goal_y0), goal_w, goal_h, fill=False, linewidth=4, edgecolor="#ffd43b"))
    ax.add_patch(
        Rectangle(
            (final_x0, final_y0),
            final_w,
            final_h,
            fill=False,
            linewidth=3,
            edgecolor="#e03131" if not record["success"] else "#1971c2",
        )
    )
    ax.scatter(start_x, start_y, s=120, marker="s", c="#2fb344", edgecolors="white", linewidths=1.3, zorder=6)
    ax.text(goal_x0 + goal_w / 2, goal_y0 + goal_h / 2, "GOAL", color="#111", fontsize=11, fontweight="bold", ha="center")


def figure_to_pil(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    buffer.close()
    return image


def render_gif(record, aerial_np, current_patch_np, goal_patch_np, save_path):
    frames = []
    traj = record["traj"]

    for end_idx in range(1, len(traj) + 1):
        fig = plt.figure(figsize=(10, 5))
        gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.0], wspace=0.18)
        ax_aerial = fig.add_subplot(gs[0, 0])
        ax_patch = fig.add_subplot(gs[0, 1])

        partial = dict(record)
        partial["traj"] = traj[:end_idx]
        partial["final"] = traj[end_idx - 1]
        draw_aerial_panel(ax_aerial, aerial_np, partial)

        ax_patch.imshow(current_patch_np if end_idx == len(traj) else goal_patch_np)
        ax_patch.set_title(f"Step {end_idx - 1}")
        ax_patch.set_xticks([])
        ax_patch.set_yticks([])

        fig.suptitle(f"Active Navigation Trajectory Replay | img_{record['img_idx']}", fontsize=14, fontweight="bold")
        frames.append(figure_to_pil(fig))
        plt.close(fig)

    if frames:
        frames[0].save(
            save_path,
            save_all=True,
            append_images=frames[1:],
            duration=700,
            loop=0,
        )


def render_record(record, metadata, dist_val, save_dir):
    aerial_path, patch_dir = resolve_sample_assets(record["img_idx"])
    aerial_image = Image.open(aerial_path).convert("RGB")
    aerial_np = np.asarray(aerial_image)

    goal_patch_np = load_patch_image(aerial_image, patch_dir, record["goal"])
    start_patch_np = load_patch_image(aerial_image, patch_dir, record["start"])
    final_patch_np = load_patch_image(aerial_image, patch_dir, record["final"])
    visit_matrix = compute_visit_matrix(record["traj"])

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(
        3,
        3,
        width_ratios=[1.2, 2.4, 1.1],
        height_ratios=[1.0, 1.0, 1.0],
        wspace=0.22,
        hspace=0.28,
    )

    ax_map = fig.add_subplot(gs[0, 0])
    ax_heatmap = fig.add_subplot(gs[1, 0])
    ax_stats = fig.add_subplot(gs[2, 0])
    ax_aerial = fig.add_subplot(gs[:, 1])
    ax_goal = fig.add_subplot(gs[0, 2])
    ax_start = fig.add_subplot(gs[1, 2])
    ax_final = fig.add_subplot(gs[2, 2])

    draw_global_map(ax_map, metadata, record["img_idx"])
    draw_visit_heatmap(ax_heatmap, visit_matrix)
    draw_stats_panel(ax_stats, record)
    draw_aerial_panel(ax_aerial, aerial_np, record)
    draw_patch_panel(ax_goal, goal_patch_np, "Goal Patch", "#ffd43b")
    draw_patch_panel(ax_start, start_patch_np, "Start Patch", "#2fb344")
    draw_patch_panel(ax_final, final_patch_np, "Final Patch", "#e03131" if not record["success"] else "#1971c2")

    fig.suptitle(
        f"Active Navigation Inference Visualization | dist={dist_val} | {'SUCCESS' if record['success'] else 'FAILED'}",
        fontsize=18,
        fontweight="bold",
    )

    stem = f"dist_{dist_val}_img_{record['img_idx']}_cfg_{record['config_idx']}_{'success' if record['success'] else 'fail'}"
    png_path = save_dir / f"{stem}.png"
    gif_path = save_dir / f"{stem}.gif"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    render_gif(record, aerial_np, final_patch_np, goal_patch_np, gif_path)
    return png_path, gif_path


def visualize_records(records, dist_val, save_root):
    if not isinstance(records, list) or not records:
        print(f"[WARN] No records available for dist={dist_val}")
        return []

    metadata = load_metadata()
    dist_dir = save_root / f"dist_{dist_val}"
    dist_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for record in pick_visualization_records(records):
        outputs.append(render_record(record, metadata, dist_val, dist_dir))

    summary_path = dist_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(records[: min(len(records), 50)], f, ensure_ascii=False, indent=2)

    return outputs


if __name__ == "__main__":
    output_root = PROJECT_ROOT / "vis_results"
    output_root.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(os.path.join(cfg.train.ckpt_folder, cfg.train.expt_folder)):
        os.makedirs(os.path.join(cfg.train.ckpt_folder, cfg.train.expt_folder))

    with open(os.path.join(cfg.train.ckpt_folder, cfg.train.expt_folder, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f)

    seed_everything(cfg.train.hparams.random_seed)

    ppo_agent = PPO(
        cfg.train.hparams.lr_actor,
        cfg.train.hparams.lr_critic,
        cfg.train.hparams.lr_llm,
        cfg.train.hparams.gamma,
        cfg.train.hparams.K_epochs,
        cfg.train.hparams.eps_clip,
        cfg.train.hparams.lr_gamma,
    ).to(DEVICE)

    ppo_agent.load_state_dict(torch.load(cfg.train.checkpoint_path, map_location=DEVICE))
    ppo_agent.eval()

    valid_path = cfg.data.test_path
    print(f"Loaded checkpoint: {cfg.train.checkpoint_path}")
    print(f"Dataset: {cfg.dataset}")
    print(f"Device: {DEVICE}")
    print(f"Validation data: {valid_path}")
    if MAX_DEBUG_IMAGES:
        print(f"[DEBUG] Limiting validation to first {MAX_DEBUG_IMAGES} images")

    dataset_obj = np.load(valid_path, allow_pickle=True)[()]
    dataset_size = len(dataset_obj.keys())

    for dist_val in range(cfg.min_c, cfg.max_c):
        seed_everything(cfg.train.hparams.random_seed)

        if cfg.dataset == "swissviewmonuments":
            config = generate_config_unseen(
                valid_path,
                GOAL_PATCH_LIST,
                patch_size=cfg.data.patch_size,
                dist=dist_val,
                n_config_per_img=cfg.num_config_per_img,
            )
            cur_val_success, records = ppo_agent.validate_unseen(
                config,
                valid_path,
                n_config_per_img=cfg.num_config_per_img,
                max_images=MAX_DEBUG_IMAGES,
            )
            total_images = MAX_DEBUG_IMAGES or cfg.sample_number
        else:
            config = generate_config(
                valid_path,
                patch_size=cfg.data.patch_size,
                dist=dist_val,
                n_config_per_img=cfg.num_config_per_img,
            )
            cur_val_success, records = ppo_agent.validate(
                config,
                valid_path,
                n_config_per_img=cfg.num_config_per_img,
                max_images=MAX_DEBUG_IMAGES,
            )
            total_images = MAX_DEBUG_IMAGES or dataset_size

        total_trials = total_images * cfg.num_config_per_img
        print(f"dist={dist_val} success_ratio: {cur_val_success / max(total_trials, 1):.4f}")
        outputs = visualize_records(records, dist_val, output_root)
        for png_path, gif_path in outputs:
            print(f"[VIS] Saved {png_path}")
            print(f"[VIS] Saved {gif_path}")
        
