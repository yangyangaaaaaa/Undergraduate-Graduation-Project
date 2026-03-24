import csv
import json
import os
from functools import lru_cache

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

matplotlib.use("Agg")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def append_csv_row(csv_path, fieldnames, row):
    ensure_dir(os.path.dirname(csv_path))
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def write_rows_to_csv(csv_path, fieldnames, rows):
    ensure_dir(os.path.dirname(csv_path))
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(json_path, payload):
    ensure_dir(os.path.dirname(json_path))
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


@lru_cache(maxsize=None)
def load_metadata(meta_path):
    with open(meta_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_aerial_image_path(dataset_name, sample_index, cfg):
    if dataset_name == "swissview":
        meta_path = os.path.normpath(cfg.visual.swissview100_meta)
    elif dataset_name == "swissviewmonuments":
        meta_path = os.path.normpath(cfg.visual.swissviewmonuments_meta)
    else:
        raise ValueError(f"Visualization is only supported for SwissView datasets, got {dataset_name}")

    metadata = load_metadata(meta_path)
    record = metadata[sample_index]
    return os.path.normpath(os.path.join(os.path.dirname(meta_path), record["aerial_view"]))


def parse_selected_ids(raw_value, default_ids=None):
    if raw_value is None or raw_value == "":
        return list(default_ids or [])
    if isinstance(raw_value, (list, tuple)):
        return [int(item) for item in raw_value]
    return [int(item.strip()) for item in str(raw_value).split(",") if item.strip()]


def patch_bounds(index, patch_size, width, height):
    row, col = divmod(index, patch_size)
    patch_w = width / patch_size
    patch_h = height / patch_size
    return col * patch_w, row * patch_h, patch_w, patch_h


def patch_center(index, patch_size, width, height):
    x, y, patch_w, patch_h = patch_bounds(index, patch_size, width, height)
    return x + patch_w / 2.0, y + patch_h / 2.0


def draw_grid(ax, patch_size, width, height):
    patch_w = width / patch_size
    patch_h = height / patch_size
    for idx in range(1, patch_size):
        ax.axvline(idx * patch_w, color="white", linewidth=0.8, alpha=0.35)
        ax.axhline(idx * patch_h, color="white", linewidth=0.8, alpha=0.35)


def highlight_patch(ax, patch_index, patch_size, width, height, color="orange", linewidth=3):
    x, y, patch_w, patch_h = patch_bounds(patch_index, patch_size, width, height)
    ax.add_patch(Rectangle((x, y), patch_w, patch_h, fill=False, edgecolor=color, linewidth=linewidth))


def _trial_colors(num_trials):
    cmap = plt.get_cmap("tab10")
    return [cmap(idx % 10) for idx in range(num_trials)]


def _draw_start_goal(ax, start_patch, goal_patch, patch_size, width, height):
    start_x, start_y = patch_center(start_patch, patch_size, width, height)
    goal_x, goal_y = patch_center(goal_patch, patch_size, width, height)
    ax.scatter([start_x], [start_y], c="white", edgecolors="black", s=90, marker="o", linewidths=1.5, zorder=5)
    ax.scatter([goal_x], [goal_y], c="#e53935", edgecolors="black", s=130, marker="^", linewidths=1.5, zorder=6)


def _draw_search_area(ax, image_array, start_patch, goal_patch, patch_size, highlight_index=None, title="Search area"):
    height, width = image_array.shape[:2]
    ax.imshow(image_array)
    draw_grid(ax, patch_size, width, height)
    _draw_start_goal(ax, start_patch, goal_patch, patch_size, width, height)
    if highlight_index is not None:
        highlight_patch(ax, highlight_index, patch_size, width, height, color="orange")
    ax.set_title(title)
    ax.set_axis_off()


def _draw_path_panel(ax, image_array, trajectories, patch_size, start_patch, goal_patch, title="Path visualization"):
    height, width = image_array.shape[:2]
    ax.imshow(image_array)
    draw_grid(ax, patch_size, width, height)
    colors = _trial_colors(len(trajectories))
    for trial_index, trajectory in enumerate(trajectories):
        centers = [patch_center(idx, patch_size, width, height) for idx in trajectory["patch_sequence"]]
        xs = [point[0] for point in centers]
        ys = [point[1] for point in centers]
        ax.plot(xs, ys, color=colors[trial_index], linewidth=2.5, marker="o", markersize=3.5, alpha=0.9)
    _draw_start_goal(ax, start_patch, goal_patch, patch_size, width, height)
    ax.set_title(title)
    ax.set_axis_off()


def save_path_overlay(image_path, trajectories, patch_size, out_path, title=None):
    image_array = np.array(Image.open(image_path).convert("RGB"))
    start_patch = trajectories[0]["start_patch"]
    goal_patch = trajectories[0]["goal_patch"]
    fig, ax = plt.subplots(figsize=(7, 7))
    _draw_path_panel(ax, image_array, trajectories, patch_size, start_patch, goal_patch, title or "Path visualization")
    fig.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_grid_heatmap(matrix, out_path, title, cmap="viridis", highlight_index=None):
    matrix = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    image = ax.imshow(matrix, cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", color="white", fontsize=9)
    if highlight_index is not None:
        row, col = divmod(highlight_index, matrix.shape[1])
        ax.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False, edgecolor="orange", linewidth=2.5))
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_composite(image_path, trajectories, reward_matrix, patch_size, out_path, title=None):
    image_array = np.array(Image.open(image_path).convert("RGB"))
    start_patch = trajectories[0]["start_patch"]
    goal_patch = trajectories[0]["goal_patch"]
    reward_matrix = np.asarray(reward_matrix, dtype=float)
    highlight_index = int(np.argmax(reward_matrix))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    _draw_search_area(axes[0], image_array, start_patch, goal_patch, patch_size, highlight_index=highlight_index)
    _draw_path_panel(axes[1], image_array, trajectories, patch_size, start_patch, goal_patch)

    heatmap = axes[2].imshow(reward_matrix, cmap="magma")
    axes[2].set_title("Intrinsic reward")
    axes[2].set_xticks(range(reward_matrix.shape[1]))
    axes[2].set_yticks(range(reward_matrix.shape[0]))
    axes[2].add_patch(
        Rectangle(
            (highlight_index % reward_matrix.shape[1] - 0.5, highlight_index // reward_matrix.shape[1] - 0.5),
            1,
            1,
            fill=False,
            edgecolor="orange",
            linewidth=2.5,
        )
    )
    for row in range(reward_matrix.shape[0]):
        for col in range(reward_matrix.shape[1]):
            axes[2].text(col, row, f"{reward_matrix[row, col]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(heatmap, ax=axes[2], fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
