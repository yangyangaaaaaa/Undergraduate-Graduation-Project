#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build training-stage case-study figures for the mixed reward mechanism.

These figures are designed for thesis defense storytelling:
- every selected case is a real training route sample;
- compared methods use the same seed, episode, image, start patch, and goal patch;
- route panels are drawn on the real overhead image when a verified image mapping exists;
- reward/gate/PBRS are described only as training-stage signals.
"""

from __future__ import annotations

import ast
import io
import json
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
FIGURES = RESULTS / "figures" / "defense_reward_training_stage" / "case_studies"
IMAGE_CACHE = FIGURES / "image_cache"
TABLES = RESULTS / "tables" / "defense_reward_training_stage"
REPORTS = RESULTS / "reports"

TRAIN_LOG_ROOT = Path(
    r"F:\bishe\GeoExplorer\analysis\pipeline_20260603_defense_reward_training_curves\training_logs"
)
MIX_INDEX_PATH = Path(r"F:\bishe\GeoExplorer\staging\dataset_mix_20260418\masa_plus_mmgag_index.csv")
MASA_METADATA_PATH = Path(r"F:\bishe\GeoExplorer\analysis\pipeline_20260507\shared_masa_compare_round1\metadata.csv")
MMGAG_INDEX_PATH = Path(r"F:\bishe\GeoExplorer\data\mm_gag\processed\mmgag_index.csv")

TARGET_CASE_COUNT = 24
PATCH_SIZE = 5
MAX_IMAGE_WHITE_RATIO = 0.01
MAX_PATCH_WHITE_RATIO = 0.08

INK = "#111827"
MUTED = "#5B6777"
GRID = "#E5E7EB"
PAPER = "#F7F9FC"
CARD = "#FFFFFF"
BLUE = "#1764AB"
ORANGE = "#D27A20"
GREEN = "#168A63"
TEAL = "#2098A3"
RED = "#B84A48"
PURPLE = "#7C5CC4"
YELLOW = "#F2C94C"
GRAY = "#6B7280"

METHODS = [
    "external_only",
    "intrinsic_only",
    "mixed_no_gate_no_pbrs",
    "mixed_gate_only",
    "mixed_pbrs_only",
    "proposed_linear_gate_pbrs",
]

METHOD_STYLE = {
    "external_only": {"label": "仅外部奖励", "short": "外部", "color": ORANGE},
    "intrinsic_only": {"label": "仅内在奖励", "short": "内在", "color": PURPLE},
    "mixed_no_gate_no_pbrs": {"label": "外部+内在直接相加", "short": "直接相加", "color": GREEN},
    "mixed_gate_only": {"label": "外部+门控内在", "short": "门控内在", "color": TEAL},
    "mixed_pbrs_only": {"label": "外部+内在+PBRS", "short": "仅加 PBRS", "color": RED},
    "proposed_linear_gate_pbrs": {"label": "本文方法", "short": "本文方法", "color": BLUE},
}

PANEL_METHODS = [
    "external_only",
    "intrinsic_only",
    "mixed_no_gate_no_pbrs",
    "mixed_gate_only",
    "mixed_pbrs_only",
    "proposed_linear_gate_pbrs",
]


@dataclass(frozen=True)
class ImageAsset:
    dataset: str
    image_id: str
    image_path: Path
    source_note: str


def ensure_dirs() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    IMAGE_CACHE.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)


def clear_stale_case_outputs() -> None:
    for pattern in ["case_*.png", "case_*.svg", "case_study_contact_sheet.png"]:
        for path in FIGURES.glob(pattern):
            if path.is_file():
                path.unlink()


def setup_style() -> None:
    for font in [
        r"C:\Windows\Fonts\times.ttf",
        r"C:\Windows\Fonts\timesbd.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]:
        path = Path(font)
        if path.exists():
            font_manager.fontManager.addfont(str(path))
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "axes.facecolor": CARD,
            "font.family": ["Times New Roman", "SimSun"],
            "font.serif": ["Times New Roman", "SimSun"],
            "font.sans-serif": ["Times New Roman", "SimSun"],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 9.0,
        }
    )


def method_from_run(run_name: str) -> str | None:
    for method in METHODS:
        if run_name.startswith(method + "_seed"):
            return method
    return None


def seed_from_run(run_name: str) -> int | None:
    if "_seed" not in run_name:
        return None
    try:
        return int(run_name.split("_seed", 1)[1].split("_", 1)[0])
    except ValueError:
        return None


def parse_list(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, float) and math.isnan(value):
        return []
    if not isinstance(value, str) or not value.strip():
        return []
    text = value.strip()
    try:
        out = json.loads(text)
    except json.JSONDecodeError:
        try:
            out = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return []
    return list(out) if isinstance(out, (list, tuple)) else []


def read_routes() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in sorted(TRAIN_LOG_ROOT.glob("*/training_route_samples.csv")):
        method = method_from_run(path.parent.name)
        seed = seed_from_run(path.parent.name)
        if method is None or seed is None:
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["method"] = method
        df["seed"] = seed
        df["run_name"] = path.parent.name
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No route samples found under {TRAIN_LOG_ROOT}")
    data = pd.concat(rows, ignore_index=True)
    numeric_cols = [
        "episode",
        "image_index",
        "time_step",
        "run_progress",
        "optimal_steps",
        "success",
        "path_len",
        "deviation_from_opt",
        "initial_patch",
        "goal_patch",
        "final_patch",
        "initial_dist",
        "final_dist",
        "min_dist",
        "mean_dist",
        "progress_steps",
        "regress_steps",
        "stay_steps",
        "unique_positions",
        "revisit_count",
        "total_reward",
        "reward_ex_sum",
        "reward_in_gated_sum",
        "pbrs_bonus_sum",
        "gate_weight_mean",
    ]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data


def build_candidate_table(routes: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["seed", "episode", "image_index", "distance_bucket", "initial_patch", "goal_patch"]
    records: list[dict] = []
    for key, group in routes.groupby(key_cols):
        if group["method"].nunique() != len(METHODS):
            continue
        by_method = {str(row.method): row for row in group.itertuples(index=False)}
        proposed = by_method["proposed_linear_gate_pbrs"]
        if int(proposed.success) != 1:
            continue
        controls = [m for m in METHODS if m != "proposed_linear_gate_pbrs"]
        fail_count = sum(int(by_method[m].success) == 0 for m in controls)
        if fail_count < 2:
            continue
        control_final = float(np.mean([float(by_method[m].final_dist) for m in controls]))
        max_revisit = float(max(float(by_method[m].revisit_count) for m in controls))
        max_regress = float(max(float(by_method[m].regress_steps) for m in controls))
        loop_controls = sum(
            float(by_method[m].revisit_count) >= 4 or float(by_method[m].regress_steps) >= 3 for m in controls
        )
        near_miss = sum(
            int(by_method[m].success) == 0 and float(by_method[m].final_dist) <= 1 for m in controls
        )
        distance_bucket = str(proposed.distance_bucket)
        try:
            dist_value = int(distance_bucket.replace("C", ""))
        except ValueError:
            dist_value = int(proposed.optimal_steps)
        proposed_clean = (
            int(float(proposed.regress_steps) == 0)
            + int(float(proposed.revisit_count) <= 1)
            + int(float(proposed.path_len) <= float(proposed.optimal_steps) + 1)
        )
        score = (
            fail_count * 10.0
            + (dist_value - 5) * 3.0
            + control_final * 1.2
            + max_revisit
            + max_regress
            + loop_controls * 2.0
            + near_miss
            + proposed_clean * 3.0
        )
        row = {
            "score": score,
            "fail_count": fail_count,
            "loop_controls": loop_controls,
            "near_miss_controls": near_miss,
            "seed": int(key[0]),
            "episode": int(key[1]),
            "image_index": int(key[2]),
            "distance_bucket": distance_bucket,
            "initial_patch": int(key[4]),
            "goal_patch": int(key[5]),
            "run_progress": float(proposed.run_progress),
            "proposed_path_len": int(proposed.path_len),
            "proposed_regress_steps": int(proposed.regress_steps),
            "proposed_revisit_count": int(proposed.revisit_count),
            "proposed_total_reward": float(proposed.total_reward),
            "control_mean_final_dist": control_final,
            "control_max_revisit": max_revisit,
            "control_max_regress": max_regress,
        }
        for method in METHODS:
            item = by_method[method]
            row[f"{method}_success"] = int(item.success)
            row[f"{method}_final_dist"] = int(item.final_dist)
            row[f"{method}_path_len"] = int(item.path_len)
            row[f"{method}_progress_steps"] = int(item.progress_steps)
            row[f"{method}_regress_steps"] = int(item.regress_steps)
            row[f"{method}_revisit_count"] = int(item.revisit_count)
            row[f"{method}_total_reward"] = float(item.total_reward)
        records.append(row)
    candidates = pd.DataFrame(records)
    if candidates.empty:
        return candidates
    return candidates.sort_values(
        ["fail_count", "distance_bucket", "score", "run_progress"], ascending=[False, False, False, True]
    ).reset_index(drop=True)


def select_cases(candidates: pd.DataFrame, max_cases: int = TARGET_CASE_COUNT) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    candidates = candidates.copy()
    if "image_asset_available" in candidates.columns:
        candidates = candidates[candidates["image_asset_available"].astype(bool)].copy()
    if {"image_white_ratio", "image_max_patch_white_ratio"}.issubset(candidates.columns):
        candidates = candidates[
            candidates["image_white_ratio"].astype(float).le(MAX_IMAGE_WHITE_RATIO)
            & candidates["image_max_patch_white_ratio"].astype(float).le(MAX_PATCH_WHITE_RATIO)
        ].copy()
    if candidates.empty:
        return candidates
    selected = []
    seen_tasks = set()
    seen_images = set()
    priority = candidates.copy()
    priority["dist_num"] = priority["distance_bucket"].astype(str).str.replace("C", "", regex=False).astype(int)
    priority["quality_adjusted_score"] = (
        priority["score"].astype(float)
        - priority.get("image_white_ratio", 0).astype(float) * 100.0
        - priority.get("image_max_patch_white_ratio", 0).astype(float) * 15.0
    )
    all_fail = priority[priority["fail_count"].eq(5)]
    # First pass: high-impact, with reasonable diversity.
    for _, row in all_fail.sort_values(["quality_adjusted_score", "dist_num"], ascending=False).iterrows():
        task_key = (row["image_index"], row["initial_patch"], row["goal_patch"])
        if task_key in seen_tasks:
            continue
        if len(selected) < 12 and row["image_index"] in seen_images and len(seen_images) >= 8:
            continue
        selected.append(row)
        seen_tasks.add(task_key)
        seen_images.add(int(row["image_index"]))
        if len(selected) >= min(18, max_cases):
            break
    # Second pass: fill with strong C8/C7 cases, including late-training loop escapes.
    for _, row in priority.sort_values(["quality_adjusted_score", "dist_num"], ascending=False).iterrows():
        if len(selected) >= max_cases:
            break
        task_key = (row["image_index"], row["initial_patch"], row["goal_patch"])
        if task_key in seen_tasks:
            continue
        if int(row["fail_count"]) < 4:
            continue
        selected.append(row)
        seen_tasks.add(task_key)
        seen_images.add(int(row["image_index"]))
    # Final pass: fill any remaining slots.
    for _, row in priority.sort_values("quality_adjusted_score", ascending=False).iterrows():
        if len(selected) >= max_cases:
            break
        task_key = (row["image_index"], row["initial_patch"], row["goal_patch"])
        if task_key in seen_tasks:
            continue
        selected.append(row)
        seen_tasks.add(task_key)
    out = pd.DataFrame(selected).reset_index(drop=True)
    out.insert(0, "case_id", [f"case_{i:02d}" for i in range(1, len(out) + 1)])
    return out


def load_image_mapping() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mix_index = pd.read_csv(MIX_INDEX_PATH)
    masa_metadata = pd.read_csv(MASA_METADATA_PATH)
    if MMGAG_INDEX_PATH.exists():
        mmgag_index = pd.read_csv(MMGAG_INDEX_PATH)
    else:
        mmgag_index = pd.DataFrame()
    return mix_index, masa_metadata, mmgag_index


def download_url(url: str, out_path: Path) -> None:
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=45) as response:
        payload = response.read()
    out_path.write_bytes(payload)


def resolve_image_asset(
    image_index: int,
    mix_index: pd.DataFrame,
    masa_metadata: pd.DataFrame,
    mmgag_index: pd.DataFrame,
) -> ImageAsset | None:
    key = f"img_{int(image_index)}"
    match = mix_index[mix_index["merged_key"].astype(str).eq(key)]
    if match.empty:
        return None
    row = match.iloc[0]
    dataset = str(row["source_dataset"])
    source_key = str(row["source_key"])
    source_num = int(source_key.split("_", 1)[1])
    if dataset == "masa":
        train_meta = masa_metadata[masa_metadata["split"].astype(str).eq("train")].reset_index(drop=True)
        if source_num >= len(train_meta):
            return None
        meta = train_meta.iloc[source_num]
        image_id = str(meta["image_id"])
        url = str(meta["image_souce_url"])
        out_path = IMAGE_CACHE / f"masa_{image_id}.tiff"
        download_url(url, out_path)
        return ImageAsset("MASA", image_id, out_path, "MASA 训练集真实航拍图")
    if dataset == "mmgag" and not mmgag_index.empty:
        mm = mmgag_index[mmgag_index["idx"].astype(int).eq(source_num)]
        if mm.empty:
            return None
        path = Path(str(mm.iloc[0]["stitched_image"]))
        if path.exists():
            return ImageAsset("MM-GAG", str(mm.iloc[0].get("sample_id", source_key)), path, "MM-GAG 真实拼接俯视图")
    return None


def open_overhead_image(asset: ImageAsset) -> Image.Image:
    image = Image.open(asset.image_path).convert("RGB")
    image = ImageOps.exif_transpose(image)
    return image.resize((1500, 1500), Image.Resampling.BICUBIC)


def measure_image_quality(asset: ImageAsset) -> dict:
    image = open_overhead_image(asset).resize((300, 300), Image.Resampling.BICUBIC)
    arr = np.asarray(image)
    white = (arr[:, :, 0] > 245) & (arr[:, :, 1] > 245) & (arr[:, :, 2] > 245)
    patch_ratios = []
    cell = 300 // PATCH_SIZE
    for row in range(PATCH_SIZE):
        for col in range(PATCH_SIZE):
            crop = white[row * cell : (row + 1) * cell, col * cell : (col + 1) * cell]
            patch_ratios.append(float(crop.mean()))
    return {
        "image_white_ratio": float(white.mean()),
        "image_max_patch_white_ratio": float(max(patch_ratios)),
        "image_mean_patch_white_ratio": float(np.mean(patch_ratios)),
        "image_texture_std": float(arr.reshape(-1, 3).std(axis=0).mean()),
    }


def enrich_candidates_with_image_quality(
    candidates: pd.DataFrame,
    mix_index: pd.DataFrame,
    masa_metadata: pd.DataFrame,
    mmgag_index: pd.DataFrame,
) -> pd.DataFrame:
    quality_rows = []
    for image_index in sorted(candidates["image_index"].astype(int).unique()):
        asset = resolve_image_asset(image_index, mix_index, masa_metadata, mmgag_index)
        if asset is None:
            quality_rows.append(
                {
                    "image_index": image_index,
                    "image_asset_available": False,
                    "dataset": "",
                    "image_id": "",
                    "image_path": "",
                    "image_white_ratio": np.nan,
                    "image_max_patch_white_ratio": np.nan,
                    "image_mean_patch_white_ratio": np.nan,
                    "image_texture_std": np.nan,
                }
            )
            continue
        quality = measure_image_quality(asset)
        quality_rows.append(
            {
                "image_index": image_index,
                "image_asset_available": True,
                "dataset": asset.dataset,
                "image_id": asset.image_id,
                "image_path": str(asset.image_path),
                **quality,
            }
        )
    quality_df = pd.DataFrame(quality_rows)
    quality_df.to_csv(TABLES / "reward_guided_case_studies_image_quality.csv", index=False, encoding="utf-8-sig")
    return candidates.merge(quality_df, on="image_index", how="left")


def patch_xy(patch: int, patch_size: int = PATCH_SIZE) -> tuple[float, float]:
    row, col = divmod(int(patch), patch_size)
    return float(col), float(row)


def draw_grid(ax: plt.Axes) -> None:
    for line in np.arange(-0.5, PATCH_SIZE + 0.5, 1.0):
        ax.axhline(line, color="white", linewidth=0.75, alpha=0.65, zorder=2)
        ax.axvline(line, color="white", linewidth=0.75, alpha=0.65, zorder=2)
        ax.axhline(line, color="black", linewidth=0.25, alpha=0.22, zorder=2)
        ax.axvline(line, color="black", linewidth=0.25, alpha=0.22, zorder=2)


def draw_route_panel(ax: plt.Axes, image: Image.Image, row: pd.Series, method: str) -> None:
    style = METHOD_STYLE[method]
    color = style["color"]
    seq = [int(x) for x in parse_list(row["patch_sequence"])]
    if len(seq) >= 2:
        goal = int(seq[0])
        path = [int(x) for x in seq[1:]]
    else:
        goal = int(row["goal_patch"])
        path = [int(row["initial_patch"]), int(row["final_patch"])]
    success = int(row["success"]) == 1
    ax.imshow(image, extent=(-0.5, PATCH_SIZE - 0.5, PATCH_SIZE - 0.5, -0.5), zorder=0)
    ax.set_xlim(-0.5, PATCH_SIZE - 0.5)
    ax.set_ylim(PATCH_SIZE - 0.5, -0.5)
    ax.set_aspect("equal")
    draw_grid(ax)
    if path:
        xy = [patch_xy(p) for p in path]
        xs, ys = zip(*xy)
        line_width = 3.2 if method == "proposed_linear_gate_pbrs" else 2.4
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=line_width,
            alpha=0.96,
            zorder=5,
            path_effects=[pe.Stroke(linewidth=line_width + 2.4, foreground="white", alpha=0.92), pe.Normal()],
        )
        for i in range(len(xy) - 1):
            if xy[i] == xy[i + 1]:
                continue
            arrow = FancyArrowPatch(
                xy[i],
                xy[i + 1],
                arrowstyle="-|>",
                mutation_scale=12,
                color=color,
                linewidth=0,
                alpha=0.95,
                zorder=6,
                shrinkA=7,
                shrinkB=7,
            )
            arrow.set_path_effects([pe.Stroke(linewidth=2.5, foreground="white", alpha=0.9), pe.Normal()])
            ax.add_patch(arrow)
        sx, sy = xy[0]
        fx, fy = xy[-1]
        ax.scatter([sx], [sy], s=115, marker="o", color="#10B981", edgecolor="white", linewidth=1.8, zorder=8)
        ax.text(sx, sy, "起", ha="center", va="center", fontsize=9.5, fontweight="bold", color="white", zorder=9)
        final_color = "#10B981" if success else "#EF4444"
        ax.scatter([fx], [fy], s=120, marker="X", color=final_color, edgecolor="white", linewidth=1.4, zorder=8)
        ax.text(fx, fy + 0.31, "终", ha="center", va="center", fontsize=8.5, fontweight="bold", color="white", zorder=9)
    gx, gy = patch_xy(goal)
    ax.scatter([gx], [gy], s=155, marker="*", color=YELLOW, edgecolor=INK, linewidth=1.0, zorder=10)
    ax.text(gx, gy + 0.35, "目标", ha="center", va="center", fontsize=8.5, fontweight="bold", color=INK, zorder=11)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(2.0 if method == "proposed_linear_gate_pbrs" else 0.8)
        spine.set_color(color if method == "proposed_linear_gate_pbrs" else "#D1D5DB")
    state = "到达" if success else "未到达"
    title_color = BLUE if method == "proposed_linear_gate_pbrs" else INK
    ax.set_title(
        f"{style['label']} | {state} | 终距 {int(row['final_dist'])} | 回退 {int(row['regress_steps'])}",
        color=title_color,
        fontsize=10.4,
        pad=5,
    )


def plot_distance_trace(ax: plt.Axes, rows_by_method: dict[str, pd.Series]) -> None:
    for method in PANEL_METHODS:
        row = rows_by_method[method]
        dist = [float(x) for x in parse_list(row["dist_sequence"])]
        if not dist:
            continue
        style = METHOD_STYLE[method]
        is_key = method == "proposed_linear_gate_pbrs"
        x = np.arange(len(dist))
        ax.plot(
            x,
            dist,
            marker="o" if is_key else None,
            markersize=4.5,
            linewidth=3.2 if is_key else 1.7,
            color=style["color"],
            alpha=1.0 if is_key else 0.70,
            label=style["short"],
            zorder=5 if is_key else 3,
        )
    ax.set_title("距离变化：越低越接近目标")
    ax.set_xlabel("行动步")
    ax.set_ylabel("到目标的网格距离")
    ax.set_ylim(-0.15, 8.35)
    ax.set_yticks(range(0, 9, 2))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(ncol=3, frameon=False, loc="upper right")


def plot_reward_decomposition(ax: plt.Axes, proposed: pd.Series) -> None:
    ex = np.array([float(x) for x in parse_list(proposed["step_reward_ex"])], dtype=float)
    gated = np.array([float(x) for x in parse_list(proposed["step_reward_in_gated"])], dtype=float)
    pbrs = np.array([float(x) for x in parse_list(proposed["step_pbrs_bonus"])], dtype=float)
    total = np.array([float(x) for x in parse_list(proposed["step_reward_total"])], dtype=float)
    n = max(len(ex), len(gated), len(pbrs), len(total))
    if n == 0:
        ax.text(0.5, 0.5, "无奖励明细", ha="center", va="center", transform=ax.transAxes, color=MUTED)
        return
    def pad(arr: np.ndarray) -> np.ndarray:
        if len(arr) >= n:
            return arr[:n]
        return np.pad(arr, (0, n - len(arr)), constant_values=np.nan)
    ex, gated, pbrs, total = map(pad, [ex, gated, pbrs, total])
    x = np.arange(1, n + 1)
    ax.axhline(0, color="#9CA3AF", linewidth=1.0)
    ax.bar(x - 0.22, ex, width=0.22, color=RED, alpha=0.78, label="外部惩罚")
    ax.bar(x, gated, width=0.22, color=GREEN, alpha=0.82, label="门控内在")
    ax.bar(x + 0.22, pbrs, width=0.22, color=BLUE, alpha=0.82, label="PBRS方向信号")
    ax.plot(x, total, color=INK, marker="o", markersize=4.0, linewidth=2.0, label="总奖励")
    ax.set_title("奖励分解：本文方法的训练信号")
    ax.set_xlabel("行动步")
    ax.set_ylabel("每步奖励")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(ncol=2, frameon=False, loc="best")


def add_case_header(fig: plt.Figure, case_num: int, case_row: pd.Series, asset: ImageAsset) -> None:
    fail_count = int(case_row["fail_count"])
    progress = float(case_row["run_progress"]) * 100
    dist = str(case_row["distance_bucket"])
    title = f"训练案例 {case_num:02d}：本文方法到达目标，{fail_count} 个对照方法未到达"
    subtitle = (
        f"真实训练样本 | {asset.source_note} | 图像 {int(case_row['image_index'])} ({asset.image_id}) | "
        f"seed={int(case_row['seed'])}, episode={int(case_row['episode'])}, 训练进度={progress:.1f}%, "
        f"{dist}, 起点={int(case_row['initial_patch'])}, 目标={int(case_row['goal_patch'])}"
    )
    fig.text(0.035, 0.968, title, ha="left", va="top", fontsize=22, fontweight="bold", color=INK)
    fig.text(0.035, 0.932, subtitle, ha="left", va="top", fontsize=10.6, color=MUTED)
    fig.lines.append(plt.Line2D([0.035, 0.965], [0.902, 0.902], transform=fig.transFigure, color="#CBD5E1", lw=1.2))
    box_specs = [
        ("同一任务", "六种奖励设置共用同一图像、起点和目标"),
        ("本文方法", f"{int(case_row['proposed_path_len'])} 步到达，回退 {int(case_row['proposed_regress_steps'])} 次"),
        ("对照方法", f"平均终距 {float(case_row['control_mean_final_dist']):.1f}，最多重复 {int(case_row['control_max_revisit'])} 次"),
    ]
    x0s = [0.035, 0.365, 0.675]
    for x0, (head, body) in zip(x0s, box_specs):
        fig.text(
            x0,
            0.872,
            f"{head}：{body}",
            ha="left",
            va="center",
            fontsize=10.4,
            bbox=dict(boxstyle="round,pad=0.32,rounding_size=0.06", facecolor="white", edgecolor="#D1D5DB"),
        )


def build_case_figure(
    case_num: int,
    case_row: pd.Series,
    rows_by_method: dict[str, pd.Series],
    asset: ImageAsset,
) -> Path:
    image = open_overhead_image(asset)
    fig = plt.figure(figsize=(15.4, 11.6))
    add_case_header(fig, case_num, case_row, asset)
    gs = GridSpec(
        4,
        3,
        figure=fig,
        left=0.045,
        right=0.965,
        top=0.835,
        bottom=0.065,
        wspace=0.10,
        hspace=0.31,
        height_ratios=[1.0, 1.0, 0.78, 0.06],
    )
    for idx, method in enumerate(PANEL_METHODS):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        draw_route_panel(ax, image, rows_by_method[method], method)
    ax_dist = fig.add_subplot(gs[2, :2])
    plot_distance_trace(ax_dist, rows_by_method)
    ax_reward = fig.add_subplot(gs[2, 2])
    plot_reward_decomposition(ax_reward, rows_by_method["proposed_linear_gate_pbrs"])
    fig.text(
        0.045,
        0.028,
        "说明：奖励、门控和 PBRS 仅在训练阶段提供学习信号；正式测试阶段只加载训练后的策略 checkpoint。",
        fontsize=10.2,
        color=MUTED,
        ha="left",
        va="bottom",
    )
    stem = f"{case_row['case_id']}_seed{int(case_row['seed'])}_ep{int(case_row['episode'])}_img{int(case_row['image_index'])}_{case_row['distance_bucket']}_s{int(case_row['initial_patch'])}_g{int(case_row['goal_patch'])}"
    png_path = FIGURES / f"{stem}.png"
    svg_path = FIGURES / f"{stem}.svg"
    fig.savefig(png_path, dpi=260, bbox_inches="tight", pad_inches=0.16)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    return png_path


def build_index_contact_sheet(case_paths: list[Path], selected: pd.DataFrame) -> Path | None:
    if not case_paths:
        return None
    thumb_w, thumb_h = 420, 316
    cols = 4
    rows = math.ceil(len(case_paths) / cols)
    sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), "#F7F9FC")
    for i, path in enumerate(case_paths):
        img = Image.open(path).convert("RGB")
        img.thumbnail((thumb_w - 12, thumb_h - 40), Image.Resampling.LANCZOS)
        x = (i % cols) * thumb_w + 6
        y = (i // cols) * thumb_h + 34
        sheet.paste(img, (x, y))
    out = FIGURES / "case_study_contact_sheet.png"
    sheet.save(out, quality=95)
    return out


def write_report(selected: pd.DataFrame, case_paths: list[Path]) -> None:
    selected_path = TABLES / "reward_guided_case_studies_selected.csv"
    lines = [
        "# 训练阶段奖励引导典型案例说明",
        "",
        "本批图用于答辩解释混合奖励机制的训练阶段作用。所有案例均来自 `training_route_samples.csv` 的真实训练采样记录，并且比较的是同一 seed、同一 episode、同一图像、同一起点和同一目标。",
        "",
        "图中路线叠加在实际俯视图上；底部距离曲线展示每一步是否靠近目标；奖励分解只展示本文方法在训练阶段收到的外部惩罚、门控内在奖励、PBRS 方向信号和总奖励。",
        "",
        "重要表述：奖励、距离门控和 PBRS 只用于训练阶段指导 PPO 学习。正式测试或论文表格评估时，不再调用奖励函数，而是加载训练好的策略 checkpoint 选择动作。",
        "",
        f"- 选中案例数：{len(selected)}",
        f"- 图像质量筛选：整图白色空白占比 ≤ {MAX_IMAGE_WHITE_RATIO:.2f}，单个网格白色空白占比 ≤ {MAX_PATCH_WHITE_RATIO:.2f}",
        f"- 选中表：`{selected_path}`",
        f"- 图像质量表：`{TABLES / 'reward_guided_case_studies_image_quality.csv'}`",
        f"- 图片目录：`{FIGURES}`",
        "",
        "## 推荐讲解顺序",
        "",
    ]
    for _, row in selected.head(12).iterrows():
        case_id = str(row["case_id"])
        fail_count = int(row["fail_count"])
        lines.append(
            f"- `{case_id}`：{row['distance_bucket']}，训练进度 {float(row['run_progress']) * 100:.1f}%；"
            f"本文方法到达目标，{fail_count} 个对照未到达，适合说明奖励信号能把中长距离动作推向正确方向。"
        )
    lines.extend(["", "## 输出图片", ""])
    for path in case_paths:
        lines.append(f"- `{path}`")
    (REPORTS / "reward_guided_case_studies_zh.md").write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> int:
    ensure_dirs()
    clear_stale_case_outputs()
    setup_style()
    routes = read_routes()
    candidates = build_candidate_table(routes)
    if candidates.empty:
        raise RuntimeError("No suitable case candidates found.")
    mix_index, masa_metadata, mmgag_index = load_image_mapping()
    candidates = enrich_candidates_with_image_quality(candidates, mix_index, masa_metadata, mmgag_index)
    candidates.to_csv(TABLES / "reward_guided_case_studies_candidates.csv", index=False, encoding="utf-8-sig")
    selected = select_cases(candidates, TARGET_CASE_COUNT)
    selected_records = []
    case_paths: list[Path] = []
    key_cols = ["seed", "episode", "image_index", "distance_bucket", "initial_patch", "goal_patch"]
    for case_num, (_, case_row) in enumerate(selected.iterrows(), start=1):
        asset = resolve_image_asset(int(case_row["image_index"]), mix_index, masa_metadata, mmgag_index)
        if asset is None:
            continue
        mask = np.ones(len(routes), dtype=bool)
        for col in key_cols:
            mask &= routes[col].astype(str).eq(str(case_row[col])).to_numpy()
        case_routes = routes.loc[mask].copy()
        if case_routes["method"].nunique() != len(METHODS):
            continue
        rows_by_method = {method: case_routes[case_routes["method"].eq(method)].iloc[0] for method in METHODS}
        png_path = build_case_figure(case_num, case_row, rows_by_method, asset)
        row = dict(case_row)
        row.update(
            {
                "dataset": asset.dataset,
                "image_id": asset.image_id,
                "image_path": str(asset.image_path),
                "figure_path": str(png_path),
            }
        )
        selected_records.append(row)
        case_paths.append(png_path)
    selected_out = pd.DataFrame(selected_records)
    selected_out.to_csv(TABLES / "reward_guided_case_studies_selected.csv", index=False, encoding="utf-8-sig")
    contact_sheet = build_index_contact_sheet(case_paths, selected_out)
    write_report(selected_out, case_paths)
    print(
        json.dumps(
            {
                "candidate_count": int(len(candidates)),
                "selected_count": int(len(selected_out)),
                "figures": [str(p) for p in case_paths],
                "contact_sheet": str(contact_sheet) if contact_sheet else None,
                "candidate_table": str(TABLES / "reward_guided_case_studies_candidates.csv"),
                "selected_table": str(TABLES / "reward_guided_case_studies_selected.csv"),
                "report": str(REPORTS / "reward_guided_case_studies_zh.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
