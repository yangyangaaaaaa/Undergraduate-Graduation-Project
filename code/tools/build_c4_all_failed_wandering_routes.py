#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Select and render C=4 all-failed wandering route examples."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[2]
BISHE_ROOT = ROOT.parent
GEO_ROOT = BISHE_ROOT / "GeoExplorer"

TRAJ_CSV = ROOT / "results" / "tables" / "main_benchmark" / "trajectory_records.csv"
SWISSVIEW_JSON = ROOT / "geoexploxer_edit" / "SwissView" / "SwissViewMonuments.json"
SWISSVIEW_ROOT = ROOT / "geoexploxer_edit" / "SwissView"
ASSET_CACHE = GEO_ROOT / "analysis" / "pipeline_20260517_anchor0624_visualization" / "asset_cache" / "aerial_view"

OUT_DIR = (
    ROOT
    / "results"
    / "figures"
    / "defense_reward_training_stage"
    / "c4_all_failed_wandering_routes"
)
TABLE_OUT = ROOT / "results" / "tables" / "trajectory_analysis" / "c4_all_failed_wandering_candidates.csv"
REPORT_OUT = ROOT / "results" / "reports" / "c4_all_failed_wandering_routes_zh.md"
SELECT_N = 8

CANVAS = (2560, 1440)
PANEL = 520
PANEL_Y = 742
PANEL_XS = [90, 1020, 1950]
GRID_REF_X = 90
GRID_REF_Y = 190
SIM_X = 720
SIM_Y = 220
WHITE = "#FFFFFF"
INK = "#111827"
MUTED = "#64748B"
GRID = "#FFFFFF"
ROUTE_COLOR = (194, 65, 12, 236)
ROUTE_DOT = (154, 52, 18, 238)
ORANGE = "#C2410C"
ORANGE_SOFT = "#F59E0B"
START = "#009E73"
TARGET = "#16A34A"
FINAL = "#111827"
BLUE = "#0072B2"
RED = "#D55E00"
SHADOW = (15, 23, 42, 82)

METHOD_ORDER = ["GOMAA-Geo", "GeoExplorer-pristine", "GeoExplorer-anchor0624"]
METHOD_LABEL = {
    "GOMAA-Geo": "GOMAA-Geo",
    "GeoExplorer-pristine": "GeoExplorer",
    "GeoExplorer-anchor0624": "本文方法",
}


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc") if bold else Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


F_TITLE = font(54, True)
F_SUB = font(30, True)
F_METHOD = font(34, True)
F_BODY = font(25, True)
F_SMALL = font(21, True)
F_NUM = font(18, True)
F_TINY = font(16, True)


def parse_list(value: object) -> list[int]:
    if isinstance(value, list):
        return [int(x) for x in value]
    return [int(x) for x in ast.literal_eval(str(value))]


def fit_cover(im: Image.Image, size: tuple[int, int]) -> Image.Image:
    im = im.convert("RGB")
    w, h = im.size
    tw, th = size
    scale = max(tw / w, th / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = im.resize((nw, nh), Image.Resampling.LANCZOS)
    left = max(0, (nw - tw) // 2)
    top = max(0, (nh - th) // 2)
    return resized.crop((left, top, left + tw, top + th))


def crop_cell_from_map(map_im: Image.Image, idx: int, out_size: int = 180) -> Image.Image:
    return map_im.crop(cell_box(idx, size=map_im.width)).resize((out_size, out_size), Image.Resampling.LANCZOS)


def cell_feature(crop: Image.Image) -> np.ndarray:
    rgb = crop.convert("RGB").resize((32, 32), Image.Resampling.LANCZOS)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    hist_parts = []
    for channel in range(3):
        hist, _ = np.histogram(arr[:, :, channel], bins=12, range=(0.0, 1.0), density=False)
        hist_parts.append(hist.astype(np.float32))
    gray = np.asarray(ImageOps.grayscale(rgb).resize((12, 12), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
    gx = np.abs(np.diff(gray, axis=1)).mean(axis=0)
    gy = np.abs(np.diff(gray, axis=0)).mean(axis=1)
    feat = np.concatenate(
        [
            np.concatenate(hist_parts) / max(1.0, arr.shape[0] * arr.shape[1]),
            arr.mean(axis=(0, 1)),
            arr.std(axis=(0, 1)),
            gray.reshape(-1) * 0.35,
            gx * 0.8,
            gy * 0.8,
        ]
    ).astype(np.float32)
    norm = np.linalg.norm(feat)
    return feat / norm if norm else feat


def cell_similarity(map_im: Image.Image, a: int, b: int) -> float:
    fa = cell_feature(crop_cell_from_map(map_im, a, out_size=96))
    fb = cell_feature(crop_cell_from_map(map_im, b, out_size=96))
    return float(np.clip(np.dot(fa, fb), 0.0, 1.0))


def cell_center(idx: int, size: int = PANEL, grid: int = 5) -> tuple[int, int]:
    row, col = divmod(int(idx), grid)
    return int((col + 0.5) * size / grid), int((row + 0.5) * size / grid)


def manhattan(a: int, b: int, grid: int = 5) -> int:
    ar, ac = divmod(int(a), grid)
    br, bc = divmod(int(b), grid)
    return abs(ar - br) + abs(ac - bc)


def cell_box(idx: int, size: int = PANEL, grid: int = 5, pad_ratio: float = 0.04) -> tuple[int, int, int, int]:
    row, col = divmod(int(idx), grid)
    cell = size / grid
    pad = int(cell * pad_ratio)
    left = int(col * cell) + pad
    top = int(row * cell) + pad
    right = int((col + 1) * cell) - pad
    bottom = int((row + 1) * cell) - pad
    return left, top, right, bottom


def image_path_for(idx: int, metadata: list[dict[str, str]]) -> Path | None:
    cached = ASSET_CACHE / f"img_{idx:03d}.png"
    if cached.exists():
        return cached
    rel = metadata[idx].get("aerial_view", "")
    if rel:
        path = SWISSVIEW_ROOT / rel
        if path.exists():
            return path
    return None


def route_metrics(traj: list[int], final_distance: int, path_length: int, detour_steps: int) -> dict[str, float]:
    repeats = len(traj) - len(set(traj))
    immediate = sum(1 for a, _, c in zip(traj, traj[1:], traj[2:]) if a == c)
    turns = 0
    for a, b, c in zip(traj, traj[1:], traj[2:]):
        ar, ac = divmod(a, 5)
        br, bc = divmod(b, 5)
        cr, cc = divmod(c, 5)
        v1 = (br - ar, bc - ac)
        v2 = (cr - br, cc - bc)
        if v1 != v2:
            turns += 1
    return {
        "repeat_visits": float(repeats),
        "immediate_backtracks": float(immediate),
        "turns": float(turns),
        "final_distance": float(final_distance),
        "path_length": float(path_length),
        "detour_steps": float(detour_steps),
    }


def wandering_pair(traj: list[int]) -> tuple[int, int, int]:
    pingpong: dict[tuple[int, int], int] = {}
    edge_counts: dict[tuple[int, int], int] = {}
    for a, b in zip(traj, traj[1:]):
        if a == b:
            continue
        edge = tuple(sorted((int(a), int(b))))
        edge_counts[edge] = edge_counts.get(edge, 0) + 1
    for a, b, c in zip(traj, traj[1:], traj[2:]):
        if a == c and a != b:
            pair = tuple(sorted((int(a), int(b))))
            pingpong[pair] = pingpong.get(pair, 0) + 1
    if pingpong:
        pair, count = max(pingpong.items(), key=lambda item: (item[1], edge_counts.get(item[0], 0)))
        return pair[0], pair[1], int(count)
    if edge_counts:
        pair, count = max(edge_counts.items(), key=lambda item: item[1])
        return pair[0], pair[1], max(1, int(count) - 1)
    return int(traj[0]), int(traj[-1]), 0


def loop_stats(traj: list[int]) -> dict[str, object]:
    count = 0
    best_segment: list[int] = []
    best_span = 0
    best_unique = 0
    for i in range(len(traj)):
        for j in range(i + 3, len(traj)):
            if int(traj[i]) != int(traj[j]):
                continue
            segment = [int(x) for x in traj[i : j + 1]]
            unique_n = len(set(segment))
            if unique_n < 3:
                continue
            count += 1
            span = j - i
            if (unique_n, span) > (best_unique, best_span):
                best_segment = segment
                best_span = span
                best_unique = unique_n
    return {
        "loop_count": int(count),
        "loop_span": int(best_span),
        "loop_unique": int(best_unique),
        "loop_segment": best_segment,
    }


def most_target_like_cells(map_im: Image.Image, cells: list[int], goal: int, n: int = 2) -> list[tuple[int, float]]:
    unique_cells = []
    for cell in cells:
        cell = int(cell)
        if cell == int(goal) or cell in unique_cells:
            continue
        unique_cells.append(cell)
    scored = [(cell, cell_similarity(map_im, cell, goal)) for cell in unique_cells]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:n]


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fnt: ImageFont.ImageFont, fill: str = INK) -> None:
    draw.text(xy, text, font=fnt, fill=fill)


def draw_centered(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, fnt: ImageFont.ImageFont, fill: str = INK) -> None:
    box = draw.textbbox((0, 0), text, font=fnt)
    draw.text((x - (box[2] - box[0]) / 2, y), text, font=fnt, fill=fill)


def most_repeated_cell(traj: list[int], goal: int) -> tuple[int, int]:
    counts: dict[int, int] = {}
    first_seen: dict[int, int] = {}
    for step, idx in enumerate(traj):
        counts[idx] = counts.get(idx, 0) + 1
        first_seen.setdefault(idx, step)
    candidates = sorted(
        counts.items(),
        key=lambda item: (item[1], item[0] != goal, -first_seen[item[0]]),
        reverse=True,
    )
    return int(candidates[0][0]), int(candidates[0][1])


def offset_point(pt: tuple[int, int], vec: tuple[float, float], amount: float) -> tuple[int, int]:
    return int(round(pt[0] + vec[0] * amount)), int(round(pt[1] + vec[1] * amount))


def draw_round_segment(
    draw: ImageDraw.ImageDraw,
    p0: tuple[int, int],
    p1: tuple[int, int],
    *,
    fill: tuple[int, int, int, int],
    width: int,
) -> None:
    draw.line((p0, p1), fill=fill, width=width)
    r = max(1, width // 2)
    draw.ellipse((p0[0] - r, p0[1] - r, p0[0] + r, p0[1] + r), fill=fill)
    draw.ellipse((p1[0] - r, p1[1] - r, p1[0] + r, p1[1] + r), fill=fill)


def segment_offsets(route: list[int]) -> list[tuple[float, float, float]]:
    edge_totals: dict[tuple[int, int], int] = {}
    for a, b in zip(route, route[1:]):
        key = tuple(sorted((int(a), int(b))))
        edge_totals[key] = edge_totals.get(key, 0) + 1

    edge_counts: dict[tuple[int, int], int] = {}
    offsets: list[tuple[float, float, float]] = []
    for a, b in zip(route, route[1:]):
        key = tuple(sorted((int(a), int(b))))
        edge_counts[key] = edge_counts.get(key, 0) + 1
        occurrence = edge_counts[key] - 1
        total = max(1, edge_totals[key])
        ax, ay = cell_center(int(a))
        bx, by = cell_center(int(b))
        dx, dy = bx - ax, by - ay
        length = max(1.0, math.hypot(dx, dy))
        perp = (-dy / length, dx / length)
        spread = 15.0 if total <= 3 else 19.0
        amount = (occurrence - (total - 1) / 2) * spread
        offsets.append((perp[0], perp[1], amount))
    return offsets


def draw_parallel_route(draw: ImageDraw.ImageDraw, traj: list[int], color: tuple[int, int, int, int]) -> None:
    offsets = segment_offsets(traj)
    for pass_color, width in [(SHADOW, 14), (color, 8)]:
        for (a, b), (ox, oy, amount) in zip(zip(traj, traj[1:]), offsets):
            p0 = offset_point(cell_center(int(a)), (ox, oy), amount)
            p1 = offset_point(cell_center(int(b)), (ox, oy), amount)
            draw_round_segment(draw, p0, p1, fill=pass_color, width=width)


def draw_route_panel(base: Image.Image, row: pd.Series) -> Image.Image:
    traj = parse_list(row["traj"])
    route_color = ROUTE_COLOR
    im = fit_cover(base, (PANEL, PANEL)).convert("RGBA")
    overlay = Image.new("RGBA", (PANEL, PANEL), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for i in range(1, 5):
        pos = int(PANEL * i / 5)
        draw.line((pos, 0, pos, PANEL), fill=(15, 23, 42, 72), width=2)
        draw.line((0, pos, PANEL, pos), fill=(15, 23, 42, 72), width=2)

    if len(traj) >= 2:
        draw_parallel_route(draw, traj, route_color)

    seen: dict[int, int] = {}
    focus_idx, _ = most_repeated_cell(traj, int(row["goal"]))
    fx0, fy0, fx1, fy1 = cell_box(focus_idx, pad_ratio=0.08)
    gx0, gy0, gx1, gy1 = cell_box(int(row["goal"]), pad_ratio=0.08)
    draw.rounded_rectangle((fx0, fy0, fx1, fy1), radius=18, outline=(249, 115, 22, 235), width=6)
    draw.rounded_rectangle((gx0, gy0, gx1, gy1), radius=18, outline=(204, 153, 0, 245), width=6)
    for step, idx in enumerate(traj):
        seen[idx] = seen.get(idx, 0) + 1
        x, y = cell_center(idx)
        visit_i = seen[idx] - 1
        angle = visit_i * math.pi / 3
        ox = int(math.cos(angle) * min(20, 6 * visit_i))
        oy = int(math.sin(angle) * min(20, 6 * visit_i))
        px, py = x + ox, y + oy
        r = 8
        fill = ROUTE_DOT
        if step == 0:
            r = 21
            fill = START
        elif idx == int(row["goal"]):
            r = 20
            fill = TARGET
        elif idx == focus_idx:
            r = 13
            fill = ORANGE_SOFT
        elif step == len(traj) - 1:
            r = 19
            fill = FINAL
        draw.ellipse((px - r, py - r, px + r, py + r), fill=fill)

    sx, sy = cell_center(int(row["start"]))
    gx, gy = cell_center(int(row["goal"]))
    draw.ellipse((sx - 27, sy - 27, sx + 27, sy + 27), outline=(0, 112, 82, 255), width=6)
    draw.ellipse((gx - 30, gy - 30, gx + 30, gy + 30), outline=(22, 101, 52, 255), width=7)

    return Image.alpha_composite(im, overlay).convert("RGB")


def draw_module_comparison(canvas: Image.Image, draw: ImageDraw.ImageDraw, base: Image.Image, row: pd.Series, x: int) -> None:
    traj = parse_list(row["traj"])
    goal = int(row["goal"])
    focus_idx, focus_count = most_repeated_cell(traj, goal)
    map_base = fit_cover(base, (PANEL, PANEL)).convert("RGB")
    crop_size = 118
    crop_y = 1138
    left_x = x + 130
    right_x = x + 410

    focus_crop = map_base.crop(cell_box(focus_idx)).resize((crop_size, crop_size), Image.Resampling.LANCZOS)
    target_crop = map_base.crop(cell_box(goal)).resize((crop_size, crop_size), Image.Resampling.LANCZOS)
    canvas.paste(focus_crop, (left_x, crop_y))
    canvas.paste(target_crop, (right_x, crop_y))

    draw.rounded_rectangle((left_x - 5, crop_y - 5, left_x + crop_size + 5, crop_y + crop_size + 5), radius=8, outline=ORANGE, width=5)
    draw.rounded_rectangle((right_x - 5, crop_y - 5, right_x + crop_size + 5, crop_y + crop_size + 5), radius=8, outline=TARGET, width=5)
    draw_centered(draw, left_x + crop_size // 2, crop_y - 34, "重复最多模块", F_TINY, ORANGE)
    draw_centered(draw, right_x + crop_size // 2, crop_y - 34, "目标模块", F_TINY, TARGET)
    draw_centered(draw, left_x + crop_size // 2, crop_y + crop_size + 12, f"格{focus_idx} / {focus_count}次", F_TINY, MUTED)
    draw_centered(draw, right_x + crop_size // 2, crop_y + crop_size + 12, f"格{goal}", F_TINY, MUTED)


def draw_grid_reference(canvas: Image.Image, draw: ImageDraw.ImageDraw, base: Image.Image, case_row: pd.Series) -> Image.Image:
    map_im = fit_cover(base, (PANEL, PANEL)).convert("RGB")
    overlay = Image.new("RGBA", (PANEL, PANEL), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    for i in range(1, 5):
        pos = int(PANEL * i / 5)
        od.line((pos, 0, pos, PANEL), fill=(15, 23, 42, 120), width=2)
        od.line((0, pos, PANEL, pos), fill=(15, 23, 42, 120), width=2)
    for idx in range(25):
        x, y = cell_center(idx)
        label = str(idx)
        box = od.textbbox((0, 0), label, font=F_TINY)
        pad = 5
        od.rounded_rectangle((x - 25, y - 44, x + 25, y - 16), radius=6, fill=(255, 255, 255, 190))
        od.text((x - (box[2] - box[0]) / 2, y - 43), label, font=F_TINY, fill=(15, 23, 42, 230))

    start = int(case_row["start"])
    goal = int(case_row["goal"])
    a = int(case_row["wander_a"])
    b = int(case_row["wander_b"])
    for idx, label, color in [
        (start, "S", (0, 112, 82, 240)),
        (goal, "T", (22, 163, 74, 245)),
        (a, "A", (245, 158, 11, 245)),
        (b, "B", (194, 65, 12, 245)),
    ]:
        x0, y0, x1, y1 = cell_box(idx, pad_ratio=0.07)
        od.rounded_rectangle((x0, y0, x1, y1), radius=18, outline=color, width=7)
        x, y = cell_center(idx)
        od.ellipse((x - 22, y - 22, x + 22, y + 22), fill=color)
        box = od.textbbox((0, 0), label, font=F_SMALL)
        od.text((x - (box[2] - box[0]) / 2, y - (box[3] - box[1]) / 2 - 2), label, font=F_SMALL, fill=WHITE)

    out = Image.alpha_composite(map_im.convert("RGBA"), overlay).convert("RGB")
    canvas.paste(out, (GRID_REF_X, GRID_REF_Y))
    draw_centered(draw, GRID_REF_X + PANEL // 2, GRID_REF_Y + PANEL + 18, "原图 5x5 分块", F_BODY, INK)
    return map_im


def draw_similarity_pair(canvas: Image.Image, draw: ImageDraw.ImageDraw, map_im: Image.Image, case_row: pd.Series) -> None:
    target = int(case_row["goal"])
    a = int(case_row["wander_a"])
    b = int(case_row["wander_b"])
    sim_a = float(case_row["wander_a_similarity"])
    sim_b = float(case_row["wander_b_similarity"])
    similar = int(case_row["similar_cell"])
    mode = str(case_row.get("case_mode", "来回徘徊"))
    crop_size = 188
    gap = 44
    y = SIM_Y + 92
    xs = [SIM_X, SIM_X + crop_size + gap, SIM_X + (crop_size + gap) * 2]
    entries = [
        ("疑似模块 A", a, sim_a, ORANGE_SOFT),
        ("疑似模块 B", b, sim_b, ORANGE),
        ("目标模块", target, 1.0, TARGET),
    ]
    draw_text(draw, (SIM_X, SIM_Y), "疑似模块与目标模块对比", F_METHOD, INK)
    draw_text(draw, (SIM_X, SIM_Y + 45), f"来自 {METHOD_LABEL[str(case_row['selected_method'])]} 的{mode}片段；相似度越高越像目标。", F_BODY, MUTED)
    for x, (label, idx, sim, color) in zip(xs, entries):
        crop = crop_cell_from_map(map_im, idx, out_size=crop_size)
        canvas.paste(crop, (x, y))
        outline = TARGET if idx == similar else color
        draw.rounded_rectangle((x - 6, y - 6, x + crop_size + 6, y + crop_size + 6), radius=10, outline=outline, width=6)
        draw_centered(draw, x + crop_size // 2, y + crop_size + 16, f"{label} / 格{idx}", F_SMALL, INK)
        if idx != target:
            draw_centered(draw, x + crop_size // 2, y + crop_size + 50, f"与目标相似度 {sim:.2f}", F_TINY, TARGET if idx == similar else MUTED)
        else:
            draw_centered(draw, x + crop_size // 2, y + crop_size + 50, "真实目标", F_TINY, TARGET)

    if mode == "局部绕圈":
        statement = f"更像目标的是格{similar}；路线围绕格{a}/格{b}所在局部形成闭环，最终没有命中目标格{target}。"
    else:
        statement = f"更像目标的是格{similar}；路线在格{a}和格{b}之间反复徘徊，最终没有命中目标格{target}。"
    draw_text(draw, (SIM_X, SIM_Y + 385), statement, F_BODY, RED)


def select_cases() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    metadata = json.loads(SWISSVIEW_JSON.read_text(encoding="utf-8"))
    df = pd.read_csv(TRAJ_CSV)
    df = df[(df["dataset"].eq("swissviewmonuments")) & (df["distance"].eq(4))].copy()

    scored: list[dict[str, object]] = []
    groups: dict[str, pd.DataFrame] = {}
    for case_id, group in df.groupby("case_id"):
        if set(METHOD_ORDER) - set(group["method"].astype(str)):
            continue
        group = group[group["method"].isin(METHOD_ORDER)].copy()
        if bool(group["success"].astype(bool).any()):
            continue
        img_idx = int(group.iloc[0]["img_idx"])
        image_path = image_path_for(img_idx, metadata)
        if image_path is None:
            continue
        base = Image.open(image_path).convert("RGB")
        map_im = fit_cover(base, (PANEL, PANEL)).convert("RGB")

        metric_rows = []
        panel_candidates = []
        for _, row in group.iterrows():
            traj = parse_list(row["traj"])
            metrics = route_metrics(
                traj,
                final_distance=int(row["final_distance"]),
                path_length=int(row["path_length"]),
                detour_steps=int(row["detour_steps"]),
            )
            metric_rows.append(metrics)
            a, b, pair_count = wandering_pair(traj)
            goal = int(row["goal"])
            sim_a = cell_similarity(map_im, a, goal)
            sim_b = cell_similarity(map_im, b, goal)
            best_sim = max(sim_a, sim_b)
            similar_cell = a if sim_a >= sim_b else b
            d_a_goal = manhattan(a, goal)
            d_b_goal = manhattan(b, goal)
            min_d_goal = min(d_a_goal, d_b_goal)
            near_bonus = max(0, 3 - min_d_goal) * 70
            target_in_pair_penalty = 140 if goal in (a, b) else 0
            panel_score = (
                pair_count * 36
                + best_sim * 95
                + near_bonus
                + metrics["immediate_backtracks"] * 5
                + metrics["repeat_visits"] * 2
                - target_in_pair_penalty
            )
            panel_candidates.append(
                {
                    "case_mode": "来回徘徊",
                    "selected_method": str(row["method"]),
                    "wander_a": int(a),
                    "wander_b": int(b),
                    "wander_pair_count": int(pair_count),
                    "wander_a_similarity": sim_a,
                    "wander_b_similarity": sim_b,
                    "similar_cell": int(similar_cell),
                    "best_target_similarity": best_sim,
                    "wander_a_goal_dist": int(d_a_goal),
                    "wander_b_goal_dist": int(d_b_goal),
                    "min_wander_goal_dist": int(min_d_goal),
                    "loop_count": 0,
                    "loop_span": 0,
                    "loop_unique": 0,
                    "panel_score": panel_score,
                }
            )

            loop = loop_stats(traj)
            if int(loop["loop_count"]) > 0:
                target_like = most_target_like_cells(map_im, list(loop["loop_segment"]), goal, n=2)
                if len(target_like) == 1:
                    target_like.append((target_like[0][0], target_like[0][1]))
                if len(target_like) >= 2:
                    la, sim_la = target_like[0]
                    lb, sim_lb = target_like[1]
                    d_la_goal = manhattan(la, goal)
                    d_lb_goal = manhattan(lb, goal)
                    min_loop_d = min(d_la_goal, d_lb_goal)
                    loop_near_bonus = max(0, 3 - min_loop_d) * 82
                    loop_score = (
                        int(loop["loop_count"]) * 34
                        + int(loop["loop_span"]) * 7
                        + int(loop["loop_unique"]) * 13
                        + max(sim_la, sim_lb) * 100
                        + loop_near_bonus
                        + metrics["turns"] * 4
                    )
                    panel_candidates.append(
                        {
                            "case_mode": "局部绕圈",
                            "selected_method": str(row["method"]),
                            "wander_a": int(la),
                            "wander_b": int(lb),
                            "wander_pair_count": int(loop["loop_count"]),
                            "wander_a_similarity": float(sim_la),
                            "wander_b_similarity": float(sim_lb),
                            "similar_cell": int(la if sim_la >= sim_lb else lb),
                            "best_target_similarity": float(max(sim_la, sim_lb)),
                            "wander_a_goal_dist": int(d_la_goal),
                            "wander_b_goal_dist": int(d_lb_goal),
                            "min_wander_goal_dist": int(min_loop_d),
                            "loop_count": int(loop["loop_count"]),
                            "loop_span": int(loop["loop_span"]),
                            "loop_unique": int(loop["loop_unique"]),
                            "panel_score": loop_score,
                        }
                    )
        sum_back = sum(m["immediate_backtracks"] for m in metric_rows)
        sum_repeat = sum(m["repeat_visits"] for m in metric_rows)
        sum_turns = sum(m["turns"] for m in metric_rows)
        avg_final = sum(m["final_distance"] for m in metric_rows) / len(metric_rows)
        sum_detour = sum(m["detour_steps"] for m in metric_rows)
        best_panel = max(panel_candidates, key=lambda item: item["panel_score"])
        score = (
            float(best_panel["panel_score"])
            + sum_back * 2.0
            + sum_repeat * 1.5
            + sum_turns * 0.4
            + avg_final * 1.5
            + sum_detour * 0.25
        )

        ordered = group.set_index("method").loc[METHOD_ORDER].reset_index()
        groups[case_id] = ordered
        scored.append(
            {
                "case_id": case_id,
                "img_idx": img_idx,
                "start": int(group.iloc[0]["start"]),
                "goal": int(group.iloc[0]["goal"]),
                "image_path": str(image_path),
                "score": score,
                "sum_immediate_backtracks": sum_back,
                "sum_repeat_visits": sum_repeat,
                "sum_turns": sum_turns,
                "avg_final_distance": avg_final,
                "sum_detour_steps": sum_detour,
                **best_panel,
            }
        )

    scored_df = pd.DataFrame(scored).sort_values(
        ["score", "best_target_similarity", "wander_pair_count", "sum_immediate_backtracks"],
        ascending=False,
    )
    return scored_df, groups


def render_case(case_row: pd.Series, group: pd.DataFrame, rank: int) -> Path:
    base = Image.open(case_row["image_path"]).convert("RGB")
    canvas = Image.new("RGB", CANVAS, WHITE)
    draw = ImageDraw.Draw(canvas)
    mode = str(case_row.get("case_mode", "来回徘徊"))
    title = f"C=4 全失败：相似模块诱发{mode} {rank:02d}"
    subtitle = (
        f"{case_row['case_id']} | 起点 {int(case_row['start'])} -> 目标 {int(case_row['goal'])} | "
        f"{METHOD_LABEL[str(case_row['selected_method'])]} 的{mode}集中在格{int(case_row['wander_a'])}/格{int(case_row['wander_b'])}"
    )
    draw_text(draw, (84, 54), title, F_TITLE)
    draw_text(draw, (86, 126), subtitle, F_SUB, MUTED)
    map_im = draw_grid_reference(canvas, draw, base, case_row)
    draw_similarity_pair(canvas, draw, map_im, case_row)

    for x, (_, row) in zip(PANEL_XS, group.iterrows()):
        panel = draw_route_panel(base, row)
        canvas.paste(panel, (x, PANEL_Y))
        method = str(row["method"])
        metrics = route_metrics(
            parse_list(row["traj"]),
            final_distance=int(row["final_distance"]),
            path_length=int(row["path_length"]),
            detour_steps=int(row["detour_steps"]),
        )
        draw_centered(draw, x + PANEL // 2, PANEL_Y + PANEL + 18, METHOD_LABEL[method], F_METHOD, BLUE if method == "GeoExplorer-anchor0624" else INK)
        line1 = f"失败 | 10步跑满 | 最短4步 | 终距 {int(row['final_distance'])}"
        line2 = (
            f"绕行 {int(row['detour_steps'])} | 重访 {int(metrics['repeat_visits'])} | "
            f"折返 {int(metrics['immediate_backtracks'])} | 转向 {int(metrics['turns'])}"
        )
        draw_centered(draw, x + PANEL // 2, PANEL_Y + PANEL + 60, line1, F_SMALL, RED)
        draw_centered(draw, x + PANEL // 2, PANEL_Y + PANEL + 91, line2, F_TINY, MUTED)

    legend = "路线统一为沉稳橙红并按重复边平行错位；青绿=起点，绿色=目标，黑色=最终位置，琥珀框=徘徊模块。"
    draw_text(draw, (86, 1404), legend, F_SMALL, MUTED)

    out = OUT_DIR / f"c4_all_failed_wandering_{rank:02d}_{case_row['case_id']}.png"
    canvas.save(out, quality=95)
    return out


def build_contact_sheet(selected: pd.DataFrame, outputs: list[Path]) -> Path:
    thumb_w, thumb_h = 560, 315
    canvas = Image.new("RGB", (2560, 1580), WHITE)
    draw = ImageDraw.Draw(canvas)
    draw_text(draw, (84, 54), "C=4 全失败徘徊路线候选总览", F_TITLE)
    draw_text(draw, (86, 126), "优先选择目标附近来回徘徊、且徘徊模块与目标模块外观相似的全失败样例。", F_SUB, MUTED)
    positions = []
    for row_i in range(2):
        for col_i in range(4):
            positions.append((80 + col_i * 620, 220 + row_i * 650))
    for i, (row, path, pos) in enumerate(zip(selected.itertuples(index=False), outputs, positions), start=1):
        im = Image.open(path).convert("RGB").resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x, y = pos
        canvas.paste(im, (x, y))
        draw_text(draw, (x, y + thumb_h + 18), f"{i:02d}. {row.case_id}", F_BODY, INK)
        info = (
            f"{row.case_mode} A/B={int(row.wander_a)}/{int(row.wander_b)} | "
            f"近目标{int(row.min_wander_goal_dist)}格 | 相似度{float(row.best_target_similarity):.2f}"
        )
        draw_text(draw, (x, y + thumb_h + 56), info, F_SMALL, MUTED)
    out = OUT_DIR / "c4_all_failed_wandering_contact_sheet.png"
    canvas.save(out, quality=95)
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("c4_all_failed_wandering*.png"):
        old.unlink()

    scored, groups = select_cases()
    target_not_in_pair = scored["goal"].ne(scored["wander_a"]) & scored["goal"].ne(scored["wander_b"])
    near_pingpong = target_not_in_pair & scored["min_wander_goal_dist"].le(2) & scored["wander_pair_count"].ge(3)
    local_loop = (
        target_not_in_pair
        & scored["case_mode"].eq("局部绕圈")
        & scored["loop_count"].ge(1)
        & scored["loop_unique"].ge(3)
        & scored["min_wander_goal_dist"].le(3)
    )
    priority = scored[near_pingpong | local_loop].copy()
    priority = priority.sort_values(
        ["min_wander_goal_dist", "case_mode", "best_target_similarity", "wander_pair_count", "score"],
        ascending=[True, True, False, False, False],
    )
    fallback = scored[~scored["case_id"].isin(priority["case_id"])].copy()
    selected = pd.concat([priority, fallback], ignore_index=True).head(SELECT_N).copy()
    outputs: list[Path] = []
    for rank, (_, row) in enumerate(selected.iterrows(), start=1):
        outputs.append(render_case(row, groups[str(row["case_id"])], rank))
    contact = build_contact_sheet(selected, outputs)

    selected = selected.copy()
    selected["figure_path"] = [str(p) for p in outputs]
    selected.to_csv(TABLE_OUT, index=False, encoding="utf-8-sig")

    lines = [
        "# C=4 全失败徘徊路线候选",
        "",
        "筛选条件：`distance=4`，同一个 `case_id` 下 GOMAA-Geo、GeoExplorer-pristine、GeoExplorer-anchor0624 三种方法全部失败；优先选择立即折返、重复访问、转向和最终距离较高的样例。",
        "绘图更新：重复经过同一路径的线段做平行错位；每个方法面板下方单独拉出“重复最多模块”和“目标模块”进行局部对比；`GeoExplorer-pristine` 的可视标签改为 `GeoExplorer`。",
        "",
        f"- 总览图：`{contact}`",
        f"- 候选表：`{TABLE_OUT}`",
        "",
        "| Rank | Case | Start | Goal | Backtrack | Revisit | Avg FinalDist | Figure |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for rank, row in enumerate(selected.itertuples(index=False), start=1):
        lines.append(
            f"| {rank} | `{row.case_id}` | {int(row.start)} | {int(row.goal)} | "
            f"{int(row.sum_immediate_backtracks)} | {int(row.sum_repeat_visits)} | "
            f"{float(row.avg_final_distance):.2f} | `{row.figure_path}` |"
        )
    REPORT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"selected={len(selected)}")
    print(f"contact={contact}")
    print(f"table={TABLE_OUT}")
    for path in outputs:
        print(f"figure={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
