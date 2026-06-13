"""Build PPT-ready static visuals for the wandering case and xBD route settings."""

from __future__ import annotations

import ast
import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(os.environ.get("UGP_ROOT", Path(__file__).resolve().parents[2])).resolve()
OUT_DIR = ROOT / "results" / "figures" / "defense_reward_training_stage" / "static_case_focus"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAJECTORY_RECORDS = ROOT / "results" / "tables" / "main_benchmark" / "trajectory_records.csv"
CANDIDATES = ROOT / "results" / "tables" / "trajectory_analysis" / "c4_all_failed_wandering_candidates.csv"
XBD_ASSETS = ROOT / "results" / "figures" / "chapter2_dataset" / "manual_redraw_assets" / "04_xBD"

CASE_ID = "img072_d4_s14_g22_r0"

CANVAS = (1920, 1080)
WHITE = "#FFFFFF"
INK = "#111827"
MUTED = "#6B7280"
LIGHT = "#E5E7EB"
FAINT = "#F8FAFC"
ROUTE = "#C66A16"
ROUTE_DARK = "#6F3B08"
START = "#047857"
TARGET = "#D4A017"
CURRENT = "#111827"
WANDER_A = "#D9480F"
WANDER_B = "#7A4E00"
BLUE = "#2563EB"
GRID = 5

ROUTE_MAP_X = 90
ROUTE_MAP_Y = 110
ROUTE_MAP_SIZE = 820
ROUTE_PANEL_X = 960
ROUTE_PANEL_Y = 104
ROUTE_PANEL_SIZE = 360
ROUTE_PANEL_GAP_X = 44
ROUTE_PANEL_GAP_Y = 40
ROUTE_LAYOUT_W = ROUTE_PANEL_X + ROUTE_PANEL_SIZE * 2 + ROUTE_PANEL_GAP_X - ROUTE_MAP_X
FOOTER_Y = 955


def font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


F = {
    "title": font(46),
    "section": font(30),
    "label": font(28),
    "body": font(25),
    "small": font(20),
    "tiny": font(17),
    "marker": font(22),
}


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def draw_text_fit(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    max_width: int,
    fnt: ImageFont.ImageFont,
    fill: str = INK,
    line_gap: int = 6,
) -> int:
    chars = list(str(text))
    lines: list[str] = []
    current = ""
    for ch in chars:
        trial = current + ch
        if current and text_size(draw, trial, fnt)[0] > max_width:
            lines.append(current)
            current = ch
        else:
            current = trial
    if current:
        lines.append(current)
    y = xy[1]
    for line in lines:
        draw.text((xy[0], y), line, font=fnt, fill=fill)
        y += text_size(draw, line, fnt)[1] + line_gap
    return y


def fit_cover(im: Image.Image, size: tuple[int, int]) -> Image.Image:
    im = im.convert("RGB")
    w, h = im.size
    tw, th = size
    scale = max(tw / w, th / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = im.resize((nw, nh), Image.Resampling.LANCZOS)
    left = (nw - tw) // 2
    top = (nh - th) // 2
    return resized.crop((left, top, left + tw, top + th))


def fit_contain(im: Image.Image, size: tuple[int, int], bg: str = WHITE) -> Image.Image:
    im = im.convert("RGB")
    w, h = im.size
    tw, th = size
    scale = min(tw / w, th / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = im.resize((nw, nh), Image.Resampling.LANCZOS)
    out = Image.new("RGB", size, bg)
    out.paste(resized, ((tw - nw) // 2, (th - nh) // 2))
    return out


def cell_box_size(size: tuple[int, int], idx: int, grid: int = GRID, inset: float = 0.02) -> tuple[int, int, int, int]:
    row, col = divmod(int(idx), grid)
    cw = size[0] / grid
    ch = size[1] / grid
    dx = int(cw * inset)
    dy = int(ch * inset)
    return (
        int(col * cw) + dx,
        int(row * ch) + dy,
        int((col + 1) * cw) - dx,
        int((row + 1) * ch) - dy,
    )


def cell_box_image(im: Image.Image, idx: int, grid: int = GRID, inset: float = 0.02) -> tuple[int, int, int, int]:
    return cell_box_size((im.width, im.height), idx, grid=grid, inset=inset)


def cell_center(size: tuple[int, int], idx: int, grid: int = GRID) -> tuple[float, float]:
    row, col = divmod(int(idx), grid)
    return (col + 0.5) * size[0] / grid, (row + 0.5) * size[1] / grid


def crop_cell(im: Image.Image, idx: int, size: int = ROUTE_PANEL_SIZE, grid: int = GRID) -> Image.Image:
    return fit_cover(im.crop(cell_box_image(im, idx, grid=grid, inset=0.015)), (size, size))


def manhattan(a: int, b: int, grid: int = GRID) -> int:
    ar, ac = divmod(int(a), grid)
    br, bc = divmod(int(b), grid)
    return abs(ar - br) + abs(ac - bc)


def parse_list(value: str) -> list[int]:
    parsed = ast.literal_eval(str(value))
    return [int(x) for x in parsed]


def load_case() -> dict[str, object]:
    candidate: dict[str, object] | None = None
    with CANDIDATES.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("case_id") == CASE_ID:
                candidate = dict(row)
                break
    if candidate is None:
        raise RuntimeError(f"Missing candidate row for {CASE_ID}")

    record: dict[str, object] | None = None
    with TRAJECTORY_RECORDS.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("case_id") == CASE_ID and row.get("method") == candidate.get("selected_method"):
                record = dict(row)
                break
    if record is None:
        raise RuntimeError(f"Missing selected trajectory for {CASE_ID}")

    merged = {**candidate, **record}
    for key in ["start", "goal", "img_idx", "path_length", "final_distance", "optimal_steps", "detour_steps"]:
        merged[key] = int(float(merged[key]))
    for key in ["wander_a", "wander_b", "wander_pair_count"]:
        merged[key] = int(float(merged[key]))
    for key in ["best_target_similarity"]:
        merged[key] = float(merged[key])
    merged["traj"] = parse_list(str(merged["traj"]))
    return merged


def draw_grid(draw: ImageDraw.ImageDraw, xy: tuple[int, int], size: tuple[int, int], grid: int = GRID) -> None:
    x, y = xy
    w, h = size
    for i in range(1, grid):
        xx = x + int(w * i / grid)
        yy = y + int(h * i / grid)
        draw.line((xx, y, xx, y + h), fill=(255, 255, 255, 170), width=2)
        draw.line((x, yy, x + w, yy), fill=(255, 255, 255, 170), width=2)


def edge_key(a: int, b: int) -> tuple[int, int]:
    return (min(a, b), max(a, b))


def normal_for_edge(a: int, b: int, size: tuple[int, int], grid: int = GRID) -> tuple[float, float]:
    low, high = edge_key(a, b)
    x1, y1 = cell_center(size, low, grid=grid)
    x2, y2 = cell_center(size, high, grid=grid)
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy) or 1.0
    return -dy / length, dx / length


def draw_arrow(draw: ImageDraw.ImageDraw, p1: tuple[float, float], p2: tuple[float, float], color: tuple[int, int, int, int]) -> None:
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < 24:
        return
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    cx, cy = x1 + dx * 0.66, y1 + dy * 0.66
    tip = (cx + ux * 15, cy + uy * 15)
    left = (cx - ux * 10 + px * 9, cy - uy * 10 + py * 9)
    right = (cx - ux * 10 - px * 9, cy - uy * 10 - py * 9)
    draw.polygon([tip, left, right], fill=color)


def draw_parallel_route(
    draw: ImageDraw.ImageDraw,
    traj: list[int],
    size: tuple[int, int],
    grid: int = GRID,
) -> None:
    counts = Counter(edge_key(a, b) for a, b in zip(traj, traj[1:]))
    seen: defaultdict[tuple[int, int], int] = defaultdict(int)
    centers = [cell_center(size, idx, grid=grid) for idx in traj]
    for i, (a, b) in enumerate(zip(traj, traj[1:])):
        key = edge_key(a, b)
        total = counts[key]
        lane = seen[key]
        seen[key] += 1
        offset = 0.0
        if total > 1:
            offset = (lane - (total - 1) / 2) * 16.0
        nx, ny = normal_for_edge(a, b, size, grid=grid)
        x1, y1 = centers[i]
        x2, y2 = centers[i + 1]
        p1 = (x1 + nx * offset, y1 + ny * offset)
        p2 = (x2 + nx * offset, y2 + ny * offset)
        draw.line((p1, p2), fill=(111, 59, 8, 150), width=15)
    seen.clear()
    for i, (a, b) in enumerate(zip(traj, traj[1:])):
        key = edge_key(a, b)
        total = counts[key]
        lane = seen[key]
        seen[key] += 1
        offset = 0.0
        if total > 1:
            offset = (lane - (total - 1) / 2) * 16.0
        nx, ny = normal_for_edge(a, b, size, grid=grid)
        x1, y1 = centers[i]
        x2, y2 = centers[i + 1]
        p1 = (x1 + nx * offset, y1 + ny * offset)
        p2 = (x2 + nx * offset, y2 + ny * offset)
        draw.line((p1, p2), fill=(198, 106, 22, 235), width=9)
        draw_arrow(draw, p1, p2, (198, 106, 22, 235))
    for idx in traj:
        x, y = cell_center(size, idx, grid=grid)
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=(198, 106, 22, 230))


def draw_cell_highlight(
    draw: ImageDraw.ImageDraw,
    idx: int,
    label: str,
    color: str,
    size: tuple[int, int],
    grid: int = GRID,
    width: int = 7,
    fill_alpha: int = 36,
) -> None:
    x0, y0, x1, y1 = cell_box_size(size, idx, grid=grid, inset=0.035)
    rgba = Image.new("RGBA", size, (0, 0, 0, 0))
    od = ImageDraw.Draw(rgba)
    rgb = tuple(int(color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
    od.rectangle((x0, y0, x1, y1), fill=(*rgb, fill_alpha), outline=(*rgb, 245), width=width)
    draw.bitmap((0, 0), rgba)
    tag_h = 32
    tag_w = max(86, text_size(draw, label, F["marker"])[0] + 22)
    tx = min(x1 - tag_w - 8, x0 + 8)
    ty = y0 + 8
    draw.rounded_rectangle((tx, ty, tx + tag_w, ty + tag_h), radius=8, fill=(*rgb, 232))
    draw.text((tx + 11, ty + 4), label, font=F["marker"], fill=WHITE)


def draw_route_map(
    base: Image.Image,
    traj: list[int],
    start: int,
    goal: int,
    *,
    wander: tuple[int, int] | None = None,
    size: tuple[int, int] = (ROUTE_MAP_SIZE, ROUTE_MAP_SIZE),
    grid: int = GRID,
) -> Image.Image:
    im = fit_cover(base, size).convert("RGBA")
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw_grid(draw, (0, 0), size, grid=grid)
    draw_cell_highlight(draw, goal, "目标", TARGET, size, grid=grid, width=8, fill_alpha=42)
    if wander is not None:
        draw_cell_highlight(draw, wander[0], "徘徊A", WANDER_A, size, grid=grid, width=7, fill_alpha=34)
        draw_cell_highlight(draw, wander[1], "徘徊B", WANDER_B, size, grid=grid, width=7, fill_alpha=30)
    draw_parallel_route(draw, traj, size, grid=grid)
    sx, sy = cell_center(size, start, grid=grid)
    fx, fy = cell_center(size, traj[-1], grid=grid)
    gx, gy = cell_center(size, goal, grid=grid)
    draw.ellipse((sx - 18, sy - 18, sx + 18, sy + 18), fill=(4, 120, 87, 245))
    draw.text((sx - 8, sy - 14), "S", font=F["small"], fill=WHITE)
    draw.ellipse((gx - 20, gy - 20, gx + 20, gy + 20), outline=(212, 160, 23, 255), width=7)
    draw.ellipse((fx - 17, fy - 17, fx + 17, fy + 17), fill=(17, 24, 39, 245))
    draw.text((fx - 8, fy - 14), "F", font=F["small"], fill=WHITE)
    return Image.alpha_composite(im, overlay).convert("RGB")


def route_distances(traj: list[int], goal: int, grid: int = GRID) -> list[int]:
    return [manhattan(idx, goal, grid=grid) for idx in traj]


def make_distance_curve(traj: list[int], goal: int, size: int = 1024, grid: int = GRID) -> Image.Image:
    distances = route_distances(traj, goal, grid=grid)
    max_step = max(1, len(distances) - 1)
    max_d = max(max(distances), 1)
    im = Image.new("RGB", (size, size), WHITE)
    draw = ImageDraw.Draw(im)
    title = "距离目标"
    tw, _ = text_size(draw, title, font(max(46, size // 18)))
    draw.text(((size - tw) / 2, int(size * 0.08)), title, font=font(max(46, size // 18)), fill=INK)
    x0, y0, x1, y1 = int(size * 0.14), int(size * 0.28), int(size * 0.88), int(size * 0.72)
    for i in range(5):
        yy = y0 + (y1 - y0) * i / 4
        draw.line((x0, yy, x1, yy), fill=LIGHT, width=max(2, size // 240))
    draw.line((x0, y0, x0, y1), fill="#9CA3AF", width=max(3, size // 170))
    draw.line((x0, y1, x1, y1), fill="#9CA3AF", width=max(3, size // 170))

    def point(i: int, d: int) -> tuple[float, float]:
        return x0 + (x1 - x0) * i / max_step, y1 - (y1 - y0) * d / max_d

    pts = [point(i, d) for i, d in enumerate(distances)]
    if len(pts) >= 2:
        draw.line(pts, fill="#D1D5DB", width=max(5, size // 80), joint="curve")
        draw.line(pts, fill=ROUTE, width=max(9, size // 52), joint="curve")
    for px, py in pts:
        r = max(8, size // 74)
        draw.ellipse((px - r, py - r, px + r, py + r), fill=ROUTE)
    px, py = pts[-1]
    r = max(14, size // 45)
    draw.ellipse((px - r, py - r, px + r, py + r), fill=CURRENT)
    draw.text((x0, y1 + int(size * 0.055)), "步骤", font=font(max(26, size // 35)), fill=MUTED)
    draw.text((x0 - int(size * 0.06), y0 - int(size * 0.02)), str(max_d), font=font(max(26, size // 35)), fill=MUTED)
    draw.text((x0 - int(size * 0.04), y1 - int(size * 0.02)), "0", font=font(max(26, size // 35)), fill=MUTED)
    summary = f"第 {len(distances) - 1} 步 / {max_step} 步    距离={distances[-1]}"
    sf = font(max(34, size // 27))
    sw, _ = text_size(draw, summary, sf)
    draw.text(((size - sw) / 2, int(size * 0.80)), summary, font=sf, fill=INK)
    footer = "曲线下降表示接近目标，反复升降表示局部徘徊"
    ff = font(max(26, size // 36))
    fw, _ = text_size(draw, footer, ff)
    draw.text(((size - fw) / 2, int(size * 0.88)), footer, font=ff, fill=MUTED)
    return im


def draw_panel_grid(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    panels: list[tuple[str, Image.Image, str]],
) -> None:
    positions = [
        (ROUTE_PANEL_X, ROUTE_PANEL_Y),
        (ROUTE_PANEL_X + ROUTE_PANEL_SIZE + ROUTE_PANEL_GAP_X, ROUTE_PANEL_Y),
        (ROUTE_PANEL_X, ROUTE_PANEL_Y + ROUTE_PANEL_SIZE + 38 + ROUTE_PANEL_GAP_Y),
        (
            ROUTE_PANEL_X + ROUTE_PANEL_SIZE + ROUTE_PANEL_GAP_X,
            ROUTE_PANEL_Y + ROUTE_PANEL_SIZE + 38 + ROUTE_PANEL_GAP_Y,
        ),
    ]
    for (label, im, color), (x, y) in zip(panels, positions):
        draw.text((x, y), label, font=F["label"], fill=color)
        panel = fit_cover(im, (ROUTE_PANEL_SIZE, ROUTE_PANEL_SIZE))
        canvas.paste(panel, (x, y + 38))
        draw.rectangle((x, y + 38, x + ROUTE_PANEL_SIZE, y + 38 + ROUTE_PANEL_SIZE), outline=LIGHT, width=2)


def draw_footer(draw: ImageDraw.ImageDraw, title: str, rows: list[tuple[str, str, str | None]]) -> None:
    x = ROUTE_MAP_X
    y = FOOTER_Y
    width = ROUTE_LAYOUT_W
    draw.line((x, y, x + width, y), fill=LIGHT, width=2)
    draw.text((x, y + 22), title, font=F["section"], fill=INK)
    item_x = x + 190
    item_w = max(190, (width - 205) // max(1, len(rows)))
    for i, (label, value, color) in enumerate(rows):
        cx = item_x + i * item_w
        draw.text((cx, y + 18), label, font=F["tiny"], fill=MUTED)
        vx = cx
        if color:
            rgb = tuple(int(color.lstrip("#")[j : j + 2], 16) for j in (0, 2, 4))
            draw.ellipse((cx, y + 52, cx + 12, y + 64), fill=rgb)
            vx += 22
        draw_text_fit(draw, (vx, y + 42), value, item_w - (vx - cx) - 8, F["body"], INK, line_gap=2)


def build_route_layout(
    *,
    title: str,
    search_image: Image.Image,
    target_image: Image.Image,
    traj: list[int],
    start: int,
    goal: int,
    panels: list[tuple[str, Image.Image, str]],
    footer_title: str,
    footer_rows: list[tuple[str, str, str | None]],
    wander: tuple[int, int] | None = None,
) -> Image.Image:
    canvas = Image.new("RGB", CANVAS, WHITE)
    draw = ImageDraw.Draw(canvas)
    draw.text((56, 34), title, font=F["title"], fill=INK)
    route_map = draw_route_map(search_image, traj, start, goal, wander=wander)
    canvas.paste(route_map, (ROUTE_MAP_X, ROUTE_MAP_Y))
    draw_panel_grid(canvas, draw, panels)
    draw_footer(draw, footer_title, footer_rows)
    return canvas


def build_wandering_case(case: dict[str, object]) -> Path:
    image_path = Path(str(case["image_path"]))
    base = Image.open(image_path).convert("RGB")
    traj = list(case["traj"])
    start = int(case["start"])
    goal = int(case["goal"])
    wander = (int(case["wander_a"]), int(case["wander_b"]))
    panels = [
        ("目标模块", crop_cell(base, goal), TARGET),
        ("徘徊模块 A", crop_cell(base, wander[0]), WANDER_A),
        ("徘徊模块 B", crop_cell(base, wander[1]), WANDER_B),
        ("距离曲线", make_distance_curve(traj, goal), ROUTE),
    ]
    route_text = "→".join(str(x) for x in traj)
    canvas = build_route_layout(
        title="短距离失败案例：局部徘徊路线",
        search_image=base,
        target_image=base,
        traj=traj,
        start=start,
        goal=goal,
        panels=panels,
        footer_title="案例信息",
        footer_rows=[
            ("方法", "GeoExplorer", BLUE),
            ("轨迹", route_text, ROUTE),
            ("徘徊", f"{wander[0]}↔{wander[1]} 重复 {int(case['wander_pair_count'])} 次", WANDER_A),
            ("结果", f"失败，最终距离={int(case['final_distance'])}", CURRENT),
        ],
        wander=wander,
    )
    out = OUT_DIR / "c4_img072_wandering_single_route_static.png"
    canvas.save(out, quality=96)
    return out


def mark_target_only(im: Image.Image, goal: int, size: tuple[int, int]) -> Image.Image:
    panel = fit_cover(im, size).convert("RGBA")
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    x0, y0, x1, y1 = cell_box_size(size, goal, inset=0.02)
    draw.rectangle((x0, y0, x1, y1), outline=(212, 160, 23, 255), width=4)
    return Image.alpha_composite(panel, overlay).convert("RGB")


def build_xbd_target_pair(pre: Image.Image, post: Image.Image, goal: int = 0) -> Path:
    canvas = Image.new("RGB", CANVAS, WHITE)
    draw = ImageDraw.Draw(canvas)
    panel_size = 850
    y = 115
    left_x = 60
    right_x = 1010
    for im, x in [(pre, left_x), (post, right_x)]:
        marked = mark_target_only(im, goal, (panel_size, panel_size))
        canvas.paste(marked, (x, y))
        draw.rectangle((x, y, x + panel_size, y + panel_size), outline="#E7E7E7", width=1)
    out = OUT_DIR / "xbd_pre_post_original_target_only.png"
    canvas.save(out, quality=96)
    return out


def build_xbd_route(
    *,
    title: str,
    search_image: Image.Image,
    target_image: Image.Image,
    start: int,
    goal: int,
    traj: list[int],
    role: str,
    cue_prefix: str,
    setting_text: str,
) -> Path:
    panels = [
        (f"{cue_prefix}目标模块", crop_cell(target_image, goal), TARGET),
        ("到达模块", crop_cell(search_image, traj[-1]), CURRENT),
        ("起点模块", crop_cell(search_image, start), START),
        ("距离曲线", make_distance_curve(traj, goal), ROUTE),
    ]
    canvas = build_route_layout(
        title=title,
        search_image=search_image,
        target_image=target_image,
        traj=traj,
        start=start,
        goal=goal,
        panels=panels,
        footer_title="xBD 路线设置",
        footer_rows=[
            ("设置", setting_text, BLUE),
            ("起点/目标", f"{start} → {goal}", TARGET),
            ("轨迹", "→".join(str(x) for x in traj), ROUTE),
            ("结果", "到达目标", START),
        ],
    )
    out = OUT_DIR / f"{role}.png"
    canvas.save(out, quality=96)
    return out


def inspect_outputs(paths: Iterable[Path]) -> list[dict[str, object]]:
    rows = []
    for path in paths:
        with Image.open(path) as im:
            stat = Image.Image.getbbox(im.convert("L"))
            rows.append(
                {
                    "path": str(path),
                    "size": list(im.size),
                    "bytes": path.stat().st_size,
                    "nonblank_bbox": list(stat) if stat else None,
                }
            )
    return rows


def main() -> None:
    case = load_case()
    pre = Image.open(XBD_ASSETS / "xBD_pre_disaster_target_panel_ready_square.png").convert("RGB")
    post = Image.open(XBD_ASSETS / "xBD_post_disaster_search_panel_ready_square.png").convert("RGB")

    outputs = [
        build_wandering_case(case),
        build_xbd_target_pair(pre, post, goal=0),
        build_xbd_route(
            title="xBD 灾前搜索路线静态图",
            search_image=pre,
            target_image=pre,
            start=20,
            goal=4,
            traj=[20, 15, 10, 5, 0, 1, 2, 3, 4],
            role="xbd_pre_route_static",
            cue_prefix="灾前",
            setting_text="灾前目标，灾前搜索",
        ),
        build_xbd_route(
            title="xBD 灾后搜索路线静态图",
            search_image=post,
            target_image=pre,
            start=24,
            goal=0,
            traj=[24, 19, 14, 9, 4, 3, 2, 1, 0],
            role="xbd_disaster_route_static",
            cue_prefix="灾前",
            setting_text="灾前目标，灾后搜索",
        ),
    ]
    manifest = {
        "generated_by": str(Path(__file__).resolve()),
        "case_id": CASE_ID,
        "outputs": inspect_outputs(outputs),
        "notes": [
            "No comparison method is drawn in the C4 wandering case.",
            "The repeated 20-21 wandering edge is rendered with parallel lanes.",
            "xBD target-pair figure marks only the target module and no route.",
        ],
    }
    manifest_path = OUT_DIR / "static_case_focus_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
