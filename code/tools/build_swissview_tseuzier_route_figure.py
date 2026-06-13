from __future__ import annotations

import ast
import csv
import json
import math
import shutil
from pathlib import Path

from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "results" / "figures" / "chapter2_dataset"
BASE_IMAGE = FIG_DIR / "search_area_5x5_grid_swissview_tseuzier_no_target.png"
OUT_IMAGE = FIG_DIR / "search_area_5x5_route_swissview_tseuzier.png"
OUT_META = FIG_DIR / "search_area_5x5_route_swissview_tseuzier_metadata.json"
TRAJECTORY_TABLE = REPO_ROOT / "results" / "tables" / "main_benchmark" / "trajectory_records.csv"
PPT_PACK_DIR = REPO_ROOT / "results" / "figures" / "ppt_candidate_pack_20260606"
PPT_CATEGORY = "01_开场_方法与数据集"

SELECTED_METHOD = "GOMAA-Geo"
SELECTED_CASE_ID = "img328_d4_s11_g03_r0"
SELECTED_IMG_IDX = "328"


def find_selected_record() -> dict[str, str]:
    with TRAJECTORY_TABLE.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (
                row.get("method") == SELECTED_METHOD
                and row.get("case_id") == SELECTED_CASE_ID
                and row.get("img_idx") == SELECTED_IMG_IDX
                and row.get("success") == "True"
            ):
                return row
    raise RuntimeError(
        f"Selected route not found: method={SELECTED_METHOD}, "
        f"case_id={SELECTED_CASE_ID}, img_idx={SELECTED_IMG_IDX}"
    )


def cell_center(index: int, width: int, height: int, grid_size: int = 5) -> tuple[float, float]:
    row, col = divmod(index, grid_size)
    return ((col + 0.5) * width / grid_size, (row + 0.5) * height / grid_size)


def scaled(points: list[tuple[float, float]], scale: int) -> list[tuple[float, float]]:
    return [(x * scale, y * scale) for x, y in points]


def draw_round_line(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    color: tuple[int, int, int, int],
    width: int,
) -> None:
    draw.line(points, fill=color, width=width, joint="curve")
    radius = width / 2
    for x, y in points:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def arrow_triangle(
    start: tuple[float, float],
    end: tuple[float, float],
    length: float,
    width: float,
) -> list[tuple[float, float]]:
    sx, sy = start
    ex, ey = end
    dx, dy = ex - sx, ey - sy
    norm = math.hypot(dx, dy)
    if norm == 0:
        return []
    ux, uy = dx / norm, dy / norm
    px, py = -uy, ux
    mx, my = sx + dx * 0.62, sy + dy * 0.62
    tip = (mx + ux * length / 2, my + uy * length / 2)
    base = (mx - ux * length / 2, my - uy * length / 2)
    return [
        tip,
        (base[0] + px * width / 2, base[1] + py * width / 2),
        (base[0] - px * width / 2, base[1] - py * width / 2),
    ]


def draw_node(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    radius: int,
    fill: tuple[int, int, int, int],
    outline: tuple[int, int, int, int],
    outline_width: int,
) -> None:
    x, y = xy
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=outline,
    )
    inner = radius - outline_width
    draw.ellipse((x - inner, y - inner, x + inner, y + inner), fill=fill)


def build_route_figure() -> dict[str, object]:
    record = find_selected_record()
    route = ast.literal_eval(record["traj"])

    base = Image.open(BASE_IMAGE).convert("RGBA")
    width, height = base.size
    points = [cell_center(int(i), width, height) for i in route]

    scale = 4
    overlay = Image.new("RGBA", (width * scale, height * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    big_points = scaled(points, scale)

    route_blue = (0, 143, 213, 255)
    white = (255, 255, 255, 245)
    shadow = (0, 0, 0, 105)
    green = (20, 162, 105, 255)
    red = (218, 52, 57, 255)

    draw_round_line(draw, big_points, shadow, 58 * scale)
    draw_round_line(draw, big_points, white, 44 * scale)
    draw_round_line(draw, big_points, route_blue, 24 * scale)

    for p0, p1 in zip(big_points, big_points[1:]):
        outer = arrow_triangle(p0, p1, length=72 * scale, width=54 * scale)
        inner = arrow_triangle(p0, p1, length=58 * scale, width=38 * scale)
        if outer and inner:
            draw.polygon(outer, fill=white)
            draw.polygon(inner, fill=route_blue)

    for point in big_points[1:-1]:
        draw_node(
            draw,
            point,
            radius=24 * scale,
            fill=(255, 255, 255, 245),
            outline=route_blue,
            outline_width=8 * scale,
        )

    draw_node(
        draw,
        big_points[0],
        radius=50 * scale,
        fill=green,
        outline=white,
        outline_width=12 * scale,
    )
    draw_node(
        draw,
        big_points[-1],
        radius=54 * scale,
        fill=red,
        outline=white,
        outline_width=12 * scale,
    )

    overlay = overlay.resize((width, height), Image.Resampling.LANCZOS)
    out = Image.alpha_composite(base, overlay).convert("RGB")
    out.save(OUT_IMAGE, quality=95)

    ppt_target_dir = PPT_PACK_DIR / PPT_CATEGORY
    ppt_target_dir.mkdir(parents=True, exist_ok=True)
    ppt_target = ppt_target_dir / OUT_IMAGE.name
    shutil.copy2(OUT_IMAGE, ppt_target)

    coordinates = [
        {
            "step": step,
            "cell_index": int(index),
            "row": int(index) // 5,
            "col": int(index) % 5,
            "x": round(points[step][0], 3),
            "y": round(points[step][1], 3),
        }
        for step, index in enumerate(route)
    ]
    metadata = {
        "output_image": str(OUT_IMAGE),
        "ppt_pack_copy": str(ppt_target),
        "base_image": str(BASE_IMAGE),
        "trajectory_table": str(TRAJECTORY_TABLE),
        "dataset": record["dataset"],
        "img_idx": int(record["img_idx"]),
        "case_id": record["case_id"],
        "method": record["method"],
        "distance": int(record["distance"]),
        "start": int(record["start"]),
        "goal": int(record["goal"]),
        "success": record["success"] == "True",
        "final_distance": int(record["final_distance"]),
        "path_length": int(record["path_length"]),
        "optimal_steps": int(record["optimal_steps"]),
        "detour_steps": int(record["detour_steps"]),
        "trajectory": route,
        "grid_mapping": "5x5 row-major cell index over the full 1500x1500 figure",
        "coordinates": coordinates,
        "visual_style": {
            "route": "solid cyan-blue with white outline and direction arrows",
            "start_marker": "green circle",
            "goal_marker": "red circle",
            "note": "Built on the no-target dashed-grid base; no standalone yellow target dot is reused.",
        },
    }
    OUT_META.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    result = build_route_figure()
    print(json.dumps(result, ensure_ascii=False, indent=2))
