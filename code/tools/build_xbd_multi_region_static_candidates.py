"""Create multiple static xBD search-region candidates from local xBD archives."""

from __future__ import annotations

import csv
import io
import json
import math
import re
import sys
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from build_static_case_focus_visuals import (  # noqa: E402
    BLUE,
    CANVAS,
    CURRENT,
    GRID,
    INK,
    LIGHT,
    MUTED,
    ROUTE,
    START,
    TARGET,
    WHITE,
    F,
    build_route_layout,
    cell_box_size,
    crop_cell,
    fit_cover,
    make_distance_curve,
    manhattan,
)


ROOT = Path(r"F:\bishe\Undergraduate-Graduation-Project").resolve()
XBD_INDEX = Path(r"F:\bishe\GeoExplorer\data\xbd\processed\audit_paper_test800_local\xbd_index.csv")
OUT_DIR = ROOT / "results" / "figures" / "defense_reward_training_stage" / "static_case_focus" / "xbd_multi_region_candidates"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COUNT = 6
PANEL_SIZE = 900
PANEL_Y = 90
LEFT_X = 45
RIGHT_X = 975

DAMAGE_WEIGHT = {
    "destroyed": 8.0,
    "major-damage": 6.0,
    "minor-damage": 3.0,
    "no-damage": 0.25,
    "un-classified": 1.0,
}

COORD_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+)")


def label_member(image_member: str) -> str:
    return image_member.replace("/images/", "/labels/").replace(".png", ".json")


def parse_wkt_points(wkt: str) -> list[tuple[float, float]]:
    nums = [float(x) for x in COORD_RE.findall(wkt)]
    return list(zip(nums[0::2], nums[1::2]))


def polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]):
        area += x1 * y2 - x2 * y1
    return abs(area) / 2


def polygon_centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    if not points:
        return 0.0, 0.0
    area2 = 0.0
    cx = 0.0
    cy = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]):
        cross = x1 * y2 - x2 * y1
        area2 += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    if abs(area2) < 1e-6:
        return sum(x for x, _ in points) / len(points), sum(y for _, y in points) / len(points)
    return cx / (3 * area2), cy / (3 * area2)


def centroid_to_cell(cx: float, cy: float, width: int = 1024, height: int = 1024, grid: int = GRID) -> int:
    col = max(0, min(grid - 1, int(cx / width * grid)))
    row = max(0, min(grid - 1, int(cy / height * grid)))
    return row * grid + col


def score_post_label(label: dict) -> tuple[int, float, dict[int, dict[str, float]]]:
    width = int(label.get("metadata", {}).get("width") or 1024)
    height = int(label.get("metadata", {}).get("height") or 1024)
    cells: dict[int, dict[str, float]] = defaultdict(lambda: {"score": 0.0, "count": 0, "area": 0.0, "damage_count": 0})
    for feature in label.get("features", {}).get("xy", []):
        props = feature.get("properties", {})
        if props.get("feature_type") != "building":
            continue
        subtype = str(props.get("subtype", "no-damage"))
        points = parse_wkt_points(str(feature.get("wkt", "")))
        if not points:
            continue
        area = polygon_area(points)
        if area < 50:
            continue
        cx, cy = polygon_centroid(points)
        cell = centroid_to_cell(cx, cy, width=width, height=height)
        weight = DAMAGE_WEIGHT.get(subtype, 1.0)
        cells[cell]["score"] += weight + min(area / 5000.0, 2.0)
        cells[cell]["count"] += 1
        cells[cell]["area"] += area
        if subtype not in {"no-damage", ""}:
            cells[cell]["damage_count"] += 1
    if not cells:
        return 0, 0.0, cells
    best_cell, stats = max(cells.items(), key=lambda kv: (kv[1]["score"], kv[1]["damage_count"], kv[1]["count"]))
    return best_cell, float(stats["score"]), cells


def read_json_from_tar(tf: tarfile.TarFile, member: str) -> dict:
    handle = tf.extractfile(member)
    if handle is None:
        raise RuntimeError(f"Cannot read {member}")
    with handle:
        return json.load(handle)


def read_image_from_tar(tf: tarfile.TarFile, member: str) -> Image.Image:
    handle = tf.extractfile(member)
    if handle is None:
        raise RuntimeError(f"Cannot read {member}")
    with handle:
        data = handle.read()
    return Image.open(io.BytesIO(data)).convert("RGB")


def load_rows() -> list[dict[str, str]]:
    with XBD_INDEX.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def select_candidates(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    by_archive: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("partition") != "test":
            continue
        by_archive[row["archive_path"]].append(row)

    scored: list[dict[str, object]] = []
    for archive_path, group in by_archive.items():
        with tarfile.open(archive_path, "r") as tf:
            for row in group:
                post_label_member = label_member(row["post_member"])
                try:
                    label = read_json_from_tar(tf, post_label_member)
                    goal, score, cells = score_post_label(label)
                except Exception:
                    continue
                best_stats = cells.get(goal, {})
                if score < 5 or best_stats.get("count", 0) < 1:
                    continue
                scored.append(
                    {
                        **row,
                        "goal": goal,
                        "score": score,
                        "building_count": int(best_stats.get("count", 0)),
                        "damage_count": int(best_stats.get("damage_count", 0)),
                    }
                )

    scored.sort(
        key=lambda r: (
            -float(r["score"]),
            -int(r["damage_count"]),
            -int(r["building_count"]),
            str(r["disaster"]),
            str(r["pair_id"]),
        )
    )

    selected: list[dict[str, object]] = []
    disaster_counts: defaultdict[str, int] = defaultdict(int)
    for max_per_disaster in [1, 2, 3]:
        for row in scored:
            disaster = str(row["disaster"])
            if row in selected or disaster_counts[disaster] >= max_per_disaster:
                continue
            selected.append(row)
            disaster_counts[disaster] += 1
            if len(selected) >= COUNT:
                return selected
    return selected[:COUNT]


def draw_target_box_only(im: Image.Image, goal: int, size: tuple[int, int] = (PANEL_SIZE, PANEL_SIZE)) -> Image.Image:
    panel = fit_cover(im, size).convert("RGBA")
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    x0, y0, x1, y1 = cell_box_size(size, goal, inset=0.018)
    draw.rectangle((x0, y0, x1, y1), outline=(212, 160, 23, 255), width=4)
    return Image.alpha_composite(panel, overlay).convert("RGB")


def build_compare_figure(pre: Image.Image, post: Image.Image, goal: int, out: Path) -> Path:
    canvas = Image.new("RGB", CANVAS, WHITE)
    draw = ImageDraw.Draw(canvas)
    for im, x in [(pre, LEFT_X), (post, RIGHT_X)]:
        marked = draw_target_box_only(im, goal)
        canvas.paste(marked, (x, PANEL_Y))
        draw.rectangle((x, PANEL_Y, x + PANEL_SIZE, PANEL_Y + PANEL_SIZE), outline="#E7E7E7", width=1)
    canvas.save(out, quality=96)
    return out


def choose_start(goal: int) -> int:
    corners = [0, 4, 20, 24]
    return max(corners, key=lambda c: (manhattan(c, goal), c))


def shortest_route(start: int, goal: int, grid: int = GRID) -> list[int]:
    row, col = divmod(start, grid)
    gr, gc = divmod(goal, grid)
    route = [start]
    while row != gr:
        row += 1 if gr > row else -1
        route.append(row * grid + col)
    while col != gc:
        col += 1 if gc > col else -1
        route.append(row * grid + col)
    return route


def build_route_figure(pre: Image.Image, post: Image.Image, row: dict[str, object], out: Path) -> Path:
    goal = int(row["goal"])
    start = choose_start(goal)
    traj = shortest_route(start, goal)
    panels = [
        ("灾前目标模块", crop_cell(pre, goal), TARGET),
        ("灾后到达模块", crop_cell(post, traj[-1]), CURRENT),
        ("灾后起点模块", crop_cell(post, start), START),
        ("距离曲线", make_distance_curve(traj, goal), ROUTE),
    ]
    canvas = build_route_layout(
        title=f"xBD 多区域静态路线：{row['disaster']}",
        search_image=post,
        target_image=pre,
        traj=traj,
        start=start,
        goal=goal,
        panels=panels,
        footer_title="xBD 区域信息",
        footer_rows=[
            ("区域", str(row["pair_id"]), BLUE),
            ("目标格", str(goal), TARGET),
            ("标签依据", f"损伤建筑 {int(row['damage_count'])} / score {float(row['score']):.1f}", MUTED),
            ("路线", "→".join(str(x) for x in traj), ROUTE),
        ],
    )
    canvas.save(out, quality=96)
    return out


def extract_images_for_candidates(candidates: list[dict[str, object]]) -> list[tuple[dict[str, object], Image.Image, Image.Image]]:
    by_archive: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in candidates:
        by_archive[str(row["archive_path"])].append(row)

    items: list[tuple[dict[str, object], Image.Image, Image.Image]] = []
    for archive_path, group in by_archive.items():
        with tarfile.open(archive_path, "r") as tf:
            for row in group:
                pre = read_image_from_tar(tf, str(row["pre_member"]))
                post = read_image_from_tar(tf, str(row["post_member"]))
                items.append((row, pre, post))
    return items


def inspect(paths: Iterable[Path]) -> list[dict[str, object]]:
    rows = []
    for path in paths:
        with Image.open(path) as im:
            small = im.resize((192, 108)).convert("RGB")
            nonwhite = sum(1 for r, g, b in small.getdata() if not (r > 246 and g > 246 and b > 246))
            mark = sum(1 for r, g, b in small.getdata() if r > 170 and 110 < g < 190 and b < 90)
            rows.append({"path": str(path), "size": list(im.size), "bytes": path.stat().st_size, "nonwhite_sample": nonwhite, "mark_sample": mark})
    return rows


def main() -> None:
    rows = load_rows()
    candidates = select_candidates(rows)
    extracted = extract_images_for_candidates(candidates)
    outputs: list[Path] = []
    manifest_candidates = []
    for idx, (row, pre, post) in enumerate(extracted, start=1):
        stem = f"xbd_region_{idx:02d}_{row['pair_id']}"
        compare_out = OUT_DIR / f"{stem}_compare_target_only.png"
        route_out = OUT_DIR / f"{stem}_route_static.png"
        build_compare_figure(pre, post, int(row["goal"]), compare_out)
        build_route_figure(pre, post, row, route_out)
        outputs.extend([compare_out, route_out])
        manifest_candidates.append(
            {
                "index": idx,
                "pair_id": row["pair_id"],
                "disaster": row["disaster"],
                "goal": int(row["goal"]),
                "score": round(float(row["score"]), 3),
                "damage_count": int(row["damage_count"]),
                "building_count": int(row["building_count"]),
                "compare": str(compare_out),
                "route": str(route_out),
            }
        )
    manifest = {
        "generated_by": str(Path(__file__).resolve()),
        "source_index": str(XBD_INDEX),
        "count": len(manifest_candidates),
        "selection_rule": "test partition, diverse disasters, highest damaged-building concentration per 5x5 cell",
        "comparison_style": "no text, no fill, thin target rectangle only",
        "candidates": manifest_candidates,
        "outputs": inspect(outputs),
    }
    manifest_path = OUT_DIR / "xbd_multi_region_candidates_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
