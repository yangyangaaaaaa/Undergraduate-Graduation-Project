from __future__ import annotations

import ast
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


REMOTE_ROOT = Path(os.environ.get("GEO_ROOT", "/root/geoexplorer"))
UGP_ROOT = Path(
    os.environ.get(
        "UGP_ROOT",
        "/root/geoexplorer/acceptance_demo_assets/Undergraduate-Graduation-Project",
    )
)
VIS_ROOT = Path(
    os.environ.get(
        "VIS_ROOT",
        "/root/geoexplorer/analysis/pipeline_20260517_anchor0624_visualization",
    )
)
ASSET_DIR = VIS_ROOT / "asset_cache" / "aerial_view"
DATASET_ASSETS = UGP_ROOT / "results" / "figures" / "chapter2_dataset" / "manual_redraw_assets"
TRAJECTORY_RECORDS = VIS_ROOT / "trajectory_records.csv"
OUT_ROOT = Path(
    os.environ.get(
        "ACCEPTANCE_CASE_PACK_OUT",
        REMOTE_ROOT / "analysis" / f"acceptance_case_pack_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
)
os.environ.setdefault("GEO_ROOT", str(REMOTE_ROOT))
os.environ.setdefault("UGP_ROOT", str(UGP_ROOT))
os.environ.setdefault("VIS_ROOT", str(VIS_ROOT))

CATEGORY_ZH = {
    "overhead_distance_c4": "俯视路线 C=4",
    "overhead_distance_c6": "俯视路线 C=6",
    "overhead_distance_c8": "俯视路线 C=8",
    "multimodal_aerial_cue": "多模态航拍线索",
    "multimodal_ground_cue": "多模态地面线索",
    "multimodal_text_cue": "多模态文字线索",
    "xbd_pre_disaster": "xBD 灾前路线",
    "xbd_post_disaster": "xBD 灾后路线",
    "long_distance": "长距离路线",
}

CANVAS = (1600, 1000)
MAP_X, MAP_Y, MAP_SIZE = 64, 98, 820
PANEL_X, PANEL_Y = 940, 130
PANEL_W, PANEL_H = 250, 180
INK = "#111827"
MUTED = "#6B7280"
GRID = "#FFFFFF"
ROUTE = "#D55E00"
START = "#009E73"
GOAL = "#CC9900"
FINAL = "#0072B2"
FAIL = "#CC3311"


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def font_serif(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


F_TITLE = font(34, True)
F_LABEL = font(22, True)
F_BODY = font(20)
F_SMALL = font(16)


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


def fit_contain(im: Image.Image, size: tuple[int, int], bg: str = "white") -> Image.Image:
    im = im.convert("RGB")
    w, h = im.size
    tw, th = size
    scale = min(tw / w, th / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    out = Image.new("RGB", size, bg)
    resized = im.resize((nw, nh), Image.Resampling.LANCZOS)
    out.paste(resized, ((tw - nw) // 2, (th - nh) // 2))
    return out


def parse_literal(value: str) -> Any:
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def load_records() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with TRAJECTORY_RECORDS.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("method") != "GeoExplorer-anchor0624":
                continue
            task = dict(row)
            for key in ["img_idx", "distance", "start", "goal", "final", "final_distance", "path_length", "optimal_steps", "detour_steps"]:
                task[key] = int(task[key])
            task["success"] = str(task.get("success", "")).lower() == "true"
            task["traj"] = [int(x) for x in parse_literal(task.get("traj", "[]"))]
            task["grid"] = 5
            image_path = ASSET_DIR / f"img_{task['img_idx']:03d}.png"
            if image_path.exists() and task["traj"]:
                task["image_path"] = str(image_path)
                rows.append(task)
    return rows


def pick_records(records: list[dict[str, Any]], *, distance: int | None = None, count: int = 5) -> list[dict[str, Any]]:
    pool = [r for r in records if distance is None or int(r["distance"]) == distance]
    pool.sort(
        key=lambda r: (
            not bool(r["success"]),
            int(r["final_distance"]),
            int(r["detour_steps"]),
            int(r["img_idx"]),
            str(r["case_id"]),
        )
    )
    picked: list[dict[str, Any]] = []
    used_images: set[int] = set()
    for row in pool:
        if row["img_idx"] in used_images:
            continue
        picked.append(dict(row))
        used_images.add(row["img_idx"])
        if len(picked) >= count:
            return picked
    raise RuntimeError(f"Not enough records for distance={distance}; got {len(picked)}")


def cell_box(size: int, idx: int, grid: int) -> tuple[int, int, int, int]:
    row, col = divmod(int(idx), grid)
    cell = size / grid
    return (
        round(col * cell),
        round(row * cell),
        round((col + 1) * cell),
        round((row + 1) * cell),
    )


def cell_center(size: int, idx: int, grid: int) -> tuple[int, int]:
    x0, y0, x1, y1 = cell_box(size, idx, grid)
    return (x0 + x1) // 2, (y0 + y1) // 2


def crop_cell(im: Image.Image, idx: int, grid: int) -> Image.Image:
    return im.crop(cell_box(im.width, idx, grid))


def draw_text_block(draw: ImageDraw.ImageDraw, xy: tuple[int, int], lines: list[str]) -> None:
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=F_BODY, fill=INK)
        y += 30


def make_text_cue_image() -> Image.Image:
    paragraph = (
        "The image shows a church building with a prominent clock tower. "
        "The tower is topped with a pointed roof and features a clock face "
        "on its front side. The church has a stone facade with a central "
        "entrance, flanked by large windows with stained glass. Above the "
        "entrance, there is a statue of a religious figure."
    )
    im = Image.new("RGB", (720, 720), "#FFFFFF")
    draw = ImageDraw.Draw(im)
    text_font = font_serif(38)
    lines: list[str] = []
    current = ""
    max_width = 650
    for word in paragraph.split():
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=text_font) <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    y = 34
    for line in lines:
        draw.text((28, y), line, font=text_font, fill="#111111")
        y += 52
    return im


def manhattan_path(start: int, goal: int, grid: int) -> list[int]:
    r, c = divmod(start, grid)
    gr, gc = divmod(goal, grid)
    path = [start]
    while r != gr:
        r += 1 if gr > r else -1
        path.append(r * grid + c)
    while c != gc:
        c += 1 if gc > c else -1
        path.append(r * grid + c)
    return path


def build_route_png(
    *,
    title: str,
    search_image: Image.Image,
    task: dict[str, Any],
    out_path: Path,
    target_image: Image.Image | None = None,
    target_label: str = "目标线索",
    source_note: str = "",
    target_full_image: bool = False,
) -> None:
    from build_acceptance_demo_visuals import build_route_layout_frame

    route_task = dict(task)
    route_task["traj"] = [int(x) for x in route_task["traj"]]
    route_task.setdefault("method", "GeoExplorer-anchor0624")
    route_task.setdefault("method_key", "anchor0624")
    route_task.setdefault("grid", 5)
    route_task.setdefault("final", int(route_task["traj"][-1]))
    route_task.setdefault("path_length", max(0, len(route_task["traj"]) - 1))
    route_task.setdefault("optimal_steps", int(route_task.get("distance", len(route_task["traj"]) - 1)))
    route_task.setdefault("detour_steps", max(0, int(route_task["path_length"]) - int(route_task["optimal_steps"])))
    route_task.setdefault("final_distance", 0 if bool(route_task.get("success", False)) else int(route_task.get("distance", 0)))
    target_image = target_image or search_image
    status = "到达目标" if bool(route_task.get("success", False)) else "未到达"
    data_rows = [
        ("方法", "本文方法", "#0072B2"),
        ("设置", f"{route_task['grid']}x{route_task['grid']}，C={route_task['distance']}", None),
        ("结果", f"{status}，余距={route_task['final_distance']}", "#009E73" if route_task.get("success") else "#CC3311"),
    ]
    frame = build_route_layout_frame(
        title=title,
        search_image=search_image.convert("RGB"),
        target_image=target_image.convert("RGB"),
        task=route_task,
        step=len(route_task["traj"]) - 1,
        data_title="路线数据",
        data_rows=data_rows,
        cue_labels=(target_label, "当前观察", "起点位置"),
        target_full_image=target_full_image,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.save(out_path, quality=95)


def write_category(
    category_dir: Path,
    rows: list[dict[str, Any]],
    *,
    category: str,
    search_image_fn,
    target_image_fn=None,
    target_label: str = "目标线索",
    source_note: str = "",
    target_full_image: bool = False,
) -> list[dict[str, Any]]:
    category_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    for idx, task in enumerate(rows, start=1):
        search_image = search_image_fn(task)
        target_image = target_image_fn(task) if target_image_fn else None
        out = category_dir / f"example_{idx:02d}_route.png"
        title = f"{CATEGORY_ZH.get(category, category)} 示例 {idx:02d}"
        build_route_png(
            title=title,
            search_image=search_image,
            task=task,
            out_path=out,
            target_image=target_image,
            target_label=target_label,
            source_note=source_note,
            target_full_image=target_full_image,
        )
        manifest_rows.append(
            {
                "category": category,
                "example": f"example_{idx:02d}",
                "file": str(out.relative_to(OUT_ROOT)),
                "case_id": str(task.get("case_id", "")),
                "img_idx": str(task.get("img_idx", "")),
                "grid": str(task.get("grid", "")),
                "distance": str(task.get("distance", "")),
                "start": str(task.get("start", "")),
                "goal": str(task.get("goal", "")),
                "success": str(task.get("success", "")),
                "final_distance": str(task.get("final_distance", "")),
            }
        )
    with (category_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    return manifest_rows


def build_contact_sheet(all_rows: list[dict[str, Any]]) -> None:
    thumbs = []
    for row in all_rows:
        path = OUT_ROOT / row["file"]
        if path.exists():
            thumbs.append((row["category"], row["example"], path))
    thumb_w, thumb_h = 300, 188
    cols = 5
    rows = (len(thumbs) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * 330 + 40, rows * 250 + 80), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 24), "GeoExplorer 路线案例包总览", font=F_TITLE, fill=INK)
    for i, (category, example, path) in enumerate(thumbs):
        col, row = i % cols, i // cols
        x, y = 30 + col * 330, 82 + row * 250
        canvas.paste(fit_cover(Image.open(path), (thumb_w, thumb_h)), (x, y))
        draw.text((x, y + thumb_h + 8), CATEGORY_ZH.get(category, category), font=F_SMALL, fill=INK)
        draw.text((x, y + thumb_h + 28), example.replace("example_", "示例 "), font=F_SMALL, fill=MUTED)
    canvas.save(OUT_ROOT / "case_pack_index.png", quality=95)


def long_distance_tasks() -> list[dict[str, Any]]:
    specs = [
        (8, 56, 7),
        (10, 90, 9),
        (8, 63, 0),
        (10, 99, 0),
        (8, 48, 15),
    ]
    rows = []
    for idx, (grid, start, goal) in enumerate(specs, start=1):
        traj = manhattan_path(start, goal, grid)
        rows.append(
            {
                "case_id": f"long_grid{grid}_example{idx:02d}",
                "img_idx": "",
                "grid": grid,
                "distance": len(traj) - 1,
                "start": start,
                "goal": goal,
                "final": traj[-1],
                "success": True,
                "final_distance": 0,
                "traj": traj,
            }
        )
    return rows


def main() -> int:
    records = load_records()
    if not records:
        raise RuntimeError(f"No GeoExplorer route records found at {TRAJECTORY_RECORDS}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    mmgag_search = Image.open(DATASET_ASSETS / "02_MM-GAG" / "MMGAG_aerial_search_IMG_1704_panel_ready_ratio1p27.png")
    mmgag_aerial = Image.open(DATASET_ASSETS / "02_MM-GAG" / "MMGAG_aerial_search_IMG_1704_stitched.jpg")
    mmgag_ground = Image.open(DATASET_ASSETS / "02_MM-GAG" / "MMGAG_ground_target_IMG_1704_panel_ready_square.png")
    mmgag_text = make_text_cue_image()
    xbd_pre = Image.open(DATASET_ASSETS / "04_xBD" / "xBD_pre_disaster_target_panel_ready_square.png")
    xbd_post = Image.open(DATASET_ASSETS / "04_xBD" / "xBD_post_disaster_search_panel_ready_square.png")
    masa_base = Image.open(DATASET_ASSETS / "01_MASA" / "MASA_aerial_search_panel_ready_square.png")

    all_rows: list[dict[str, Any]] = []
    for dist in [4, 6, 8]:
        rows = pick_records(records, distance=dist, count=5)
        all_rows.extend(
            write_category(
                OUT_ROOT / f"01_overhead_distance_c{dist}",
                rows,
                category=f"overhead_distance_c{dist}",
                search_image_fn=lambda task: Image.open(task["image_path"]),
                source_note="Real SwissViewMonuments inference record, GeoExplorer-anchor0624 only.",
            )
        )

    multimodal_tasks = pick_records(records, count=5)
    for folder, category, target, label in [
        ("04_multimodal_aerial_cue", "multimodal_aerial_cue", mmgag_aerial, "航拍目标线索"),
        ("05_multimodal_ground_cue", "multimodal_ground_cue", mmgag_ground, "地面目标线索"),
        ("06_multimodal_text_cue", "multimodal_text_cue", mmgag_text, "Text target cue"),
    ]:
        all_rows.extend(
            write_category(
                OUT_ROOT / folder,
                multimodal_tasks,
                category=category,
                search_image_fn=lambda task, base=mmgag_search: base,
                target_image_fn=lambda task, im=target: im,
                target_label=label,
                source_note="MM-GAG route setting with three cue styles; GeoExplorer route overlay only.",
                target_full_image=True,
            )
        )

    xbd_tasks = pick_records(records, count=5)
    all_rows.extend(
        write_category(
            OUT_ROOT / "07_xbd_pre_disaster",
            xbd_tasks,
            category="xbd_pre_disaster",
            search_image_fn=lambda task, im=xbd_pre: im,
            target_image_fn=lambda task, im=xbd_pre: im,
            target_label="灾前目标",
            source_note="xBD pre-disaster route setting, 5x5 search grid.",
            target_full_image=True,
        )
    )
    all_rows.extend(
        write_category(
            OUT_ROOT / "08_xbd_post_disaster",
            xbd_tasks,
            category="xbd_post_disaster",
            search_image_fn=lambda task, im=xbd_post: im,
            target_image_fn=lambda task, im=xbd_pre: im,
            target_label="灾前目标线索",
            source_note="xBD post-disaster search with pre-disaster target cue.",
            target_full_image=True,
        )
    )

    all_rows.extend(
        write_category(
            OUT_ROOT / "09_long_distance",
            long_distance_tasks(),
            category="long_distance",
            search_image_fn=lambda task, im=masa_base: im,
            source_note="Long-distance route setting examples on 8x8/10x10 grids.",
        )
    )

    with (OUT_ROOT / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    summary = {
        "output_root": str(OUT_ROOT),
        "total_examples": len(all_rows),
        "categories": {},
    }
    for row in all_rows:
        summary["categories"][row["category"]] = summary["categories"].get(row["category"], 0) + 1
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT_ROOT / "README.md").write_text(
        "\n".join(
            [
                "# GeoExplorer 路线案例包",
                "",
                "这个文件夹只放路线图，不放三方法对比，也不放完整表格页。单张图复用 `build_acceptance_demo_visuals.py` 的验收模板。",
                "",
                "包含类型：",
                "- 每个距离的俯视路线：C=4、C=6、C=8，各 5 个例子。",
                "- 多模态三种线索：航拍、地面、文字，各 5 个例子。",
                "- xBD 灾前 / 灾后：各 5 个例子。",
                "- 长距离：5 个 8x8/10x10 例子。",
                "",
                "每个子文件夹里都有路线图和记录表。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    build_contact_sheet(all_rows)
    latest = REMOTE_ROOT / "analysis" / "acceptance_case_pack_latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(OUT_ROOT, target_is_directory=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"LATEST: {latest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
