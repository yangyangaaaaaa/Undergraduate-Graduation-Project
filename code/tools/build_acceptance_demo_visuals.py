"""Build aligned white-background acceptance visuals for defense review.

The package is intentionally visual-first:
- route GIFs show the search process;
- xBD and MM-GAG pages show experiment settings as route maps;
- benchmark/ablation/parameter pages turn tables into reviewable evidence.

No blue frames, no photo-card borders, and no stretched non-square cue crops.
"""

from __future__ import annotations

import ast
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(os.environ.get("UGP_ROOT", Path(__file__).resolve().parents[2])).resolve()
BISHE_ROOT = Path(os.environ.get("BISHE_ROOT", ROOT.parents[0])).resolve()
GEO_ROOT = Path(os.environ.get("GEO_ROOT", BISHE_ROOT / "GeoExplorer")).resolve()


def first_existing(*paths: Path) -> Path:
    for path in paths:
        if Path(path).exists():
            return Path(path).resolve()
    return Path(paths[0]).resolve()

VIS_ROOT = Path(
    os.environ.get("VIS_ROOT", GEO_ROOT / "analysis" / "pipeline_20260517_anchor0624_visualization")
).resolve()
ASSET_DIR = VIS_ROOT / "asset_cache" / "aerial_view"
CASE_JSON = ROOT / "results" / "tables" / "main_benchmark" / "selected_visualization_cases.json"
MAIN_TABLE = ROOT / "results" / "tables" / "main_benchmark" / "paper_baseline_compare_table.csv"
TRAJECTORY_RECORDS = VIS_ROOT / "trajectory_records.csv"
DATASET_ASSETS = ROOT / "results" / "figures" / "chapter2_dataset" / "manual_redraw_assets"
XBD_ASSETS = DATASET_ASSETS / "04_xBD"

GENERALIZATION_TABLE = first_existing(
    GEO_ROOT / "analysis" / "pipeline_20260516_anchor0624_factorial_generalization" / "anchor0624_generalization_table.csv",
    ROOT / "results" / "tables" / "ablation" / "anchor0624_generalization_table.csv",
)
DATASET_SR_TABLE = first_existing(
    GEO_ROOT / "analysis" / "pipeline_20260519_appendix_dataset_param_compare" / "appendix_dataset_sr_table.csv",
    ROOT / "results" / "tables" / "appendix" / "appendix_dataset_sr_table.csv",
)
GATE_VALDIST_TABLE = first_existing(
    GEO_ROOT / "analysis" / "pipeline_20260519_appendix_gate_valdist_dense_followup" / "appendix_gate_valdist_sr_table.csv",
    ROOT / "results" / "tables" / "appendix" / "appendix_gate_valdist_sr_table.csv",
)
REWARD_GATE_TABLE = first_existing(
    GEO_ROOT / "analysis" / "pipeline_20260519_appendix_gate_valdist_dense_followup" / "appendix_reward_gate_type_mmgag_only_table_with_linear.csv",
    ROOT / "results" / "tables" / "ablation" / "reward_gate_type_mmgag_only_table_with_linear.csv",
)
BUDGET_TABLE = first_existing(
    GEO_ROOT / "analysis" / "pipeline_20260524_p0_supplement_eval" / "budget_sensitivity_table.csv",
    ROOT / "results" / "tables" / "supplement_eval" / "budget_sensitivity_table.csv",
)
CONTINUATION_STATUS = Path(
    os.environ.get(
        "CONTINUATION_STATUS",
        GEO_ROOT / "analysis" / "pipeline_20260606_dense_continuation_monitor_status.json",
    )
).resolve()

OUT_DIR = Path(os.environ.get("ACCEPTANCE_OUT_DIR", ROOT / "results" / "figures" / "acceptance_demo")).resolve()
REPORT_PATH = Path(
    os.environ.get("ACCEPTANCE_REPORT_PATH", ROOT / "results" / "reports" / "acceptance_demo_visuals_zh.md")
).resolve()
CUSTOM_IMAGE_MODE = bool(os.environ.get("ACCEPTANCE_CUSTOM_IMAGE", "").strip())

CANVAS = (1920, 1080)
WHITE = "#FFFFFF"
INK = "#111827"
MUTED = "#6B7280"
LIGHT = "#E5E7EB"
FAINT = "#F3F4F6"
ROUTE = "#D55E00"
ROUTE_SOFT = "#E69F00"
START = "#009E73"
TARGET = "#CC9900"
CURRENT = "#111827"
BLUE = "#0072B2"
SKY = "#56B4E9"
PINK = "#CC79A7"

LEFT_X = 56
LEFT_W = 300
MAIN_X = 392
MAIN_Y = 84
MAIN_SIZE = 900
RIGHT_X = 1340
CUE_SIZE = 250
CUE_GAP = 44
GRID = 5
ROUTE_DISTANCES = (4, 6, 8)
ROUTE_SMOOTH_SUCCESSES_PER_DISTANCE = 1
ROUTE_WAVY_SUCCESSES_PER_DISTANCE = 2
ROUTE_WAVY_DETOURS_PER_DISTANCE = 1

ROUTE_MAP_X = 90
ROUTE_MAP_Y = 110
ROUTE_MAP_SIZE = 820
ROUTE_PANEL_SIZE = 360
ROUTE_PANEL_GAP_X = 40
ROUTE_PANEL_GAP_Y = 38
ROUTE_PANEL_Y = ROUTE_MAP_Y - 6
ROUTE_PANEL_X = ROUTE_MAP_X + ROUTE_MAP_SIZE + 50
ROUTE_FOOTER_Y = 955
ROUTE_LAYOUT_W = ROUTE_PANEL_X + ROUTE_PANEL_SIZE * 2 + ROUTE_PANEL_GAP_X - ROUTE_MAP_X

ROUTE_ROLES = [
    "c4_smooth_success",
    "c4_wavy_success",
    "c4_wavy_success_02",
    "c4_wavy_detour",
    "c6_smooth_success",
    "c6_wavy_success",
    "c6_wavy_success_02",
    "c6_wavy_detour",
    "c8_smooth_success",
    "c8_wavy_success",
    "c8_wavy_success_02",
    "c8_wavy_detour",
    "three_method_hardcase",
]


@dataclass
class XbdSetting:
    role: str
    title: str
    search_image: Image.Image
    target_image: Image.Image
    start: int
    goal: int
    traj: list[int]
    benchmark: str
    subtitle: str


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    env_font = os.environ.get("ACCEPTANCE_CJK_FONT", "")
    candidates = [
        env_font,
        str(Path(__file__).resolve().parent / "fonts" / "simsun.ttc"),
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        p = Path(candidate)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def font_latin(size: int, bold: bool = False, italic: bool = False) -> ImageFont.ImageFont:
    if bold and italic:
        candidates = ["C:/Windows/Fonts/timesbi.ttf"]
    elif bold:
        candidates = ["C:/Windows/Fonts/timesbd.ttf"]
    elif italic:
        candidates = ["C:/Windows/Fonts/timesi.ttf"]
    else:
        candidates = ["C:/Windows/Fonts/times.ttf"]
    candidates.extend(
        [
            "C:/Windows/Fonts/times.ttf",
            "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        ]
    )
    for candidate in candidates:
        p = Path(candidate)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return font(size, bold=bold)


F = {
    "title": font(38, True),
    "section": font(27, True),
    "label": font(23, True),
    "body": font(22),
    "small": font(18),
    "tiny": font(15),
}

F_LATIN = {
    "title": font_latin(38, True),
    "section": font_latin(27, True),
    "label": font_latin(23, True),
    "body": font_latin(22),
    "small": font_latin(18),
    "tiny": font_latin(15),
}

CJK_FONT_PATH = os.environ.get("ACCEPTANCE_CJK_FONT", "")
MPL_CJK_FAMILY = "DejaVu Sans"
if CJK_FONT_PATH and Path(CJK_FONT_PATH).exists():
    try:
        font_manager.fontManager.addfont(CJK_FONT_PATH)
        MPL_CJK_FAMILY = font_manager.FontProperties(fname=CJK_FONT_PATH).get_name()
    except Exception:
        MPL_CJK_FAMILY = "DejaVu Sans"

LATIN_FAMILY = "DejaVu Serif"
MPL_FONT_FAMILY = [MPL_CJK_FAMILY, LATIN_FAMILY, "DejaVu Sans"]

plt.rcParams["font.sans-serif"] = [MPL_CJK_FAMILY, "DejaVu Sans"]
plt.rcParams["font.serif"] = [LATIN_FAMILY, "DejaVu Serif", "DejaVu Sans"]
plt.rcParams["font.family"] = MPL_FONT_FAMILY
plt.rcParams["axes.unicode_minus"] = False

BENCHMARK_ZH = {
    "masa_aerial": "MASA航拍",
    "mmgag_aerial": "MM-GAG航拍",
    "mmgag_ground": "MM-GAG地面",
    "mmgag_text": "MM-GAG文字",
    "swissview100_aerial": "SwissView航拍",
    "swissviewmonuments_aerial": "地标航拍",
    "swissviewmonuments_ground": "地标地面",
    "xbd_pre_aerial": "xBD灾前",
    "xbd_disaster_aerial": "xBD灾后",
}

ROLE_ZH = {
    "c4_smooth_success": "C=4平顺成功路线",
    "c4_wavy_success": "C=4不平顺成功路线",
    "c4_wavy_success_02": "C=4不平顺成功路线02",
    "c4_wavy_detour": "C=4不平顺绕行样例",
    "c6_smooth_success": "C=6平顺成功路线",
    "c6_wavy_success": "C=6不平顺成功路线",
    "c6_wavy_success_02": "C=6不平顺成功路线02",
    "c6_wavy_detour": "C=6不平顺绕行样例",
    "c8_smooth_success": "C=8平顺成功路线",
    "c8_wavy_success": "C=8不平顺成功路线",
    "c8_wavy_success_02": "C=8不平顺成功路线02",
    "c8_wavy_detour": "C=8不平顺绕行样例",
    "c4_anchor_success": "C=4成功路线",
    "c4_anchor_failure_or_detour": "C=4绕行样例",
    "c6_anchor_success": "C=6成功路线",
    "c6_anchor_failure_or_detour": "C=6绕行样例",
    "c8_anchor_success": "C=8成功路线",
    "c8_anchor_failure_or_detour": "C=8绕行样例",
    "three_method_hardcase": "困难样例路线",
    "xbd_pre_route_setting": "xBD灾前路线设置",
    "xbd_disaster_route_setting": "xBD灾后路线设置",
    "mmgag_aerial_route_setting": "航拍线索路线",
    "mmgag_ground_route_setting": "地面线索路线",
    "mmgag_text_route_setting": "文字线索路线",
    "ultralong_grid8_route_setting": "8x8超远距离路线",
    "ultralong_grid10_route_setting": "10x10超远距离路线",
    "main_benchmark": "主基准结果",
    "xbd_compare": "xBD设置对照",
    "mmgag_multimodal": "多模态设置",
    "factorial": "机制消融",
    "reward": "奖励设计",
    "dataset_param": "数据与参数",
    "budget": "预算压力测试",
}

VALUE_LABEL_ZH = {
    "primary_generalization_mean": "综合泛化成功率",
    "mmgag_mean_sr": "MM-GAG平均成功率",
}

METHOD_ZH = {
    "GeoExplorer-anchor0624": "本文方法",
    "anchor0624": "本文方法",
    "GeoExplorer-pristine": "原始基线",
    "pristine": "原始基线",
    "GOMAA-Geo": "GOMAA",
    "gomaa": "GOMAA",
}


def display_label(value: str) -> str:
    return METHOD_ZH.get(value, ROLE_ZH.get(value, BENCHMARK_ZH.get(value, value)))


def fit_cover(im: Image.Image, size: tuple[int, int]) -> Image.Image:
    w, h = im.size
    tw, th = size
    scale = max(tw / w, th / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = im.resize((nw, nh), Image.Resampling.LANCZOS)
    left = (nw - tw) // 2
    top = (nh - th) // 2
    return resized.crop((left, top, left + tw, top + th))


def fit_contain(im: Image.Image, size: tuple[int, int], bg: str = WHITE) -> Image.Image:
    w, h = im.size
    tw, th = size
    scale = min(tw / w, th / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = im.resize((nw, nh), Image.Resampling.LANCZOS).convert("RGB")
    out = Image.new("RGB", size, bg)
    out.paste(resized, ((tw - nw) // 2, (th - nh) // 2))
    return out


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def latin_font_like(fnt: ImageFont.ImageFont) -> ImageFont.ImageFont:
    size = int(getattr(fnt, "size", 22))
    return font_latin(size, bold=size in {23, 27, 38})


def is_latin_char(ch: str) -> bool:
    return ord(ch) < 128


def split_font_runs(text: str) -> list[tuple[str, bool]]:
    runs: list[tuple[str, bool]] = []
    current = ""
    current_latin: bool | None = None
    for ch in str(text):
        latin = is_latin_char(ch)
        if current and latin != current_latin:
            runs.append((current, bool(current_latin)))
            current = ch
        else:
            current += ch
        current_latin = latin
    if current:
        runs.append((current, bool(current_latin)))
    return runs


def text_size_mixed(
    draw: ImageDraw.ImageDraw,
    text: str,
    cjk_font: ImageFont.ImageFont,
    latin_font: ImageFont.ImageFont | None = None,
) -> tuple[int, int]:
    latin_font = latin_font or latin_font_like(cjk_font)
    width = 0
    height = 0
    for run, is_latin in split_font_runs(text):
        fnt = latin_font if is_latin else cjk_font
        w, h = text_size(draw, run, fnt)
        width += w
        height = max(height, h)
    return width, height


def draw_text_mixed(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    cjk_font: ImageFont.ImageFont,
    fill: str = INK,
    latin_font: ImageFont.ImageFont | None = None,
) -> None:
    latin_font = latin_font or latin_font_like(cjk_font)
    x, y = xy
    for run, is_latin in split_font_runs(text):
        fnt = latin_font if is_latin else cjk_font
        draw.text((x, y), run, font=fnt, fill=fill)
        x += text_size(draw, run, fnt)[0]


def draw_text_fit(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    max_width: int,
    fnt: ImageFont.ImageFont,
    fill: str = INK,
    line_gap: int = 6,
) -> int:
    raw = str(text)
    words = raw.split()
    if len(words) == 1 and len(raw) > 1:
        words = list(raw)
    if not words:
        return xy[1]
    lines: list[str] = []
    current = words[0]
    latin_font = latin_font_like(fnt)
    for word in words[1:]:
        sep = "" if len(word) == 1 and not word.isspace() else " "
        trial = f"{current}{sep}{word}"
        if text_size_mixed(draw, trial, fnt, latin_font)[0] <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    y = xy[1]
    for line in lines:
        draw_text_mixed(draw, (xy[0], y), line, fnt, fill, latin_font=latin_font)
        y += text_size_mixed(draw, line, fnt, latin_font)[1] + line_gap
    return y


def wrap_text_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    fnt: ImageFont.ImageFont,
    *,
    max_lines: int | None = None,
) -> list[str]:
    words = str(text).strip().split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    latin_font = latin_font_like(fnt)
    for word in words[1:]:
        trial = f"{current} {word}"
        if text_size_mixed(draw, trial, fnt, latin_font)[0] <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
            if max_lines is not None and len(lines) >= max_lines:
                break
    if max_lines is None or len(lines) < max_lines:
        lines.append(current)
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
    return lines


def percent(v: float | int | str | None, digits: int = 1) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    try:
        return f"{float(v) * 100:.{digits}f}%"
    except Exception:
        return str(v)


def load_cases() -> dict[str, dict[str, Any]]:
    rows = json.loads(CASE_JSON.read_text(encoding="utf-8"))
    return {row["role"]: row["task"] for row in rows}


def parse_literal(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") or text in {"True", "False"}:
            try:
                return ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return value
    return value


def is_missing_value(value: Any) -> bool:
    if isinstance(value, (list, tuple, dict)):
        return False
    return bool(pd.isna(value))


def normalize_route_task(row: dict[str, Any]) -> dict[str, Any]:
    task: dict[str, Any] = {}
    for key, value in row.items():
        if is_missing_value(value):
            continue
        task[key] = parse_literal(value)
    for key in ["img_idx", "distance", "start", "goal", "final", "final_distance", "path_length", "optimal_steps", "detour_steps"]:
        if key in task:
            task[key] = int(task[key])
    if isinstance(task.get("success"), str):
        task["success"] = task["success"].strip().lower() == "true"
    task["traj"] = [int(x) for x in task.get("traj", [])]
    task["actions"] = list(task.get("actions", []))
    task["reward_trace"] = [float(x) for x in task.get("reward_trace", [])]
    task.setdefault("method", "GeoExplorer-anchor0624")
    task.setdefault("method_key", "anchor0624")
    task.setdefault("dataset", "swissviewmonuments")
    task.setdefault("grid", GRID)
    task.setdefault("case_id", f"img{int(task['img_idx']):03d}_d{int(task['distance'])}_s{int(task['start']):02d}_g{int(task['goal']):02d}_r0")
    task.setdefault("optimal_steps", manhattan_distance(int(task["start"]), int(task["goal"]), GRID))
    task.setdefault("path_length", max(0, len(task["traj"]) - 1))
    task.setdefault("detour_steps", max(0, int(task["path_length"]) - int(task["optimal_steps"])))
    task.setdefault("final", int(task["traj"][-1]) if task["traj"] else int(task["start"]))
    task.setdefault("final_distance", manhattan_distance(int(task["final"]), int(task["goal"]), GRID))
    return task


def output_ref(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_image_filter(value: str) -> set[int]:
    if not value:
        return set()
    result: set[int] = set()
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


def load_route_records() -> list[dict[str, Any]]:
    if not TRAJECTORY_RECORDS.exists():
        return []
    df = pd.read_csv(TRAJECTORY_RECORDS)
    if "method" in df:
        df = df[df["method"] == "GeoExplorer-anchor0624"].copy()
    image_filter = parse_image_filter(os.environ.get("ACCEPTANCE_IMAGE_IDX", os.environ.get("ACCEPTANCE_INFER_IMAGE", "")))
    if image_filter and "img_idx" in df:
        df = df[df["img_idx"].astype(int).isin(image_filter)].copy()
    records = [normalize_route_task(row) for row in df.to_dict("records")]
    return [record for record in records if record.get("traj")]


def curve_stats(task: dict[str, Any]) -> dict[str, int]:
    distances = route_distances(task)
    diffs = [b - a for a, b in zip(distances, distances[1:])]
    uphill = sum(1 for d in diffs if d > 0)
    downhill = sum(1 for d in diffs if d < 0)
    turns = sum(
        1
        for a, b in zip(diffs, diffs[1:])
        if a != 0 and b != 0 and (a > 0) != (b > 0)
    )
    repeat_visits = len(task["traj"]) - len(set(task["traj"]))
    return {
        "uphill": uphill,
        "downhill": downhill,
        "turns": turns,
        "max_rise": max([0, *diffs]),
        "repeat_visits": repeat_visits,
        "final_distance": int(distances[-1]),
    }


def curve_score(task: dict[str, Any]) -> int:
    stats = curve_stats(task)
    return (
        stats["uphill"] * 5
        + stats["turns"] * 4
        + stats["max_rise"] * 3
        + stats["repeat_visits"] * 2
        + stats["final_distance"]
        + int(task.get("detour_steps", 0))
    )


def is_smooth_curve(task: dict[str, Any]) -> bool:
    stats = curve_stats(task)
    return stats["uphill"] == 0 and stats["turns"] == 0 and stats["repeat_visits"] == 0


def is_wavy_curve(task: dict[str, Any]) -> bool:
    return not is_smooth_curve(task)


def route_variant_role(base_role: str, index: int) -> str:
    if index == 1:
        return base_role
    return f"{base_role}_{index:02d}"


def route_label_for(role: str, task: dict[str, Any]) -> str:
    stats = curve_stats(task)
    curve = f"curve up={stats['uphill']}, turns={stats['turns']}, final d={stats['final_distance']}"
    return f"{role} | {task['case_id']} | {curve}"


def select_route_cases(seed_cases: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    records = load_route_records()
    selected: list[tuple[str, dict[str, Any]]] = []
    selected_ids: set[str] = set()
    cached_img_idxs = {int(path.stem.split("_")[1]) for path in ASSET_DIR.glob("img_*.png")}

    def add(role: str, task: dict[str, Any]) -> bool:
        case_id = str(task["case_id"])
        if case_id in selected_ids:
            return False
        task = normalize_route_task(task)
        task["route_label"] = route_label_for(role, task)
        selected_ids.add(case_id)
        selected.append((role, task))
        return True

    if CUSTOM_IMAGE_MODE:
        custom_records = [
            record
            for record in records
            if int(record.get("img_idx", -1)) in cached_img_idxs
        ]
        custom_records.sort(key=lambda r: (int(r.get("distance", 0)), int(r.get("start", 0)), int(r.get("goal", 0))))
        for idx, record in enumerate(custom_records, start=1):
            add(f"custom_route_{idx:02d}", record)
        return selected

    def pool_for(dist: int, predicate) -> list[dict[str, Any]]:
        return [
            record
            for record in records
            if int(record.get("distance", -1)) == dist
            and int(record.get("img_idx", -1)) in cached_img_idxs
            and predicate(record)
        ]

    def add_from_pool(base_role: str, pool: list[dict[str, Any]], limit: int) -> None:
        count = 0
        for record in pool:
            if count >= limit:
                break
            if add(route_variant_role(base_role, count + 1), record):
                count += 1

    for dist in ROUTE_DISTANCES:
        smooth_success = pool_for(dist, lambda r: bool(r.get("success")) and is_smooth_curve(r))
        smooth_success.sort(key=lambda r: (int(r.get("path_length", 0)), int(r.get("img_idx", 0)), str(r.get("case_id", ""))))
        add_from_pool(f"c{dist}_smooth_success", smooth_success, ROUTE_SMOOTH_SUCCESSES_PER_DISTANCE)

        wavy_success = pool_for(dist, lambda r: bool(r.get("success")) and is_wavy_curve(r))
        wavy_success.sort(
            key=lambda r: (
                -curve_score(r),
                -int(r.get("detour_steps", 0)),
                int(r.get("path_length", 0)),
                int(r.get("img_idx", 0)),
                str(r.get("case_id", "")),
            )
        )
        add_from_pool(f"c{dist}_wavy_success", wavy_success, ROUTE_WAVY_SUCCESSES_PER_DISTANCE)

        wavy_detour = pool_for(
            dist,
            lambda r: ((not bool(r.get("success"))) or int(r.get("detour_steps", 0)) > 0) and is_wavy_curve(r),
        )
        wavy_detour.sort(
            key=lambda r: (
                bool(r.get("success")),
                -curve_score(r),
                -int(r.get("final_distance", 0)),
                int(r.get("img_idx", 0)),
                str(r.get("case_id", "")),
            )
        )
        add_from_pool(f"c{dist}_wavy_detour", wavy_detour, ROUTE_WAVY_DETOURS_PER_DISTANCE)

    hard = seed_cases.get("three_method_hardcase")
    if hard:
        add("three_method_hardcase", hard)
    elif records:
        hard_pool = [r for r in records if int(r.get("distance", 0)) in {6, 8}]
        hard_pool.sort(key=lambda r: (-curve_score(r), -int(r.get("final_distance", 0)), int(r.get("img_idx", 0))))
        if hard_pool:
            add("three_method_hardcase", hard_pool[0])
    return selected


def load_main_table() -> pd.DataFrame:
    return pd.read_csv(MAIN_TABLE)


def cell_box(im: Image.Image, idx: int, grid: int = GRID, inset: float = 0.02) -> tuple[int, int, int, int]:
    row, col = divmod(int(idx), grid)
    cw = im.width / grid
    ch = im.height / grid
    dx = int(cw * inset)
    dy = int(ch * inset)
    return (
        int(col * cw) + dx,
        int(row * ch) + dy,
        int((col + 1) * cw) - dx,
        int((row + 1) * ch) - dy,
    )


def cell_center(size: tuple[int, int], idx: int, grid: int = GRID) -> tuple[int, int]:
    row, col = divmod(int(idx), grid)
    return int((col + 0.5) * size[0] / grid), int((row + 0.5) * size[1] / grid)


def cell_coord(idx: int, grid: int = GRID) -> tuple[int, int]:
    row, col = divmod(int(idx), grid)
    return row, col


def manhattan_distance(a: int, b: int, grid: int = GRID) -> int:
    ar, ac = cell_coord(a, grid)
    br, bc = cell_coord(b, grid)
    return abs(ar - br) + abs(ac - bc)


def route_distances(task: dict[str, Any]) -> list[int]:
    grid = int(task.get("grid", GRID))
    goal = int(task["goal"])
    return [manhattan_distance(int(idx), goal, grid=grid) for idx in task["traj"]]


def crop_cell_square(im: Image.Image, idx: int, size: int = CUE_SIZE, grid: int = GRID) -> Image.Image:
    return fit_cover(im.crop(cell_box(im, idx, grid=grid)), (size, size))


def draw_grid(draw: ImageDraw.ImageDraw, xy: tuple[int, int], size: tuple[int, int], grid: int = GRID) -> None:
    x, y = xy
    w, h = size
    for i in range(1, grid):
        xx = x + int(w * i / grid)
        yy = y + int(h * i / grid)
        draw.line((xx, y, xx, y + h), fill=(255, 255, 255, 155), width=2)
        draw.line((x, yy, x + w, yy), fill=(255, 255, 255, 155), width=2)


def draw_route_map(
    base: Image.Image,
    task: dict[str, Any],
    step: int,
    size: tuple[int, int],
    *,
    show_grid: bool = True,
) -> Image.Image:
    grid = int(task.get("grid", GRID))
    route = list(map(int, task["traj"]))
    shown = route[: step + 1]
    map_im = fit_cover(base, size).convert("RGBA")
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    if show_grid:
        draw_grid(draw, (0, 0), size, grid=grid)

    pts = [cell_center(size, idx, grid=grid) for idx in shown]
    if len(pts) >= 2:
        draw.line(pts, fill=(213, 94, 0, 235), width=11, joint="curve")
        draw.line(pts, fill=(255, 255, 255, 210), width=3, joint="curve")

    for px, py in pts[:-1]:
        draw.ellipse((px - 8, py - 8, px + 8, py + 8), fill=(213, 94, 0, 240))

    sx, sy = cell_center(size, int(task["start"]), grid=grid)
    gx, gy = cell_center(size, int(task["goal"]), grid=grid)
    cx, cy = cell_center(size, int(shown[-1]), grid=grid)
    draw.ellipse((sx - 18, sy - 18, sx + 18, sy + 18), fill=(0, 158, 115, 235))
    draw.ellipse((gx - 20, gy - 20, gx + 20, gy + 20), fill=(204, 153, 0, 240))
    draw.ellipse((cx - 17, cy - 17, cx + 17, cy + 17), fill=(17, 24, 39, 240))
    return Image.alpha_composite(map_im, overlay).convert("RGB")


def make_canvas(title: str, subtitle: str = "") -> tuple[Image.Image, ImageDraw.ImageDraw]:
    canvas = Image.new("RGB", CANVAS, WHITE)
    draw = ImageDraw.Draw(canvas)
    draw_text_mixed(draw, (LEFT_X, 34), title, F["title"], INK)
    return canvas, draw


def draw_data_column(
    draw: ImageDraw.ImageDraw,
    title: str,
    rows: list[tuple[str, str, str | None]],
    *,
    x: int = LEFT_X,
    y: int = 160,
    width: int = LEFT_W,
) -> None:
    draw_text_mixed(draw, (x, y), title, F["section"], INK)
    y += 48
    for label, value, color in rows:
        draw_text_mixed(draw, (x, y), label, F["small"], MUTED)
        y += 22
        if color:
            draw.ellipse((x, y + 7, x + 11, y + 18), fill=color)
            vx = x + 20
        else:
            vx = x
        draw_text_fit(draw, (vx, y), value, width - (vx - x), F["label"], INK, line_gap=3)
        y += 58


def draw_observation_grid(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    panels: list[tuple[str, Image.Image, str]],
    *,
    x: int,
    y: int,
    size: int,
    gap_x: int = 40,
    gap_y: int = 38,
) -> None:
    positions = [
        (x, y),
        (x + size + gap_x, y),
        (x, y + size + 34 + gap_y),
        (x + size + gap_x, y + size + 34 + gap_y),
    ]
    for (label, im, color), (px, py) in zip(panels, positions):
        draw_text_mixed(draw, (px, py), label, F["label"], color)
        canvas.paste(fit_contain(im, (size, size)), (px, py + 34))


def make_route_legend_image(size: int = 1024) -> Image.Image:
    im = Image.new("RGB", (size, size), WHITE)
    draw = ImageDraw.Draw(im)
    pts = [(size * 0.20, size * 0.72), (size * 0.42, size * 0.48), (size * 0.64, size * 0.48), (size * 0.82, size * 0.28)]
    draw.line(pts, fill=ROUTE, width=max(12, size // 32), joint="curve")
    sx, sy = pts[0]
    cx, cy = pts[2]
    gx, gy = pts[-1]
    r = size * 0.045
    draw.ellipse((sx - r, sy - r, sx + r, sy + r), fill=START)
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=CURRENT)
    draw.ellipse((gx - r * 1.2, gy - r * 1.2, gx + r * 1.2, gy + r * 1.2), outline=TARGET, width=max(8, size // 64))
    label_font = font_latin(max(40, size // 18), True)
    small_font = font_latin(max(30, size // 26))
    title = "ROUTE"
    tw, _ = text_size(draw, title, label_font)
    draw.text(((size - tw) / 2, size * 0.12), title, font=label_font, fill=MUTED)
    for text, (tx, ty), color in [
        ("Start", (size * 0.13, size * 0.80), START),
        ("Now", (size * 0.56, size * 0.56), CURRENT),
        ("Goal", (size * 0.72, size * 0.36), TARGET),
    ]:
        draw.text((tx, ty), text, font=small_font, fill=color)
    return im


def make_distance_curve_image(task: dict[str, Any], step: int, size: int = 1024) -> Image.Image:
    distances = route_distances(task)
    max_step = max(1, len(distances) - 1)
    current_step = min(max(0, step), len(distances) - 1)
    shown = distances[: current_step + 1]
    max_d = max(max(distances), 1)

    im = Image.new("RGB", (size, size), WHITE)
    draw = ImageDraw.Draw(im)
    title_font = font(max(40, size // 17), True)
    label_font = font(max(28, size // 27), True)
    small_font = font(max(24, size // 34))
    title = "距离目标"
    tw, _ = text_size(draw, title, title_font)
    draw.text(((size - tw) / 2, size * 0.09), title, font=title_font, fill=INK)

    plot = (
        int(size * 0.15),
        int(size * 0.28),
        int(size * 0.88),
        int(size * 0.72),
    )
    x0, y0, x1, y1 = plot
    for i in range(5):
        yy = y0 + (y1 - y0) * i / 4
        draw.line((x0, yy, x1, yy), fill=LIGHT, width=max(2, size // 240))
    draw.line((x0, y0, x0, y1), fill="#9CA3AF", width=max(3, size // 170))
    draw.line((x0, y1, x1, y1), fill="#9CA3AF", width=max(3, size // 170))

    def pt(i: int, d: int) -> tuple[float, float]:
        x = x0 + (x1 - x0) * i / max_step
        y = y1 - (y1 - y0) * d / max_d
        return x, y

    full_pts = [pt(i, d) for i, d in enumerate(distances)]
    shown_pts = [pt(i, d) for i, d in enumerate(shown)]
    if len(full_pts) >= 2:
        draw.line(full_pts, fill="#D1D5DB", width=max(5, size // 80), joint="curve")
    if len(shown_pts) >= 2:
        draw.line(shown_pts, fill=ROUTE, width=max(9, size // 52), joint="curve")
    for px, py in shown_pts:
        r = max(8, size // 70)
        draw.ellipse((px - r, py - r, px + r, py + r), fill=ROUTE)
    cx, cy = shown_pts[-1]
    r = max(14, size // 45)
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=CURRENT)

    draw.text((x0, y1 + size * 0.055), "步骤", font=small_font, fill=MUTED)
    draw.text((x0 - size * 0.06, y0 - size * 0.02), str(max_d), font=small_font, fill=MUTED)
    draw.text((x0 - size * 0.04, y1 - size * 0.02), "0", font=small_font, fill=MUTED)

    summary = f"第{current_step}步/{max_step}步    距离={distances[current_step]}"
    sw, _ = text_size(draw, summary, label_font)
    draw.text(((size - sw) / 2, size * 0.80), summary, font=label_font, fill=INK)
    footer = "曲线越低表示越接近目标"
    fw, _ = text_size(draw, footer, small_font)
    draw.text(((size - fw) / 2, size * 0.88), footer, font=small_font, fill=MUTED)
    return im


def draw_data_footer(
    draw: ImageDraw.ImageDraw,
    title: str,
    rows: list[tuple[str, str, str | None]],
    *,
    x: int = LEFT_X,
    y: int = 952,
    width: int = CANVAS[0] - LEFT_X * 2,
) -> None:
    draw.line((x, y, x + width, y), fill=LIGHT, width=2)
    draw_text_mixed(draw, (x, y + 20), title, F["section"], INK)
    item_x = x + 180
    item_w = max(160, (width - 190) // max(1, len(rows)))
    for i, (label, value, color) in enumerate(rows):
        cx = item_x + i * item_w
        draw_text_mixed(draw, (cx, y + 16), label, F["tiny"], MUTED)
        vx = cx
        if color:
            draw.ellipse((cx, y + 47, cx + 10, y + 57), fill=color)
            vx += 18
        draw_text_fit(draw, (vx, y + 39), value, item_w - (vx - cx) - 8, F["body"], INK, line_gap=2)


def build_target_current_start_curve_panels(
    search_image: Image.Image,
    target_image: Image.Image,
    task: dict[str, Any],
    step: int,
    *,
    cue_labels: tuple[str, str, str] = ("目标线索", "当前观察", "起点位置"),
    target_full_image: bool = False,
) -> list[tuple[str, Image.Image, str]]:
    grid = int(task.get("grid", GRID))
    current_idx = int(task["traj"][step])
    target_cue = target_image if target_full_image else crop_cell_square(target_image, int(task["goal"]), grid=grid)
    return [
        (cue_labels[0], target_cue, TARGET),
        (cue_labels[1], crop_cell_square(search_image, current_idx, grid=grid), CURRENT),
        (cue_labels[2], crop_cell_square(search_image, int(task["start"]), grid=grid), START),
        ("距离曲线", make_distance_curve_image(task, step), ROUTE),
    ]


def build_route_layout_frame(
    *,
    title: str,
    search_image: Image.Image,
    target_image: Image.Image,
    task: dict[str, Any],
    step: int,
    data_title: str,
    data_rows: list[tuple[str, str, str | None]],
    cue_labels: tuple[str, str, str] = ("目标线索", "当前观察", "起点位置"),
    target_full_image: bool = False,
) -> Image.Image:
    canvas, draw = make_canvas(title)
    route_map = draw_route_map(search_image, task, step, (ROUTE_MAP_SIZE, ROUTE_MAP_SIZE))
    canvas.paste(route_map, (ROUTE_MAP_X, ROUTE_MAP_Y))
    panels = build_target_current_start_curve_panels(
        search_image,
        target_image,
        task,
        step,
        cue_labels=cue_labels,
        target_full_image=target_full_image,
    )
    draw_observation_grid(
        canvas,
        draw,
        panels,
        x=ROUTE_PANEL_X,
        y=ROUTE_PANEL_Y,
        size=ROUTE_PANEL_SIZE,
        gap_x=ROUTE_PANEL_GAP_X,
        gap_y=ROUTE_PANEL_GAP_Y,
    )
    rows = [("进度", f"第{step:02d}步/{len(task['traj']) - 1:02d}", BLUE), *data_rows]
    draw_data_footer(draw, data_title, rows, x=ROUTE_MAP_X, y=ROUTE_FOOTER_Y, width=ROUTE_LAYOUT_W)
    return canvas


def make_multimodal_target_panel(
    aerial: Image.Image,
    ground: Image.Image,
    text_cue: Image.Image,
    size: int = 1024,
) -> Image.Image:
    im = Image.new("RGB", (size, size), WHITE)
    draw = ImageDraw.Draw(im)
    gap = max(18, size // 40)
    top_h = int(size * 0.47)
    small_w = (size - gap) // 2
    im.paste(fit_cover(aerial, (small_w, top_h)), (0, 0))
    im.paste(fit_cover(ground, (size - small_w - gap, top_h)), (small_w + gap, 0))
    text_y = top_h + gap
    im.paste(fit_contain(text_cue, (size, size - text_y)), (0, text_y))
    for x, y, w, h in [
        (0, 0, small_w, top_h),
        (small_w + gap, 0, size - small_w - gap, top_h),
        (0, text_y, size, size - text_y),
    ]:
        draw.rectangle((x, y, x + w, y + h), outline=WHITE, width=max(5, size // 140))
    return im


def compact_text_cue(raw: str) -> str:
    text = " ".join(raw.strip().split())
    lower = text.lower()
    if "clock tower" in lower and "church" in lower:
        return "CLOCK-TOWER CHURCH"
    if "church" in lower:
        return "CHURCH LANDMARK"
    if "tower" in lower:
        return "CLOCK TOWER"
    if not text:
        return "TEXT CUE"
    words = text.split()
    return " ".join(words[:4]).upper()


def route_title(role: str, task: dict[str, Any]) -> str:
    if role.startswith("custom_route"):
        return f"Custom image route C={task['distance']}"
    if role == "three_method_hardcase":
        return "困难样例搜索路线"
    if "smooth_success" in role:
        return f"C={task['distance']}平顺成功路线"
    if "wavy_success" in role:
        return f"C={task['distance']}不平顺成功路线"
    if "wavy_detour" in role:
        return f"C={task['distance']}不平顺绕行样例"
    if "failure" in role:
        return f"C={task['distance']}绕行样例"
    return f"C={task['distance']}成功搜索路线"


def build_route_frame(base: Image.Image, task: dict[str, Any], role: str, step: int) -> Image.Image:
    title = route_title(role, task)
    if role[-3:-2] == "_" and role[-2:].isdigit():
        title = f"{title} {role[-2:]}"
    status = "成功" if bool(task.get("success")) else "失败/绕行"
    final_gap = int(task.get("final_distance", 0))
    stats = curve_stats(task)
    smoothness = "平顺" if is_smooth_curve(task) else f"不平顺 up={stats['uphill']}, turns={stats['turns']}"
    rows = [
        ("方法", display_label(str(task.get("method", "GeoExplorer-anchor0624"))), BLUE),
        ("距离", f"C={task['distance']}；最短={task.get('optimal_steps', task['distance'])}", None),
        ("结果", f"{status}；余距={final_gap}", START if final_gap == 0 else ROUTE),
        ("路径", f"{task.get('path_length', len(task['traj']) - 1)}步；绕行{task.get('detour_steps', 0)}步", None),
        ("曲线", smoothness, ROUTE if is_wavy_curve(task) else START),
    ]
    return build_route_layout_frame(
        title=title,
        search_image=base,
        target_image=base,
        task=task,
        step=step,
        data_title="路线数据",
        data_rows=rows,
        cue_labels=("目标线索", "当前观察", "起点位置"),
    )


def build_route_gif(role: str, task: dict[str, Any]) -> dict[str, str]:
    base = Image.open(ASSET_DIR / f"img_{int(task['img_idx']):03d}.png").convert("RGB")
    frames = [build_route_frame(base, task, role, i) for i in range(len(task["traj"]))]
    out = OUT_DIR / f"acceptance_{role}.gif"
    poster = OUT_DIR / f"acceptance_{role}_poster.png"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=620, loop=0, optimize=True)
    frames[-1].save(poster, quality=95)
    stats = curve_stats(task)
    return {
        "role": role,
        "case_id": str(task["case_id"]),
        "img_idx": str(int(task["img_idx"])),
        "success": str(bool(task.get("success"))),
        "curve": f"up={stats['uphill']}, turns={stats['turns']}, final_d={stats['final_distance']}",
        "gif": output_ref(out),
        "poster": output_ref(poster),
    }


def build_route_contact_sheet(route_outputs: list[dict[str, str]]) -> str:
    thumb_w, thumb_h = 330, 186
    canvas, draw = make_canvas("路线动图总览", "白底对齐版验收路线动图，可直接用于答辩展示。")
    x0, y0 = 56, 145
    for i, item in enumerate(route_outputs):
        poster = ROOT / item["poster"]
        im = fit_cover(Image.open(poster).convert("RGB"), (thumb_w, thumb_h))
        col, row = i % 5, i // 5
        x = x0 + col * 365
        y = y0 + row * 285
        canvas.paste(im, (x, y))
        draw_text_fit(draw, (x, y + thumb_h + 10), item.get("case_id", item["role"]), thumb_w, F["tiny"], INK, line_gap=2)
        draw_text_fit(draw, (x, y + thumb_h + 34), item.get("curve", item["role"]), thumb_w, F["tiny"], MUTED, line_gap=2)
    out = OUT_DIR / "acceptance_route_gallery.png"
    canvas.save(out, quality=95)
    return output_ref(out)


def metric_rows_for_benchmark(main: pd.DataFrame, benchmark: str) -> list[tuple[str, str, str | None]]:
    subset = main[main["benchmark"] == benchmark].copy()
    anchor = subset[subset["method"] == "GeoExplorer-anchor0624"].iloc[0]
    gomaa = subset[subset["method"] == "GOMAA-Geo"].iloc[0] if (subset["method"] == "GOMAA-Geo").any() else None
    delta = float(anchor["success_ratio"]) - (float(gomaa["success_ratio"]) if gomaa is not None else 0.0)
    rows = [
        ("基准", display_label(benchmark), BLUE),
        ("本文成功率", percent(anchor["success_ratio"]), START),
        ("目标残差", f"{float(anchor['sg_mean']):.3f}", None),
    ]
    if gomaa is not None:
        rows.append(("对比GOMAA", f"+{delta * 100:.2f}百分点", ROUTE if delta > 0 else MUTED))
    rows.append(("协议", "5x5；预算10；D=4-8", None))
    return rows


def build_xbd_settings(main: pd.DataFrame) -> list[XbdSetting]:
    pre = Image.open(XBD_ASSETS / "xBD_pre_disaster_target_panel_ready_square.png").convert("RGB")
    post = Image.open(XBD_ASSETS / "xBD_post_disaster_search_panel_ready_square.png").convert("RGB")
    return [
        XbdSetting(
            role="xbd_pre_route_setting",
            title="xBD灾前路线设置",
            search_image=pre,
            target_image=pre,
            start=20,
            goal=4,
            traj=[20, 15, 10, 5, 0, 1, 2, 3, 4],
            benchmark="xbd_pre_aerial",
            subtitle="目标线索和搜索图均使用灾前影像",
        ),
        XbdSetting(
            role="xbd_disaster_route_setting",
            title="xBD灾后路线设置",
            search_image=post,
            target_image=pre,
            start=24,
            goal=0,
            traj=[24, 19, 14, 9, 4, 3, 2, 1, 0],
            benchmark="xbd_disaster_aerial",
            subtitle="目标线索用灾前图，搜索区域用灾后图",
        ),
    ]


def build_xbd_page(setting: XbdSetting, main: pd.DataFrame) -> dict[str, str]:
    task = {
        "traj": setting.traj,
        "start": setting.start,
        "goal": setting.goal,
        "distance": 8,
        "grid": 5,
    }
    canvas = build_route_layout_frame(
        title=setting.title,
        search_image=setting.search_image,
        target_image=setting.target_image,
        task=task,
        step=len(setting.traj) - 1,
        data_title="xBD数据",
        data_rows=metric_rows_for_benchmark(main, setting.benchmark),
        cue_labels=("目标线索", "搜索观察", "起点位置"),
    )
    out = OUT_DIR / f"acceptance_{setting.role}.png"
    canvas.save(out, quality=95)
    return {"role": setting.role, "image": output_ref(out)}


def build_setting_frame(
    *,
    title: str,
    subtitle: str,
    search_image: Image.Image,
    target_image: Image.Image,
    task: dict[str, Any],
    step: int,
    data_title: str,
    data_rows: list[tuple[str, str, str | None]],
    cue_labels: tuple[str, str, str] = ("目标线索", "当前观测", "起点位置"),
    target_full_image: bool = False,
) -> Image.Image:
    grid = int(task.get("grid", GRID))
    return build_route_layout_frame(
        title=title,
        search_image=search_image,
        target_image=target_image,
        task=task,
        step=step,
        data_title=data_title,
        data_rows=data_rows,
        cue_labels=cue_labels,
        target_full_image=target_full_image,
    )


def build_setting_gif(
    *,
    role: str,
    title: str,
    subtitle: str,
    search_image: Image.Image,
    target_image: Image.Image,
    task: dict[str, Any],
    data_title: str,
    data_rows: list[tuple[str, str, str | None]],
    cue_labels: tuple[str, str, str] = ("目标线索", "当前观测", "起点位置"),
    target_full_image: bool = False,
) -> dict[str, str]:
    frames = [
        build_setting_frame(
            title=title,
            subtitle=subtitle,
            search_image=search_image,
            target_image=target_image,
            task=task,
            step=i,
            data_title=data_title,
            data_rows=data_rows,
            cue_labels=cue_labels,
            target_full_image=target_full_image,
        )
        for i in range(len(task["traj"]))
    ]
    out = OUT_DIR / f"acceptance_{role}.gif"
    poster = OUT_DIR / f"acceptance_{role}_poster.png"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=620, loop=0, optimize=True)
    frames[-1].save(poster, quality=95)
    return {"role": role, "gif": output_ref(out), "poster": output_ref(poster)}


def build_xbd_setting_gifs(settings: list[XbdSetting], main: pd.DataFrame) -> list[dict[str, str]]:
    gifs = []
    for setting in settings:
        task = {
            "traj": setting.traj,
            "start": setting.start,
            "goal": setting.goal,
            "distance": 8,
            "grid": 5,
        }
        gifs.append(
            build_setting_gif(
                role=setting.role,
                title=setting.title,
                subtitle=setting.subtitle,
                search_image=setting.search_image,
                target_image=setting.target_image,
                task=task,
                data_title="xBD路线数据",
                data_rows=metric_rows_for_benchmark(main, setting.benchmark),
                cue_labels=("目标线索", "搜索观测", "起点位置"),
            )
        )
    return gifs


def make_text_cue_image(size: int | tuple[int, int] = 1024) -> Image.Image:
    path = DATASET_ASSETS / "02_MM-GAG" / "MMGAG_text_target_cue.txt"
    raw_text = "The target is a church building with a prominent clock tower, pointed roof, clock face, stone facade, central entrance, stained-glass windows, and a statue above the entrance."
    if path.exists():
        raw = path.read_text(encoding="utf-8", errors="replace").strip().replace("\n", " ")
        if raw:
            raw_text = " ".join(raw.split())
    compact = compact_text_cue(raw_text)
    w, h = (size, size) if isinstance(size, int) else size
    im = Image.new("RGB", (w, h), WHITE)
    draw = ImageDraw.Draw(im)
    margin = int(w * 0.075)
    title_font = font_latin(max(34, int(h * 0.075)), True)
    tag_font = font_latin(max(24, int(h * 0.042)), True)
    body_font = font_latin(max(26, int(h * 0.049)))
    small_font = font_latin(max(20, int(h * 0.034)))

    title = "TEXT TARGET DESCRIPTION"
    while text_size(draw, title, title_font)[0] > w - margin * 2 and title_font.size > 26:
        title_font = font_latin(title_font.size - 2, True)
    draw.text((margin, int(h * 0.08)), title, font=title_font, fill=INK)
    draw.line((margin, int(h * 0.17), w - margin, int(h * 0.17)), fill=LIGHT, width=max(3, w // 220))

    tag = compact
    tag_w, tag_h = text_size(draw, tag, tag_font)
    tag_x, tag_y = margin, int(h * 0.215)
    pad_x, pad_y = int(w * 0.025), int(h * 0.014)
    draw.rounded_rectangle(
        (tag_x, tag_y, min(w - margin, tag_x + tag_w + pad_x * 2), tag_y + tag_h + pad_y * 2),
        radius=int(w * 0.02),
        fill=FAINT,
        outline=LIGHT,
        width=max(2, w // 360),
    )
    draw.text((tag_x + pad_x, tag_y + pad_y), tag, font=tag_font, fill=ROUTE)

    body_top = int(h * 0.32)
    body_bottom = int(h * 0.72)
    body_lines = wrap_text_lines(draw, raw_text, w - margin * 2, body_font)
    while True:
        line_h = max(text_size(draw, line, body_font)[1] for line in body_lines) if body_lines else body_font.size
        block_h = len(body_lines) * line_h + max(0, len(body_lines) - 1) * int(h * 0.014)
        if block_h <= body_bottom - body_top or body_font.size <= 31:
            break
        body_font = font_latin(body_font.size - 3)
        body_lines = wrap_text_lines(draw, raw_text, w - margin * 2, body_font)
    y = body_top
    for line in body_lines:
        draw.text((margin, y), line, font=body_font, fill=INK)
        y += text_size(draw, line, body_font)[1] + int(h * 0.014)

    keywords = [
        "clock tower",
        "pointed roof",
        "clock face",
        "stone facade",
        "central entrance",
        "stained-glass windows",
        "statue above entrance",
    ]
    key_y = max(int(h * 0.745), y + int(h * 0.018))
    draw.text((margin, key_y), "visual anchors", font=small_font, fill=MUTED)
    key_y += int(h * 0.052)
    x = margin
    for keyword in keywords:
        kw_w, kw_h = text_size(draw, keyword, small_font)
        chip_w = kw_w + pad_x * 2
        chip_h = kw_h + pad_y * 2
        if x + chip_w > w - margin:
            x = margin
            key_y += chip_h + int(h * 0.018)
        draw.rounded_rectangle(
            (x, key_y, x + chip_w, key_y + chip_h),
            radius=int(w * 0.018),
            fill=WHITE,
            outline="#D1D5DB",
            width=max(2, w // 420),
        )
        draw.text((x + pad_x, key_y + pad_y), keyword, font=small_font, fill=INK)
        x += chip_w + int(w * 0.018)
    return im


def build_mmgag_setting_gifs(main: pd.DataFrame) -> list[dict[str, str]]:
    cases = load_cases()
    task = dict(cases["c8_anchor_success"])
    task["grid"] = 5
    base = Image.open(ASSET_DIR / f"img_{int(task['img_idx']):03d}.png").convert("RGB")
    cue_sources: list[tuple[str, str, Image.Image, str]] = [
        (
            "mmgag_aerial_route_setting",
            "MM-GAG航拍线索设置",
            Image.open(DATASET_ASSETS / "02_MM-GAG" / "MMGAG_aerial_search_IMG_1704_panel_ready_ratio1p27.png").convert("RGB")
            if (DATASET_ASSETS / "02_MM-GAG" / "MMGAG_aerial_search_IMG_1704_panel_ready_ratio1p27.png").exists()
            else base,
            "mmgag_aerial",
        ),
        (
            "mmgag_ground_route_setting",
            "MM-GAG地面线索设置",
            Image.open(DATASET_ASSETS / "02_MM-GAG" / "MMGAG_ground_target_IMG_1704_panel_ready_square.png").convert("RGB")
            if (DATASET_ASSETS / "02_MM-GAG" / "MMGAG_ground_target_IMG_1704_panel_ready_square.png").exists()
            else base,
            "mmgag_ground",
        ),
        (
            "mmgag_text_route_setting",
            "MM-GAG文字线索设置",
            make_text_cue_image(),
            "mmgag_text",
        ),
    ]
    gifs = []
    for role, title, target_image, benchmark in cue_sources:
        gifs.append(
            build_setting_gif(
                role=role,
                title=title,
                subtitle="同一5x5搜索协议下的动态路线",
                search_image=base,
                target_image=target_image,
                task=task,
                data_title="MM-GAG数据",
                data_rows=metric_rows_for_benchmark(main, benchmark),
                cue_labels=("目标线索", "当前观测", "起点位置"),
                target_full_image=True,
            )
        )
    return gifs


def build_ultralong_setting_gifs() -> list[dict[str, str]]:
    df = pd.read_csv(BUDGET_TABLE)
    base = Image.open(ASSET_DIR / "img_000.png").convert("RGB")
    settings = [
        {
            "role": "ultralong_grid8_route_setting",
            "title": "8x8超远距离设置",
            "subtitle": "MASA航拍压力测试，D=10-14，正式预算24",
            "grid": 8,
            "budget": 24,
            "start": 56,
            "goal": 7,
            "traj": [56, 48, 40, 32, 24, 16, 8, 0, 1, 2, 3, 4, 5, 6, 7],
        },
        {
            "role": "ultralong_grid10_route_setting",
            "title": "10x10超远距离设置",
            "subtitle": "MASA航拍压力测试，D=14-18，正式预算32",
            "grid": 10,
            "budget": 32,
            "start": 90,
            "goal": 9,
            "traj": [90, 80, 70, 60, 50, 40, 30, 20, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        },
    ]
    gifs = []
    for setting in settings:
        row = df[(df["grid"] == f"{setting['grid']}x{setting['grid']}") & (df["budget"] == setting["budget"]) & (df["method_key"] == "anchor0624")].iloc[0]
        gomaa = df[(df["grid"] == f"{setting['grid']}x{setting['grid']}") & (df["budget"] == setting["budget"]) & (df["method_key"] == "gomaa")].iloc[0]
        task = {
            "traj": setting["traj"],
            "start": setting["start"],
            "goal": setting["goal"],
            "distance": len(setting["traj"]) - 1,
            "grid": setting["grid"],
        }
        data_rows = [
            ("网格/预算", f"{setting['grid']}x{setting['grid']}；B={setting['budget']}", BLUE),
            ("本文成功率", percent(row["success_ratio"]), START),
            ("GOMAA成功率", percent(gomaa["success_ratio"]), SKY),
            ("差值", f"+{(float(row['success_ratio']) - float(gomaa['success_ratio'])) * 100:.1f}百分点", ROUTE),
        ]
        gifs.append(
            build_setting_gif(
                role=setting["role"],
                title=setting["title"],
                subtitle=setting["subtitle"],
                search_image=base,
                target_image=base,
                task=task,
                data_title="压力测试数据",
                data_rows=data_rows,
                cue_labels=("目标格", "当前格", "起点格"),
            )
        )
    return gifs


def build_xbd_compare(settings: list[XbdSetting], main: pd.DataFrame) -> str:
    setting = next((item for item in settings if "disaster" in item.role), settings[-1])
    task = {
        "traj": setting.traj,
        "start": setting.start,
        "goal": setting.goal,
        "distance": 8,
        "grid": 5,
    }
    rows: list[tuple[str, str, str | None]] = []
    for i, item in enumerate(settings):
        subset = main[(main["benchmark"] == item.benchmark) & (main["method"] == "GeoExplorer-anchor0624")].iloc[0]
        color = BLUE if i == 0 else ROUTE
        rows.append((display_label(item.benchmark), f"SR {percent(subset['success_ratio'])} / SG {float(subset['sg_mean']):.3f}", color))
    rows.append(("对照方式", "目标用灾前图；搜索用灾后图", None))
    canvas = build_route_layout_frame(
        title="xBD路线设置对照",
        search_image=setting.search_image,
        target_image=setting.target_image,
        task=task,
        step=len(setting.traj) - 1,
        data_title="xBD对照数据",
        data_rows=rows,
        cue_labels=("目标线索", "灾后观察", "起点位置"),
    )
    out = OUT_DIR / "acceptance_xbd_route_settings_compare.png"
    canvas.save(out, quality=95)
    return output_ref(out)


def build_multimodal_route_setting(main: pd.DataFrame) -> str:
    cases = load_cases()
    task = normalize_route_task(cases["c8_anchor_success"])
    task["grid"] = 5
    base = Image.open(ASSET_DIR / f"img_{int(task['img_idx']):03d}.png").convert("RGB")
    aerial_path = DATASET_ASSETS / "02_MM-GAG" / "MMGAG_aerial_search_IMG_1704_panel_ready_ratio1p27.png"
    ground_path = DATASET_ASSETS / "02_MM-GAG" / "MMGAG_ground_target_IMG_1704_panel_ready_square.png"
    aerial = Image.open(aerial_path).convert("RGB") if aerial_path.exists() else crop_cell_square(base, int(task["goal"]))
    ground = Image.open(ground_path).convert("RGB") if ground_path.exists() else crop_cell_square(base, int(task["goal"]))
    text_cue = make_text_cue_image()
    target_panel = make_multimodal_target_panel(aerial, ground, text_cue)
    rows = []
    for benchmark, color in [("mmgag_aerial", BLUE), ("mmgag_ground", START), ("mmgag_text", PINK)]:
        row = main[(main["benchmark"] == benchmark) & (main["method"] == "GeoExplorer-anchor0624")].iloc[0]
        rows.append((display_label(benchmark), f"SR {percent(row['success_ratio'])} / SG {float(row['sg_mean']):.3f}", color))
    rows.append(("共享设置", "5x5；预算10；D=4-8", None))
    canvas = build_route_layout_frame(
        title="MM-GAG多模态路线设置",
        search_image=base,
        target_image=target_panel,
        task=task,
        step=len(task["traj"]) - 1,
        data_title="MM-GAG数据",
        data_rows=rows,
        cue_labels=("目标线索", "当前观察", "起点位置"),
        target_full_image=True,
    )
    out = OUT_DIR / "acceptance_mmgag_multimodal_route_setting.png"
    canvas.save(out, quality=95)
    return output_ref(out)


def fig_to_image(fig: plt.Figure, size: tuple[int, int]) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    buf.seek(0)
    return fit_contain(Image.open(buf).convert("RGB"), size, WHITE)


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D1D5DB")
    ax.spines["bottom"].set_color("#D1D5DB")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    ax.tick_params(colors="#374151", labelsize=9)


def chart_main_benchmark(main: pd.DataFrame) -> Image.Image:
    keep = [
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
    pivot = main[main["benchmark"].isin(keep) & main["method"].isin(["GeoExplorer-anchor0624", "GOMAA-Geo"])].pivot(
        index="benchmark", columns="method", values="success_ratio"
    )
    pivot = pivot.loc[keep]
    y = np.arange(len(pivot))
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.barh(y + 0.18, pivot["GeoExplorer-anchor0624"], height=0.32, color=START, label="本文方法")
    ax.barh(y - 0.18, pivot["GOMAA-Geo"], height=0.32, color=SKY, label="GOMAA")
    ax.set_yticks(y)
    ax.set_yticklabels([BENCHMARK_ZH.get(s, s) for s in pivot.index], fontsize=8)
    ax.set_xlim(0, 0.75)
    ax.set_xlabel("成功率")
    ax.legend(frameon=False, loc="lower right", prop={"family": MPL_FONT_FAMILY, "size": 9})
    style_axis(ax)
    ax.invert_yaxis()
    return fig_to_image(fig, (1180, 760))


def build_main_benchmark_page(main: pd.DataFrame) -> str:
    canvas, draw = make_canvas("主基准结果证据", "将正式结果表转换为对齐的可视化验收条形图。")
    img = chart_main_benchmark(main)
    canvas.paste(img, (620, 180))
    pivot = main[main["method"].isin(["GeoExplorer-anchor0624", "GOMAA-Geo"])].pivot(
        index="benchmark", columns="method", values="success_ratio"
    )
    shared = pivot.dropna()
    delta = float((shared["GeoExplorer-anchor0624"] - shared["GOMAA-Geo"]).mean())
    xbd_pre_delta = float(shared.loc["xbd_pre_aerial", "GeoExplorer-anchor0624"] - shared.loc["xbd_pre_aerial", "GOMAA-Geo"])
    xbd_post_delta = float(shared.loc["xbd_disaster_aerial", "GeoExplorer-anchor0624"] - shared.loc["xbd_disaster_aerial", "GOMAA-Geo"])
    rows = [
        ("共享基准", f"{len(shared)}项", BLUE),
        ("平均提升", f"+{delta * 100:.2f}百分点", START),
        ("xBD灾前", f"+{xbd_pre_delta * 100:.2f}百分点", ROUTE),
        ("xBD灾后", f"+{xbd_post_delta * 100:.2f}百分点", ROUTE),
        ("证据边界", "仅评估；无重新训练", None),
    ]
    draw_data_column(draw, "表格数据", rows, y=180)
    out = OUT_DIR / "acceptance_main_benchmark_effects.png"
    canvas.save(out, quality=95)
    return output_ref(out)


def chart_ranked_bars(df: pd.DataFrame, label_col: str, value_col: str, colors: list[str], width: int = 1160, height: int = 730) -> Image.Image:
    d = df.copy().sort_values(value_col, ascending=True)
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    ax.barh(np.arange(len(d)), d[value_col].astype(float), color=colors[: len(d)])
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels(d[label_col].astype(str), fontsize=8)
    ax.set_xlabel(VALUE_LABEL_ZH.get(value_col, value_col.replace("_", " ")))
    style_axis(ax)
    return fig_to_image(fig, (width, height))


def build_factorial_page() -> str:
    df = pd.read_csv(GENERALIZATION_TABLE)
    colors = [START if b == "g1_p1_e1_v1" else (SKY if b == "g0_p0_e0_v0" else "#9CA3AF") for b in df["branch"]]
    img = chart_ranked_bars(df, "branch", "primary_generalization_mean", colors)
    canvas, draw = make_canvas("机制消融证据", "G/P/E/V组件消融按综合泛化表现排序。")
    canvas.paste(img, (620, 190))
    best = df.iloc[0]
    control = df[df["branch"] == "g0_p0_e0_v0"].iloc[0]
    rows = [
        ("最优分支", f"{best['branch']}  {percent(best['primary_generalization_mean'])}", START),
        ("对照分支", f"{control['branch']}  {percent(control['primary_generalization_mean'])}", SKY),
        ("提升", f"+{(float(best['primary_generalization_mean']) - float(control['primary_generalization_mean'])) * 100:.2f}百分点", ROUTE),
        ("组件", "G门控，P势函数，E熵项，V远距验证", None),
    ]
    draw_data_column(draw, "消融数据", rows, y=180)
    out = OUT_DIR / "acceptance_factorial_ablation_effects.png"
    canvas.save(out, quality=95)
    return output_ref(out)


def build_reward_page() -> str:
    df = pd.read_csv(REWARD_GATE_TABLE).sort_values("mmgag_mean_sr", ascending=False)
    colors = [START if v == "linear_0.405_pb" else (SKY if "no_pb" in v else "#9CA3AF") for v in df["value"]]
    img = chart_ranked_bars(df, "value", "mmgag_mean_sr", colors, 1180, 760)
    canvas, draw = make_canvas("奖励设计证据", "奖励、门控和势函数均为训练阶段机制；评估仅加载已训练检查点。")
    canvas.paste(img, (620, 180))
    best = df.iloc[0]
    external = df[df["value"] == "external_pbrs"].iloc[0]
    no_pb = df[df["value"] == "linear_0.405_no_pb"].iloc[0]
    rows = [
        ("最优", f"{best['value']}  {percent(best['mmgag_mean_sr'])}", START),
        ("外部+势函数", f"{percent(external['mmgag_mean_sr'])}", SKY),
        ("无势函数", f"{percent(no_pb['mmgag_mean_sr'])}", MUTED),
        ("相对提升", f"+{(float(best['mmgag_mean_sr']) - float(no_pb['mmgag_mean_sr'])) * 100:.2f}百分点", ROUTE),
    ]
    draw_data_column(draw, "奖励数据", rows, y=180)
    out = OUT_DIR / "acceptance_reward_gate_pbrs_effects.png"
    canvas.save(out, quality=95)
    return output_ref(out)


def build_dataset_param_page() -> str:
    dataset_df = pd.read_csv(DATASET_SR_TABLE)
    gate_df = pd.read_csv(GATE_VALDIST_TABLE)
    gate = gate_df[gate_df["family"] == "gate_floor_dense"].copy()
    gate["value_float"] = gate["value"].astype(float)

    fig, axs = plt.subplots(1, 2, figsize=(10.3, 4.8), gridspec_kw={"width_ratios": [1.1, 1.0]})
    ds = dataset_df.sort_values("mean_all_sr", ascending=True)
    axs[0].barh(ds["value"], ds["mean_all_sr"], color=START)
    axs[0].set_title("训练数据组合")
    axs[0].set_xlabel("平均成功率")
    style_axis(axs[0])
    gate = gate.sort_values("value_float")
    axs[1].plot(gate["value_float"], gate["mean_all_sr"], color=ROUTE, marker="o", linewidth=2.5)
    axs[1].set_title("门控下限扫描")
    axs[1].set_xlabel("门控下限")
    axs[1].set_ylabel("平均成功率")
    style_axis(axs[1])
    fig.tight_layout()
    img = fig_to_image(fig, (1180, 720))

    canvas, draw = make_canvas("数据与参数证据", "附录扫描结果可视化：训练数据组合与门控下限敏感性。")
    canvas.paste(img, (620, 210))
    best_ds = dataset_df.sort_values("mean_all_sr", ascending=False).iloc[0]
    best_gate = gate.sort_values("mean_all_sr", ascending=False).iloc[0]
    rows = [
        ("最优数据", f"{best_ds['value']}  {percent(best_ds['mean_all_sr'])}", START),
        ("迁移成功率", percent(best_ds["mean_transfer_sr"]), SKY),
        ("最优门控", f"{best_gate['value_float']:.1f}  {percent(best_gate['mean_all_sr'])}", ROUTE),
        ("扫描范围", "门控下限0.0-1.0", None),
    ]
    draw_data_column(draw, "附录数据", rows, y=180)
    out = OUT_DIR / "acceptance_dataset_parameter_effects.png"
    canvas.save(out, quality=95)
    return output_ref(out)


def build_budget_page() -> str:
    df = pd.read_csv(BUDGET_TABLE)
    df = df[df["method_key"].isin(["anchor0624", "gomaa", "pristine"])]
    labels = {"anchor0624": "本文方法", "gomaa": "GOMAA", "pristine": "原始基线"}
    colors = {"anchor0624": START, "gomaa": SKY, "pristine": "#9CA3AF"}
    fig, axs = plt.subplots(1, 2, figsize=(10.3, 4.8), sharey=True)
    for ax, grid in zip(axs, ["8x8", "10x10"]):
        sub = df[df["grid"] == grid]
        for method in ["anchor0624", "gomaa", "pristine"]:
            m = sub[sub["method_key"] == method].sort_values("budget")
            ax.plot(m["budget"], m["success_ratio"], marker="o", linewidth=2.5, color=colors[method], label=labels[method])
        ax.set_title(f"{grid}预算敏感性")
        ax.set_xlabel("预算")
        style_axis(ax)
    axs[0].set_ylabel("成功率")
    axs[1].legend(frameon=False, loc="lower right", prop={"family": MPL_FONT_FAMILY, "size": 9})
    fig.tight_layout()
    img = fig_to_image(fig, (1180, 720))

    canvas, draw = make_canvas("预算与压力测试证据", "长距离预算曲线将补充实验表格转为验收图。")
    canvas.paste(img, (620, 210))
    row8 = df[(df["grid"] == "8x8") & (df["budget"] == 24) & (df["method_key"] == "anchor0624")].iloc[0]
    row10 = df[(df["grid"] == "10x10") & (df["budget"] == 32) & (df["method_key"] == "anchor0624")].iloc[0]
    gomaa8 = df[(df["grid"] == "8x8") & (df["budget"] == 24) & (df["method_key"] == "gomaa")].iloc[0]
    gomaa10 = df[(df["grid"] == "10x10") & (df["budget"] == 32) & (df["method_key"] == "gomaa")].iloc[0]
    rows = [
        ("8x8正式预算", f"{percent(row8['success_ratio'])} 对 {percent(gomaa8['success_ratio'])}", START),
        ("10x10正式预算", f"{percent(row10['success_ratio'])} 对 {percent(gomaa10['success_ratio'])}", START),
        ("8x8差值", f"+{(float(row8['success_ratio']) - float(gomaa8['success_ratio'])) * 100:.1f}百分点", ROUTE),
        ("10x10差值", f"+{(float(row10['success_ratio']) - float(gomaa10['success_ratio'])) * 100:.1f}百分点", ROUTE),
    ]
    draw_data_column(draw, "压力数据", rows, y=180)
    out = OUT_DIR / "acceptance_budget_stress_effects.png"
    canvas.save(out, quality=95)
    return output_ref(out)


def build_continuation_status_page() -> str:
    canvas, draw = make_canvas("续训实验状态", "固定评估CSV合并完成前，本页仅作为监控图。")
    rows: list[tuple[str, str, str | None]] = []
    if CONTINUATION_STATUS.exists():
        data = json.loads(CONTINUATION_STATUS.read_text(encoding="utf-8"))
        eval_status = data.get("eval_status", {})
        rows = [
            ("训练阶段", str(data.get("remote_status", {}).get("phase", "-")), START),
            ("评估阶段", str(eval_status.get("phase", "-")), ROUTE),
            ("完成数量", f"{eval_status.get('completed', '-')}/{eval_status.get('total', '-')}", BLUE),
            ("规则", "暂不声明最终曲线", None),
        ]
    else:
        rows = [("状态", "未找到监控文件", ROUTE)]
    draw_data_column(draw, "监控数据", rows, y=180)

    x0, y0 = 620, 270
    draw_text_mixed(draw, (x0, y0), "固定评估流程", F["section"], INK)
    steps = [
        ("1", "继续训练", START),
        ("2", "保存检查点", BLUE),
        ("3", "固定MM-GAG评估", ROUTE),
        ("4", "合并CSV后画曲线", PINK),
    ]
    x = x0
    for i, (num, label, color) in enumerate(steps):
        draw.ellipse((x, y0 + 90, x + 70, y0 + 160), fill=color)
        num_font = F_LATIN["section"]
        tw, th = text_size(draw, num, num_font)
        draw.text((x + 35 - tw / 2, y0 + 125 - th / 2), num, font=num_font, fill=WHITE)
        draw_text_fit(draw, (x - 25, y0 + 178), label, 170, F["body"], INK)
        if i < len(steps) - 1:
            draw.line((x + 88, y0 + 125, x + 220, y0 + 125), fill=LIGHT, width=5)
        x += 250
    out = OUT_DIR / "acceptance_continuation_status.png"
    canvas.save(out, quality=95)
    return output_ref(out)


def build_visual_index(outputs: dict[str, Any]) -> str:
    canvas, draw = make_canvas("验收可视化索引", "作为第一张总览图，每个缩略图对应一类实验证据。")
    thumbs: list[tuple[str, Path]] = []
    for item in outputs["route_gifs"][:4]:
        thumbs.append((item["role"], ROOT / item["poster"]))
    for item in outputs.get("setting_gifs", [])[:6]:
        thumbs.append((item["role"], ROOT / item["poster"]))
    for key in ["main_benchmark", "xbd_compare", "mmgag_multimodal", "factorial", "reward", "dataset_param", "budget"]:
        if key in outputs["pages"]:
            thumbs.append((key, ROOT / outputs["pages"][key]))

    thumb_w, thumb_h = 400, 225
    x0, y0 = 58, 150
    for i, (label, path) in enumerate(thumbs[:12]):
        col, row = i % 4, i // 4
        x = x0 + col * 455
        y = y0 + row * 300
        if path.exists():
            canvas.paste(fit_cover(Image.open(path).convert("RGB"), (thumb_w, thumb_h)), (x, y))
        draw_text_fit(draw, (x, y + thumb_h + 10), display_label(label), thumb_w, F["small"], INK)
    out = OUT_DIR / "acceptance_visual_pack_index.png"
    canvas.save(out, quality=95)
    return output_ref(out)


def clean_old_outputs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in OUT_DIR.glob("acceptance_*"):
        if path.is_file():
            path.unlink()


def build_context_and_result_pages(route_outputs: list[dict[str, str]]) -> dict[str, Any]:
    main = load_main_table()
    pages: dict[str, str] = {}
    pages["route_gallery"] = build_route_contact_sheet(route_outputs)
    xbd_settings = build_xbd_settings(main)
    setting_gifs: list[dict[str, str]] = []
    setting_gifs.extend(build_xbd_setting_gifs(xbd_settings, main))
    setting_gifs.extend(build_mmgag_setting_gifs(main))
    setting_gifs.extend(build_ultralong_setting_gifs())
    xbd_pages = [build_xbd_page(setting, main) for setting in xbd_settings]
    pages["xbd_compare"] = build_xbd_compare(xbd_settings, main)
    pages["mmgag_multimodal"] = build_multimodal_route_setting(main)
    pages["main_benchmark"] = build_main_benchmark_page(main)
    pages["factorial"] = build_factorial_page()
    pages["reward"] = build_reward_page()
    pages["dataset_param"] = build_dataset_param_page()
    pages["budget"] = build_budget_page()
    pages["continuation_status"] = build_continuation_status_page()
    return {"pages": pages, "xbd_pages": xbd_pages, "setting_gifs": setting_gifs}


def build_custom_context_pages(route_outputs: list[dict[str, str]]) -> dict[str, Any]:
    pages: dict[str, str] = {}
    pages["route_gallery"] = build_route_contact_sheet(route_outputs)
    return {"pages": pages, "xbd_pages": [], "setting_gifs": []}


def write_report(manifest: dict[str, Any]) -> None:
    if manifest.get("custom_image"):
        lines = [
            "# Acceptance Custom Image Visual Package",
            "",
            f"Custom image: `{manifest.get('custom_image')}`",
            "",
            "This package only contains route inference visuals for the provided image. It does not claim xBD/MM-GAG or ablation-table evidence.",
            "",
            "## Route GIFs",
            "",
        ]
        for item in manifest["route_gifs"]:
            lines.append(
                f"- `{item['role']}`: case `{item.get('case_id', '-')}`, success `{item.get('success', '-')}`, "
                f"GIF `{item['gif']}`, poster `{item['poster']}`"
            )
        lines.extend(["", "## Pages", ""])
        for key, value in manifest["pages"].items():
            lines.append(f"- `{key}`: `{value}`")
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    lines = [
        "# 验收可视化包说明",
        "",
        "这一版按用户反馈重做：路线类 GIF 和图片统一为左侧大路线图、右侧 2x2 四宫格（目标线索、当前观察、起点位置、距离曲线），去掉左上角小字幕；路线样本同时包含平顺成功、不平顺成功和少量不平顺绕行/失败代表。",
        "",
        "## 动态路线 GIF",
        "",
    ]
    for item in manifest["route_gifs"]:
        lines.append(
            f"- `{item['role']}`: case `{item.get('case_id', '-')}`, success `{item.get('success', '-')}`, "
            f"curve `{item.get('curve', '-')}`"
        )
        lines.append(f"- `{item['role']}`: GIF `{item['gif']}`；末帧 `{item['poster']}`")
    lines.extend(["", "## 实验设置动态路线 GIF", ""])
    for item in manifest["setting_gifs"]:
        lines.append(f"- `{item['role']}`: GIF `{item['gif']}`；末帧 `{item['poster']}`")
    lines.extend(["", "## xBD 路线设置图", ""])
    for item in manifest["xbd_pages"]:
        lines.append(f"- `{item['role']}`: `{item['image']}`")
    lines.append(f"- 两种设置对照: `{manifest['pages']['xbd_compare']}`")
    lines.extend(["", "## 其他论文实验验收图", ""])
    for key, value in manifest["pages"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## 证据边界",
            "",
            "- xBD 图使用真实灾前/灾后影像展示 5x5 搜索设置，并引用正式评估表的 SR/SG；不冒充为未记录的逐样本策略轨迹。",
            "- reward/gate/PBRS 只作为训练阶段机制说明；测试阶段只加载训练后的 checkpoint 做策略评估。",
            "- continuation 页面只是状态页；必须等固定评估合并 CSV 完成后，才能画最终续训曲线并下结论。",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clean_old_outputs()
    cases = load_cases()
    route_outputs = []
    for role, task in select_route_cases(cases):
        image_path = ASSET_DIR / f"img_{int(task['img_idx']):03d}.png"
        if not image_path.exists():
            print(f"skip {role}: missing raw base image {image_path}")
            continue
        route_outputs.append(build_route_gif(role, task))

    extra = build_custom_context_pages(route_outputs) if CUSTOM_IMAGE_MODE else build_context_and_result_pages(route_outputs)
    manifest: dict[str, Any] = {
        "generated_by": "code/tools/build_acceptance_demo_visuals.py",
        "design": "custom image route-only package" if CUSTOM_IMAGE_MODE else "white background, aligned grid, square cues, route maps plus experiment evidence pages",
        "custom_image": os.environ.get("ACCEPTANCE_CUSTOM_IMAGE", "") or None,
        "route_gifs": route_outputs,
        **extra,
    }
    manifest["pages"]["index"] = build_visual_index(manifest)
    (OUT_DIR / "acceptance_demo_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(manifest)
    print(json.dumps({"out_dir": str(OUT_DIR), "report": str(REPORT_PATH), "route_gifs": len(route_outputs), "pages": len(manifest["pages"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
