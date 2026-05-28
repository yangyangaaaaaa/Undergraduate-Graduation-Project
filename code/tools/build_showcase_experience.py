#!/usr/bin/env python
"""Build experience-level showcase media for the GitHub landing page.

The polished chart layer is useful as a reproducible figure bank, but the
project landing page needs a stronger visual narrative.  This script creates
cinematic, data-backed assets that combine real trajectory media, concise
method schematics, and high-signal result callouts.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps, ImageSequence


ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "results" / "tables"
SHOWCASE_DIR = ROOT / "results" / "figures" / "showcase"
EXP_DIR = SHOWCASE_DIR / "experience"
THEATER_DIR = EXP_DIR / "trajectory_theater_gifs"
GIF_DIR = SHOWCASE_DIR / "trajectories" / "gifs"

INK = "#09111F"
PAPER = "#F7F8F4"
MIST = "#D8E5E7"
BLUE = "#1F77B4"
CYAN = "#6EC6E8"
ORANGE = "#E66A1A"
GREEN = "#13A37F"
YELLOW = "#F6C85F"
RED = "#C43E3E"
SLATE = "#627083"
VIOLET = "#7E6BD1"
WHITE = "#FFFFFF"

METHODS = [
    ("anchor0624", "本文方法", BLUE, "到达目标"),
    ("gomaa", "GOMAA-Geo", ORANGE, "未到达目标"),
    ("pristine", "GeoExplorer", GREEN, "偏离目标"),
]

BENCHMARK_DISPLAY_LABEL = {
    "masa_aerial": "MASA 航拍",
    "mmgag_aerial": "MM-GAG 航拍",
    "mmgag_ground": "MM-GAG 地面",
    "mmgag_text": "MM-GAG 文本",
    "swissview100_aerial": "SwissView100 航拍",
    "swissviewmonuments_aerial": "SwissMon 航拍",
    "swissviewmonuments_ground": "SwissMon 地面",
    "xbd_pre_aerial": "xBD 灾前",
    "xbd_disaster_aerial": "xBD 灾后",
}


def display_benchmark(value: object) -> str:
    text = str(value)
    return BENCHMARK_DISPLAY_LABEL.get(text, text.replace("_", " "))


def ensure_dirs() -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    THEATER_DIR.mkdir(parents=True, exist_ok=True)


def font(size: int, bold: bool = False, serif: bool = False) -> ImageFont.FreeTypeFont:
    if serif:
        candidates = ["C:/Windows/Fonts/timesbd.ttf" if bold else "C:/Windows/Fonts/times.ttf"]
    else:
        candidates = [
            "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
        ]
    for candidate in candidates:
        p = Path(candidate)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


F = {
    "display": font(78, bold=True),
    "h1": font(54, bold=True),
    "h2": font(34, bold=True),
    "h3": font(26, bold=True),
    "body": font(24),
    "small": font(18),
    "tiny": font(15),
    "number": font(70, bold=True, serif=True),
}


def read_csv(rel: str) -> pd.DataFrame:
    return pd.read_csv(TABLE_DIR / rel)


def normalize_method(value: object) -> str:
    text = str(value)
    return {
        "GeoExplorer-anchor0624": "Ours",
        "anchor0624": "Ours",
        "gomaa": "GOMAA-Geo",
        "GeoExplorer-pristine": "GeoExplorer",
        "pristine": "GeoExplorer",
    }.get(text, text)


def rounded_mask(size: tuple[int, int], radius: int) -> Image.Image:
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=radius, fill=255)
    return mask


def paste_round(
    canvas: Image.Image,
    im: Image.Image,
    xy: tuple[int, int],
    size: tuple[int, int],
    radius: int = 34,
    border: str | None = None,
    border_width: int = 3,
) -> None:
    im = fit_cover(im.convert("RGB"), size)
    shadow = Image.new("RGBA", (size[0] + 46, size[1] + 46), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.rounded_rectangle((23, 23, 23 + size[0], 23 + size[1]), radius=radius, fill=(3, 10, 20, 110))
    shadow = shadow.filter(ImageFilter.GaussianBlur(18))
    canvas.alpha_composite(shadow, (xy[0] - 23, xy[1] - 18))
    mask = rounded_mask(size, radius)
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    layer.paste(im, (0, 0), mask)
    canvas.alpha_composite(layer, xy)
    if border:
        draw = ImageDraw.Draw(canvas)
        draw.rounded_rectangle(
            (xy[0], xy[1], xy[0] + size[0], xy[1] + size[1]),
            radius=radius,
            outline=border,
            width=border_width,
        )


def fit_cover(im: Image.Image, size: tuple[int, int]) -> Image.Image:
    w, h = im.size
    tw, th = size
    scale = max(tw / w, th / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    im = im.resize((nw, nh), Image.Resampling.LANCZOS)
    left = (nw - tw) // 2
    top = (nh - th) // 2
    return im.crop((left, top, left + tw, top + th))


def fit_contain(im: Image.Image, size: tuple[int, int], bg: str = WHITE) -> Image.Image:
    w, h = im.size
    tw, th = size
    scale = min(tw / w, th / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    im = im.resize((nw, nh), Image.Resampling.LANCZOS).convert("RGB")
    out = Image.new("RGB", size, bg)
    out.paste(im, ((tw - nw) // 2, (th - nh) // 2))
    return out


def draw_shadow_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font_obj: ImageFont.ImageFont,
    fill: str | tuple[int, int, int, int],
    stroke: int = 2,
) -> None:
    draw.text(xy, text, font=font_obj, fill=fill, stroke_width=stroke, stroke_fill=(0, 0, 0, 185))


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font_obj: ImageFont.ImageFont,
    fill: str,
    width: int,
    line_gap: int = 8,
) -> int:
    x, y = xy
    has_spaces = " " in text
    words = text.split() if has_spaces else list(text)
    sep = " " if has_spaces else ""
    line = ""
    for word in words:
        candidate = f"{line}{sep}{word}".strip() if has_spaces else f"{line}{word}"
        if draw.textlength(candidate, font=font_obj) <= width or not line:
            line = candidate
        else:
            draw.text((x, y), line, font=font_obj, fill=fill)
            y += font_obj.size + line_gap
            line = word
    if line:
        draw.text((x, y), line, font=font_obj, fill=fill)
        y += font_obj.size + line_gap
    return y


def gradient_bg(size: tuple[int, int], top: str = "#08101E", bottom: str = "#11283A") -> Image.Image:
    w, h = size
    a = np.array(Image.new("RGB", (1, 1), top), dtype=np.float32)[0, 0]
    b = np.array(Image.new("RGB", (1, 1), bottom), dtype=np.float32)[0, 0]
    ys = np.linspace(0, 1, h)[:, None]
    arr = ((1 - ys) * a + ys * b).astype(np.uint8)
    arr = np.repeat(arr[:, None, :], w, axis=1)
    return Image.fromarray(arr, "RGB").convert("RGBA")


def draw_chip(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    label: str,
    value: str,
    color: str,
    width: int = 310,
    height: int = 128,
) -> None:
    x, y = xy
    draw.rounded_rectangle((x, y, x + width, y + height), radius=26, fill=(255, 255, 255, 26), outline=(255, 255, 255, 58), width=2)
    draw.text((x + 26, y + 20), label, font=F["small"], fill="#5D6B7C")
    draw.text((x + 26, y + 50), value, font=F["number"], fill=color)


def load_gif_frames(base: str, suffix: str) -> tuple[list[Image.Image], int]:
    p = GIF_DIR / f"{base}__{suffix}.gif"
    im = Image.open(p)
    frames = [frame.convert("RGB") for frame in ImageSequence.Iterator(im)]
    durations = [int(frame.info.get("duration", 240)) for frame in ImageSequence.Iterator(im)]
    return frames, int(np.median(durations)) if durations else 240


def crop_map_frame(frame: Image.Image) -> Image.Image:
    # The source GIF includes a white title band and border.  Cropping keeps the
    # meaningful aerial evidence and lets the new layout own the labels.
    return frame.crop((100, 112, 613, 582))


def first_last_frames(base: str) -> dict[str, Image.Image]:
    out = {}
    for suffix, _, _, _ in METHODS:
        frames, _ = load_gif_frames(base, suffix)
        out[f"{suffix}_first"] = crop_map_frame(frames[0])
        out[f"{suffix}_last"] = crop_map_frame(frames[-1])
    return out


def compute_metrics() -> dict[str, float]:
    main = read_csv("main_benchmark/paper_baseline_compare_table.csv")
    main["method_clean"] = main["method"].map(normalize_method)
    pair = main[main["method_clean"].isin(["Ours", "GOMAA-Geo"])].copy()
    pivot = pair.pivot_table(index="benchmark", columns="method_clean", values="success_ratio", aggfunc="first").dropna()
    mmgag = pivot.loc[[x for x in ["mmgag_aerial", "mmgag_ground", "mmgag_text"] if x in pivot.index]]

    ultra = read_csv("ultra_long/ultra_long_v2_summary.csv")
    ultra["method_clean"] = ultra["method_key"].map(normalize_method)
    ultra_delta = ultra[ultra["method_clean"].eq("Ours")]["success_ratio"].mean() - ultra[ultra["method_clean"].eq("GOMAA-Geo")]["success_ratio"].mean()

    ablation = read_csv("ablation/anchor0624_generalization_table.csv")
    full = float(ablation[ablation["branch"].eq("g1_p1_e1_v1")]["primary_generalization_mean"].iloc[0])
    control = float(ablation[ablation["branch"].eq("g0_p0_e0_v0")]["primary_generalization_mean"].iloc[0])

    reward = read_csv("ablation/reward_gate_type_mmgag_only_table_with_linear.csv")
    gate = float(reward[reward["value"].eq("linear_0.405_pb")]["mmgag_mean_sr"].iloc[0])
    external = float(reward[reward["value"].eq("external_pbrs")]["mmgag_mean_sr"].iloc[0])

    return {
        "shared_mean": float(pivot["Ours"].mean()),
        "shared_gain": float((pivot["Ours"] - pivot["GOMAA-Geo"]).mean()),
        "mmgag_gain": float((mmgag["Ours"] - mmgag["GOMAA-Geo"]).mean()),
        "ultra_gain": float(ultra_delta),
        "ablation_full": full,
        "ablation_gain": full - control,
        "gate_sr": gate,
        "gate_gain": gate - external,
    }


def draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: str, width: int = 6) -> None:
    draw.line((start, end), fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    length = 22
    spread = 0.55
    p1 = (end[0] - length * math.cos(angle - spread), end[1] - length * math.sin(angle - spread))
    p2 = (end[0] - length * math.cos(angle + spread), end[1] - length * math.sin(angle + spread))
    draw.polygon([end, p1, p2], fill=color)


def build_hero() -> None:
    metrics = compute_metrics()
    frames = first_last_frames("three_method_hardcase__img189_d6_s20_g14_r0")
    bg_src = fit_cover(frames["anchor0624_last"], (2560, 1440)).filter(ImageFilter.GaussianBlur(14))
    bg = ImageEnhance_like(bg_src, brightness=0.45, contrast=1.20).convert("RGBA")
    overlay = gradient_bg((2560, 1440), "#06101D", "#0D2437")
    canvas = Image.blend(bg, overlay, 0.72)
    draw = ImageDraw.Draw(canvas)

    # Subtle satellite grid.
    for x in range(0, 2560, 160):
        draw.line((x, 0, x, 1440), fill=(255, 255, 255, 12), width=1)
    for y in range(0, 1440, 160):
        draw.line((0, y, 2560, y), fill=(255, 255, 255, 10), width=1)

    draw.text((120, 96), "好奇心驱动", font=F["display"], fill=WHITE)
    draw.text((120, 178), "无人机主动目标定位", font=font(62, bold=True), fill=WHITE)
    draw_wrapped(
        draw,
        (124, 275),
        "智能体在离散航拍网格中主动搜索目标。训练阶段通过距离门控好奇心奖励和势函数奖励塑形学习策略，推理阶段使用训练好的 Actor-Critic 网络选择动作。",
        F["body"],
        "#D9E6F2",
        1040,
        line_gap=10,
    )

    chips = [
        ("平均成功率", f"{metrics['shared_mean']:.3f}", "#7CC7FF"),
        ("相对提升", f"+{metrics['shared_gain']:.3f}", "#6FE0B5"),
        ("跨模态提升", f"+{metrics['mmgag_gain']:.3f}", "#C3B5FF"),
        ("长距离提升", f"+{metrics['ultra_gain']:.3f}", "#FFD66B"),
    ]
    for i, chip in enumerate(chips):
        draw_chip(draw, (124 + (i % 2) * 340, 535 + (i // 2) * 150), *chip)

    # Main trajectory evidence.
    paste_round(canvas, frames["anchor0624_last"], (1380, 250), (890, 740), radius=42, border="#7CC7FF", border_width=5)
    draw.rounded_rectangle((1428, 298, 1820, 368), radius=24, fill=(31, 119, 180, 225))
    draw.text((1455, 314), "同一困难样例：到达目标", font=F["h3"], fill=WHITE)
    draw.text((1415, 1040), "轨迹证据", font=F["h2"], fill=WHITE)
    draw_wrapped(
        draw,
        (1415, 1092),
        "成功轨迹不只是路径更短，更关键是持续朝目标推进；对比方法更容易出现绕行和重复访问。",
        F["body"],
        "#D9E6F2",
        850,
    )

    # Pipeline strip.
    strip_y = 1220
    stages = [
        ("目标线索", "航拍 / 地面 / 文本"),
        ("历史序列", "动作 + 观测"),
        ("Transformer", "状态与下一特征预测"),
        ("Actor-Critic", "合法动作分布"),
        ("移动", "更新网格位置"),
    ]
    x = 120
    for i, (title, subtitle) in enumerate(stages):
        w = 365 if i == 2 else 275
        draw.rounded_rectangle((x, strip_y, x + w, strip_y + 125), radius=28, fill=(255, 255, 255, 235), outline=(255, 255, 255, 90), width=2)
        draw.text((x + 24, strip_y + 22), title, font=F["h3"], fill=INK)
        draw.text((x + 24, strip_y + 66), subtitle, font=F["small"], fill=SLATE)
        if i < len(stages) - 1:
            draw_arrow(draw, (x + w + 18, strip_y + 62), (x + w + 82, strip_y + 62), "#8BBCE8", width=5)
        x += w + 96

    canvas.convert("RGB").save(EXP_DIR / "hero_experience.png", quality=95)


def ImageEnhance_like(im: Image.Image, brightness: float = 1.0, contrast: float = 1.0) -> Image.Image:
    from PIL import ImageEnhance

    im = ImageEnhance.Brightness(im).enhance(brightness)
    im = ImageEnhance.Contrast(im).enhance(contrast)
    return im


def build_evidence_wall() -> None:
    metrics = compute_metrics()
    main = read_csv("main_benchmark/paper_baseline_compare_table.csv")
    main["method_clean"] = main["method"].map(normalize_method)
    traj = read_csv("trajectory_analysis/trajectory_behavior_by_distance.csv")
    budget = read_csv("supplement_eval/budget_sensitivity_table.csv")
    budget["method_clean"] = budget["method"].map(normalize_method)

    pair = main[main["method_clean"].isin(["Ours", "GOMAA-Geo"])].copy()
    pivot = pair.pivot_table(index="benchmark", columns="method_clean", values="success_ratio", aggfunc="first").dropna()
    pivot["gain"] = pivot["Ours"] - pivot["GOMAA-Geo"]
    pivot = pivot.sort_values("gain", ascending=True)

    canvas = Image.new("RGBA", (2400, 1350), "#F6F8F5")
    draw = ImageDraw.Draw(canvas)

    draw.text((90, 56), "核心结果证据墙", font=F["h1"], fill=INK)
    draw.text((94, 126), "从总体性能、跨模态适应、长距离搜索和轨迹行为四个角度说明方法改进。", font=F["body"], fill="#445166")
    metric_tiles = [
        ("平均成功率", f"{metrics['shared_mean']:.3f}", BLUE),
        ("平均提升", f"+{metrics['shared_gain']:.3f}", GREEN),
        ("完整机制收益", f"+{metrics['ablation_gain']:.3f}", VIOLET),
        ("长距离提升", f"+{metrics['ultra_gain']:.3f}", YELLOW),
    ]
    for i, (label, value, color) in enumerate(metric_tiles):
        x = 1160 + i * 290
        y = 44
        draw.rounded_rectangle((x, y, x + 250, y + 112), radius=20, fill=WHITE, outline="#DCE3E4", width=2)
        draw.rectangle((x + 18, y + 18, x + 25, y + 92), fill=color)
        draw.text((x + 42, y + 20), label, font=F["small"], fill="#627083")
        draw.text((x + 42, y + 44), value, font=F["number"], fill=color)

    def section(x: int, y: int, w: int, h: int, title: str, tag: str, color: str, note: str) -> None:
        draw.rounded_rectangle((x, y, x + w, y + h), radius=18, fill=WHITE, outline="#DDE4E5", width=2)
        draw.rectangle((x + 24, y + 26, x + 34, y + 80), fill=color)
        draw.text((x + 52, y + 24), tag, font=F["h3"], fill=color)
        draw.text((x + 105, y + 22), title, font=F["h2"], fill=INK)
        draw.line((x + 30, y + 92, x + w - 30, y + 92), fill="#E5EAEC", width=2)
        draw_wrapped(draw, (x + 34, y + h - 72), note, F["small"], "#445166", w - 68, line_gap=6)

    # 01 benchmark gains.
    x, y, w, h = 90, 230, 1030, 460
    section(x, y, w, h, "主实验：各数据集上的成功率提升", "01", BLUE, "绿色条表示本文方法相对 GOMAA-Geo 的成功率增益，多个数据集同时提升说明改进不是单一场景偶然现象。")
    bar_x, bar_y = x + 360, y + 116
    bar_w, row_h = 485, 30
    min_gain, max_gain = float(pivot["gain"].min()), float(pivot["gain"].max())
    scale = bar_w / max(0.001, max_gain - min_gain)
    for i, (bench, row) in enumerate(pivot.iterrows()):
        yy = bar_y + i * row_h
        label = display_benchmark(bench)
        draw.text((x + 32, yy - 4), label, font=F["tiny"], fill="#293649")
        zero_x = bar_x + int((0 - min_gain) * scale)
        val_x = bar_x + int((row["gain"] - min_gain) * scale)
        draw.line((bar_x, yy + 12, bar_x + bar_w, yy + 12), fill="#EDF1F2", width=3)
        draw.line((zero_x, yy + 12, val_x, yy + 12), fill=GREEN if row["gain"] >= 0 else RED, width=16)
        draw.ellipse((val_x - 8, yy + 4, val_x + 8, yy + 20), fill=GREEN if row["gain"] >= 0 else RED)
        draw.text((bar_x + bar_w + 18, yy - 4), f"{row['gain']:+.3f}", font=F["tiny"], fill=GREEN if row["gain"] >= 0 else RED)
    draw.line((bar_x + int((0 - min_gain) * scale), bar_y - 12, bar_x + int((0 - min_gain) * scale), bar_y + len(pivot) * row_h), fill="#6B7280", width=2)

    # 02 MM-GAG cross-modal.
    x, y, w, h = 1190, 230, 1120, 460
    section(x, y, w, h, "跨模态目标：图像与文本线索都能受益", "02", GREEN, "航拍图、地面图和文本描述三类目标线索下均保持优势，说明目标表示与历史序列建模具有较好的跨模态适应性。")
    targets = [("mmgag_aerial", "航拍目标"), ("mmgag_ground", "地面目标"), ("mmgag_text", "文本目标")]
    for i, (bench, label) in enumerate(targets):
        ours = float(main[(main["benchmark"].eq(bench)) & (main["method_clean"].eq("Ours"))]["success_ratio"].iloc[0])
        gomaa = float(main[(main["benchmark"].eq(bench)) & (main["method_clean"].eq("GOMAA-Geo"))]["success_ratio"].iloc[0])
        yy = y + 140 + i * 76
        x1 = x + 250 + int(gomaa * 760)
        x2 = x + 250 + int(ours * 760)
        draw.text((x + 40, yy - 18), label, font=F["body"], fill=INK)
        draw.line((x + 250, yy, x + 250 + int(0.70 * 760), yy), fill="#EDF1F2", width=4)
        draw.line((x1, yy, x2, yy), fill="#BFD7EA", width=18)
        draw.ellipse((x1 - 16, yy - 16, x1 + 16, yy + 16), fill=ORANGE)
        draw.ellipse((x2 - 18, yy - 18, x2 + 18, yy + 18), fill=BLUE)
        draw.text((x2 + 34, yy - 18), f"+{ours-gomaa:.3f}", font=F["h3"], fill=GREEN)
    draw.text((x + 360, y + 352), "橙色为 GOMAA-Geo，蓝色为本文方法", font=F["small"], fill=SLATE)

    # 03 long-range budget curves.
    x, y, w, h = 90, 745, 1030, 505
    section(x, y, w, h, "长距离搜索：预算变化下仍保持优势", "03", ORANGE, "8x8 与 10x10 网格中，预算增加后两种方法都会变好，但本文方法曲线整体更高，说明优势并非来自某一个固定预算点。")
    for idx, grid in enumerate(["8x8", "10x10"]):
        sub = budget[(budget["grid"].eq(grid)) & (budget["method_clean"].isin(["Ours", "GOMAA-Geo"]))].copy()
        if sub.empty:
            continue
        bx = x + 65 + idx * 465
        by = y + 135
        bw, bh = 380, 230
        draw.text((bx, y + 100), grid, font=F["h3"], fill=INK)
        draw.line((bx, by + bh, bx + bw, by + bh), fill="#A9B4BE", width=2)
        draw.line((bx, by, bx, by + bh), fill="#A9B4BE", width=2)
        budgets = sorted(sub["budget"].unique())
        for method, color in [("GOMAA-Geo", ORANGE), ("Ours", BLUE)]:
            hit = sub[sub["method_clean"].eq(method)].sort_values("budget")
            pts = []
            for _, row in hit.iterrows():
                xx = bx + int((budgets.index(row["budget"]) / max(1, len(budgets) - 1)) * bw)
                yy = by + bh - int(row["success_ratio"] * bh)
                pts.append((xx, yy))
            if len(pts) > 1:
                draw.line(pts, fill=color, width=7)
            for pt in pts:
                draw.ellipse((pt[0] - 9, pt[1] - 9, pt[0] + 9, pt[1] + 9), fill=color)
            if pts:
                label = "本文方法" if method == "Ours" else method
                dx = -100 if method == "Ours" else -72
                dy = -48 if method == "Ours" else 14
                draw.text((pts[-1][0] + dx, pts[-1][1] + dy), label, font=F["tiny"], fill=color)
        draw.text((bx, by + bh + 22), f"B={budgets[0]}..{budgets[-1]}", font=F["small"], fill=SLATE)

    # 04 trajectory behavior.
    x, y, w, h = 1190, 745, 1120, 505
    section(x, y, w, h, "搜索轨迹：更稳定地接近目标", "04", VIOLET, "C=8 困难样例中，本文方法成功率更高、接近目标比例更高，同时重复访问更少，轨迹行为更符合主动搜索目标。")
    sub = traj[traj["distance"].eq(8)].copy()
    metrics_bars = [
        ("success_rate", "成功率"),
        ("progress_ratio", "接近目标"),
        ("monotonic_step_rate", "单调接近"),
        ("revisit_rate", "重复访问"),
    ]
    for i, (metric, name) in enumerate(metrics_bars):
        yy = y + 126 + i * 62
        draw.text((x + 44, yy + 5), name, font=F["body"], fill=INK)
        for method, color, offset in [("GOMAA-Geo", ORANGE, 0), ("GeoExplorer-anchor0624", BLUE, 28)]:
            val = float(sub[sub["method"].eq(method)][metric].iloc[0])
            bar_len = int(val * 760)
            draw.rounded_rectangle((x + 230, yy + offset, x + 230 + bar_len, yy + offset + 20), radius=10, fill=color)
            draw.text((x + 1010, yy + offset - 4), f"{val:.2f}", font=F["tiny"], fill=color)
    draw.text((x + 230, y + 392), "橙色为 GOMAA-Geo，蓝色为本文方法", font=F["small"], fill=SLATE)

    canvas.convert("RGB").save(EXP_DIR / "evidence_wall_experience.png", quality=95)


def build_trajectory_storyboard(base: str = "three_method_hardcase__img189_d6_s20_g14_r0") -> None:
    frames = first_last_frames(base)
    canvas = gradient_bg((2400, 1040), "#07101D", "#142A3D")
    draw = ImageDraw.Draw(canvas)
    draw.text((88, 58), "同一困难样例下的三种搜索行为", font=F["h1"], fill=WHITE)
    draw.text((92, 126), "起点、目标和预算完全一致；标签直接嵌入图像，便于对比真实搜索路径。", font=F["body"], fill="#C8D6E4")
    panel_w, panel_h = 710, 650
    gutter = 16
    start_x, y0 = 75, 235
    for j, (suffix, label, color, outcome) in enumerate(METHODS):
        x = start_x + j * (panel_w + gutter)
        panel = fit_cover(frames[f"{suffix}_last"], (panel_w, panel_h)).convert("RGBA")
        canvas.alpha_composite(panel, (x, y0))
        draw.rectangle((x, y0, x + 8, y0 + panel_h), fill=color)
        draw_shadow_text(draw, (x + 34, y0 + 32), label, F["h2"], WHITE, stroke=3)
        badge = "到达目标" if suffix == "anchor0624" else outcome
        draw_shadow_text(draw, (x + 34, y0 + panel_h - 70), badge, F["h2"], color, stroke=3)
    draw.text(
        (105, 950),
        "绿色框为起点，黄色框为目标；编号标记表示搜索顺序。对比重点是路径本身，因此版式不再使用白色说明卡片或装饰边框。",
        font=F["body"],
        fill="#C8D6E4",
    )
    canvas.convert("RGB").save(EXP_DIR / "trajectory_storyboard_experience.png", quality=95)


def theater_frame(
    base: str,
    i: int,
    n: int,
    loaded: list[list[Image.Image]],
    title: str,
) -> Image.Image:
    panel_w, panel_h = 568, 520
    gutter = 16
    canvas = Image.new("RGBA", (panel_w * 3 + gutter * 2, panel_h), "#07101D")
    draw = ImageDraw.Draw(canvas)

    for j, (suffix, label, color, outcome) in enumerate(METHODS):
        x0 = j * (panel_w + gutter)
        frame = crop_map_frame(loaded[j][min(i, len(loaded[j]) - 1)])
        panel = fit_cover(frame, (panel_w, panel_h))
        canvas.alpha_composite(panel.convert("RGBA"), (x0, 0))
        # Text is burned into the image instead of being placed in cards.
        draw_shadow_text(draw, (x0 + 16, 12), label, font(22, bold=True), WHITE, stroke=2)
        draw_shadow_text(draw, (x0 + panel_w - 118, 14), f"步 {i:02d}/{n - 1:02d}", F["tiny"], "#EAF3FB", stroke=2)
        status = "到达目标" if suffix == "anchor0624" and i == n - 1 else (outcome if i == n - 1 else "搜索中")
        meta = f"{status} | C=6 | 图像 189 | 步数 {i}/{n - 1}"
        draw_shadow_text(draw, (x0 + 16, panel_h - 30), meta, F["tiny"], "#F2F7FA", stroke=2)

    # Ultra-thin progress indicator, kept inside the image instead of adding a
    # separate caption area.
    progress_w = canvas.size[0]
    draw.rectangle((0, panel_h - 4, progress_w, panel_h), fill=(5, 10, 18, 160))
    draw.rectangle((0, panel_h - 4, int(progress_w * i / max(1, n - 1)), panel_h), fill=BLUE)
    return canvas.convert("P", palette=Image.Palette.ADAPTIVE, colors=128)


def build_theater_gifs() -> None:
    base_names = sorted(
        p.name.removesuffix("__anchor0624.gif")
        for p in GIF_DIR.glob("*__anchor0624.gif")
        if (GIF_DIR / (p.name.removesuffix("__anchor0624.gif") + "__gomaa.gif")).exists()
        and (GIF_DIR / (p.name.removesuffix("__anchor0624.gif") + "__pristine.gif")).exists()
    )
    for base in base_names:
        loaded = []
        durations = []
        for suffix, _, _, _ in METHODS:
            frames, duration = load_gif_frames(base, suffix)
            loaded.append(frames)
            durations.append(duration)
        n = max(len(x) for x in loaded)
        frames_out = [theater_frame(base, i, n, loaded, base) for i in range(n)]
        out = THEATER_DIR / f"{base}__theater.gif"
        frames_out[0].save(
            out,
            save_all=True,
            append_images=frames_out[1:],
            duration=max(220, int(np.median(durations))),
            loop=0,
            optimize=True,
        )


def write_manifest() -> None:
    files = sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in EXP_DIR.rglob("*") if p.is_file())
    (EXP_DIR / "experience_manifest.json").write_text(
        json.dumps(
            {
                "generated_by": "code/tools/build_showcase_experience.py",
                "style": "cinematic scientific landing-page assets",
                "file_count": len(files),
                "files": files,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def build_experience() -> None:
    ensure_dirs()
    build_hero()
    build_evidence_wall()
    build_trajectory_storyboard()
    build_theater_gifs()
    write_manifest()
    print(f"Experience showcase written to {EXP_DIR}")


if __name__ == "__main__":
    build_experience()
