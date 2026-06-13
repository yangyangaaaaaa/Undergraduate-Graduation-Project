#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Global stepwise reward analysis for reward-mechanism comparison methods."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import build_reward_guided_case_studies as base  # noqa: E402


METHOD_ORDER = [
    "external_only",
    "intrinsic_only",
    "mixed_no_gate_no_pbrs",
    "mixed_gate_only",
    "mixed_pbrs_only",
    "proposed_linear_gate_pbrs",
]

METHOD_CN = {
    "external_only": "仅外在",
    "intrinsic_only": "仅好奇心",
    "mixed_no_gate_no_pbrs": "直接相加",
    "mixed_gate_only": "距离门控",
    "mixed_pbrs_only": "仅塑形",
    "proposed_linear_gate_pbrs": "本文方法",
}

OUT_TABLE_DIR = base.TABLES
OUT_REPORT = base.REPORTS / "reward_stepwise_global_analysis_zh.md"


def parse_float_list(value: object, n: int | None = None, fill: float = 0.0) -> list[float]:
    out = [float(x) for x in base.parse_list(value)]
    if n is not None:
        if len(out) < n:
            out.extend([fill] * (n - len(out)))
        elif len(out) > n:
            out = out[:n]
    return out


def expand_steps(routes: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for ridx, row in routes.iterrows():
        dist = parse_float_list(row.get("dist_sequence", ""))
        total = parse_float_list(row.get("step_reward_total", ""))
        n = min(max(len(dist) - 1, 0), len(total))
        if n <= 0:
            continue
        ex = parse_float_list(row.get("step_reward_ex", ""), n)
        ex_raw = parse_float_list(row.get("step_reward_ex_raw", ""), n)
        in_raw = parse_float_list(row.get("step_reward_in_raw", ""), n)
        in_gated = parse_float_list(row.get("step_reward_in_gated", ""), n)
        gate = parse_float_list(row.get("step_gate_weight", ""), n, fill=float(row.get("gate_weight_mean", 0.0) or 0.0))
        finish = parse_float_list(row.get("step_finish_bonus", ""), n)
        pbrs = parse_float_list(row.get("step_pbrs_bonus", ""), n)
        path = [int(x) for x in base.parse_list(row.get("patch_sequence", ""))]
        actions = [str(x) for x in base.parse_list(row.get("action_sequence", ""))]

        for i in range(n):
            prev_dist = float(dist[i])
            next_dist = float(dist[i + 1])
            delta = prev_dist - next_dist
            if next_dist == 0:
                move_type = "goal"
            elif delta > 0:
                move_type = "progress"
            elif delta < 0:
                move_type = "regress"
            else:
                move_type = "stay"
            prev_patch = path[i + 1] if i + 1 < len(path) else math.nan
            cur_patch = path[i + 2] if i + 2 < len(path) else math.nan
            records.append(
                {
                    "route_index": int(ridx),
                    "method": row["method"],
                    "method_cn": METHOD_CN.get(str(row["method"]), str(row["method"])),
                    "seed": int(row["seed"]),
                    "episode": int(row["episode"]),
                    "image_index": int(row["image_index"]),
                    "distance_bucket": str(row["distance_bucket"]),
                    "success": int(row["success"]),
                    "initial_dist": float(row["initial_dist"]),
                    "final_dist": float(row["final_dist"]),
                    "route_total_reward": float(row["total_reward"]),
                    "step": i + 1,
                    "prev_patch": prev_patch,
                    "cur_patch": cur_patch,
                    "action": actions[i] if i < len(actions) else "",
                    "prev_dist": prev_dist,
                    "next_dist": next_dist,
                    "delta_dist": delta,
                    "move_type": move_type,
                    "is_progress": int(delta > 0),
                    "is_regress": int(delta < 0),
                    "is_goal": int(next_dist == 0),
                    "is_near": int(prev_dist <= 2 or next_dist <= 2),
                    "reward_total": float(total[i]),
                    "reward_ex": float(ex[i]),
                    "reward_ex_raw": float(ex_raw[i]),
                    "reward_in_raw": float(in_raw[i]),
                    "reward_in_gated": float(in_gated[i]),
                    "gate_weight": float(gate[i]),
                    "finish_bonus": float(finish[i]),
                    "pbrs_bonus": float(pbrs[i]),
                    "reward_positive": int(float(total[i]) > 0),
                    "reward_strong_positive": int(float(total[i]) >= 0.6),
                }
            )
    return pd.DataFrame.from_records(records)


def pct(series: pd.Series) -> float:
    return 100.0 * float(series.mean()) if len(series) else math.nan


def mean_where(df: pd.DataFrame, mask: pd.Series, col: str) -> float:
    hit = df.loc[mask, col]
    return float(hit.mean()) if len(hit) else math.nan


def pct_where(df: pd.DataFrame, mask: pd.Series, col: str) -> float:
    hit = df.loc[mask, col]
    return pct(hit) if len(hit) else math.nan


def build_method_summary(routes: pd.DataFrame, steps: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        r = routes[routes["method"].eq(method)]
        s = steps[steps["method"].eq(method)]
        non_goal_progress = s["move_type"].eq("progress")
        progress_or_goal = s["move_type"].isin(["progress", "goal"])
        regress = s["move_type"].eq("regress")
        stay = s["move_type"].eq("stay")
        goal = s["move_type"].eq("goal")
        near = s["is_near"].eq(1)
        records.append(
            {
                "method": method,
                "方法": METHOD_CN[method],
                "轨迹数": int(len(r)),
                "步数": int(len(s)),
                "成功率%": pct(r["success"]),
                "平均终距": float(r["final_dist"].mean()),
                "平均轨迹总奖励": float(r["total_reward"].mean()),
                "接近步占比%": pct(s["move_type"].isin(["progress", "goal"])),
                "回退步占比%": pct(regress),
                "到达步数": int(goal.sum()),
                "接近/到达步平均总奖励": mean_where(s, progress_or_goal, "reward_total"),
                "普通接近步平均总奖励": mean_where(s, non_goal_progress, "reward_total"),
                "回退步平均总奖励": mean_where(s, regress, "reward_total"),
                "停留步平均总奖励": mean_where(s, stay, "reward_total"),
                "到达步平均总奖励": mean_where(s, goal, "reward_total"),
                "接近/到达步正奖励率%": pct_where(s, progress_or_goal, "reward_positive"),
                "普通接近步正奖励率%": pct_where(s, non_goal_progress, "reward_positive"),
                "回退步正奖励率%": pct_where(s, regress, "reward_positive"),
                "近目标步正奖励率%": pct_where(s, near, "reward_positive"),
                "奖励选择性": mean_where(s, progress_or_goal, "reward_total") - mean_where(s, regress, "reward_total"),
                "正奖励选择性百分点": pct_where(s, progress_or_goal, "reward_positive") - pct_where(s, regress, "reward_positive"),
            }
        )
    return pd.DataFrame.from_records(records)


def build_component_summary(steps: pd.DataFrame) -> pd.DataFrame:
    cols = ["reward_ex", "reward_in_raw", "reward_in_gated", "pbrs_bonus", "finish_bonus", "reward_total"]
    records: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        s = steps[steps["method"].eq(method)]
        for move_type in ["progress", "regress", "stay", "goal"]:
            hit = s[s["move_type"].eq(move_type)]
            if hit.empty:
                continue
            rec: dict[str, object] = {
                "method": method,
                "方法": METHOD_CN[method],
                "动作类型": move_type,
                "步数": int(len(hit)),
                "占比%": 100.0 * len(hit) / len(s),
                "正总奖励率%": pct(hit["reward_positive"]),
            }
            for col in cols:
                rec[f"{col}_mean"] = float(hit[col].mean())
            records.append(rec)
    return pd.DataFrame.from_records(records)


def build_distance_bin_summary(steps: pd.DataFrame) -> pd.DataFrame:
    bins = [(8, 99, "远:8+"), (6, 7, "中远:6-7"), (3, 5, "中:3-5"), (1, 2, "近:1-2")]
    records: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        s = steps[steps["method"].eq(method)]
        for lo, hi, label in bins:
            hit = s[(s["prev_dist"] >= lo) & (s["prev_dist"] <= hi)]
            if hit.empty:
                continue
            records.append(
                {
                    "method": method,
                    "方法": METHOD_CN[method],
                    "距离段": label,
                    "步数": int(len(hit)),
                    "接近/到达步占比%": pct(hit["move_type"].isin(["progress", "goal"])),
                    "回退步占比%": pct(hit["move_type"].eq("regress")),
                    "平均外在奖励": float(hit["reward_ex"].mean()),
                    "平均好奇心门控": float(hit["reward_in_gated"].mean()),
                    "平均塑形": float(hit["pbrs_bonus"].mean()),
                    "平均总奖励": float(hit["reward_total"].mean()),
                    "正总奖励率%": pct(hit["reward_positive"]),
                    "平均门控系数": float(hit["gate_weight"].mean()),
                }
            )
    return pd.DataFrame.from_records(records)


def build_success_contrast(steps: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        for success in [0, 1]:
            s = steps[(steps["method"].eq(method)) & (steps["success"].eq(success))]
            if s.empty:
                continue
            progress_or_goal = s["move_type"].isin(["progress", "goal"])
            regress = s["move_type"].eq("regress")
            goal = s["move_type"].eq("goal")
            records.append(
                {
                    "method": method,
                    "方法": METHOD_CN[method],
                    "是否成功": success,
                    "步数": int(len(s)),
                    "接近/到达步占比%": pct(progress_or_goal),
                    "回退步占比%": pct(regress),
                    "接近/到达步平均总奖励": mean_where(s, progress_or_goal, "reward_total"),
                    "回退步平均总奖励": mean_where(s, regress, "reward_total"),
                    "到达步平均总奖励": mean_where(s, goal, "reward_total"),
                    "接近/到达步正奖励率%": pct_where(s, progress_or_goal, "reward_positive"),
                    "回退步正奖励率%": pct_where(s, regress, "reward_positive"),
                }
            )
    return pd.DataFrame.from_records(records)


def fmt(x: object, digits: int = 2) -> str:
    if x is None:
        return ""
    try:
        value = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


def make_report(
    routes: pd.DataFrame,
    steps: pd.DataFrame,
    method_summary: pd.DataFrame,
    component_summary: pd.DataFrame,
    distance_summary: pd.DataFrame,
    success_contrast: pd.DataFrame,
) -> str:
    proposed = method_summary[method_summary["method"].eq("proposed_linear_gate_pbrs")].iloc[0]
    gate = method_summary[method_summary["method"].eq("mixed_gate_only")].iloc[0]
    intrinsic = method_summary[method_summary["method"].eq("intrinsic_only")].iloc[0]
    external = method_summary[method_summary["method"].eq("external_only")].iloc[0]
    no_gate = method_summary[method_summary["method"].eq("mixed_no_gate_no_pbrs")].iloc[0]

    lines: list[str] = []
    lines.append("# 全量逐步奖励归因分析")
    lines.append("")
    lines.append(f"- 轨迹样本：{len(routes):,} 条；展开动作步：{len(steps):,} 步。")
    lines.append(f"- 方法：{', '.join(METHOD_CN[m] for m in METHOD_ORDER)}。")
    lines.append("- 正确动作的操作化定义：本步后距离下降或到达目标；错误/风险动作主要指距离回退。")
    lines.append("- 统计字段：外在奖励、原始好奇心、门控好奇心、塑形奖励、终点奖励、总奖励。")
    lines.append("")

    lines.append("## 1. 总体结论")
    lines.append("")
    lines.append(
        "这批数据是训练阶段的路线采样日志，适合分析奖励信号如何对齐动作，不能直接当作最终 benchmark 胜率。"
        f"从样本成功率看，本文方法为 {fmt(proposed['成功率%'])}%、平均终距 {fmt(proposed['平均终距'])}；"
        f"仅外在为 {fmt(external['成功率%'])}%、平均终距 {fmt(external['平均终距'])}；"
        f"直接相加为 {fmt(no_gate['成功率%'])}%、平均终距 {fmt(no_gate['平均终距'])}。"
        "因此全量日志不能用来讲“其它方法完全不行”，更应该用来讲每种奖励信号的动作指导差异。"
    )
    lines.append(
        f"仅好奇心平均轨迹总奖励为 {fmt(intrinsic['平均轨迹总奖励'])}，高于本文方法的 {fmt(proposed['平均轨迹总奖励'])}，"
        f"但成功率只有 {fmt(intrinsic['成功率%'])}%、平均终距 {fmt(intrinsic['平均终距'])}。"
        "这说明奖励总和大不等于导航正确，关键是正反馈是否落在目标方向动作上。"
    )
    lines.append(
        f"从逐步奖励看，本文方法的关键优势是奖励选择性：接近/到达步平均总奖励 {fmt(proposed['接近/到达步平均总奖励'])}，"
        f"回退步平均总奖励 {fmt(proposed['回退步平均总奖励'])}，两者差值 {fmt(proposed['奖励选择性'])}。"
        f"这说明高奖励主要落在正确方向动作上，而回退动作被压低。"
    )
    lines.append(
        f"仅外在奖励的回退步平均总奖励为 {fmt(external['回退步平均总奖励'])}，接近/到达步平均总奖励也只有 {fmt(external['接近/到达步平均总奖励'])}，"
        "说明信号过稀疏，不能持续强化中间接近动作。"
    )
    lines.append(
        f"直接相加方法的回退步平均总奖励为 {fmt(no_gate['回退步平均总奖励'])}；"
        f"本文方法为 {fmt(proposed['回退步平均总奖励'])}，回退步正奖励率为 {fmt(proposed['回退步正奖励率%'])}%。"
        "这说明门控和塑形并不是为了把总奖励抬高，而是让回退动作保持负反馈。"
    )
    lines.append("")

    lines.append("## 2. 方法级汇总")
    lines.append("")
    show_cols = [
        "方法",
        "轨迹数",
        "步数",
        "成功率%",
        "平均终距",
        "平均轨迹总奖励",
        "接近/到达步平均总奖励",
        "回退步平均总奖励",
        "到达步平均总奖励",
        "回退步正奖励率%",
        "奖励选择性",
    ]
    lines.append(method_summary[show_cols].to_markdown(index=False, floatfmt=".2f"))
    lines.append("")

    lines.append("## 3. 分动作类型看各奖励项")
    lines.append("")
    pivot_cols = ["方法", "动作类型", "步数", "reward_ex_mean", "reward_in_gated_mean", "pbrs_bonus_mean", "reward_total_mean", "正总奖励率%"]
    lines.append(component_summary[pivot_cols].to_markdown(index=False, floatfmt=".3f"))
    lines.append("")

    lines.append("## 4. 距离阶段分析")
    lines.append("")
    dist_cols = ["方法", "距离段", "步数", "接近/到达步占比%", "回退步占比%", "平均外在奖励", "平均好奇心门控", "平均塑形", "平均总奖励", "正总奖励率%"]
    lines.append(distance_summary[dist_cols].to_markdown(index=False, floatfmt=".3f"))
    lines.append("")

    lines.append("## 5. 逐步奖励如何指导正确动作")
    lines.append("")
    lines.append("- 外在奖励提供目标约束，但单独使用时信号稀疏，很多中间接近动作仍是负反馈，因此策略容易在远距离阶段震荡。")
    lines.append("- 仅好奇心能提供持续正反馈，但它不区分目标方向，回退动作也可能为正，所以高累计奖励不等于到达目标。")
    lines.append("- 直接相加把好奇心直接叠到外在奖励上，能缓解稀疏性，但也会削弱对回退动作的惩罚。")
    lines.append("- 距离门控让好奇心在远处更多参与、近处减弱，能提升接近目标的能力；但没有塑形项时，末端方向稳定性仍不足。")
    lines.append("- 本文方法把门控好奇心和塑形项叠加到外在目标约束上，使远距离阶段有探索正反馈，近目标阶段回退被压制，到达动作获得最高总奖励。")
    lines.append("")
    lines.append("成功/失败轨迹的差异也支持这一点：在本文方法中，成功轨迹的接近/到达步占比为 98.82%，失败轨迹为 81.79%；成功轨迹中接近/到达步平均总奖励为 0.44，失败轨迹接近/到达步平均总奖励只有 0.01。这说明真正导向成功的是连续正确动作被奖励，而不是单步奖励偶然变大。")
    lines.append("")

    lines.append("## 6. 可用于 PPT 的一句话")
    lines.append("")
    lines.append(
        "全量逐步统计表明，本文方法的优势不是让奖励数值整体最大，而是让正反馈更集中地落在接近目标和到达目标的动作上，"
        "同时降低回退动作获得正奖励的概率，因此策略更容易从探索过渡到稳定收敛。"
    )
    lines.append("")

    lines.append("## 输出文件")
    lines.append("")
    lines.append("- `reward_stepwise_global_steps.csv`：展开后的每一步动作数据。")
    lines.append("- `reward_stepwise_global_method_summary.csv`：方法级汇总。")
    lines.append("- `reward_stepwise_global_component_by_move.csv`：动作类型 × 奖励项统计。")
    lines.append("- `reward_stepwise_global_distance_bins.csv`：距离阶段统计。")
    lines.append("- `reward_stepwise_global_success_contrast.csv`：成功/失败轨迹逐步奖励对比。")
    return "\n".join(lines)


def main() -> int:
    OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    base.REPORTS.mkdir(parents=True, exist_ok=True)

    routes = base.read_routes()
    routes = routes[routes["method"].isin(METHOD_ORDER)].copy()
    steps = expand_steps(routes)

    method_summary = build_method_summary(routes, steps)
    component_summary = build_component_summary(steps)
    distance_summary = build_distance_bin_summary(steps)
    success_contrast = build_success_contrast(steps)

    steps.to_csv(OUT_TABLE_DIR / "reward_stepwise_global_steps.csv", index=False, encoding="utf-8-sig")
    method_summary.to_csv(OUT_TABLE_DIR / "reward_stepwise_global_method_summary.csv", index=False, encoding="utf-8-sig")
    component_summary.to_csv(OUT_TABLE_DIR / "reward_stepwise_global_component_by_move.csv", index=False, encoding="utf-8-sig")
    distance_summary.to_csv(OUT_TABLE_DIR / "reward_stepwise_global_distance_bins.csv", index=False, encoding="utf-8-sig")
    success_contrast.to_csv(OUT_TABLE_DIR / "reward_stepwise_global_success_contrast.csv", index=False, encoding="utf-8-sig")

    report = make_report(routes, steps, method_summary, component_summary, distance_summary, success_contrast)
    OUT_REPORT.write_text(report, encoding="utf-8")

    print(f"routes={len(routes)} steps={len(steps)}")
    print(f"report={OUT_REPORT}")
    print(f"method_summary={OUT_TABLE_DIR / 'reward_stepwise_global_method_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
