from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


ACTION_LIST = {0: "up", 1: "right", 2: "down", 3: "left"}


@dataclass
class AgentBundle:
    method: str
    cfg: object | None
    action_list: dict[int, str]
    sequence_cls: object | None
    agent: object | None
    llm: object | None


def natural_img_key(key: str) -> tuple[int, str]:
    try:
        return int(str(key).split("_")[-1]), str(key)
    except Exception:
        return math.inf, str(key)


def parse_int_list(raw: str, default: list[int]) -> list[int]:
    if not raw.strip():
        return list(default)
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def wilson_interval(success: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = success / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / total) + ((z * z) / (4.0 * total * total)))
    return max(0.0, center - margin), min(1.0, center + margin)


def get_dist(p1: int, p2: int, patch_size: int = 5) -> int:
    r1, c1 = divmod(int(p1), patch_size)
    r2, c2 = divmod(int(p2), patch_size)
    return abs(r1 - r2) + abs(c1 - c2)


def valid_actions(current_patch: int, patch_size: int) -> list[int]:
    actions = []
    row, col = divmod(int(current_patch), patch_size)
    if row > 0:
        actions.append(0)
    if col < patch_size - 1:
        actions.append(1)
    if row < patch_size - 1:
        actions.append(2)
    if col > 0:
        actions.append(3)
    return actions


def step_patch(current_patch: int, action: int, patch_size: int) -> int:
    row, col = divmod(int(current_patch), patch_size)
    if action == 0 and row > 0:
        return current_patch - patch_size
    if action == 1 and col < patch_size - 1:
        return current_patch + 1
    if action == 2 and row < patch_size - 1:
        return current_patch + patch_size
    if action == 3 and col > 0:
        return current_patch - 1
    return current_patch


def extract_image_item(bank, img_key: str, img_order: int):
    if isinstance(bank, np.ndarray) and bank.shape == ():
        return bank[()][img_key]
    if hasattr(bank, "keys"):
        return bank[img_key]
    return bank[img_order]


def build_pairs_by_distance(patch_size: int, distances: list[int]) -> dict[int, list[tuple[int, int]]]:
    patch_count = patch_size ** 2
    pairs_by_distance: dict[int, list[tuple[int, int]]] = {}
    for dist in distances:
        pairs = []
        for goal_patch in range(patch_count):
            for current_patch in range(patch_count):
                if get_dist(current_patch, goal_patch, patch_size) == dist:
                    pairs.append((goal_patch, current_patch))
        pairs_by_distance[dist] = pairs
    return pairs_by_distance


def build_task_bank(
    dataset_dict,
    patch_size: int,
    distances: list[int],
    repeats: int,
    seed: int,
    max_images: int | None,
    fixed_goal_mode: str,
) -> tuple[list[dict], list[dict]]:
    keys = sorted(dataset_dict[()].keys(), key=natural_img_key)
    if max_images is not None:
        keys = keys[:max_images]
    rng = np.random.default_rng(seed)
    pairs_by_distance = build_pairs_by_distance(patch_size, distances)
    tasks = []
    skipped = []

    for img_order, img_key in enumerate(keys):
        if fixed_goal_mode == "monuments":
            goal_patch = img_order % (patch_size ** 2)
            for dist in distances:
                candidates = [cur for goal, cur in pairs_by_distance[dist] if goal == goal_patch]
                if not candidates:
                    skipped.append(
                        {
                            "img_order": int(img_order),
                            "img_key": str(img_key),
                            "distance": int(dist),
                            "goal_patch": int(goal_patch),
                            "reason": "no_current_patch_at_requested_distance",
                        }
                    )
                    continue
                for rep in range(repeats):
                    current_patch = candidates[int(rng.integers(len(candidates)))]
                    tasks.append(
                        {
                            "img_order": int(img_order),
                            "img_key": str(img_key),
                            "distance": int(dist),
                            "repeat_idx": int(rep),
                            "goal_patch": int(goal_patch),
                            "current_patch": int(current_patch),
                        }
                    )
            continue

        for dist in distances:
            pairs = pairs_by_distance[dist]
            if not pairs:
                raise ValueError(f"No patch pairs for distance={dist}, patch_size={patch_size}")
            for rep in range(repeats):
                goal_patch, current_patch = pairs[int(rng.integers(len(pairs)))]
                tasks.append(
                    {
                        "img_order": int(img_order),
                        "img_key": str(img_key),
                        "distance": int(dist),
                        "repeat_idx": int(rep),
                        "goal_patch": int(goal_patch),
                        "current_patch": int(current_patch),
                    }
                )
    return tasks, skipped


def load_bundle(args: argparse.Namespace) -> AgentBundle:
    if args.method == "random":
        return AgentBundle("random", None, ACTION_LIST, None, None, None)

    repo_dir = Path(args.repo_dir).resolve()
    sys.path.insert(0, str(repo_dir))
    device = torch.device(args.device)

    if args.method == "geoexplorer":
        from config import cfg, action_list  # type: ignore
        from data_utils import Sequence  # type: ignore
        from models.ppo import PPO  # type: ignore
        from utils import seed_everything  # type: ignore

        cfg.dataset = args.dataset
        cfg.data.patch_size = args.patch_size
        cfg.data.test_path = args.test_path
        cfg.train.device = args.device
        cfg.train.checkpoint_path = args.checkpoint
        if args.llm_checkpoint:
            cfg.train.llm_checkpoint = args.llm_checkpoint
        cfg.train.hparams.max_ep_len = args.budget
        seed_everything(cfg.train.hparams.random_seed)
        agent = PPO(
            cfg.train.hparams.lr_actor,
            cfg.train.hparams.lr_critic,
            cfg.train.hparams.lr_llm,
            cfg.train.hparams.gamma,
            cfg.train.hparams.K_epochs,
            cfg.train.hparams.eps_clip,
            cfg.train.hparams.lr_gamma,
            ent_coef=getattr(cfg.train.hparams, "ent_coef", 0.01),
        ).to(device)
        agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
        agent.eval()
        return AgentBundle(args.method, cfg, action_list, Sequence, agent, agent.llm)

    if args.method == "gomaa":
        from gomaa_geo.config import cfg, action_list  # type: ignore
        from gomaa_geo.data_utils import Sequence  # type: ignore
        from gomaa_geo.models import PPO  # type: ignore
        from gomaa_geo.utils import seed_everything  # type: ignore

        cfg.data.patch_size = args.patch_size
        cfg.data.test_path = args.test_path
        cfg.train.checkpoint_path = args.checkpoint
        cfg.train.llm_checkpoint = args.llm_checkpoint
        cfg.train.hparams.max_ep_len = args.budget
        seed_everything(cfg.train.hparams.random_seed)
        agent = PPO(
            cfg.train.hparams.lr_actor,
            cfg.train.hparams.lr_critic,
            cfg.train.hparams.lr_llm,
            cfg.train.hparams.gamma,
            cfg.train.hparams.K_epochs,
            cfg.train.hparams.eps_clip,
            cfg.train.hparams.lr_gamma,
        ).to(device)
        agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
        agent.eval()
        return AgentBundle(args.method, cfg, action_list, Sequence, agent, agent.llm)

    if args.method == "dit":
        from dit_agl.config import cfg, action_list  # type: ignore
        from dit_agl.data_utils import Sequence  # type: ignore
        from dit_agl.models.pretrain_model import MaskedActionModeling  # type: ignore
        from dit_agl.utils import seed_everything  # type: ignore

        cfg.data.patch_size = args.patch_size
        cfg.data.test_path = args.test_path
        cfg.train.llm_checkpoint = args.llm_checkpoint
        cfg.train.hparams.max_ep_len = args.budget
        seed_everything(cfg.train.hparams.random_seed)
        model = MaskedActionModeling.load_from_checkpoint(
            args.llm_checkpoint,
            train_dataset=None,
            val_dataset=None,
        ).to(device)
        model.eval()
        return AgentBundle(args.method, cfg, action_list, Sequence, model, model.llm)

    raise ValueError(f"Unsupported method: {args.method}")


def build_sequence(
    bundle: AgentBundle,
    env_embeds,
    goal_patch: int,
    goal_mode: str,
    goal_item=None,
    pre_goal_item=None,
    patch_size: int = 5,
):
    if bundle.sequence_cls is None:
        raise ValueError("Sequence class is required for model-based evaluation")
    seq = bundle.sequence_cls(env_embeds, num_patches=patch_size)
    if goal_mode == "aerial":
        if pre_goal_item is not None:
            seq.init_with_goal_embed(pre_goal_item[goal_patch], goal_patch)
        else:
            seq.init_with_goal_image(goal_patch)
    elif goal_mode in {"ground", "text"}:
        if goal_item is None:
            raise ValueError(f"goal_item is required for goal_mode={goal_mode}")
        seq.init_with_goal_embed(goal_item, goal_patch)
    else:
        raise ValueError(f"Unsupported goal_mode: {goal_mode}")
    return seq


def model_action(bundle: AgentBundle, seq, args: argparse.Namespace) -> int:
    inputs = seq.get_input_for_model(device=args.device)
    with torch.no_grad():
        if bundle.method == "geoexplorer":
            if inputs["actions"] == []:
                state, _, _ = bundle.llm(
                    inputs_embeds=inputs["inputs_embeds"],
                    patch_sequence=inputs["patch_sequence"][:, 1:],
                    patch_size=args.patch_size,
                )
            else:
                state, _, _ = bundle.llm(
                    inputs_embeds=inputs["inputs_embeds"],
                    actions=[inputs["actions"]],
                    patch_sequence=inputs["patch_sequence"][:, 1:],
                    patch_size=args.patch_size,
                )
            return int(bundle.agent.select_greedy_action(state, seq.patch_sequence, args.patch_size))

        if bundle.method == "gomaa":
            if inputs["actions"] == []:
                state = bundle.llm(
                    inputs_embeds=inputs["inputs_embeds"],
                    patch_sequence=inputs["patch_sequence"][:, 1:],
                    patch_size=args.patch_size,
                )
            else:
                state = bundle.llm(
                    inputs_embeds=inputs["inputs_embeds"],
                    actions=[inputs["actions"]],
                    patch_sequence=inputs["patch_sequence"][:, 1:],
                    patch_size=args.patch_size,
                )
            return int(bundle.agent.select_greedy_action(state, seq.patch_sequence, args.patch_size))

        if bundle.method == "dit":
            if inputs["actions"] == []:
                state = bundle.llm(
                    inputs_embeds=inputs["inputs_embeds"],
                    patch_sequence=inputs["patch_sequence"][:, 1:],
                    patch_size=args.patch_size,
                )
            else:
                state = bundle.llm(
                    inputs_embeds=inputs["inputs_embeds"],
                    actions=[inputs["actions"]],
                    patch_sequence=inputs["patch_sequence"][:, 1:],
                    patch_size=args.patch_size,
                )
            return int(torch.argmax(bundle.llm.distance_pred(state).reshape(-1)).item())

    raise ValueError(f"Unsupported model action method: {bundle.method}")


def evaluate_random(tasks: list[dict], args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed + 17)
    return evaluate_tasks(None, None, None, None, tasks, args, random_rng=rng)


def evaluate_tasks(
    bundle,
    dataset_dict,
    goal_embeds,
    pre_goal_dict,
    tasks: list[dict],
    args: argparse.Namespace,
    random_rng=None,
) -> dict:
    by_dist: dict[int, list[dict]] = defaultdict(list)
    total_success = 0
    final_distances = []
    success_steps = []
    success_deviation = []

    for task in tasks:
        goal_patch = int(task["goal_patch"])
        current_patch = int(task["current_patch"])
        optimal_steps = get_dist(current_patch, goal_patch, args.patch_size)
        success = False

        if random_rng is not None:
            patch = current_patch
            for _ in range(args.budget):
                choices = valid_actions(patch, args.patch_size)
                action = int(choices[int(random_rng.integers(len(choices)))])
                patch = step_patch(patch, action, args.patch_size)
                if patch == goal_patch:
                    success = True
                    total_success += 1
                    steps = _ + 1
                    success_steps.append(float(steps))
                    success_deviation.append(float(steps - optimal_steps))
                    break
            final_patch = patch
        else:
            env_embeds = extract_image_item(dataset_dict, task["img_key"], int(task["img_order"]))
            goal_item = None if goal_embeds is None else extract_image_item(goal_embeds, task["img_key"], int(task["img_order"]))
            pre_goal_item = (
                None if pre_goal_dict is None else extract_image_item(pre_goal_dict, task["img_key"], int(task["img_order"]))
            )
            seq = build_sequence(
                bundle,
                env_embeds,
                goal_patch,
                goal_mode=args.goal_mode,
                goal_item=goal_item,
                pre_goal_item=pre_goal_item,
                patch_size=args.patch_size,
            )
            seq.update_sequence_with_satellite_image_token(current_patch)
            for _ in range(args.budget):
                action = model_action(bundle, seq, args)
                seq.update_sequence_with_action(bundle.action_list[action])
                if seq.patch_sequence[-1] == goal_patch:
                    success = True
                    total_success += 1
                    steps = len(seq.action_sequence)
                    success_steps.append(float(steps))
                    success_deviation.append(float(steps - optimal_steps))
                    break
            final_patch = int(seq.patch_sequence[-1])

        final_distance = get_dist(final_patch, goal_patch, args.patch_size)
        final_distances.append(float(final_distance))
        by_dist[int(task["distance"])].append(
            {
                "distance": int(task["distance"]),
                "success": bool(success),
                "optimal_steps": int(optimal_steps),
                "final_distance": int(final_distance),
            }
        )

    total_trials = len(tasks)
    ci_low, ci_high = wilson_interval(total_success, total_trials)
    per_dist = []
    for dist in sorted(by_dist):
        rows = by_dist[dist]
        dist_success = sum(1 for row in rows if row["success"])
        dist_trials = len(rows)
        dist_ci_low, dist_ci_high = wilson_interval(dist_success, dist_trials)
        per_dist.append(
            {
                "distance": int(dist),
                "trials": int(dist_trials),
                "success": int(dist_success),
                "success_ratio": float(dist_success / max(dist_trials, 1)),
                "success_ratio_ci_low": float(dist_ci_low),
                "success_ratio_ci_high": float(dist_ci_high),
                "sg_mean": float(np.mean([row["final_distance"] for row in rows])) if rows else math.nan,
            }
        )

    return {
        "mode": "random" if random_rng is not None else "greedy",
        "total_trials": int(total_trials),
        "success": int(total_success),
        "success_ratio": float(total_success / max(total_trials, 1)),
        "success_ratio_ci_low": float(ci_low),
        "success_ratio_ci_high": float(ci_high),
        "sg_mean": float(np.mean(final_distances)) if final_distances else math.nan,
        "avg_steps_on_success": float(np.mean(success_steps)) if success_steps else math.nan,
        "avg_deviation_on_success": float(np.mean(success_deviation)) if success_deviation else math.nan,
        "per_dist": per_dist,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper-aligned evaluator for external baselines and anchor0624.")
    parser.add_argument("--method", required=True, choices=["random", "geoexplorer", "gomaa", "dit"])
    parser.add_argument("--method-label", required=True)
    parser.add_argument("--repo-dir", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--llm-checkpoint", default="")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--goal-mode", default="aerial", choices=["aerial", "ground", "text"])
    parser.add_argument("--test-path", required=True)
    parser.add_argument("--pre-goal-path", default="", help="Optional pre-disaster aerial goal bank for xBD-disaster.")
    parser.add_argument("--goal-embeds", default="")
    parser.add_argument("--fixed-goal-mode", default="none", choices=["none", "monuments"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--patch-size", type=int, default=5)
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--distances", default="4,5,6,7,8")
    parser.add_argument("--repeats-per-dist", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--paper-table", default="")
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    distances = parse_int_list(args.distances, [4, 5, 6, 7, 8])
    max_images = args.max_images if args.max_images > 0 else None
    dataset_dict = np.load(args.test_path, allow_pickle=True)
    pre_goal_dict = np.load(args.pre_goal_path, allow_pickle=True) if args.pre_goal_path else None
    goal_embeds = np.load(args.goal_embeds, allow_pickle=True) if args.goal_embeds else None
    tasks, skipped = build_task_bank(
        dataset_dict,
        args.patch_size,
        distances,
        repeats=args.repeats_per_dist,
        seed=args.seed,
        max_images=max_images,
        fixed_goal_mode=args.fixed_goal_mode,
    )

    if args.method == "random":
        evaluation = evaluate_random(tasks, args)
    else:
        bundle = load_bundle(args)
        evaluation = evaluate_tasks(bundle, dataset_dict, goal_embeds, pre_goal_dict, tasks, args)

    payload = {
        "method": args.method_label,
        "method_type": args.method,
        "benchmark": args.benchmark,
        "paper_table": args.paper_table,
        "dataset": args.dataset,
        "goal_mode": args.goal_mode,
        "test_path": args.test_path,
        "pre_goal_path": args.pre_goal_path or None,
        "goal_embeds_path": args.goal_embeds or None,
        "checkpoint_path": args.checkpoint or None,
        "llm_checkpoint": args.llm_checkpoint or None,
        "protocol": {
            "grid": f"{args.patch_size}x{args.patch_size}",
            "budget": int(args.budget),
            "distance_buckets": distances,
            "goal_mode": args.goal_mode,
            "policy_mode": "random" if args.method == "random" else "greedy_argmax",
            "task_bank_seed": int(args.seed),
            "repeats_per_distance": int(args.repeats_per_dist),
            "fixed_goal_mode": args.fixed_goal_mode,
            "pre_disaster_goal": bool(args.pre_goal_path),
        },
        "num_images": len({task["img_key"] for task in tasks}),
        "num_tasks": len(tasks),
        "skipped_tasks": skipped,
        "success_ratio": evaluation["success_ratio"],
        "sg_mean": evaluation["sg_mean"],
        "modes": [evaluation],
        "per_distance": evaluation["per_dist"],
        "task_bank_preview": tasks[: min(12, len(tasks))],
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
