import argparse
import json
import os

import numpy as np
import torch

from config import cfg
from data_utils import Sequence
from models.ppo import PPO
from utils import generate_config, generate_config_unseen, seed_everything
from utils.visualization import (
    ensure_dir,
    parse_selected_ids,
    resolve_aerial_image_path,
    save_composite,
    save_grid_heatmap,
    save_json,
    save_path_overlay,
    write_rows_to_csv,
)

device = torch.device("cuda:0")

GOAL_PATCH_LIST = list(range(25)) * 16
METRIC_FIELDS = [
    "dataset",
    "modality",
    "distance",
    "reward_mode",
    "policy_mode",
    "success_ratio",
    "avg_reward",
    "avg_steps_success",
    "avg_deviation",
    "avg_final_distance",
    "avg_unique_visited",
    "num_success",
    "num_episodes",
]


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Unsupported boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="Validate GeoExplorer and optionally export visualizations.")
    parser.add_argument("--checkpoint_path", default=cfg.train.checkpoint_path)
    parser.add_argument("--output_dir", default=cfg.visual.output_dir)
    parser.add_argument("--save_vis", type=str2bool, default=False)
    parser.add_argument("--num_trials", type=int, default=cfg.visual.default_num_trials)
    parser.add_argument("--policy_mode", choices=["greedy", "stochastic", "entropy", "random"], default=cfg.visual.default_policy_mode)
    parser.add_argument("--selected_ids", default=None, help="Comma-separated sample indices for visualization.")
    parser.add_argument("--save_reward_map", type=str2bool, default=True)
    parser.add_argument("--reward_mode", choices=["ex_only", "static_mix", "adaptive_mix"], default=cfg.train.reward_mode)
    return parser.parse_args()


def build_agent(checkpoint_path):
    agent = PPO(
        cfg.train.hparams.lr_actor,
        cfg.train.hparams.lr_critic,
        cfg.train.hparams.lr_llm,
        cfg.train.hparams.gamma,
        cfg.train.hparams.K_epochs,
        cfg.train.hparams.eps_clip,
        cfg.train.hparams.lr_gamma,
    ).cuda()
    agent.load_state_dict(torch.load(checkpoint_path))
    agent.eval()
    return agent


def build_sequence_for_visualization(sample_index, goal_patch, current_patch, modality, valid_path, ground_path=None):
    sat_dict = np.load(valid_path, allow_pickle=True)
    if modality == "aerial":
        sat_embeds = sat_dict[()][f"img_{sample_index}"]
        seq = Sequence(sat_embeds, num_patches=cfg.data.patch_size)
        seq.init_with_goal_image(goal_patch)
    elif modality == "ground":
        ground_dict = np.load(ground_path, allow_pickle=True)
        sat_embeds = sat_dict[()][f"img_{sample_index}"].reshape(25, -1)
        seq = Sequence(sat_embeds, num_patches=cfg.data.patch_size)
        seq.init_with_goal_embed(ground_dict[()][f"img_{sample_index}"], goal_patch)
    else:
        raise ValueError(f"Unsupported modality for visualization: {modality}")
    seq.update_sequence_with_satellite_image_token(current_patch)
    return seq


def export_visualizations(agent, dataset_name, modality, config_map, valid_path, output_dir, selected_ids, policy_mode, num_trials, save_reward_map, ground_path=None, dist_value=None):
    for sample_index in selected_ids:
        if f"img_{sample_index}" not in config_map:
            continue
        goal_patch, current_patch = config_map[f"img_{sample_index}"][0]
        if goal_patch == 999:
            continue

        trajectories = []
        reward_accumulator = np.zeros((cfg.data.patch_size, cfg.data.patch_size), dtype=float)
        for trial_index in range(num_trials):
            seq = build_sequence_for_visualization(sample_index, goal_patch, current_patch, modality, valid_path, ground_path=ground_path)
            rollout = agent.rollout_episode(
                seq,
                goal_patch,
                cfg.train.hparams.max_ep_len,
                cfg.data.patch_size,
                policy_mode=policy_mode,
                phase="val",
            )
            trajectories.append(
                {
                    "trial_index": trial_index,
                    "start_patch": rollout["start_patch"],
                    "goal_patch": rollout["goal_patch"],
                    "patch_sequence": rollout["path_sequence"],
                    "action_sequence": rollout["action_sequence"],
                    "success": rollout["success"],
                    "num_steps": rollout["num_steps"],
                    "final_distance": rollout["final_distance"],
                    "step_records": rollout["step_records"],
                }
            )
            reward_accumulator += rollout["reward_matrix"]

        reward_matrix = reward_accumulator / max(num_trials, 1)
        sample_dir = os.path.join(
            output_dir,
            "samples",
            dataset_name,
            modality,
            f"dist_{dist_value}",
            f"img_{sample_index}",
        )
        ensure_dir(sample_dir)
        image_path = resolve_aerial_image_path(dataset_name, sample_index, cfg)
        save_json(
            os.path.join(sample_dir, "trajectory.json"),
            {
                "dataset": dataset_name,
                "modality": modality,
                "distance": dist_value,
                "reward_mode": cfg.train.reward_mode,
                "policy_mode": policy_mode,
                "goal_patch": goal_patch,
                "current_patch": current_patch,
                "trials": trajectories,
            },
        )
        save_path_overlay(image_path, trajectories, cfg.data.patch_size, os.path.join(sample_dir, "path_overlay.png"))
        if save_reward_map:
            save_grid_heatmap(
                reward_matrix,
                os.path.join(sample_dir, "reward_heatmap.png"),
                title="Intrinsic reward per patch",
                cmap="magma",
                highlight_index=int(np.argmax(reward_matrix)),
            )
            save_composite(
                image_path,
                trajectories,
                reward_matrix,
                cfg.data.patch_size,
                os.path.join(sample_dir, "composite.png"),
                title=f"{dataset_name} {modality} dist={dist_value} img_{sample_index}",
            )


if __name__ == "__main__":
    args = parse_args()
    cfg.train.checkpoint_path = args.checkpoint_path
    cfg.train.reward_mode = args.reward_mode
    cfg.visual.output_dir = args.output_dir
    ensure_dir(args.output_dir)
    save_json(
        os.path.join(args.output_dir, "validate_config.json"),
        {
            "args": vars(args),
            "cfg": cfg,
        },
    )

    seed_everything(cfg.train.hparams.random_seed)
    ppo_agent = build_agent(cfg.train.checkpoint_path)

    valid_path = cfg.data.test_path
    ground_path = cfg.data.ground_embeds_path if cfg.dataset == "swissviewmonuments" else None
    selected_ids = parse_selected_ids(args.selected_ids, cfg.visual.default_selected_ids)
    metric_rows = []

    print("checkpoint", cfg.train.checkpoint_path)
    print("data", cfg.data.test_path)
    print("dataset", cfg.dataset)
    print("reward_mode", cfg.train.reward_mode)

    for dist_value in range(cfg.min_c, cfg.max_c):
        seed_everything(cfg.train.hparams.random_seed)
        summary_dir = os.path.join(args.output_dir, "summary", f"{cfg.dataset}_dist_{dist_value}")
        ensure_dir(summary_dir)

        if cfg.dataset == "swissviewmonuments":
            config_map = generate_config_unseen(
                cfg.data.test_path,
                GOAL_PATCH_LIST,
                patch_size=cfg.data.patch_size,
                dist=dist_value,
                n_config_per_img=cfg.num_config_per_img,
            )

            print("===================aerial view====================")
            aerial_summary = ppo_agent.validate_unseen(
                config_map,
                valid_path,
                n_config_per_img=cfg.num_config_per_img,
                return_details=True,
                policy_mode=args.policy_mode,
            )
            metric_rows.append(
                {
                    "dataset": cfg.dataset,
                    "modality": "aerial",
                    "distance": dist_value,
                    "reward_mode": cfg.train.reward_mode,
                    "policy_mode": args.policy_mode,
                    "success_ratio": aerial_summary["success_ratio"],
                    "avg_reward": aerial_summary["avg_reward"],
                    "avg_steps_success": aerial_summary["avg_steps_success"],
                    "avg_deviation": aerial_summary["avg_deviation"],
                    "avg_final_distance": aerial_summary["avg_final_distance"],
                    "avg_unique_visited": aerial_summary["avg_unique_visited"],
                    "num_success": aerial_summary["num_success"],
                    "num_episodes": aerial_summary["num_episodes"],
                }
            )
            save_grid_heatmap(aerial_summary["visited_matrix"], os.path.join(summary_dir, "visited_heatmap.png"), f"{cfg.dataset} aerial visited dist={dist_value}")
            save_grid_heatmap(aerial_summary["end_matrix"], os.path.join(summary_dir, "end_heatmap.png"), f"{cfg.dataset} aerial end dist={dist_value}")
            if args.save_vis:
                export_visualizations(
                    ppo_agent,
                    cfg.dataset,
                    "aerial",
                    config_map,
                    valid_path,
                    args.output_dir,
                    selected_ids,
                    args.policy_mode,
                    args.num_trials,
                    args.save_reward_map,
                    dist_value=dist_value,
                )

            print("===================ground view====================")
            ground_summary = ppo_agent.validate_ground_unseen(
                config_map,
                valid_path,
                ground_path,
                n_config_per_img=cfg.num_config_per_img,
                return_details=True,
                policy_mode=args.policy_mode,
            )
            metric_rows.append(
                {
                    "dataset": cfg.dataset,
                    "modality": "ground",
                    "distance": dist_value,
                    "reward_mode": cfg.train.reward_mode,
                    "policy_mode": args.policy_mode,
                    "success_ratio": ground_summary["success_ratio"],
                    "avg_reward": ground_summary["avg_reward"],
                    "avg_steps_success": ground_summary["avg_steps_success"],
                    "avg_deviation": ground_summary["avg_deviation"],
                    "avg_final_distance": ground_summary["avg_final_distance"],
                    "avg_unique_visited": ground_summary["avg_unique_visited"],
                    "num_success": ground_summary["num_success"],
                    "num_episodes": ground_summary["num_episodes"],
                }
            )
            ground_summary_dir = os.path.join(args.output_dir, "summary", f"{cfg.dataset}_ground_dist_{dist_value}")
            ensure_dir(ground_summary_dir)
            save_grid_heatmap(
                ground_summary["visited_matrix"],
                os.path.join(ground_summary_dir, "visited_heatmap.png"),
                f"{cfg.dataset} ground visited dist={dist_value}",
            )
            save_grid_heatmap(
                ground_summary["end_matrix"],
                os.path.join(ground_summary_dir, "end_heatmap.png"),
                f"{cfg.dataset} ground end dist={dist_value}",
            )
            if args.save_vis:
                export_visualizations(
                    ppo_agent,
                    cfg.dataset,
                    "ground",
                    config_map,
                    valid_path,
                    args.output_dir,
                    selected_ids,
                    args.policy_mode,
                    args.num_trials,
                    args.save_reward_map,
                    ground_path=ground_path,
                    dist_value=dist_value,
                )
        else:
            config_map = generate_config(
                cfg.data.test_path,
                patch_size=cfg.data.patch_size,
                dist=dist_value,
                n_config_per_img=cfg.num_config_per_img,
            )
            aerial_summary = ppo_agent.validate(
                config_map,
                valid_path,
                n_config_per_img=cfg.num_config_per_img,
                return_details=True,
                policy_mode=args.policy_mode,
            )
            metric_rows.append(
                {
                    "dataset": cfg.dataset,
                    "modality": "aerial",
                    "distance": dist_value,
                    "reward_mode": cfg.train.reward_mode,
                    "policy_mode": args.policy_mode,
                    "success_ratio": aerial_summary["success_ratio"],
                    "avg_reward": aerial_summary["avg_reward"],
                    "avg_steps_success": aerial_summary["avg_steps_success"],
                    "avg_deviation": aerial_summary["avg_deviation"],
                    "avg_final_distance": aerial_summary["avg_final_distance"],
                    "avg_unique_visited": aerial_summary["avg_unique_visited"],
                    "num_success": aerial_summary["num_success"],
                    "num_episodes": aerial_summary["num_episodes"],
                }
            )
            save_grid_heatmap(aerial_summary["visited_matrix"], os.path.join(summary_dir, "visited_heatmap.png"), f"{cfg.dataset} visited dist={dist_value}")
            save_grid_heatmap(aerial_summary["end_matrix"], os.path.join(summary_dir, "end_heatmap.png"), f"{cfg.dataset} end dist={dist_value}")
            print(f"dist={dist_value}", f"success_ratio: {aerial_summary['success_ratio']}")
            if args.save_vis:
                export_visualizations(
                    ppo_agent,
                    cfg.dataset,
                    "aerial",
                    config_map,
                    valid_path,
                    args.output_dir,
                    selected_ids,
                    args.policy_mode,
                    args.num_trials,
                    args.save_reward_map,
                    dist_value=dist_value,
                )

    write_rows_to_csv(os.path.join(args.output_dir, "metrics.csv"), METRIC_FIELDS, metric_rows)
        
