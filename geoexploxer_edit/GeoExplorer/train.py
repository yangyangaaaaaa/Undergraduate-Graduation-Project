import json
import os

import numpy as np
import torch

from config import cfg, action_list, require_data_paths
from data_utils import Sequence
from models.ppo import PPO
from utils import generate_random_dist_config, get_dist, seed_everything
from utils.visualization import append_csv_row, ensure_dir

device = torch.device("cuda:0")

STEP_FIELDNAMES = [
    "episode_index",
    "sample_index",
    "step_index",
    "current_patch",
    "goal_patch",
    "action",
    "reward_ex",
    "reward_in",
    "w_ex",
    "w_in",
    "reward_total",
    "success",
]

EPISODE_FIELDNAMES = [
    "episode_index",
    "time_step",
    "avg_reward",
    "success_ratio",
    "avg_steps_success",
    "avg_deviation",
    "avg_final_distance",
    "avg_unique_visited",
    "val_success_ratio",
    "val_avg_reward",
    "reward_mode",
]


if __name__ == "__main__":
    require_data_paths("train_path", "val_path")

    ckpt_dir = os.path.join(cfg.train.ckpt_folder, cfg.train.expt_folder)
    ensure_dir(ckpt_dir)
    ensure_dir(cfg.train.output_dir)

    with open(os.path.join(ckpt_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=2)

    seed_everything(cfg.train.hparams.random_seed)

    ppo_agent = PPO(
        cfg.train.hparams.lr_actor,
        cfg.train.hparams.lr_critic,
        cfg.train.hparams.lr_llm,
        cfg.train.hparams.gamma,
        cfg.train.hparams.K_epochs,
        cfg.train.hparams.eps_clip,
        cfg.train.hparams.lr_gamma,
    ).cuda()

    if cfg.train.load_from_checkpoint:
        ppo_agent.load_state_dict(torch.load(cfg.train.checkpoint_path))

    print_freq = 0
    print_running_reward = 0.0
    print_running_episodes = 0
    time_step = 0
    i_episode = 0
    val_success = 0

    average_steps_to_success = []
    average_deviation_from_opt = []
    average_reward = []
    num_success_history = []

    dataset_dict = np.load(cfg.data.train_path, allow_pickle=True)
    val_config = generate_random_dist_config(cfg.data.val_path, patch_size=cfg.data.patch_size, dist_possible=[7, 8])

    step_metrics_path = os.path.join(cfg.train.output_dir, cfg.train.step_log_name)
    episode_metrics_path = os.path.join(cfg.train.output_dir, cfg.train.metrics_log_name)
    text_log_path = os.path.join(ckpt_dir, cfg.train.log_name)

    with open(text_log_path, "a+", encoding="utf-8") as log_handle:
        while time_step <= cfg.train.hparams.max_training_timesteps:
            current_ep_reward = 0.0
            num_success = 0
            episode_final_distances = []
            episode_unique_visited = []

            for sample_index in range(len(dataset_dict[()].keys())):
                seq = Sequence(dataset_dict[()][f"img_{sample_index}"], num_patches=cfg.data.patch_size)

                dist = np.random.randint(1, cfg.data.patch_size * 2 - 1)
                goal_patch = np.random.randint(0, cfg.data.patch_size**2)
                current_patch = np.random.randint(0, cfg.data.patch_size**2)
                while get_dist(current_patch, goal_patch) != dist:
                    goal_patch = np.random.randint(0, cfg.data.patch_size**2)
                    current_patch = np.random.randint(0, cfg.data.patch_size**2)

                optimal_steps = get_dist(current_patch, goal_patch)
                best_dist = optimal_steps
                seq.init_with_goal_image(goal_patch)
                seq.update_sequence_with_satellite_image_token(current_patch)

                max_steps = np.random.randint(optimal_steps, cfg.train.hparams.max_ep_len + 1)
                for step_index in range(1, max_steps):
                    state, state_preds, state_gt = ppo_agent._run_llm(seq, device=str(device))
                    action = ppo_agent.select_action(state, seq.patch_sequence, cfg.data.patch_size)
                    seq.update_sequence_with_action(action_list[action])

                    current_patch_id = seq.patch_sequence[-1]
                    prev_patch_id = seq.patch_sequence[-2]
                    reward_terms = ppo_agent.compute_reward_components(
                        prev_patch_id,
                        current_patch_id,
                        goal_patch,
                        seq.patch_sequence[1:-1],
                        best_dist,
                        state_preds,
                        state_gt,
                        phase="train",
                    )
                    reward = reward_terms["reward_total"]
                    best_dist = min(best_dist, reward_terms["current_dist"])
                    done = current_patch_id == goal_patch

                    if done:
                        average_steps_to_success.append(len(seq.action_sequence))
                        average_deviation_from_opt.append(len(seq.action_sequence) - optimal_steps)
                        num_success += 1

                    ppo_agent.buffer.rewards.append(reward)
                    ppo_agent.buffer.is_terminals.append(done)

                    if cfg.train.save_reward_components:
                        append_csv_row(
                            step_metrics_path,
                            STEP_FIELDNAMES,
                            {
                                "episode_index": i_episode,
                                "sample_index": sample_index,
                                "step_index": step_index,
                                "current_patch": current_patch_id,
                                "goal_patch": goal_patch,
                                "action": action_list[action],
                                "reward_ex": reward_terms["reward_ex"],
                                "reward_in": reward_terms["reward_in"],
                                "w_ex": reward_terms["w_ex"],
                                "w_in": reward_terms["w_in"],
                                "reward_total": reward,
                                "success": done,
                            },
                        )

                    time_step += 1
                    current_ep_reward += reward

                    if time_step % cfg.train.hparams.update_timestep == 0:
                        ppo_agent.update(True, seq.patch_sequence, cfg.data.patch_size)

                    if done:
                        break

                episode_final_distances.append(get_dist(seq.patch_sequence[-1], goal_patch, cfg.data.patch_size))
                episode_unique_visited.append(len(set(seq.patch_sequence[1:])))

            print_freq += 1
            num_success_history.append(num_success)

            ppo_agent.eval()
            val_summary = ppo_agent.validate(
                val_config,
                cfg.data.val_path,
                return_details=True,
            )
            if val_summary["num_success"] >= val_success:
                torch.save(ppo_agent.state_dict(), os.path.join(ckpt_dir, cfg.train.expt_name))
                val_success = val_summary["num_success"]

            if i_episode % 50 == 0:
                torch.save(
                    ppo_agent.state_dict(),
                    os.path.join(ckpt_dir, cfg.train.expt_name_tmp) + str(i_episode) + ".pt",
                )

            ppo_agent.train()

            avg_reward_this_episode = current_ep_reward / max(len(dataset_dict[()].keys()), 1)
            average_reward.append(round(avg_reward_this_episode, 4))
            print_running_reward += current_ep_reward
            print_running_episodes += 1

            avg_steps_success = float(np.mean(average_steps_to_success[-200:])) if average_steps_to_success else None
            avg_deviation = float(np.mean(average_deviation_from_opt[-200:])) if average_deviation_from_opt else None
            success_ratio = num_success / max(len(dataset_dict[()].keys()), 1)

            append_csv_row(
                episode_metrics_path,
                EPISODE_FIELDNAMES,
                {
                    "episode_index": i_episode,
                    "time_step": time_step,
                    "avg_reward": avg_reward_this_episode,
                    "success_ratio": success_ratio,
                    "avg_steps_success": avg_steps_success,
                    "avg_deviation": avg_deviation,
                    "avg_final_distance": float(np.mean(episode_final_distances)) if episode_final_distances else None,
                    "avg_unique_visited": float(np.mean(episode_unique_visited)) if episode_unique_visited else None,
                    "val_success_ratio": val_summary["success_ratio"],
                    "val_avg_reward": val_summary["avg_reward"],
                    "reward_mode": cfg.train.reward_mode,
                },
            )

            if print_freq % 2 == 0:
                mean_reward_window = np.mean(average_reward[-20:]) if average_reward else 0.0
                if i_episode < 20:
                    success_window = sum(num_success_history[-20:]) / max((i_episode + 1) * len(dataset_dict[()].keys()), 1)
                else:
                    success_window = sum(num_success_history[-20:]) / max(20 * len(dataset_dict[()].keys()), 1)
                print(
                    "Episode : {} \t Timestep : {} \t Average Reward : {} \t Num Successess : {} \t "
                    "Success Ratio : {} \t Average Steps : {} \t Deviation OPT : {}".format(
                        i_episode,
                        time_step,
                        mean_reward_window,
                        sum(num_success_history[-20:]),
                        success_window,
                        avg_steps_success,
                        avg_deviation,
                    )
                )
                print_running_reward = 0.0
                print_running_episodes = 0

            i_episode += 1

            log_handle.write(f"CURRENT_PATCH: {current_patch}, GOAL_PATCH: {goal_patch}\n")
            log_handle.write(str(seq.patch_sequence))
            log_handle.write("\n")
            log_handle.write(str(seq.action_sequence))
            log_handle.write("\n\n")
