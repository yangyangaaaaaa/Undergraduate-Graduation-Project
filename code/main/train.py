import csv
import json
import math
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from config import cfg, action_list
from data_utils import Sequence
from models.ppo import PPO
from utils import generate_random_dist_config, get_dist, seed_everything


device = torch.device(cfg.train.device)


def safe_mean(values):
    if not values:
        return float("nan")
    return float(np.mean(values))


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_live_control(path, previous_signature):
    if not os.path.exists(path):
        return previous_signature, None
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    current_signature = raw
    if previous_signature is not None and current_signature == previous_signature:
        return previous_signature, None
    return current_signature, json.loads(raw)


def aggregate_update_stats(stats_list, fallback):
    source = stats_list if stats_list else ([fallback] if fallback else [])
    if not source:
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "oracle_bc_loss": 0.0,
            "sil_loss": 0.0,
            "update_epochs_ran": 0,
            "actor_lr": 0.0,
            "critic_lr": 0.0,
        }
    keys = [
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "oracle_bc_loss",
        "sil_loss",
        "actor_lr",
        "critic_lr",
    ]
    aggregated = {key: float(np.mean([item.get(key, 0.0) for item in source])) for key in keys}
    aggregated["update_epochs_ran"] = int(np.mean([item.get("update_epochs_ran", 0) for item in source]))
    return aggregated


def gate_weight(current_dist: int, optimal_steps: int) -> float:
    algo = getattr(cfg, "algo", None)
    mode = str(getattr(algo, "gate_mode", "none"))
    if mode in {"", "none"}:
        return 1.0
    gate_floor = float(getattr(algo, "gate_floor", 1.0))
    if mode == "constant":
        return gate_floor
    gate_power = float(getattr(algo, "gate_power", 1.0))
    gate_blend_alpha = float(getattr(algo, "gate_blend_alpha", 0.0))
    ratio = max(0.0, min(float(current_dist) / max(float(optimal_steps), 1.0), 1.0))
    if mode == "sqrt":
        shaped = math.sqrt(ratio)
    elif mode == "power":
        shaped = math.pow(ratio, gate_power)
    elif mode == "sine":
        shaped = math.sin(math.pi * ratio / 2.0)
    elif mode == "blend_lp":
        shaped = (1.0 - gate_blend_alpha) * ratio + gate_blend_alpha * math.pow(ratio, gate_power)
    else:
        shaped = ratio
    return gate_floor + (1.0 - gate_floor) * shaped


def finish_bonus(current_dist: int) -> float:
    algo = getattr(cfg, "algo", None)
    bonus_scale = float(getattr(algo, "finish_bonus_scale", 0.0))
    if bonus_scale <= 0.0:
        return 0.0
    radius = max(int(getattr(algo, "finish_bonus_radius", 1)), 1)
    if current_dist > radius:
        return 0.0
    return bonus_scale * ((radius - current_dist + 1) / radius)


def pbrs_bonus(prev_dist: int, current_dist: int) -> float:
    coef = float(getattr(getattr(cfg, "algo", None), "pbrs_coef", 0.0))
    if coef <= 0.0:
        return 0.0
    max_dist = max(1, cfg.data.patch_size * 2 - 2)
    phi_prev = -float(prev_dist) / max_dist
    phi_cur = -float(current_dist) / max_dist
    return coef * (cfg.train.hparams.gamma * phi_cur - phi_prev)


def should_commit_best(current_dist: int, best_dist: int, reward: float) -> bool:
    algo = getattr(cfg, "algo", None)
    if bool(getattr(algo, "commit_best_on_progress", False)):
        return current_dist < best_dist
    return reward >= 0.6


def choose_train_distance(time_step: int) -> int:
    mode = str(getattr(getattr(cfg, "algo", None), "curriculum_mode", "none"))
    if mode != "short_to_eval":
        return int(np.random.randint(1, cfg.data.patch_size * 2 - 1))
    total = max(int(cfg.train.hparams.max_training_timesteps), 1)
    progress = min(max(float(time_step) / total, 0.0), 1.0)
    phase1 = float(getattr(cfg.algo, "curriculum_phase1", 0.25))
    phase2 = float(getattr(cfg.algo, "curriculum_phase2", 0.65))
    if progress < phase1:
        candidates = [1, 2, 3, 4]
    elif progress < phase2:
        candidates = [3, 4, 5, 6]
    else:
        candidates = [4, 5, 6, 7, 8]
    return int(np.random.choice(candidates))


def oracle_action_toward_goal(current_patch: int, goal_patch: int, patch_size: int) -> int:
    cur_r, cur_c = divmod(int(current_patch), patch_size)
    goal_r, goal_c = divmod(int(goal_patch), patch_size)
    candidates = []
    if cur_r > goal_r:
        candidates.append(0)
    if cur_c < goal_c:
        candidates.append(1)
    if cur_r < goal_r:
        candidates.append(2)
    if cur_c > goal_c:
        candidates.append(3)
    return candidates[0] if candidates else 0


if __name__ == "__main__":
    output_dir = os.path.join(cfg.train.ckpt_folder, cfg.train.expt_folder)
    os.makedirs(output_dir, exist_ok=True)

    write_json(os.path.join(output_dir, "config.json"), cfg)

    control_path = os.path.join(output_dir, cfg.train.control_name)
    heartbeat_path = os.path.join(output_dir, cfg.train.heartbeat_name)
    if not os.path.exists(control_path):
        write_json(
            control_path,
            {
                "stop": False,
                "reward": cfg.reward,
                "factor": cfg.factor,
                "progress_metric": cfg.progress_metric,
                "gamma": cfg.train.hparams.gamma,
                "K_epochs": cfg.train.hparams.K_epochs,
                "eps_clip": cfg.train.hparams.eps_clip,
                "ent_coef": cfg.train.hparams.ent_coef,
                "vf_coef": cfg.train.hparams.vf_coef,
                "max_grad_norm": cfg.train.hparams.max_grad_norm,
                "target_kl": cfg.train.hparams.target_kl,
                "lr_actor": cfg.train.hparams.lr_actor,
                "lr_critic": cfg.train.hparams.lr_critic,
                "oracle_bc_coef": cfg.algo.oracle_bc_coef,
                "sil_coef": cfg.algo.sil_coef,
            },
        )

    seed_everything(cfg.train.hparams.random_seed)

    ppo_agent = PPO(
        cfg.train.hparams.lr_actor,
        cfg.train.hparams.lr_critic,
        cfg.train.hparams.lr_llm,
        cfg.train.hparams.gamma,
        cfg.train.hparams.K_epochs,
        cfg.train.hparams.eps_clip,
        cfg.train.hparams.lr_gamma,
        ent_coef=cfg.train.hparams.ent_coef,
        vf_coef=cfg.train.hparams.vf_coef,
        max_grad_norm=cfg.train.hparams.max_grad_norm,
        normalize_advantage=cfg.train.hparams.normalize_advantage,
        target_kl=cfg.train.hparams.target_kl,
        oracle_bc_coef=cfg.algo.oracle_bc_coef,
        sil_coef=cfg.algo.sil_coef,
    ).to(device)

    print_running_reward = 0
    print_running_episodes = 0
    time_step = 0
    i_episode = 0
    num_success = 0
    control_signature = None
    stop_requested = False
    runtime_reward = cfg.reward
    runtime_factor = cfg.factor
    runtime_progress_metric = cfg.progress_metric

    trace_log_path = os.path.join(output_dir, cfg.train.log_name)
    metrics_path = os.path.join(output_dir, cfg.train.metrics_name)
    trace_file = open(trace_log_path, "a+", encoding="utf-8")
    metrics_exists = os.path.exists(metrics_path) and os.path.getsize(metrics_path) > 0
    metrics_file = open(metrics_path, "a", newline="", encoding="utf-8")
    metrics_writer = csv.DictWriter(
        metrics_file,
        fieldnames=[
            "timestamp",
            "episode",
            "time_step",
            "current_ep_reward",
            "rolling_avg_reward",
            "episode_successes",
            "rolling_success_ratio",
            "avg_steps_recent",
            "avg_deviation_recent",
            "val_success",
            "best_val_success",
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "clip_fraction",
            "oracle_bc_loss",
            "sil_loss",
            "update_epochs_ran",
            "actor_lr",
            "critic_lr",
            "reward_factor",
            "reward_mode",
            "progress_metric",
            "elapsed_sec",
        ],
    )
    if not metrics_exists:
        metrics_writer.writeheader()
        metrics_file.flush()

    average_steps_to_success = []
    average_deviation_from_opt = []
    average_reward = []
    num_successess = []
    dataset_dict = np.load(cfg.data.train_path, allow_pickle=True)
    dataset_size = len(dataset_dict[()].keys())
    val_success = 0
    val_every_episodes = max(1, int(os.getenv("GEOEXPLORER_VAL_EVERY_EPISODES", "1")))
    val_max_images_raw = os.getenv("GEOEXPLORER_VAL_MAX_IMAGES", "").strip()
    val_max_images = int(val_max_images_raw) if val_max_images_raw else None
    last_eval_episode = -1
    val_config = generate_random_dist_config(
        cfg.data.val_path,
        patch_size=cfg.data.patch_size,
        n_config_per_img=cfg.num_config_per_img,
        dist_possible=cfg.train.hparams.val_distances,
        sample_mode="balanced",
    )

    if cfg.train.load_from_checkpoint:
        ppo_agent.load_state_dict(torch.load(cfg.train.checkpoint_path, map_location=device))

    run_start = time.time()
    print(f"[{datetime.now().isoformat()}] Starting PPO training")
    print(f"Dataset size: {dataset_size}")
    print(f"Device: {device}")
    print(f"Max timesteps: {cfg.train.hparams.max_training_timesteps}")
    print(f"Update timestep: {cfg.train.hparams.update_timestep}")
    print(f"Validation every episodes: {val_every_episodes}")
    print(f"Validation max images: {val_max_images if val_max_images is not None else 'full'}")
    print(f"Output dir: {output_dir}")

    while time_step <= cfg.train.hparams.max_training_timesteps and not stop_requested:
        episode_updates = []
        current_ep_reward = 0

        control_signature, live_control = read_live_control(control_path, control_signature)
        if live_control:
            runtime_reward = live_control.get("reward", runtime_reward)
            runtime_factor = float(live_control.get("factor", runtime_factor))
            runtime_progress_metric = live_control.get("progress_metric", runtime_progress_metric)
            cfg.progress_metric = runtime_progress_metric
            ppo_agent.apply_live_overrides(live_control)
            stop_requested = bool(live_control.get("stop", False))
            print(f"[{datetime.now().isoformat()}] Applied live control: {live_control}", flush=True)
            if stop_requested:
                break

        for i in range(dataset_size):
            seq = Sequence(dataset_dict[()][f"img_{i}"], num_patches=cfg.data.patch_size)

            dist = choose_train_distance(time_step)
            GOAL_PATCH = np.random.randint(0, cfg.data.patch_size ** 2)
            CURRENT_PATCH = np.random.randint(0, cfg.data.patch_size ** 2)
            while get_dist(CURRENT_PATCH, GOAL_PATCH) != dist:
                GOAL_PATCH = np.random.randint(0, cfg.data.patch_size ** 2)
                CURRENT_PATCH = np.random.randint(0, cfg.data.patch_size ** 2)

            optimal_steps = get_dist(CURRENT_PATCH, GOAL_PATCH)
            best_dist = optimal_steps
            seq.init_with_goal_image(GOAL_PATCH)
            seq.update_sequence_with_satellite_image_token(CURRENT_PATCH)
            episode_buffer_start = len(ppo_agent.buffer.actions)

            for _ in range(1, np.random.randint(optimal_steps, cfg.train.hparams.max_ep_len + 1)):
                inputs = seq.get_input_for_model(device=cfg.train.device)
                with torch.no_grad():
                    if inputs["actions"] == []:
                        state, state_preds, state_gt = ppo_agent.llm(
                            inputs_embeds=inputs["inputs_embeds"],
                            patch_sequence=inputs["patch_sequence"][:, 1:],
                            patch_size=cfg.data.patch_size,
                        )
                    else:
                        state, state_preds, state_gt = ppo_agent.llm(
                            inputs_embeds=inputs["inputs_embeds"],
                            actions=[inputs["actions"]],
                            patch_sequence=inputs["patch_sequence"][:, 1:],
                            patch_size=cfg.data.patch_size,
                        )

                oracle_action = None
                if float(getattr(cfg.algo, "oracle_bc_coef", 0.0)) > 0.0:
                    oracle_action = oracle_action_toward_goal(
                        seq.patch_sequence[-1],
                        seq.patch_sequence[0],
                        cfg.data.patch_size,
                    )
                action = ppo_agent.select_action(state, seq.patch_sequence, cfg.data.patch_size, oracle_action=oracle_action)
                seq.update_sequence_with_action(action_list[action])

                current_patch_id = seq.patch_sequence[-1]
                prev_patch_id = seq.patch_sequence[-2]
                goal_patch_id = seq.patch_sequence[0]
                prev_dist = get_dist(prev_patch_id, goal_patch_id)
                current_dist = get_dist(current_patch_id, goal_patch_id)
                reward_ex = ppo_agent.get_reward(
                    cfg.data.patch_size,
                    prev_patch_id,
                    current_patch_id,
                    goal_patch_id,
                    seq.patch_sequence[1:-1],
                    best_dist,
                )
                reward_in = (2 * ((F.mse_loss(state_preds, state_gt).item() - 0.8) / 0.1) - 1.0) * 0.25

                if runtime_reward == "ex":
                    reward = reward_ex
                elif runtime_reward == "in":
                    reward = reward_in * runtime_factor * gate_weight(current_dist, optimal_steps) + reward_ex
                elif runtime_reward == "intrinsic_only":
                    reward = reward_in * runtime_factor * gate_weight(current_dist, optimal_steps)
                else:
                    reward = reward_ex
                reward += finish_bonus(current_dist)
                reward += pbrs_bonus(prev_dist, current_dist)

                if should_commit_best(current_dist, best_dist, reward):
                    best_dist = current_dist

                done = current_patch_id == GOAL_PATCH
                if done:
                    average_steps_to_success.append(len(seq.action_sequence))
                    average_deviation_from_opt.append(len(seq.action_sequence) - optimal_steps)
                    num_success += 1
                    if float(getattr(cfg.algo, "sil_coef", 0.0)) > 0.0:
                        ppo_agent.mark_self_imitation_since(episode_buffer_start, weight=1.0)

                ppo_agent.buffer.rewards.append(reward)
                ppo_agent.buffer.is_terminals.append(done)

                time_step += 1
                current_ep_reward += reward

                if time_step % cfg.train.hparams.update_timestep == 0:
                    episode_updates.append(ppo_agent.update(True, seq.patch_sequence, cfg.data.patch_size, device=cfg.train.device))
                    episode_buffer_start = len(ppo_agent.buffer.actions)

                if done or time_step > cfg.train.hparams.max_training_timesteps:
                    break

            if time_step > cfg.train.hparams.max_training_timesteps or stop_requested:
                break

        num_successess.append(num_success)
        episode_successes = num_success
        num_success = 0

        ppo_agent.eval()
        should_validate = (i_episode == 0) or ((i_episode - last_eval_episode) >= val_every_episodes)
        if should_validate:
            cur_val_success = ppo_agent.validate(
                val_config,
                cfg.data.val_path,
                n_config_per_img=cfg.num_config_per_img,
                max_images=val_max_images,
            )[0]
            last_eval_episode = i_episode
            if cur_val_success >= val_success:
                torch.save(ppo_agent.state_dict(), os.path.join(output_dir, cfg.train.expt_name))
                val_success = cur_val_success
        else:
            cur_val_success = val_success

        if i_episode % 50 == 0:
            torch.save(ppo_agent.state_dict(), os.path.join(output_dir, cfg.train.expt_name_tmp) + str(i_episode) + ".pt")

        ppo_agent.train()

        print_running_reward += current_ep_reward
        print_running_episodes += 1
        rolling_avg_reward = print_running_reward / max(print_running_episodes, 1)
        average_reward.append(round(rolling_avg_reward, 2))

        if i_episode < 20:
            denom = max((i_episode + 1) * dataset_size, 1)
        else:
            denom = max(20 * dataset_size, 1)

        rolling_success_ratio = sum(num_successess[-20:]) / denom
        avg_steps_recent = safe_mean(average_steps_to_success[-200:])
        avg_deviation_recent = safe_mean(average_deviation_from_opt[-200:])
        elapsed_sec = time.time() - run_start
        update_stats = aggregate_update_stats(episode_updates, ppo_agent.last_update_stats)

        message = (
            f"Episode: {i_episode} | Timestep: {time_step} | AvgReward: {safe_mean(average_reward[-20:]) / dataset_size:.6f} "
            f"| EpisodeSuccesses: {episode_successes} | RollingSuccessRatio: {rolling_success_ratio:.6f} "
            f"| AvgSteps: {avg_steps_recent:.4f} | AvgDeviation: {avg_deviation_recent:.4f} "
            f"| ValSuccess: {cur_val_success} | BestVal: {val_success} "
            f"| KL: {update_stats['approx_kl']:.5f} | Entropy: {update_stats['entropy']:.5f} "
            f"| ElapsedMin: {elapsed_sec / 60:.2f}"
        )
        print(f"[{datetime.now().isoformat()}] {message}", flush=True)

        metrics_writer.writerow(
            {
                "timestamp": datetime.now().isoformat(),
                "episode": i_episode,
                "time_step": time_step,
                "current_ep_reward": current_ep_reward,
                "rolling_avg_reward": rolling_avg_reward,
                "episode_successes": episode_successes,
                "rolling_success_ratio": rolling_success_ratio,
                "avg_steps_recent": avg_steps_recent,
                "avg_deviation_recent": avg_deviation_recent,
                "val_success": cur_val_success,
                "best_val_success": val_success,
                "policy_loss": update_stats["policy_loss"],
                "value_loss": update_stats["value_loss"],
                "entropy": update_stats["entropy"],
                "approx_kl": update_stats["approx_kl"],
                "clip_fraction": update_stats["clip_fraction"],
                "oracle_bc_loss": update_stats["oracle_bc_loss"],
                "sil_loss": update_stats["sil_loss"],
                "update_epochs_ran": update_stats["update_epochs_ran"],
                "actor_lr": update_stats["actor_lr"],
                "critic_lr": update_stats["critic_lr"],
                "reward_factor": runtime_factor,
                "reward_mode": runtime_reward,
                "progress_metric": runtime_progress_metric,
                "elapsed_sec": elapsed_sec,
            }
        )
        metrics_file.flush()

        write_json(
            heartbeat_path,
            {
                "timestamp": datetime.now().isoformat(),
                "episode": i_episode,
                "time_step": time_step,
                "current_ep_reward": current_ep_reward,
                "rolling_success_ratio": rolling_success_ratio,
                "val_success": cur_val_success,
                "best_val_success": val_success,
                "update_stats": update_stats,
                "runtime_reward": runtime_reward,
                "runtime_factor": runtime_factor,
                "runtime_progress_metric": runtime_progress_metric,
                "gate_mode": cfg.algo.gate_mode,
                "gate_floor": cfg.algo.gate_floor,
                "pbrs_coef": cfg.algo.pbrs_coef,
                "oracle_bc_coef": cfg.algo.oracle_bc_coef,
                "sil_coef": cfg.algo.sil_coef,
                "curriculum_mode": cfg.algo.curriculum_mode,
                "control_path": control_path,
                "elapsed_sec": elapsed_sec,
            },
        )

        trace_file.write(f"CURRENT_PATCH: {CURRENT_PATCH}, GOAL_PATCH: {GOAL_PATCH}\n")
        trace_file.write(str(seq.patch_sequence))
        trace_file.write("\n")
        trace_file.write(str(seq.action_sequence))
        trace_file.write("\n\n")
        trace_file.flush()

        if print_running_episodes % 2 == 0:
            print_running_reward = 0
            print_running_episodes = 0

        i_episode += 1

        control_signature, live_control = read_live_control(control_path, control_signature)
        if live_control and live_control.get("stop", False):
            stop_requested = True

    torch.save(ppo_agent.state_dict(), os.path.join(output_dir, cfg.train.expt_name_tmp) + "latest.pt")
    metrics_file.close()
    trace_file.close()
    print(f"[{datetime.now().isoformat()}] Training finished. Best validation success: {val_success}")
