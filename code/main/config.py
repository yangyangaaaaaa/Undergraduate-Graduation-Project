import os
from pathlib import Path

from easydict import EasyDict as edict


def env_int(name, default):
    value = os.getenv(name)
    return int(value) if value is not None else default


def env_float(name, default):
    value = os.getenv(name)
    return float(value) if value is not None else default


def env_str(name, default):
    value = os.getenv(name)
    return value if value is not None and value != "" else default


def env_bool(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def env_int_list(name, default):
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return list(default)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


PROJECT_ROOT = Path(__file__).resolve().parent


def env_existing_path(name, defaults):
    value = os.getenv(name)
    if value is not None and value != "":
        return value

    for candidate in defaults:
        candidate_path = Path(candidate)
        if candidate_path.exists() or (PROJECT_ROOT / candidate_path).exists():
            return candidate

    return defaults[0]


action_list = {0: "up", 1: "right", 2: "down", 3: "left"}

cfg = edict()

cfg.data = edict()
cfg.data.patch_size = env_int("GEOEXPLORER_PATCH_SIZE", 5)
cfg.data.min_budget = env_int("GEOEXPLORER_MIN_BUDGET", 10)
cfg.data.max_budget = env_int("GEOEXPLORER_MAX_BUDGET", 11)
cfg.data.budget_step = env_int("GEOEXPLORER_BUDGET_STEP", 2)
cfg.min_c = env_int("GEOEXPLORER_MIN_C", 4)
cfg.max_c = env_int("GEOEXPLORER_MAX_C", 9)
cfg.dataset = env_str("GEOEXPLORER_DATASET", "swissview")
cfg.reward = env_str("GEOEXPLORER_REWARD", "in")
cfg.factor = env_float("GEOEXPLORER_FACTOR", 1.0)
cfg.progress_metric = env_str("GEOEXPLORER_PROGRESS_METRIC", "l2sq")

cfg.algo = edict()
cfg.algo.gate_mode = env_str("GEOEXPLORER_GATE_MODE", "none")
cfg.algo.gate_floor = env_float("GEOEXPLORER_GATE_FLOOR", 1.0)
cfg.algo.gate_power = env_float("GEOEXPLORER_GATE_POWER", 1.0)
cfg.algo.gate_blend_alpha = env_float("GEOEXPLORER_GATE_BLEND_ALPHA", 0.0)
cfg.algo.commit_best_on_progress = env_bool("GEOEXPLORER_COMMIT_BEST_ON_PROGRESS", False)
cfg.algo.finish_bonus_scale = env_float("GEOEXPLORER_FINISH_BONUS_SCALE", 0.0)
cfg.algo.finish_bonus_radius = env_int("GEOEXPLORER_FINISH_BONUS_RADIUS", 1)
cfg.algo.pbrs_coef = env_float("GEOEXPLORER_PBRS_COEF", 0.0)
cfg.algo.oracle_bc_coef = env_float("GEOEXPLORER_ORACLE_BC_COEF", 0.0)
cfg.algo.sil_coef = env_float("GEOEXPLORER_SIL_COEF", 0.0)
cfg.algo.curriculum_mode = env_str("GEOEXPLORER_CURRICULUM_MODE", "none")
cfg.algo.curriculum_phase1 = env_float("GEOEXPLORER_CURRICULUM_PHASE1", 0.25)
cfg.algo.curriculum_phase2 = env_float("GEOEXPLORER_CURRICULUM_PHASE2", 0.65)

if cfg.dataset in {"masa", "masa-budget"}:
    cfg.sample_number = 895
elif cfg.dataset == "swissview":
    cfg.sample_number = 500
elif cfg.dataset == "swissviewmonuments":
    cfg.sample_number = 15 * 25
else:
    raise ValueError(f"Unsupported dataset: {cfg.dataset}")

cfg.num_config_per_img = 1 if cfg.dataset == "swissviewmonuments" else env_int("GEOEXPLORER_NUM_CONFIG", 5)

if cfg.dataset == "masa":
    if cfg.data.patch_size == 5:
        cfg.data.train_path = env_str("GEOEXPLORER_TRAIN_DATA", "data/masa/sat_train_grid_5.npy")
        cfg.data.val_path = env_str("GEOEXPLORER_VAL_DATA", "data/masa/sat_val_grid_5.npy")
        cfg.data.test_path = env_str("GEOEXPLORER_TEST_DATA", "data/masa/sat_test_grid_5.npy")
    elif cfg.data.patch_size == 10:
        cfg.data.train_path = env_str("GEOEXPLORER_TRAIN_DATA", "data/masa/sat_train_grid_10.npy")
        cfg.data.val_path = env_str("GEOEXPLORER_VAL_DATA", "data/masa/sat_val_grid_10.npy")
        cfg.data.test_path = env_str("GEOEXPLORER_TEST_DATA", "data/masa/sat_test_grid_10.npy")
elif cfg.dataset == "swissview":
    cfg.data.train_path = env_str("GEOEXPLORER_TRAIN_DATA", "data/swissview/swissview100_sat_patches.npy")
    cfg.data.val_path = env_str("GEOEXPLORER_VAL_DATA", "data/swissview/swissview100_sat_patches.npy")
    cfg.data.test_path = env_str("GEOEXPLORER_TEST_DATA", "data/swissview/swissview100_sat_patches.npy")
elif cfg.dataset == "swissviewmonuments":
    cfg.data.ground_embeds_path = env_existing_path(
        "GEOEXPLORER_GROUND_EMBEDS",
        ["data/swissview/swissviewmonuments_grd.npy"],
    )
    cfg.data.test_path = env_existing_path(
        "GEOEXPLORER_TEST_DATA",
        [
            "data/swissview/swissviewmonuments_sat_patches.npy",
            "data/swissview/swissviewmonuments_patches.npy",
        ],
    )

cfg.pretrain = edict()
cfg.pretrain.ckpt_folder = env_str("GEOEXPLORER_PRETRAIN_CKPT_ROOT", "checkpoint")
cfg.pretrain.expt_folder = env_str("GEOEXPLORER_PRETRAIN_EXPT", "env_modeling")
cfg.pretrain.expt_name = env_str("GEOEXPLORER_PRETRAIN_NAME", "state_action")
cfg.pretrain.log_name = env_str("GEOEXPLORER_PRETRAIN_LOG", "expt_logs.txt")
cfg.pretrain.min_seq_length = env_int("GEOEXPLORER_MIN_SEQ_LENGTH", 6)

cfg.pretrain.hparams = edict()
cfg.pretrain.hparams.accelerator = env_str("GEOEXPLORER_ACCELERATOR", "gpu")
cfg.pretrain.hparams.lr = env_float("GEOEXPLORER_PRETRAIN_LR", 1e-5)
cfg.pretrain.hparams.warmup = env_int("GEOEXPLORER_PRETRAIN_WARMUP", 5)
cfg.pretrain.hparams.devices = env_int("GEOEXPLORER_PRETRAIN_DEVICES", 1)
cfg.pretrain.hparams.epochs = env_int("GEOEXPLORER_PRETRAIN_EPOCHS", 300)
cfg.pretrain.hparams.weight_decay = env_float("GEOEXPLORER_PRETRAIN_WEIGHT_DECAY", 0.0001)

cfg.train = edict()
cfg.train.ckpt_folder = env_str("GEOEXPLORER_TRAIN_CKPT_ROOT", "checkpoint")
cfg.train.expt_folder = env_str("GEOEXPLORER_TRAIN_EXPT", "env_exploration")
cfg.train.load_from_checkpoint = env_bool("GEOEXPLORER_LOAD_FROM_CHECKPOINT", False)
cfg.train.device = env_str("GEOEXPLORER_DEVICE", "cuda:0")
cfg.train.expt_name = env_str("GEOEXPLORER_TRAIN_NAME", "geoexplorer.pt")
cfg.train.expt_name_tmp = env_str("GEOEXPLORER_TRAIN_PREFIX", "geoexplorer_")
cfg.train.log_name = env_str("GEOEXPLORER_TRAIN_LOG", "expt_logs.txt")
cfg.train.metrics_name = env_str("GEOEXPLORER_TRAIN_METRICS", "training_metrics.csv")
cfg.train.heartbeat_name = env_str("GEOEXPLORER_HEARTBEAT_NAME", "heartbeat.json")
cfg.train.control_name = env_str("GEOEXPLORER_CONTROL_NAME", "control.json")
cfg.train.llm_model = env_str("GEOEXPLORER_LLM_MODEL", "tiiuae/falcon-7b")
cfg.train.num_actions = env_int("GEOEXPLORER_NUM_ACTIONS", 4)
cfg.train.llm_hidden_dim = env_int("GEOEXPLORER_LLM_HIDDEN_DIM", 1152)

cfg.train.llm_checkpoint = env_str(
    "GEOEXPLORER_LLM_CHECKPOINT",
    os.path.join(cfg.pretrain.ckpt_folder, cfg.pretrain.expt_folder, cfg.pretrain.expt_name + ".ckpt"),
)
cfg.train.checkpoint_path = env_str(
    "GEOEXPLORER_TRAIN_CHECKPOINT_PATH",
    os.path.join(cfg.train.ckpt_folder, cfg.train.expt_folder, cfg.train.expt_name),
)

cfg.train.hparams = edict()
cfg.train.hparams.max_ep_len = env_int("GEOEXPLORER_MAX_EP_LEN", cfg.data.min_budget)
cfg.train.hparams.max_training_timesteps = env_int("GEOEXPLORER_MAX_TRAINING_TIMESTEPS", int(1e8))
cfg.train.hparams.log_freq = env_int("GEOEXPLORER_LOG_FREQ", cfg.train.hparams.max_ep_len * 2)
cfg.train.hparams.save_model_freq = env_int("GEOEXPLORER_SAVE_MODEL_FREQ", int(2e4))
cfg.train.hparams.update_timestep = env_int("GEOEXPLORER_UPDATE_TIMESTEP", cfg.train.hparams.max_ep_len * 64)
cfg.train.hparams.K_epochs = env_int("GEOEXPLORER_K_EPOCHS", 4)
cfg.train.hparams.eps_clip = env_float("GEOEXPLORER_EPS_CLIP", 0.2)
cfg.train.hparams.gamma = env_float("GEOEXPLORER_GAMMA", 0.93)
cfg.train.hparams.lr_actor = env_float("GEOEXPLORER_LR_ACTOR", 0.0001)
cfg.train.hparams.lr_critic = env_float("GEOEXPLORER_LR_CRITIC", 0.0001)
cfg.train.hparams.lr_llm = env_float("GEOEXPLORER_LR_LLM", 0.0001)
cfg.train.hparams.lr_gamma = env_float("GEOEXPLORER_LR_GAMMA", 0.9999)
cfg.train.hparams.random_seed = env_int("GEOEXPLORER_RANDOM_SEED", 42)
cfg.train.hparams.ent_coef = env_float("GEOEXPLORER_ENT_COEF", 0.01)
cfg.train.hparams.vf_coef = env_float("GEOEXPLORER_VF_COEF", 0.5)
cfg.train.hparams.max_grad_norm = env_float("GEOEXPLORER_MAX_GRAD_NORM", 0.5)
cfg.train.hparams.normalize_advantage = env_bool("GEOEXPLORER_NORMALIZE_ADVANTAGE", True)
cfg.train.hparams.target_kl = env_float("GEOEXPLORER_TARGET_KL", 0.0)
cfg.train.hparams.val_distances = env_int_list("GEOEXPLORER_VAL_DISTS", [4, 5, 6, 7, 8])
