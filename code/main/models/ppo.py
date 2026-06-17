import torch
import torch.nn as nn
import sys
sys.path.append("..")
from torch.distributions import Categorical
from data_utils import Sequence
from config import cfg, action_list
from models.pretrain_model import MaskedActionModeling
from utils import get_dist
import numpy as np
import torch.nn.functional as F
try:
    import wandb  # noqa: F401
except Exception:
    wandb = None

DEFAULT_DEVICE = cfg.train.device

def grid_steps(index1, index2, grid_size=5):
    # cal row and clumn from index
    row1, col1 = divmod(index1, grid_size)
    row2, col2 = divmod(index2, grid_size)

    # use Manhattan distance to calculate steps
    steps = abs(row1 - row2) + abs(col1 - col2)
    return steps


def _safe_div(numerator, denominator):
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _format_eval_summary(label, success, total_trials, avg_steps_to_success=0.0, avg_dev_steps=0.0, avg_reward=None):
    success_ratio = _safe_div(success, total_trials)
    parts = [
        f"{label} Trials : {int(total_trials)}",
        f"{label} Success : {int(success)}/{int(total_trials)} ({success_ratio:.4f})",
    ]
    if avg_reward is not None:
        parts.append(f"{label} Avg Reward : {float(avg_reward):.4f}")
    if success > 0:
        parts.append(f"{label} Avg Steps Success : {_safe_div(avg_steps_to_success, success):.4f}")
        parts.append(f"{label} Dev : {_safe_div(avg_dev_steps, success):.4f}")
    return " \t ".join(parts)


def _format_unseen_summary(label, success, valid_trials, step_to_goal):
    success_ratio = _safe_div(success, valid_trials)
    mean_final_distance = _safe_div(step_to_goal, valid_trials)
    return (
        f"{label} Valid Trials : {int(valid_trials)}"
        f" \t {label} Success : {int(success)}/{int(valid_trials)} ({success_ratio:.4f})"
        f" \t {label} Mean Final Dist : {mean_final_distance:.4f}"
    )

class PatchCounter:
    def __init__(self):
        # 5x5 grid
        self.size = 5
        self.count_matrix = np.zeros((self.size, self.size), dtype=float)

    def visit(self, index):
        # counting
        x, y = divmod(index, self.size)
        if 0 <= index < 25:
            self.count_matrix[x, y] += 1
        else:
            print(f"index {index} out of range!")

    def visit_number(self, index, number):
        # counting
        x, y = divmod(index, self.size)
        if 0 <= index < 25:
            self.count_matrix[x, y] += number
        else:
            print(f"index {index} out of range!")

    def get_count(self, index):
        # get the count of a specific index
        x, y = divmod(index, self.size)
        if 0 <= index < 25:
            return self.count_matrix[x, y]
        else:
            print(f"index {index} out of range!")
            return None
    
    def reset(self):
        # reset the count matrix to zero
        self.count_matrix.fill(0)

    def display(self):
        # print the current count matrix
        print(self.count_matrix)

    def get_data(self):
        return self.count_matrix



class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.action_masks = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.oracle_actions = []
        self.imitation_weights = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.action_masks[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.oracle_actions[:]
        del self.imitation_weights[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim=5):
        super(ActorCritic, self).__init__()
        self.action_dim = action_dim

        self.actor = nn.Sequential(
                        nn.Linear(state_dim, state_dim//4),
                        nn.Tanh(),
                        nn.Linear(state_dim//4, state_dim//4),
                        nn.Tanh(),
                        nn.Linear(state_dim//4, action_dim),
                        nn.Softmax(dim=-1)
                    )
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, state_dim//4),
                        nn.Tanh(),
                        nn.Linear(state_dim//4, state_dim//4),
                        nn.Tanh(),
                        nn.Linear(state_dim//4, 1)
                    )

    def forward(self):
        raise NotImplementedError

    def build_action_mask(self, patch_sequence, patch_size, device):
        current_patch = int(patch_sequence[-1])
        mask = torch.ones((1, self.action_dim), dtype=torch.float32, device=device)

        if current_patch % patch_size == 0:
            mask[..., 3] = 0.0
        if current_patch % patch_size == patch_size - 1:
            mask[..., 1] = 0.0
        if current_patch < patch_size:
            mask[..., 0] = 0.0
        if current_patch >= patch_size**2 - patch_size:
            mask[..., 2] = 0.0

        return mask

    def apply_action_mask(self, action_probs, action_mask):
        masked_probs = action_probs * action_mask
        masked_probs = masked_probs.clamp_min(1e-8)
        masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return masked_probs
    
    def act(self, state, patch_sequence, patch_size):
        action_probs = self.actor(state)
        action_mask = self.build_action_mask(patch_sequence, patch_size, action_probs.device)
        action_probs = self.apply_action_mask(action_probs, action_mask)

        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach(), action_mask.detach()

    def evaluate(self, state, action, action_mask): 
        action_probs = self.actor(state)
        action_probs = self.apply_action_mask(action_probs, action_mask)

        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy

    def entropy_act(self, state, patch_sequence, patch_size):
        action_probs = self.actor(state)
        action_mask = self.build_action_mask(patch_sequence, patch_size, action_probs.device)
        action_probs = self.apply_action_mask(action_probs, action_mask)

        top_probs = torch.topk(action_probs, 2).values
        if (top_probs[0, 0]) > 0.6:
            action = torch.argmax(action_probs, dim=-1)
        else:
            dist = Categorical(action_probs+1e-6)
            action = dist.sample()

        return action.item()
    
    def greedy_act(self, state, patch_sequence, patch_size):
        action_probs = self.actor(state)
        action_mask = self.build_action_mask(patch_sequence, patch_size, action_probs.device)
        action_probs = self.apply_action_mask(action_probs, action_mask)

        action = torch.argmax(action_probs, dim=-1)

        return action.item()
    
    def stochastic_act(self, state, patch_sequence, patch_size):
        action_probs = self.actor(state)
        action_mask = self.build_action_mask(patch_sequence, patch_size, action_probs.device)
        action_probs = self.apply_action_mask(action_probs, action_mask)

        dist = Categorical(action_probs)
        action = dist.sample()

        return action.item()

    def random_act(self, state, patch_sequence, patch_size):
        action_probs = state.new_full((state.shape[0], self.action_dim), 1.0 / self.action_dim)
        action_mask = self.build_action_mask(patch_sequence, patch_size, action_probs.device)
        action_probs = self.apply_action_mask(action_probs, action_mask)

        dist = Categorical(action_probs)
        action = dist.sample()

        return action.item()
    

class PPO(nn.Module):
    def __init__(
        self,
        lr_actor,
        lr_critic,
        lr_llm,
        gamma,
        K_epochs,
        eps_clip,
        lr_gamma,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        normalize_advantage=True,
        target_kl=0.0,
        oracle_bc_coef=0.0,
        sil_coef=0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self.oracle_bc_coef = oracle_bc_coef
        self.sil_coef = sil_coef
        self.device = DEFAULT_DEVICE
        
        self.buffer = RolloutBuffer()

        self.llm_module = MaskedActionModeling.load_from_checkpoint(
            cfg.train.llm_checkpoint,
            train_dataset=None,
            val_dataset=None,
        )

        self.llm = self.llm_module.llm
        # LLM features are used as frozen state encoder in PPO.
        self.llm.eval()
        for param in self.llm.parameters():
            param.requires_grad = False

        state_dim = self.llm.config.word_embed_proj_dim
        action_dim = self.llm.config.num_actions

        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic},
                    ])

        self.schedular = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, lr_gamma)

        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        self.last_update_stats = {}

    def apply_live_overrides(self, overrides):
        if "gamma" in overrides:
            self.gamma = float(overrides["gamma"])
        if "K_epochs" in overrides:
            self.K_epochs = int(overrides["K_epochs"])
        if "eps_clip" in overrides:
            self.eps_clip = float(overrides["eps_clip"])
        if "ent_coef" in overrides:
            self.ent_coef = float(overrides["ent_coef"])
        if "vf_coef" in overrides:
            self.vf_coef = float(overrides["vf_coef"])
        if "max_grad_norm" in overrides:
            self.max_grad_norm = float(overrides["max_grad_norm"])
        if "target_kl" in overrides:
            self.target_kl = float(overrides["target_kl"])
        if "oracle_bc_coef" in overrides:
            self.oracle_bc_coef = float(overrides["oracle_bc_coef"])
        if "sil_coef" in overrides:
            self.sil_coef = float(overrides["sil_coef"])
        if "lr_actor" in overrides:
            self.optimizer.param_groups[0]["lr"] = float(overrides["lr_actor"])
        if len(self.optimizer.param_groups) > 1 and "lr_critic" in overrides:
            self.optimizer.param_groups[1]["lr"] = float(overrides["lr_critic"])

    def get_current_lrs(self):
        actor_lr = self.optimizer.param_groups[0]["lr"]
        critic_lr = self.optimizer.param_groups[1]["lr"] if len(self.optimizer.param_groups) > 1 else actor_lr
        return actor_lr, critic_lr

    ### required
    def select_action(self, state, patch_sequence, patch_size, oracle_action=None):
        
        with torch.no_grad():
            action, action_logprob, state_val, action_mask = self.policy_old.act(state, patch_sequence, patch_size)
            
        self.buffer.states.append(state.detach())
        self.buffer.actions.append(action)
        self.buffer.action_masks.append(action_mask)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)
        self.buffer.oracle_actions.append(-1 if oracle_action is None else int(oracle_action))
        self.buffer.imitation_weights.append(0.0)

        return action.item()

    def mark_self_imitation_since(self, start_index, weight=1.0):
        start_index = max(0, int(start_index))
        for idx in range(start_index, len(self.buffer.imitation_weights)):
            self.buffer.imitation_weights[idx] = max(float(self.buffer.imitation_weights[idx]), float(weight))
    
    def select_stochastic_action(self, state, patch_sequence, patch_size):
        return self.policy_old.stochastic_act(state, patch_sequence, patch_size)
    
    def select_greedy_action(self, state, patch_sequence, patch_size):
        return self.policy_old.greedy_act(state, patch_sequence, patch_size)
    
    def select_random_action(self, state, patch_sequence, patch_size):
        return self.policy_old.random_act(state, patch_sequence, patch_size)
    
    def select_entropy_action(self, state, patch_sequence, patch_size):
        return self.policy_old.entropy_act(state, patch_sequence, patch_size)
    

    def get_reward(self, patch_size, prev_patch_id, current_patch_id, goal_patch_id, patch_sequence, best_dist):

        cur_rows = current_patch_id//patch_size
        cur_cols = current_patch_id%patch_size

        goal_rows = goal_patch_id//patch_size
        goal_cols = goal_patch_id%patch_size


        current_dist = get_dist(current_patch_id, goal_patch_id)

        if current_patch_id == prev_patch_id:
            return -1
        if cur_cols==goal_cols and cur_rows==goal_rows:
            return 2
        elif current_patch_id in patch_sequence:
            return -1
        elif cfg.progress_metric == "manhattan" and current_dist < best_dist:
            return 1
        elif cfg.progress_metric == "l2sq" and (cur_rows - goal_rows)**2 + (cur_cols - goal_cols)**2 < best_dist:
            return 1
        else:
            return -1


    def update(self, flag, patch_sequence, patch_size, device=None):
        device = device or self.device

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_action_masks = torch.squeeze(torch.stack(self.buffer.action_masks, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)
        old_oracle_actions = torch.tensor(self.buffer.oracle_actions, dtype=torch.long, device=device).view(-1)
        old_imitation_weights = torch.tensor(self.buffer.imitation_weights, dtype=torch.float32, device=device).view(-1)
        old_actions_flat = old_actions.view(-1)
        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()
        if self.normalize_advantage and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # Optimize policy for K epochs
        policy_losses = []
        value_losses = []
        entropies = []
        approx_kls = []
        clip_fractions = []
        oracle_bc_losses = []
        sil_losses = []
        epochs_ran = 0
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, old_action_masks)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            policy_loss = -torch.min(surr1, surr2)
            value_loss = self.MseLoss(state_values, rewards)
            entropy_term = dist_entropy.mean()
            extra_loss = state_values.new_tensor(0.0)

            action_probs = self.policy.apply_action_mask(self.policy.actor(old_states), old_action_masks)
            action_probs = action_probs.clamp_min(1e-8)
            valid_oracle = old_oracle_actions >= 0
            if self.oracle_bc_coef > 0 and valid_oracle.any():
                oracle_targets = old_oracle_actions[valid_oracle]
                oracle_probs = action_probs[valid_oracle].gather(1, oracle_targets.view(-1, 1)).squeeze(1)
                oracle_bc_loss = -torch.log(oracle_probs).mean()
                extra_loss = extra_loss + self.oracle_bc_coef * oracle_bc_loss
                oracle_bc_losses.append(float(oracle_bc_loss.item()))
            else:
                oracle_bc_losses.append(0.0)

            valid_sil = old_imitation_weights > 0
            if self.sil_coef > 0 and valid_sil.any():
                sil_targets = old_actions_flat[valid_sil]
                sil_probs = action_probs[valid_sil].gather(1, sil_targets.view(-1, 1)).squeeze(1)
                sil_loss = (-torch.log(sil_probs) * old_imitation_weights[valid_sil]).mean()
                extra_loss = extra_loss + self.sil_coef * sil_loss
                sil_losses.append(float(sil_loss.item()))
            else:
                sil_losses.append(0.0)

            loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * dist_entropy + extra_loss
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            epochs_ran += 1

            approx_kl = (old_logprobs.detach() - logprobs).mean().item()
            clipped = ((ratios - 1.0).abs() > self.eps_clip).float().mean().item()
            policy_losses.append(policy_loss.mean().item())
            value_losses.append(float(value_loss.item()))
            entropies.append(float(entropy_term.item()))
            approx_kls.append(float(approx_kl))
            clip_fractions.append(float(clipped))

            if self.target_kl > 0 and approx_kl > self.target_kl:
                break
        
        self.schedular.step()
            
        # Copy new weights into old policy
        if True: #flag:
            self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
        actor_lr, critic_lr = self.get_current_lrs()
        self.last_update_stats = {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
            "oracle_bc_loss": float(np.mean(oracle_bc_losses)) if oracle_bc_losses else 0.0,
            "sil_loss": float(np.mean(sil_losses)) if sil_losses else 0.0,
            "update_epochs_ran": epochs_ran,
            "actor_lr": float(actor_lr),
            "critic_lr": float(critic_lr),
        }
        return self.last_update_stats

    @torch.no_grad()
    def validate_varying_budget(self, config, valid_path, tokenizer=None, n_config_per_img=5, flag='none'):
        dataset_dict = np.load(valid_path, allow_pickle=True)
        total_imgs = len(dataset_dict[()].keys())
        res = [0]*n_config_per_img

        for budget in range(cfg.data.min_budget, cfg.data.max_budget, cfg.data.budget_step):
            num_success=0
            avg_dev_steps=0 
            avg_steps_to_success=0
            for i in range(total_imgs):
                for j in range(n_config_per_img):
                    seq = Sequence(dataset_dict[()][f"img_{i}"], tokenizer, num_patches=cfg.data.patch_size)
                    GOAL_PATCH = config[f"img_{i}"][j][0]
                    CURRENT_PATCH = config[f"img_{i}"][j][1]
                    seq.init_with_goal_image(GOAL_PATCH)
                    seq.update_sequence_with_satellite_image_token(CURRENT_PATCH)
                    best_dist=get_dist(GOAL_PATCH, CURRENT_PATCH)
                    opt_steps=get_dist(GOAL_PATCH, CURRENT_PATCH)

                    for t in (range(1, budget+1)):

                        inputs = seq.get_input_for_model(device=self.device)

                        if inputs["actions"] == []:
                            state, state_preds, state_gt = self.llm(inputs_embeds=inputs["inputs_embeds"],
                            patch_sequence=inputs["patch_sequence"][:, 1:],
                            patch_size=cfg.data.patch_size)
                        else:
                            state, state_preds, state_gt = self.llm(
                                inputs_embeds=inputs["inputs_embeds"],
                                actions=[inputs["actions"]],
                                patch_sequence=inputs["patch_sequence"][:, 1:],
                                patch_size=cfg.data.patch_size)

                        if flag=='entropy':
                            action = self.select_entropy_action(state, seq.patch_sequence, cfg.data.patch_size)
                        else:
                            action = self.select_greedy_action(state, seq.patch_sequence, cfg.data.patch_size)

                        seq.update_sequence_with_action(action_list[action])
                        
                        current_patch_id = seq.patch_sequence[-1]

                        done = (current_patch_id==GOAL_PATCH)
                        if done:
                            avg_steps_to_success+=len(seq.action_sequence)
                            avg_dev_steps+=(len(seq.action_sequence)-opt_steps)
                            num_success+=1
                            #res[budget-5][j]+=1
                            break
            total_trials = total_imgs * n_config_per_img
            print(_format_eval_summary(f"Val(B={budget})", num_success, total_trials, avg_steps_to_success, avg_dev_steps))
            #wandb.log({"val success": num_success})
        return num_success, res


    @torch.no_grad()
    def validate(self, config, valid_path, tokenizer=None, n_config_per_img=5, flag='none', max_images=None):
        dataset_dict = np.load(valid_path, allow_pickle=True)
        total_imgs = len(dataset_dict[()].keys())
        if max_images is not None:
            total_imgs = min(total_imgs, max_images)
        num_success=0
        avg_dev_steps=0 
        avg_steps_to_success=0
        avg_reward=0
        res = []


        for i in range(total_imgs):
            
            for j in range(n_config_per_img):
                
                GOAL_PATCH = config[f"img_{i}"][j][0]
                CURRENT_PATCH = config[f"img_{i}"][j][1]
                seq = Sequence(dataset_dict[()][f"img_{i}"], tokenizer, num_patches=cfg.data.patch_size)
                


                seq.init_with_goal_image(GOAL_PATCH)
                seq.update_sequence_with_satellite_image_token(CURRENT_PATCH)
                best_dist=get_dist(GOAL_PATCH, CURRENT_PATCH)
                opt_steps=get_dist(GOAL_PATCH, CURRENT_PATCH)
                is_success = False
                reward_trace = []

                for t in (range(1, cfg.train.hparams.max_ep_len+1)):

                    inputs = seq.get_input_for_model(device=self.device)

                    if inputs["actions"] == []:
                        state, state_preds, state_gt = self.llm(inputs_embeds=inputs["inputs_embeds"],
                        patch_sequence=inputs["patch_sequence"][:, 1:],
                        patch_size=cfg.data.patch_size)
                    else:
                        state, state_preds, state_gt = self.llm(
                            inputs_embeds=inputs["inputs_embeds"],
                            actions=[inputs["actions"]],
                            patch_sequence=inputs["patch_sequence"][:, 1:],
                            patch_size=cfg.data.patch_size)

                    if flag=='entropy':
                        action = self.select_entropy_action(state, seq.patch_sequence, cfg.data.patch_size)
                    else:
                        action = self.select_greedy_action(state, seq.patch_sequence, cfg.data.patch_size)
                    seq.update_sequence_with_action(action_list[action])
                    
                    
                    current_patch_id = seq.patch_sequence[-1]
                    prev_patch_id = seq.patch_sequence[-2]
                    goal_patch_id = seq.patch_sequence[0]
                    reward_in = (2*((F.mse_loss(state_preds, state_gt).item() - 0.8) / 0.1) - 1.0)*0.25
                    #reward_in = -(2*(F.cosine_similarity(state_gt, state_preds).item()) -1.0)
                    reward_ex = self.get_reward(cfg.data.patch_size, prev_patch_id, current_patch_id, goal_patch_id, seq.patch_sequence[1:-1], best_dist)
                    if cfg.reward == 'ex':
                        reward = reward_ex
                    elif cfg.reward == 'in':
                        reward = reward_in * cfg.factor + reward_ex
                    else:
                        reward = reward_ex
                    reward_trace.append(float(reward))
                    current_dist = get_dist(current_patch_id, goal_patch_id)
                    if current_dist < best_dist:
                        best_dist = current_dist
                    avg_reward+=reward
                    done = (current_patch_id==GOAL_PATCH)
                    if done:
                        avg_steps_to_success+=len(seq.action_sequence)
                        avg_dev_steps+=(len(seq.action_sequence)-opt_steps)
                        num_success+=1
                        is_success = True
                        break

                trajectory = [int(patch_id) for patch_id in seq.patch_sequence[1:]]
                res.append({
                    "img_idx": i,
                    "config_idx": j,
                    "goal": int(GOAL_PATCH),
                    "start": int(CURRENT_PATCH),
                    "final": int(seq.patch_sequence[-1]),
                    "traj": trajectory,
                    "actions": list(seq.action_sequence),
                    "distance": int(opt_steps),
                    "optimal_steps": int(opt_steps),
                    "path_length": len(trajectory),
                    "final_distance": int(get_dist(seq.patch_sequence[-1], GOAL_PATCH)),
                    "success": bool(is_success),
                    "reward_trace": reward_trace,
                })

        avg_reward = avg_reward / max(total_imgs * n_config_per_img, 1)
        total_trials = total_imgs * n_config_per_img
        print(_format_eval_summary("Val", num_success, total_trials, avg_steps_to_success, avg_dev_steps, avg_reward=avg_reward))

        return num_success, res


    @torch.no_grad()
    def validate_unseen(self, config, valid_path, tokenizer=None, n_config_per_img=5, flag='none', max_images=None):
        dataset_dict = np.load(valid_path, allow_pickle=True)
        total_imgs = len(dataset_dict[()].keys())
        if max_images is not None:
            total_imgs = min(total_imgs, max_images)
        num_success=0
        avg_dev_steps=0 
        avg_steps_to_success=0
        avg_reward=0
        res = [0]*n_config_per_img

        num_pass = 0
        step_to_goal = 0


        for i in range(total_imgs):
            
            for j in range(n_config_per_img):
                
                GOAL_PATCH = config[f"img_{i}"][j][0]
                CURRENT_PATCH = config[f"img_{i}"][j][1]


                if GOAL_PATCH == 999:
                    num_pass += 1 
                    continue

                seq = Sequence(dataset_dict[()][f"img_{i}"], tokenizer, num_patches=cfg.data.patch_size)

                seq.init_with_goal_image(GOAL_PATCH)
                seq.update_sequence_with_satellite_image_token(CURRENT_PATCH)
                best_dist=get_dist(GOAL_PATCH, CURRENT_PATCH)
                opt_steps=get_dist(GOAL_PATCH, CURRENT_PATCH)

                for t in (range(1, cfg.train.hparams.max_ep_len+1)):

                    inputs = seq.get_input_for_model(device=self.device)

                    if inputs["actions"] == []:
                        state, state_preds, state_gt = self.llm(inputs_embeds=inputs["inputs_embeds"],
                        patch_sequence=inputs["patch_sequence"][:, 1:],
                        patch_size=cfg.data.patch_size)
                    else:
                        state, state_preds, state_gt = self.llm(
                            inputs_embeds=inputs["inputs_embeds"],
                            actions=[inputs["actions"]],
                            patch_sequence=inputs["patch_sequence"][:, 1:],
                            patch_size=cfg.data.patch_size)

                    if flag=='entropy':
                        action = self.select_entropy_action(state, seq.patch_sequence, cfg.data.patch_size)
                    else:
                        action = self.select_greedy_action(state, seq.patch_sequence, cfg.data.patch_size)
                    seq.update_sequence_with_action(action_list[action])
                    
                    current_patch_id = seq.patch_sequence[-1]
                    prev_patch_id = seq.patch_sequence[-2]
                    goal_patch_id = seq.patch_sequence[0]
                    done = (current_patch_id==GOAL_PATCH)
                    if done:
                        avg_steps_to_success+=len(seq.action_sequence)
                        avg_dev_steps+=(len(seq.action_sequence)-opt_steps)
                        num_success+=1
                        res[j]+=1
                        break

                step_to_goal += grid_steps(seq.patch_sequence[-1], GOAL_PATCH)
            
        valid_trials = total_imgs * n_config_per_img - num_pass
        print(_format_unseen_summary("Val", num_success, valid_trials, step_to_goal))
        
        return num_success, res

    @torch.no_grad()
    def validate_ground(self, config, sat_paths, ground_path, tokenizer=None, n_config_per_img=5):

        ground_dict = np.load(ground_path, allow_pickle=True)
        sat_dict = np.load(sat_paths, allow_pickle=True)
        total_imgs = len(sat_dict[()].keys())
        num_success=0
        avg_dev_steps=0 
        avg_steps_to_success=0
        for i in range(total_imgs):
            for j in range(n_config_per_img):
                goal_ground = ground_dict[()][f"img_{i}"]
                GOAL_PATCH = config[f"img_{i}"][j][0]
                CURRENT_PATCH = config[f"img_{i}"][j][1]
                sat_embeds = sat_dict[()][f"img_{i}"].reshape(25, -1)
                seq = Sequence(sat_embeds, tokenizer, num_patches=5)

                seq.init_with_goal_embed(goal_ground, GOAL_PATCH)
                seq.update_sequence_with_satellite_image_token(CURRENT_PATCH)
                best_dist=get_dist(GOAL_PATCH, CURRENT_PATCH)
                opt_steps=get_dist(GOAL_PATCH, CURRENT_PATCH)

                for t in (range(1, cfg.train.hparams.max_ep_len+1)):

                    inputs = seq.get_input_for_model(device=self.device)

                    if inputs["actions"] == []:
                        state, state_preds, state_gt = self.llm(inputs_embeds=inputs["inputs_embeds"],
                        patch_sequence=inputs["patch_sequence"][:, 1:],
                        patch_size=cfg.data.patch_size)
                    else:
                        state, state_preds, state_gt = self.llm(
                            inputs_embeds=inputs["inputs_embeds"],
                            actions=[inputs["actions"]],
                            patch_sequence=inputs["patch_sequence"][:, 1:],
                            patch_size=cfg.data.patch_size)

                    action = self.select_greedy_action(state, seq.patch_sequence, cfg.data.patch_size)
                    seq.update_sequence_with_action(action_list[action])
                    
                    current_patch_id = seq.patch_sequence[-1]
                    
                    done = (current_patch_id==GOAL_PATCH)
                    if done:
                        avg_steps_to_success+=len(seq.action_sequence)
                        avg_dev_steps+=(len(seq.action_sequence)-opt_steps)
                        num_success+=1
                        break

        total_trials = total_imgs * n_config_per_img
        print(_format_eval_summary("Val", num_success, total_trials, avg_steps_to_success, avg_dev_steps))
        return num_success
    

    @torch.no_grad()
    def validate_text(self, config, sat_paths, ground_path, tokenizer=None, n_config_per_img=5):

        ground_dict = np.load(ground_path, allow_pickle=True)
        sat_dict = np.load(sat_paths, allow_pickle=True)
        total_imgs = len(sat_dict[()].keys())
        num_success=0
        avg_dev_steps=0 
        avg_steps_to_success=0
        avg_reward=0
        for i in range(total_imgs):
            for j in range(n_config_per_img):
                goal_ground = ground_dict[i]
                GOAL_PATCH = config[f"img_{i}"][j][0]
                CURRENT_PATCH = config[f"img_{i}"][j][1]
                sat_embeds = sat_dict[()][f"img_{i}"].reshape(25, -1)
                seq = Sequence(sat_embeds, tokenizer, num_patches=5)

                seq.init_with_goal_embed(goal_ground, GOAL_PATCH)
                seq.update_sequence_with_satellite_image_token(CURRENT_PATCH)
                best_dist=get_dist(GOAL_PATCH, CURRENT_PATCH)
                opt_steps=get_dist(GOAL_PATCH, CURRENT_PATCH)

                for t in (range(1, cfg.train.hparams.max_ep_len+1)):

                    inputs = seq.get_input_for_model(device=self.device)

                    if inputs["actions"] == []:
                        state, state_preds, state_gt = self.llm(inputs_embeds=inputs["inputs_embeds"],
                        patch_sequence=inputs["patch_sequence"][:, 1:],
                        patch_size=cfg.data.patch_size)
                    else:
                        state, state_preds, state_gt = self.llm(
                            inputs_embeds=inputs["inputs_embeds"],
                            actions=[inputs["actions"]],
                            patch_sequence=inputs["patch_sequence"][:, 1:],
                            patch_size=cfg.data.patch_size)

                    action = self.select_greedy_action(state, seq.patch_sequence, cfg.data.patch_size)
                    seq.update_sequence_with_action(action_list[action])
                    
                    current_patch_id = seq.patch_sequence[-1]
                    done = (current_patch_id==GOAL_PATCH)
                    if done:
                        avg_steps_to_success+=len(seq.action_sequence)
                        avg_dev_steps+=(len(seq.action_sequence)-opt_steps)
                        num_success+=1
                        break
        total_trials = total_imgs * n_config_per_img
        print(_format_eval_summary("Val", num_success, total_trials, avg_steps_to_success, avg_dev_steps))
        return num_success

    @torch.no_grad()
    def validate_ground_unseen(self, config, sat_paths, ground_path, tokenizer=None, n_config_per_img=5, max_images=None):

        ground_dict = np.load(ground_path, allow_pickle=True)
        sat_dict = np.load(sat_paths, allow_pickle=True)
        total_imgs = len(sat_dict[()].keys())
        if max_images is not None:
            total_imgs = min(total_imgs, max_images)
        #print(len(ground_dict[()].keys()))
        num_success=0
        avg_dev_steps=0 
        avg_steps_to_success=0
        avg_reward=0
        num_pass = 0
        step_to_goal = 0
        for i in range(0, total_imgs):
            #print(i)
            for j in range(n_config_per_img):
                #goal_ground = ground_dict[i]#ground_dict[()][f"img_{i}"]#ground_dict[i]#
                goal_ground = ground_dict[()][f"img_{i}"]
                GOAL_PATCH = config[f"img_{i}"][j][0]
                CURRENT_PATCH = config[f"img_{i}"][j][1]
                if GOAL_PATCH == 999:
                    num_pass += 1 
                    continue


                sat_embeds = sat_dict[()][f"img_{i}"].reshape(25, -1)
                seq = Sequence(sat_embeds, tokenizer, num_patches=5)

                seq.init_with_goal_embed(goal_ground, GOAL_PATCH)
                seq.update_sequence_with_satellite_image_token(CURRENT_PATCH)
                best_dist=get_dist(GOAL_PATCH, CURRENT_PATCH)
                opt_steps=get_dist(GOAL_PATCH, CURRENT_PATCH)

                for t in (range(1, cfg.train.hparams.max_ep_len+1)):

                    inputs = seq.get_input_for_model(device=self.device)

                    if inputs["actions"] == []:
                        state, state_preds, state_gt = self.llm(inputs_embeds=inputs["inputs_embeds"],
                        patch_sequence=inputs["patch_sequence"][:, 1:],
                        patch_size=cfg.data.patch_size)
                    else:
                        state, state_preds, state_gt = self.llm(
                            inputs_embeds=inputs["inputs_embeds"],
                            actions=[inputs["actions"]],
                            patch_sequence=inputs["patch_sequence"][:, 1:],
                            patch_size=cfg.data.patch_size)

                    action = self.select_greedy_action(state, seq.patch_sequence, cfg.data.patch_size)
                    seq.update_sequence_with_action(action_list[action])
                    
                    current_patch_id = seq.patch_sequence[-1]
                    prev_patch_id = seq.patch_sequence[-2]
                    goal_patch_id = seq.patch_sequence[0]
                    done = (current_patch_id==GOAL_PATCH)
                    if done:
                        avg_steps_to_success+=len(seq.action_sequence)
                        avg_dev_steps+=(len(seq.action_sequence)-opt_steps)
                        num_success+=1
                        break
                step_to_goal += grid_steps(seq.patch_sequence[-1], GOAL_PATCH)
                #print(seq.patch_sequence)

        valid_trials = total_imgs * n_config_per_img - num_pass
        print(_format_unseen_summary("Val", num_success, valid_trials, step_to_goal))
        return num_success

    @torch.no_grad()
    def validate_text_unseen(self, config, sat_paths, text_path, tokenizer=None, n_config_per_img=5):

        ground_dict = np.load(text_path, allow_pickle=True)
        sat_dict = np.load(sat_paths, allow_pickle=True)
        total_imgs = len(sat_dict[()].keys())
        num_success=0
        avg_dev_steps=0 
        avg_steps_to_success=0
        avg_reward=0
        num_pass = 0
        step_to_goal = 0
        for i in range(0, total_imgs):
            for j in range(n_config_per_img):
                goal_ground = ground_dict[i]
                GOAL_PATCH = config[f"img_{i}"][j][0]
                CURRENT_PATCH = config[f"img_{i}"][j][1]
                if GOAL_PATCH == 999:
                    num_pass += 1 
                    continue


                sat_embeds = sat_dict[()][f"img_{i}"].reshape(25, -1)
                seq = Sequence(sat_embeds, tokenizer, num_patches=5)

                seq.init_with_goal_embed(goal_ground, GOAL_PATCH)
                seq.update_sequence_with_satellite_image_token(CURRENT_PATCH)
                best_dist=get_dist(GOAL_PATCH, CURRENT_PATCH)
                opt_steps=get_dist(GOAL_PATCH, CURRENT_PATCH)

                for t in (range(1, cfg.train.hparams.max_ep_len+1)):

                    inputs = seq.get_input_for_model(device=self.device)

                    if inputs["actions"] == []:
                        state, state_preds, state_gt = self.llm(inputs_embeds=inputs["inputs_embeds"],
                        patch_sequence=inputs["patch_sequence"][:, 1:],
                        patch_size=cfg.data.patch_size)
                    else:
                        state, state_preds, state_gt = self.llm(
                            inputs_embeds=inputs["inputs_embeds"],
                            actions=[inputs["actions"]],
                            patch_sequence=inputs["patch_sequence"][:, 1:],
                            patch_size=cfg.data.patch_size)

                    action = self.select_greedy_action(state, seq.patch_sequence, cfg.data.patch_size)
                    seq.update_sequence_with_action(action_list[action])
                    
                    current_patch_id = seq.patch_sequence[-1]
                    prev_patch_id = seq.patch_sequence[-2]
                    goal_patch_id = seq.patch_sequence[0]
                    done = (current_patch_id==GOAL_PATCH)
                    if done:
                        avg_steps_to_success+=len(seq.action_sequence)
                        avg_dev_steps+=(len(seq.action_sequence)-opt_steps)
                        num_success+=1
                        break
                step_to_goal += grid_steps(seq.patch_sequence[-1], GOAL_PATCH)

        valid_trials = total_imgs * n_config_per_img - num_pass
        print(_format_unseen_summary("Val", num_success, valid_trials, step_to_goal))
        return num_success
