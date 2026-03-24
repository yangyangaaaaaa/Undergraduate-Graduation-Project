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
import wandb

device = torch.device('cuda:0')

def grid_steps(index1, index2, grid_size=5):
    # cal row and clumn from index
    row1, col1 = divmod(index1, grid_size)
    row2, col2 = divmod(index2, grid_size)

    # use Manhattan distance to calculate steps
    steps = abs(row1 - row2) + abs(col1 - col2)
    return steps

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
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim=5):
        super(ActorCritic, self).__init__()

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
    
    def act(self, state, patch_sequence, patch_size):
        action_probs = self.actor(state)

        if patch_sequence[-1]%patch_size==0:
            action_probs=torch.multiply(action_probs, torch.FloatTensor([[1.0, 1.0, 1.0, 0.0]]).cuda())
        if patch_sequence[-1]%patch_size==patch_size-1:
            action_probs=torch.multiply(action_probs, torch.FloatTensor([[1.0, 0.0, 1.0, 1.0]]).cuda())
        if patch_sequence[-1] in torch.arange(0, patch_size):
            action_probs=torch.multiply(action_probs, torch.FloatTensor([[0.0, 1.0, 1.0, 1.0]]).cuda())
        if patch_sequence[-1] in torch.arange(patch_size**2-patch_size, patch_size**2):
            action_probs=torch.multiply(action_probs, torch.FloatTensor([[1.0, 1.0, 0.0, 1.0]]).cuda())

        dist = Categorical(action_probs+1e-6)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action): 
        action_probs = self.actor(state)

        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy

    def entropy_act(self, state, patch_sequence, patch_size):
        action_probs = self.actor(state)

        if patch_sequence[-1]%patch_size==0:
            action_probs=torch.multiply(action_probs, torch.FloatTensor([[1.0, 1.0, 1.0, 0.0]]).cuda())
        if patch_sequence[-1]%patch_size==patch_size-1:
            action_probs=torch.multiply(action_probs, torch.FloatTensor([[1.0, 0.0, 1.0, 1.0]]).cuda())
        if patch_sequence[-1] in torch.arange(0, patch_size):
            action_probs=torch.multiply(action_probs, torch.FloatTensor([[0.0, 1.0, 1.0, 1.0]]).cuda())
        if patch_sequence[-1] in torch.arange(patch_size**2-patch_size, patch_size**2):
            action_probs=torch.multiply(action_probs, torch.FloatTensor([[1.0, 1.0, 0.0, 1.0]]).cuda())

        top_probs = torch.topk(action_probs, 2).values
        if (top_probs[0, 0]) > 0.6:
            action = torch.argmax(action_probs, dim=-1)
        else:
            dist = Categorical(action_probs+1e-6)
            action = dist.sample()

        return action.item()
    
    def greedy_act(self, state, patch_sequence, patch_size):
        action_probs = self.actor(state)

        if patch_sequence[-1]%patch_size==0:
            action_probs=torch.multiply(action_probs, torch.FloatTensor([[1.0, 1.0, 1.0, 0.0]]).cuda())
        if patch_sequence[-1]%patch_size==patch_size-1:
            action_probs=torch.multiply(action_probs, torch.FloatTensor([[1.0, 0.0, 1.0, 1.0]]).cuda())
        if patch_sequence[-1] in torch.arange(0, patch_size):
            action_probs=torch.multiply(action_probs, torch.FloatTensor([[0.0, 1.0, 1.0, 1.0]]).cuda())
        if patch_sequence[-1] in torch.arange(patch_size**2-patch_size, patch_size**2):
            action_probs=torch.multiply(action_probs, torch.FloatTensor([[1.0, 1.0, 0.0, 1.0]]).cuda())

        action = torch.argmax(action_probs, dim=-1)

        return action.item()
    
    def stochastic_act(self, state, patch_sequence, patch_size):
        action_probs = self.actor(state)

        if patch_sequence[-1]%patch_size==0:
            action_probs=torch.multiply(action_probs, torch.FloatTensor([[1.0, 1.0, 1.0, 0.0]]).cuda())
        if patch_sequence[-1]%patch_size==patch_size-1:
            action_probs=torch.multiply(action_probs, torch.FloatTensor([[1.0, 0.0, 1.0, 1.0]]).cuda())
        if patch_sequence[-1] in torch.arange(0, patch_size):
            action_probs=torch.multiply(action_probs, torch.FloatTensor([[0.0, 1.0, 1.0, 1.0]]).cuda())
        if patch_sequence[-1] in torch.arange(patch_size**2-patch_size, patch_size**2):
            action_probs=torch.multiply(action_probs, torch.FloatTensor([[1.0, 1.0, 0.0, 1.0]]).cuda())

        dist = Categorical(action_probs+1e-6)
        action = dist.sample()

        return action.item()

    def random_act(self, state, patch_sequence, patch_size):
        action_probs = torch.FloatTensor([[0.25, 0.25, 0.25, 0.25]]).cuda()
        if patch_sequence[-1]%patch_size==0:
            action_probs=torch.multiply(action_probs, torch.FloatTensor([[1.0, 1.0, 1.0, 0.0]]).cuda())
        if patch_sequence[-1]%patch_size==patch_size-1:
            action_probs=torch.multiply(action_probs, torch.FloatTensor([[1.0, 0.0, 1.0, 1.0]]).cuda())
        if patch_sequence[-1] in torch.arange(0, patch_size):
            action_probs=torch.multiply(action_probs, torch.FloatTensor([[0.0, 1.0, 1.0, 1.0]]).cuda())
        if patch_sequence[-1] in torch.arange(patch_size**2-patch_size, patch_size**2):
            action_probs=torch.multiply(action_probs, torch.FloatTensor([[1.0, 1.0, 0.0, 1.0]]).cuda())
        

        dist = Categorical(action_probs+1e-6)
        action = dist.sample()

        return action.item()
    

class PPO(nn.Module):
    def __init__(self, lr_actor, lr_critic, lr_llm, gamma, K_epochs, eps_clip, lr_gamma):
        super().__init__()
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.llm_module = MaskedActionModeling.load_from_checkpoint(
            cfg.train.llm_checkpoint,
            train_dataset=None,
            val_dataset=None,
        )

        self.llm = self.llm_module.llm

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

    ### required
    def select_action(self, state, patch_sequence, patch_size):
        
        with torch.no_grad():
            action, action_logprob, state_val = self.policy_old.act(state, patch_sequence, patch_size)
            
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.item()
    
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

        current_dist = abs(cur_rows - goal_rows) + abs(cur_cols - goal_cols)

        if current_patch_id == prev_patch_id:
            return -1
        if cur_cols==goal_cols and cur_rows==goal_rows:
            return 2
        elif current_patch_id in patch_sequence:
            return -1
        elif current_dist < best_dist:
            return 1
        else:
            return -1

    def _run_llm(self, seq, device="cuda:0"):
        inputs = seq.get_input_for_model(device=device)
        llm_inputs = {
            "inputs_embeds": inputs["inputs_embeds"],
            "patch_sequence": inputs["patch_sequence"][:, 1:],
            "patch_size": cfg.data.patch_size,
        }
        if inputs["actions"]:
            llm_inputs["actions"] = [inputs["actions"]]
        return self.llm(**llm_inputs)

    def get_intrinsic_reward(self, state_preds, state_gt, phase="train"):
        if phase == "train":
            offset = cfg.train.train_intrinsic_offset
            scale = cfg.train.train_intrinsic_scale
        else:
            offset = cfg.train.val_intrinsic_offset
            scale = cfg.train.val_intrinsic_scale
        return (2 * ((F.mse_loss(state_preds, state_gt).item() - offset) / scale) - 1.0) * 0.25

    def get_reward_weights(self, current_dist):
        if cfg.train.reward_mode == "ex_only":
            return 1.0, 0.0
        if cfg.train.reward_mode == "static_mix":
            return 1.0, 1.0

        span = max(float(cfg.train.d_far - cfg.train.d_near), 1e-6)
        w_in = float(np.clip((current_dist - cfg.train.d_near) / span, 0.0, 1.0))
        w_ex = 1.0 - w_in
        return w_ex, w_in

    def compute_reward_components(
        self,
        prev_patch_id,
        current_patch_id,
        goal_patch_id,
        patch_sequence,
        best_dist,
        state_preds,
        state_gt,
        phase="train",
    ):
        reward_ex = self.get_reward(
            cfg.data.patch_size,
            prev_patch_id,
            current_patch_id,
            goal_patch_id,
            patch_sequence,
            best_dist,
        )
        reward_in = self.get_intrinsic_reward(state_preds, state_gt, phase=phase)
        current_dist = get_dist(current_patch_id, goal_patch_id, cfg.data.patch_size)
        w_ex, w_in = self.get_reward_weights(current_dist)
        reward_total = w_ex * reward_ex + w_in * cfg.train.lambda_in * reward_in
        return {
            "reward_ex": reward_ex,
            "reward_in": reward_in,
            "reward_total": reward_total,
            "w_ex": w_ex,
            "w_in": w_in,
            "current_dist": current_dist,
        }

    def _select_policy_action(self, state, patch_sequence, patch_size, policy_mode="greedy"):
        if policy_mode == "stochastic":
            return self.select_stochastic_action(state, patch_sequence, patch_size)
        if policy_mode == "random":
            return self.select_random_action(state, patch_sequence, patch_size)
        if policy_mode == "entropy":
            return self.select_entropy_action(state, patch_sequence, patch_size)
        return self.select_greedy_action(state, patch_sequence, patch_size)

    def rollout_episode(self, seq, goal_patch, max_steps, patch_size, policy_mode="greedy", phase="val", device="cuda:0"):
        best_dist = get_dist(goal_patch, seq.patch_sequence[-1], patch_size)
        optimal_steps = best_dist
        visited_matrix = np.zeros((patch_size, patch_size), dtype=float)
        end_matrix = np.zeros((patch_size, patch_size), dtype=float)
        reward_sum_matrix = np.zeros((patch_size, patch_size), dtype=float)
        reward_count_matrix = np.zeros((patch_size, patch_size), dtype=float)
        step_records = []

        start_patch = seq.patch_sequence[-1]
        start_row, start_col = divmod(start_patch, patch_size)
        visited_matrix[start_row, start_col] += 1

        for step_index in range(1, max_steps + 1):
            state, state_preds, state_gt = self._run_llm(seq, device=device)
            action = self._select_policy_action(state, seq.patch_sequence, patch_size, policy_mode=policy_mode)
            seq.update_sequence_with_action(action_list[action])

            current_patch_id = seq.patch_sequence[-1]
            prev_patch_id = seq.patch_sequence[-2]
            reward_terms = self.compute_reward_components(
                prev_patch_id,
                current_patch_id,
                goal_patch,
                seq.patch_sequence[1:-1],
                best_dist,
                state_preds,
                state_gt,
                phase=phase,
            )
            best_dist = min(best_dist, reward_terms["current_dist"])

            row, col = divmod(current_patch_id, patch_size)
            visited_matrix[row, col] += 1
            reward_sum_matrix[row, col] += reward_terms["reward_in"]
            reward_count_matrix[row, col] += 1

            step_records.append(
                {
                    "step_index": step_index,
                    "action": action_list[action],
                    "current_patch": current_patch_id,
                    "reward_ex": reward_terms["reward_ex"],
                    "reward_in": reward_terms["reward_in"],
                    "reward_total": reward_terms["reward_total"],
                    "w_ex": reward_terms["w_ex"],
                    "w_in": reward_terms["w_in"],
                }
            )

            if current_patch_id == goal_patch:
                break

        final_patch = seq.patch_sequence[-1]
        final_row, final_col = divmod(final_patch, patch_size)
        end_matrix[final_row, final_col] += 1
        reward_matrix = np.divide(
            reward_sum_matrix,
            reward_count_matrix,
            out=np.zeros_like(reward_sum_matrix),
            where=reward_count_matrix > 0,
        )
        visited_patch_sequence = list(seq.patch_sequence[1:])
        total_reward = float(sum(record["reward_total"] for record in step_records))

        return {
            "success": final_patch == goal_patch,
            "start_patch": start_patch,
            "goal_patch": goal_patch,
            "path_sequence": visited_patch_sequence,
            "action_sequence": list(seq.action_sequence),
            "step_records": step_records,
            "visited_matrix": visited_matrix,
            "end_matrix": end_matrix,
            "reward_matrix": reward_matrix,
            "total_reward": total_reward,
            "optimal_steps": optimal_steps,
            "num_steps": len(seq.action_sequence),
            "deviation_from_opt": len(seq.action_sequence) - optimal_steps,
            "final_distance": get_dist(final_patch, goal_patch, patch_size),
            "unique_visited_count": len(set(visited_patch_sequence)),
        }


    def update(self, flag, patch_sequence, patch_size, device="cuda:0"):

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
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)
        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        self.schedular.step()
            
        # Copy new weights into old policy
        if True: #flag:
            self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

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

                        inputs = seq.get_input_for_model(device='cuda:0')

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
            if num_success > 0:
                print(f"Val Success : {num_success/cfg.sample_number} \t Val Avg Steps Success : {avg_steps_to_success/num_success} \t Val Dev : {avg_dev_steps/num_success}")
            #wandb.log({"val success": num_success})
        return num_success, res


    def validate(self, config, valid_path, tokenizer=None, n_config_per_img=5, flag='none'):
        dataset_dict = np.load(valid_path, allow_pickle=True)
        total_imgs = len(dataset_dict[()].keys())
        num_success=0
        avg_dev_steps=0 
        avg_steps_to_success=0
        avg_reward=0
        res = [0]*n_config_per_img
        num_pass = 0


        for i in range(total_imgs):
            
            for j in range(n_config_per_img):
                
                GOAL_PATCH = config[f"img_{i}"][j][0]
                CURRENT_PATCH = config[f"img_{i}"][j][1]
                seq = Sequence(dataset_dict[()][f"img_{i}"], tokenizer, num_patches=cfg.data.patch_size)
                


                seq.init_with_goal_image(GOAL_PATCH)
                seq.update_sequence_with_satellite_image_token(CURRENT_PATCH)
                best_dist=get_dist(GOAL_PATCH, CURRENT_PATCH)
                opt_steps=get_dist(GOAL_PATCH, CURRENT_PATCH)

                for t in (range(1, cfg.train.hparams.max_ep_len+1)):

                    inputs = seq.get_input_for_model(device='cuda:0')

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
                    reward_in = (2*((F.mse_loss(state_preds, state_gt).item() -0.005) / 0.08) - 1.0)*0.25
                    #reward_in = -(2*(F.cosine_similarity(state_gt, state_preds).item()) -1.0)
                    reward_ex = self.get_reward(cfg.data.patch_size, prev_patch_id, current_patch_id, goal_patch_id, seq.patch_sequence[1:-1], best_dist)
                    if cfg.reward == 'ex':
                        reward = reward_ex
                    elif cfg.reward == 'in':
                        reward = reward_in + reward_ex
                    if reward>=0.5:
                        best_dist=get_dist(current_patch_id, goal_patch_id)
                    avg_reward+=reward
                    done = (current_patch_id==GOAL_PATCH)
                    if done:
                        avg_steps_to_success+=len(seq.action_sequence)
                        avg_dev_steps+=(len(seq.action_sequence)-opt_steps)
                        num_success+=1
                        res[j]+=1
                        break

        if num_success > 0:
            print(f"Val Success : {num_success} \t Val Avg Reward : {avg_reward/(total_imgs*5)} \t Val Avg Steps Success : {avg_steps_to_success/num_success} \t Val Dev : {avg_dev_steps/num_success}")
        

        return num_success, res


    def validate_unseen(self, config, valid_path, tokenizer=None, n_config_per_img=5, flag='none'):
        dataset_dict = np.load(valid_path, allow_pickle=True)
        total_imgs = len(dataset_dict[()].keys())
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

                    inputs = seq.get_input_for_model(device='cuda:0')

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
            
        print(cfg.sample_number-num_pass, num_success/(cfg.sample_number-num_pass))
        print(step_to_goal, step_to_goal/(cfg.sample_number-num_pass))
        
        return num_success, res

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

                    inputs = seq.get_input_for_model(device='cuda:0')

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

        print(f"Val Success : {num_success} \t Val Avg Steps Success : {avg_steps_to_success/num_success} \t Val Dev : {avg_dev_steps/num_success}")
        return num_success
    

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

                    inputs = seq.get_input_for_model(device='cuda:0')

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
        print(f"Val Success : {num_success} \t Val Avg Steps Success : {avg_steps_to_success/num_success} \t Val Dev : {avg_dev_steps/num_success}")
        return num_success

    def validate_ground_unseen(self, config, sat_paths, ground_path, tokenizer=None, n_config_per_img=5):

        ground_dict = np.load(ground_path, allow_pickle=True)
        sat_dict = np.load(sat_paths, allow_pickle=True)
        total_imgs = len(sat_dict[()].keys())
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

                    inputs = seq.get_input_for_model(device='cuda:0')

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

        print(cfg.sample_number*n_config_per_img-num_pass, num_success/(cfg.sample_number*n_config_per_img-num_pass))
        print(step_to_goal, step_to_goal/(cfg.sample_number*n_config_per_img-num_pass))
        return num_success


def _ppo_build_validation_summary(
    self,
    num_success,
    total_reward,
    avg_steps_to_success,
    avg_dev_steps,
    total_final_distance,
    total_unique_visited,
    res,
    total_episodes,
    num_pass,
    visited_matrix,
    end_matrix,
):
    effective_episodes = max(total_episodes - num_pass, 1)
    return {
        "num_success": num_success,
        "success_ratio": num_success / effective_episodes,
        "avg_reward": total_reward / effective_episodes,
        "avg_steps_success": (avg_steps_to_success / num_success) if num_success > 0 else None,
        "avg_deviation": (avg_dev_steps / num_success) if num_success > 0 else None,
        "avg_final_distance": total_final_distance / effective_episodes,
        "avg_unique_visited": total_unique_visited / effective_episodes,
        "results_per_config": res,
        "num_episodes": effective_episodes,
        "num_pass": num_pass,
        "visited_matrix": visited_matrix,
        "end_matrix": end_matrix,
    }


def _ppo_validate_varying_budget(self, config, valid_path, tokenizer=None, n_config_per_img=5, flag='none'):
    dataset_dict = np.load(valid_path, allow_pickle=True)
    total_imgs = len(dataset_dict[()].keys())
    policy_mode = "entropy" if flag == "entropy" else "greedy"
    res = [0] * n_config_per_img
    num_success = 0

    for budget in range(cfg.data.min_budget, cfg.data.max_budget, cfg.data.budget_step):
        num_success = 0
        avg_dev_steps = 0
        avg_steps_to_success = 0
        budget_res = [0] * n_config_per_img
        for i in range(total_imgs):
            for j in range(n_config_per_img):
                seq = Sequence(dataset_dict[()][f"img_{i}"], tokenizer, num_patches=cfg.data.patch_size)
                goal_patch = config[f"img_{i}"][j][0]
                current_patch = config[f"img_{i}"][j][1]
                seq.init_with_goal_image(goal_patch)
                seq.update_sequence_with_satellite_image_token(current_patch)
                rollout = self.rollout_episode(
                    seq,
                    goal_patch,
                    budget,
                    cfg.data.patch_size,
                    policy_mode=policy_mode,
                    phase="val",
                )
                if rollout["success"]:
                    avg_steps_to_success += rollout["num_steps"]
                    avg_dev_steps += rollout["deviation_from_opt"]
                    num_success += 1
                    budget_res[j] += 1
        if num_success > 0:
            print(
                f"Val Success : {num_success/cfg.sample_number} \t "
                f"Val Avg Steps Success : {avg_steps_to_success/num_success} \t "
                f"Val Dev : {avg_dev_steps/num_success}"
            )
        res = budget_res
    return num_success, res


def _ppo_validate(self, config, valid_path, tokenizer=None, n_config_per_img=5, flag='none', return_details=False, policy_mode="greedy"):
    dataset_dict = np.load(valid_path, allow_pickle=True)
    total_imgs = len(dataset_dict[()].keys())
    effective_policy_mode = "entropy" if flag == "entropy" else policy_mode
    num_success = 0
    avg_dev_steps = 0
    avg_steps_to_success = 0
    total_reward = 0
    total_final_distance = 0
    total_unique_visited = 0
    res = [0] * n_config_per_img
    visited_matrix = np.zeros((cfg.data.patch_size, cfg.data.patch_size), dtype=float)
    end_matrix = np.zeros((cfg.data.patch_size, cfg.data.patch_size), dtype=float)
    total_episodes = 0

    for i in range(total_imgs):
        for j in range(n_config_per_img):
            total_episodes += 1
            goal_patch = config[f"img_{i}"][j][0]
            current_patch = config[f"img_{i}"][j][1]
            seq = Sequence(dataset_dict[()][f"img_{i}"], tokenizer, num_patches=cfg.data.patch_size)
            seq.init_with_goal_image(goal_patch)
            seq.update_sequence_with_satellite_image_token(current_patch)
            rollout = self.rollout_episode(
                seq,
                goal_patch,
                cfg.train.hparams.max_ep_len,
                cfg.data.patch_size,
                policy_mode=effective_policy_mode,
                phase="val",
            )
            total_reward += rollout["total_reward"]
            total_final_distance += rollout["final_distance"]
            total_unique_visited += rollout["unique_visited_count"]
            visited_matrix += rollout["visited_matrix"]
            end_matrix += rollout["end_matrix"]
            if rollout["success"]:
                avg_steps_to_success += rollout["num_steps"]
                avg_dev_steps += rollout["deviation_from_opt"]
                num_success += 1
                res[j] += 1

    summary = self._build_validation_summary(
        num_success,
        total_reward,
        avg_steps_to_success,
        avg_dev_steps,
        total_final_distance,
        total_unique_visited,
        res,
        total_episodes,
        0,
        visited_matrix,
        end_matrix,
    )
    if summary["avg_steps_success"] is not None:
        print(
            f"Val Success : {num_success} \t "
            f"Val Avg Reward : {summary['avg_reward']} \t "
            f"Val Avg Steps Success : {summary['avg_steps_success']} \t "
            f"Val Dev : {summary['avg_deviation']}"
        )
    if return_details:
        return summary
    return num_success, res


def _ppo_validate_unseen(self, config, valid_path, tokenizer=None, n_config_per_img=5, flag='none', return_details=False, policy_mode="greedy"):
    dataset_dict = np.load(valid_path, allow_pickle=True)
    total_imgs = len(dataset_dict[()].keys())
    effective_policy_mode = "entropy" if flag == "entropy" else policy_mode
    num_success = 0
    avg_dev_steps = 0
    avg_steps_to_success = 0
    total_reward = 0
    total_final_distance = 0
    total_unique_visited = 0
    res = [0] * n_config_per_img
    visited_matrix = np.zeros((cfg.data.patch_size, cfg.data.patch_size), dtype=float)
    end_matrix = np.zeros((cfg.data.patch_size, cfg.data.patch_size), dtype=float)
    total_episodes = 0
    num_pass = 0

    for i in range(total_imgs):
        for j in range(n_config_per_img):
            total_episodes += 1
            goal_patch = config[f"img_{i}"][j][0]
            current_patch = config[f"img_{i}"][j][1]
            if goal_patch == 999:
                num_pass += 1
                continue
            seq = Sequence(dataset_dict[()][f"img_{i}"], tokenizer, num_patches=cfg.data.patch_size)
            seq.init_with_goal_image(goal_patch)
            seq.update_sequence_with_satellite_image_token(current_patch)
            rollout = self.rollout_episode(
                seq,
                goal_patch,
                cfg.train.hparams.max_ep_len,
                cfg.data.patch_size,
                policy_mode=effective_policy_mode,
                phase="val",
            )
            total_reward += rollout["total_reward"]
            total_final_distance += rollout["final_distance"]
            total_unique_visited += rollout["unique_visited_count"]
            visited_matrix += rollout["visited_matrix"]
            end_matrix += rollout["end_matrix"]
            if rollout["success"]:
                avg_steps_to_success += rollout["num_steps"]
                avg_dev_steps += rollout["deviation_from_opt"]
                num_success += 1
                res[j] += 1

    summary = self._build_validation_summary(
        num_success,
        total_reward,
        avg_steps_to_success,
        avg_dev_steps,
        total_final_distance,
        total_unique_visited,
        res,
        total_episodes,
        num_pass,
        visited_matrix,
        end_matrix,
    )
    print(summary["num_episodes"], summary["success_ratio"])
    print(total_final_distance, summary["avg_final_distance"])
    if return_details:
        return summary
    return num_success, res


def _ppo_validate_ground(self, config, sat_paths, ground_path, tokenizer=None, n_config_per_img=5, return_details=False, policy_mode="greedy"):
    ground_dict = np.load(ground_path, allow_pickle=True)
    sat_dict = np.load(sat_paths, allow_pickle=True)
    total_imgs = len(sat_dict[()].keys())
    num_success = 0
    avg_dev_steps = 0
    avg_steps_to_success = 0
    total_reward = 0
    total_final_distance = 0
    total_unique_visited = 0
    res = [0] * n_config_per_img
    visited_matrix = np.zeros((cfg.data.patch_size, cfg.data.patch_size), dtype=float)
    end_matrix = np.zeros((cfg.data.patch_size, cfg.data.patch_size), dtype=float)
    total_episodes = 0

    for i in range(total_imgs):
        for j in range(n_config_per_img):
            total_episodes += 1
            goal_ground = ground_dict[()][f"img_{i}"]
            goal_patch = config[f"img_{i}"][j][0]
            current_patch = config[f"img_{i}"][j][1]
            sat_embeds = sat_dict[()][f"img_{i}"].reshape(25, -1)
            seq = Sequence(sat_embeds, tokenizer, num_patches=cfg.data.patch_size)
            seq.init_with_goal_embed(goal_ground, goal_patch)
            seq.update_sequence_with_satellite_image_token(current_patch)
            rollout = self.rollout_episode(
                seq,
                goal_patch,
                cfg.train.hparams.max_ep_len,
                cfg.data.patch_size,
                policy_mode=policy_mode,
                phase="val",
            )
            total_reward += rollout["total_reward"]
            total_final_distance += rollout["final_distance"]
            total_unique_visited += rollout["unique_visited_count"]
            visited_matrix += rollout["visited_matrix"]
            end_matrix += rollout["end_matrix"]
            if rollout["success"]:
                avg_steps_to_success += rollout["num_steps"]
                avg_dev_steps += rollout["deviation_from_opt"]
                num_success += 1
                res[j] += 1

    summary = self._build_validation_summary(
        num_success,
        total_reward,
        avg_steps_to_success,
        avg_dev_steps,
        total_final_distance,
        total_unique_visited,
        res,
        total_episodes,
        0,
        visited_matrix,
        end_matrix,
    )
    if summary["avg_steps_success"] is not None:
        print(
            f"Val Success : {num_success} \t "
            f"Val Avg Steps Success : {summary['avg_steps_success']} \t "
            f"Val Dev : {summary['avg_deviation']}"
        )
    if return_details:
        return summary
    return num_success


def _ppo_validate_text(self, config, sat_paths, ground_path, tokenizer=None, n_config_per_img=5, return_details=False, policy_mode="greedy"):
    ground_dict = np.load(ground_path, allow_pickle=True)
    sat_dict = np.load(sat_paths, allow_pickle=True)
    total_imgs = len(sat_dict[()].keys())
    num_success = 0
    avg_dev_steps = 0
    avg_steps_to_success = 0
    total_reward = 0
    total_final_distance = 0
    total_unique_visited = 0
    res = [0] * n_config_per_img
    visited_matrix = np.zeros((cfg.data.patch_size, cfg.data.patch_size), dtype=float)
    end_matrix = np.zeros((cfg.data.patch_size, cfg.data.patch_size), dtype=float)
    total_episodes = 0

    for i in range(total_imgs):
        for j in range(n_config_per_img):
            total_episodes += 1
            goal_text = ground_dict[i]
            goal_patch = config[f"img_{i}"][j][0]
            current_patch = config[f"img_{i}"][j][1]
            sat_embeds = sat_dict[()][f"img_{i}"].reshape(25, -1)
            seq = Sequence(sat_embeds, tokenizer, num_patches=cfg.data.patch_size)
            seq.init_with_goal_embed(goal_text, goal_patch)
            seq.update_sequence_with_satellite_image_token(current_patch)
            rollout = self.rollout_episode(
                seq,
                goal_patch,
                cfg.train.hparams.max_ep_len,
                cfg.data.patch_size,
                policy_mode=policy_mode,
                phase="val",
            )
            total_reward += rollout["total_reward"]
            total_final_distance += rollout["final_distance"]
            total_unique_visited += rollout["unique_visited_count"]
            visited_matrix += rollout["visited_matrix"]
            end_matrix += rollout["end_matrix"]
            if rollout["success"]:
                avg_steps_to_success += rollout["num_steps"]
                avg_dev_steps += rollout["deviation_from_opt"]
                num_success += 1
                res[j] += 1

    summary = self._build_validation_summary(
        num_success,
        total_reward,
        avg_steps_to_success,
        avg_dev_steps,
        total_final_distance,
        total_unique_visited,
        res,
        total_episodes,
        0,
        visited_matrix,
        end_matrix,
    )
    if summary["avg_steps_success"] is not None:
        print(
            f"Val Success : {num_success} \t "
            f"Val Avg Steps Success : {summary['avg_steps_success']} \t "
            f"Val Dev : {summary['avg_deviation']}"
        )
    if return_details:
        return summary
    return num_success


def _ppo_validate_ground_unseen(self, config, sat_paths, ground_path, tokenizer=None, n_config_per_img=5, return_details=False, policy_mode="greedy"):
    ground_dict = np.load(ground_path, allow_pickle=True)
    sat_dict = np.load(sat_paths, allow_pickle=True)
    total_imgs = len(sat_dict[()].keys())
    num_success = 0
    avg_dev_steps = 0
    avg_steps_to_success = 0
    total_reward = 0
    total_final_distance = 0
    total_unique_visited = 0
    res = [0] * n_config_per_img
    visited_matrix = np.zeros((cfg.data.patch_size, cfg.data.patch_size), dtype=float)
    end_matrix = np.zeros((cfg.data.patch_size, cfg.data.patch_size), dtype=float)
    total_episodes = 0
    num_pass = 0

    for i in range(total_imgs):
        for j in range(n_config_per_img):
            total_episodes += 1
            goal_patch = config[f"img_{i}"][j][0]
            current_patch = config[f"img_{i}"][j][1]
            if goal_patch == 999:
                num_pass += 1
                continue
            goal_ground = ground_dict[()][f"img_{i}"]
            sat_embeds = sat_dict[()][f"img_{i}"].reshape(25, -1)
            seq = Sequence(sat_embeds, tokenizer, num_patches=cfg.data.patch_size)
            seq.init_with_goal_embed(goal_ground, goal_patch)
            seq.update_sequence_with_satellite_image_token(current_patch)
            rollout = self.rollout_episode(
                seq,
                goal_patch,
                cfg.train.hparams.max_ep_len,
                cfg.data.patch_size,
                policy_mode=policy_mode,
                phase="val",
            )
            total_reward += rollout["total_reward"]
            total_final_distance += rollout["final_distance"]
            total_unique_visited += rollout["unique_visited_count"]
            visited_matrix += rollout["visited_matrix"]
            end_matrix += rollout["end_matrix"]
            if rollout["success"]:
                avg_steps_to_success += rollout["num_steps"]
                avg_dev_steps += rollout["deviation_from_opt"]
                num_success += 1
                res[j] += 1

    summary = self._build_validation_summary(
        num_success,
        total_reward,
        avg_steps_to_success,
        avg_dev_steps,
        total_final_distance,
        total_unique_visited,
        res,
        total_episodes,
        num_pass,
        visited_matrix,
        end_matrix,
    )
    print(summary["num_episodes"], summary["success_ratio"])
    print(total_final_distance, summary["avg_final_distance"])
    if return_details:
        return summary
    return num_success


def _ppo_validate_text_unseen(self, config, sat_paths, text_path, tokenizer=None, n_config_per_img=5, return_details=False, policy_mode="greedy"):
    ground_dict = np.load(text_path, allow_pickle=True)
    sat_dict = np.load(sat_paths, allow_pickle=True)
    total_imgs = len(sat_dict[()].keys())
    num_success = 0
    avg_dev_steps = 0
    avg_steps_to_success = 0
    total_reward = 0
    total_final_distance = 0
    total_unique_visited = 0
    res = [0] * n_config_per_img
    visited_matrix = np.zeros((cfg.data.patch_size, cfg.data.patch_size), dtype=float)
    end_matrix = np.zeros((cfg.data.patch_size, cfg.data.patch_size), dtype=float)
    total_episodes = 0
    num_pass = 0

    for i in range(total_imgs):
        for j in range(n_config_per_img):
            total_episodes += 1
            goal_patch = config[f"img_{i}"][j][0]
            current_patch = config[f"img_{i}"][j][1]
            if goal_patch == 999:
                num_pass += 1
                continue
            goal_text = ground_dict[i]
            sat_embeds = sat_dict[()][f"img_{i}"].reshape(25, -1)
            seq = Sequence(sat_embeds, tokenizer, num_patches=cfg.data.patch_size)
            seq.init_with_goal_embed(goal_text, goal_patch)
            seq.update_sequence_with_satellite_image_token(current_patch)
            rollout = self.rollout_episode(
                seq,
                goal_patch,
                cfg.train.hparams.max_ep_len,
                cfg.data.patch_size,
                policy_mode=policy_mode,
                phase="val",
            )
            total_reward += rollout["total_reward"]
            total_final_distance += rollout["final_distance"]
            total_unique_visited += rollout["unique_visited_count"]
            visited_matrix += rollout["visited_matrix"]
            end_matrix += rollout["end_matrix"]
            if rollout["success"]:
                avg_steps_to_success += rollout["num_steps"]
                avg_dev_steps += rollout["deviation_from_opt"]
                num_success += 1
                res[j] += 1

    summary = self._build_validation_summary(
        num_success,
        total_reward,
        avg_steps_to_success,
        avg_dev_steps,
        total_final_distance,
        total_unique_visited,
        res,
        total_episodes,
        num_pass,
        visited_matrix,
        end_matrix,
    )
    print(summary["num_episodes"], summary["success_ratio"])
    print(total_final_distance, summary["avg_final_distance"])
    if return_details:
        return summary
    return num_success


PPO._build_validation_summary = _ppo_build_validation_summary
PPO.validate_varying_budget = _ppo_validate_varying_budget
PPO.validate = _ppo_validate
PPO.validate_unseen = _ppo_validate_unseen
PPO.validate_ground = _ppo_validate_ground
PPO.validate_text = _ppo_validate_text
PPO.validate_ground_unseen = _ppo_validate_ground_unseen
PPO.validate_text_unseen = _ppo_validate_text_unseen
