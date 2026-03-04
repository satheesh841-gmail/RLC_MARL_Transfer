import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from smac.env import StarCraft2Env
from collections import deque
import matplotlib.pyplot as plt
from torch.distributions import Categorical
import argparse
import os
import json
from pathlib import Path
import glob
from policy_assignment import PolicyAssignment

# GPU Device Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class TransformerMapper(nn.Module):
    
    def __init__(self, target_dim, source_dim, model_dim=128, n_heads=4, n_layers=2,
                 token_size=16, dropout=0.1):
        super(TransformerMapper, self).__init__()
        self.target_dim = target_dim
        self.source_dim = source_dim
        self.model_dim = model_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.token_size = token_size
        self.dropout = dropout

        
        self.num_tokens = (target_dim + token_size - 1) // token_size
        pad_dim = self.num_tokens * token_size - target_dim
        self.pad_dim = pad_dim

      
        self.input_proj = nn.Linear(token_size, model_dim)

      
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, model_dim))
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

       
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            batch_first=True,
            activation='relu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        
        hidden = 2 * max(source_dim, model_dim)
        self.output_head = nn.Sequential(
            nn.Linear(self.num_tokens * model_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, source_dim),
        )

        
        nn.init.xavier_uniform_(self.output_head[0].weight)
        nn.init.zeros_(self.output_head[0].bias)
        nn.init.xavier_uniform_(self.output_head[2].weight)
        nn.init.zeros_(self.output_head[2].bias)

    def forward(self, target_obs):
        # target_obs: [batch, target_dim]
        bsz = target_obs.size(0)
        x = target_obs
        if self.pad_dim > 0:
            pad = torch.zeros(bsz, self.pad_dim, device=target_obs.device, dtype=target_obs.dtype)
            x = torch.cat([x, pad], dim=-1)
        
        x = x.view(bsz, self.num_tokens, self.token_size)
        
        x = self.input_proj(x)  # [batch, num_tokens, model_dim]
        
        x = x + self.pos_embed
        
        x = self.encoder(x)  # [batch, num_tokens, model_dim]
        
        x = x.reshape(bsz, self.num_tokens * self.model_dim)
        # map to source obs dim
        return self.output_head(x)


class TransformerObservationAdapter(nn.Module):
    """Transformer-based observation adapter for source policies"""
    def __init__(self, 
                 target_obs_dim, 
                 source_obs_dim, 
                 transformer_adapter_path=None,
                 trainable=True, 
                 model_dim=128, n_heads=4, n_layers=2, token_size=16, dropout=0.1,
                 init_transformer_if_missing=True):
        super(TransformerObservationAdapter, self).__init__()
        self.target_obs_dim = target_obs_dim
        self.source_obs_dim = source_obs_dim
        self.transformer_adapter_path = transformer_adapter_path
        self.trainable = trainable
        self.adapter_type = None

        if target_obs_dim != source_obs_dim:
            if transformer_adapter_path and os.path.exists(transformer_adapter_path):
                print(f"Loading Transformer adapter from: {transformer_adapter_path}")
                try:
                    checkpoint = torch.load(transformer_adapter_path, map_location=device)
                    self.mapper = TransformerMapper(
                        target_dim=target_obs_dim,
                        source_dim=source_obs_dim,
                        model_dim=model_dim, n_heads=n_heads, n_layers=n_layers,
                        token_size=token_size, dropout=dropout,
                    )
                    key = 'transformer'
                    if key in checkpoint:
                        self.mapper.load_state_dict(checkpoint[key])
                    else:
                        self.mapper.load_state_dict(checkpoint)
                    # TRAINING vs FREEZE
                    for p in self.mapper.parameters():
                        p.requires_grad = self.trainable
                    self.mapper.train(mode=self.trainable)
                    self.needs_adaptation = True
                    self.adapter_type = "transformer"
                    print(f"Transformer adapter loaded: {target_obs_dim} → {source_obs_dim} | trainable={self.trainable}")
                except Exception as e:
                    print(f"Error loading Transformer adapter: {e}")
                    if init_transformer_if_missing and self.trainable:
                        print("Initializing new Transformer adapter (trainable)")
                        self._init_new_transformer(target_obs_dim, source_obs_dim, model_dim, n_heads, n_layers, token_size, dropout)
                    else:
                        print("Falling back to linear adapter...")
                        self._create_linear_adapter()
            else:
                # No checkpoint; optionally initialize a new transformer if training is requested
                if init_transformer_if_missing and self.trainable:
                    print(f"Initializing new Transformer adapter: {target_obs_dim} → {source_obs_dim} (trainable)")
                    self._init_new_transformer(target_obs_dim, source_obs_dim, model_dim, n_heads, n_layers, token_size, dropout)
                else:
                    print(f"Using linear adapter: {target_obs_dim} → {source_obs_dim}")
                    self._create_linear_adapter()
        else:
            # No adaptation needed
            self.adapter = nn.Identity()
            self.needs_adaptation = False
            self.adapter_type = "identity"
            print(f"No adaptation needed: {target_obs_dim} = {source_obs_dim}")

    def _init_new_transformer(self, target_obs_dim, source_obs_dim, model_dim, n_heads, n_layers, token_size, dropout):
        self.mapper = TransformerMapper(
            target_dim=target_obs_dim,
            source_dim=source_obs_dim,
            model_dim=model_dim, n_heads=n_heads, n_layers=n_layers,
            token_size=token_size, dropout=dropout,
        )
        for p in self.mapper.parameters():
            p.requires_grad = True
        self.mapper.train(True)
        self.needs_adaptation = True
        self.adapter_type = "transformer"

    def _create_linear_adapter(self):
        hidden_size = 2 * max(self.target_obs_dim, self.source_obs_dim)
        self.adapter = nn.Sequential(
            nn.Linear(self.target_obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.source_obs_dim)
        )
        nn.init.xavier_uniform_(self.adapter[0].weight)
        nn.init.xavier_uniform_(self.adapter[2].weight)
        nn.init.zeros_(self.adapter[0].bias)
        nn.init.zeros_(self.adapter[2].bias)
        self.needs_adaptation = True
        self.adapter_type = "linear"

    def forward(self, target_obs):
        if not self.needs_adaptation:
            return target_obs
        if self.adapter_type == "transformer":
            # Allow gradients if trainable; otherwise use no_grad for efficiency
            if self.trainable:
                return self.mapper(target_obs)              # grads flow
            else:
                with torch.no_grad():
                    return self.mapper(target_obs)          # frozen
        else:
            return self.adapter(target_obs)

    def get_adapter_parameters(self):
        """Get trainable parameters (linear or transformer when trainable)"""
        if self.adapter_type == "linear":
            return list(self.adapter.parameters())
        elif self.adapter_type == "transformer" and self.trainable:
            return list(self.mapper.parameters())
        else:
            return []  # frozen transformer or identity


class AttentionModule(nn.Module):
    """Soft attention module for weighting transferred features"""
    def __init__(self, input_dim, num_policies):
        super(AttentionModule, self).__init__()
        self.num_policies = num_policies
        self.projection = nn.Linear(input_dim * num_policies, 128)
        self.attention_weights = nn.Linear(128, num_policies)

    def forward(self, policy_outputs):
        # Concatenate all policy outputs
        concat_outputs = torch.cat(policy_outputs, dim=-1)
        projected = torch.relu(self.projection(concat_outputs))
        weights = torch.softmax(self.attention_weights(projected), dim=-1)
        return weights

class MALTActor(nn.Module):
    """MALT-enhanced Actor with lateral connections and Transformer-based adapters"""
    def __init__(self, obs_dim, act_dim, hidden_size=256, gru_layers=1,
                 source_policies=None, assigned_policy_indices=None, num_assigned_policies=3,
                 source_obs_dims=None, source_act_dims=None, transformer_adapter_path=None,transformer_trainable=True,
                 transformer_model_dim=128, transformer_n_heads=4, transformer_n_layers=2,
                 transformer_token_size=16, transformer_dropout=0.1
):
        super(MALTActor, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.gru_layers = gru_layers
        self.num_assigned_policies = num_assigned_policies
        self.transformer_adapter_path = transformer_adapter_path

        self.source_policies = source_policies if source_policies else []
        self.assigned_policy_indices = assigned_policy_indices if assigned_policy_indices else []
        self.source_obs_dims = source_obs_dims if source_obs_dims else []
        self.source_act_dims = source_act_dims if source_act_dims else []

        # Target agent's own networks
        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=gru_layers,
            batch_first=True
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim)
        )

        # Transformer-based adapters for source policies
        if source_policies and len(assigned_policy_indices) > 0:
            self.obs_adapters = nn.ModuleList()
            print(f"Creating Transformer adapters for {len(assigned_policy_indices)} assigned policies...")
            
            for idx in assigned_policy_indices:
                if idx < len(self.source_obs_dims):
                    source_obs_dim = self.source_obs_dims[idx]
                    adapter = TransformerObservationAdapter(
                        target_obs_dim=obs_dim,
                        source_obs_dim=source_obs_dim,
                        transformer_adapter_path=transformer_adapter_path,
                        trainable=transformer_trainable,
                        model_dim=transformer_model_dim, n_heads=transformer_n_heads, n_layers=transformer_n_layers,
                        token_size=transformer_token_size, dropout=transformer_dropout,
                        init_transformer_if_missing=True
                    )
                    self.obs_adapters.append(adapter)

                    print(f" Policy {idx}: {adapter.adapter_type} adapter ({obs_dim} -> {source_obs_dim})")
            for policy in self.source_policies:
                if policy is not None:
                    for param in policy.parameters():
                        param.requires_grad = False
            self.attention = AttentionModule(hidden_size, len(assigned_policy_indices))
        else:
            self.obs_adapters = nn.ModuleList()
            self.attention = None

        self.hidden = None

    def reset_hidden_states(self, batch_size=1):
        self.hidden = torch.zeros(self.gru_layers, batch_size, self.hidden_size).to(device)

    def forward(self, obs, return_features=False, return_attention=False):
        batch_size = obs.size(0)
        if self.hidden is None or self.hidden.size(1) != batch_size:
            self.reset_hidden_states(batch_size)
        target_features = self.feature_net(obs)
        gru_out, self.hidden = self.gru(target_features.unsqueeze(1), self.hidden)
        gru_out = gru_out.squeeze(1)

        
        if len(self.source_policies) > 0 and len(self.obs_adapters) > 0:
            policy_features = []
            for adapter, policy_idx in zip(self.obs_adapters, self.assigned_policy_indices):
                if policy_idx < len(self.source_policies):
                    source_policy = self.source_policies[policy_idx]
                    if source_policy is not None:
                        adapted_obs = adapter(obs)  # may be trainable transformer
              
                        if getattr(adapter, 'adapter_type', None) == 'transformer' and getattr(adapter, 'trainable', False):
                            if hasattr(source_policy, 'feature_net'):
                                source_features = source_policy.feature_net(adapted_obs)  # params frozen; grads flow to inputs
                            else:
                                _, source_features = source_policy(adapted_obs, return_features=True)
                        else:
                            with torch.no_grad():
                                if hasattr(source_policy, 'feature_net'):
                                    source_features = source_policy.feature_net(adapted_obs)
                                else:
                                    _, source_features = source_policy(adapted_obs, return_features=True)

                        policy_features.append(source_features)

            if policy_features:
                attention_weights = self.attention(policy_features)
                weighted_features = sum(
                    w.unsqueeze(-1) * f
                    for w, f in zip(attention_weights.unbind(1), policy_features)
                )
                combined_features = gru_out + 0.3 * weighted_features
            else:
                combined_features = gru_out
                attention_weights = None
        else:
            combined_features = gru_out
            attention_weights = None

        logits = self.policy_head(combined_features)

        if return_features and return_attention:
            return logits, combined_features, attention_weights
        elif return_features:
            return logits, combined_features
        elif return_attention:
            return logits, attention_weights
        else:
            return logits

    def get_adapter_parameters(self):
        adapter_params = []
        for adapter in self.obs_adapters:
            adapter_params.extend(adapter.get_adapter_parameters())
        return adapter_params

class MALTCritic(nn.Module):
    """Centralized Critic for multi-agent value estimation"""
    def __init__(self, state_dim, hidden_size=256, gru_layers=1):
        super(MALTCritic, self).__init__()
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        self.gru_layers = gru_layers
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=gru_layers,
            batch_first=True
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.hidden = None

    def reset_hidden_states(self, batch_size=1):
        self.hidden = torch.zeros(self.gru_layers, batch_size, self.hidden_size).to(device)

    def forward(self, state):
        batch_size = state.size(0)
        if self.hidden is None or self.hidden.size(1) != batch_size:
            self.reset_hidden_states(batch_size)
        features = self.feature_net(state)
        gru_out, self.hidden = self.gru(features.unsqueeze(1), self.hidden)
        gru_out = gru_out.squeeze(1)
        value = self.value_head(gru_out)
        return value

class MALTAgent:
    """Individual MALT agent with PPO training and Transformer adapters"""
    
    def __init__(self, 
                 agent_id, obs_dim, act_dim, state_dim, hidden_size=256,
                 lr_actor=3e-4, lr_critic=1e-3, lr_adapter=1e-3, gamma=0.99, eps_clip=0.2,
                 entropy_coef=0.01, source_policies=None, assigned_policy_indices=None,
                 source_obs_dims=None, source_act_dims=None, transformer_adapter_path=None,
                 # NEW:
                 transformer_trainable=False, transformer_lr=None,
                 transformer_model_dim=128, transformer_n_heads=4, transformer_n_layers=2,
                 transformer_token_size=16, transformer_dropout=0.1):

        self.actor = MALTActor(
            obs_dim=obs_dim, act_dim=act_dim, hidden_size=hidden_size,
            source_policies=source_policies, assigned_policy_indices=assigned_policy_indices,
            source_obs_dims=source_obs_dims, source_act_dims=source_act_dims,
            transformer_adapter_path=transformer_adapter_path,
            transformer_trainable=transformer_trainable,
            transformer_model_dim=transformer_model_dim,
            transformer_n_heads=transformer_n_heads,
            transformer_n_layers=transformer_n_layers,
            transformer_token_size=transformer_token_size,
            transformer_dropout=transformer_dropout,
        ).to(device)

        self.critic = MALTCritic(state_dim=state_dim, hidden_size=hidden_size).to(device)

        # --- Build parameter sets robustly ---
        adapter_params = self.actor.get_adapter_parameters()
        adapter_param_ids = {id(p) for p in adapter_params}
        base_actor_params = [p for p in self.actor.parameters() if p.requires_grad and id(p) not in adapter_param_ids]

        self.optimizer_actor = optim.Adam(base_actor_params, lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.gamma = gamma 
        self.eps_clip = eps_clip 
        self.entropy_coef = entropy_coef

        if adapter_params:
            # Use transformer_lr if provided; else fall back to lr_adapter
            lr_ad = transformer_lr if (transformer_lr is not None) else lr_adapter
            self.optimizer_adapter = optim.Adam(adapter_params, lr=lr_ad)
            print(f"Agent {agent_id}: Created adapter optimizer with {len(adapter_params)} parameters (lr={lr_ad})")
        else:
            self.optimizer_adapter = None


        self.buffer = {
            'obs': [],
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'dones': [],
            'avail_actions': []
        }
        self.attention_history = []

    def reset_hidden_states(self):
        self.actor.reset_hidden_states(batch_size=1)
        self.critic.reset_hidden_states(batch_size=1)

    def select_action(self, obs, state, avail_actions=None):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, attention_weights = self.actor(obs_tensor, return_attention=True)
            if avail_actions is not None:
                avail_mask = torch.FloatTensor(avail_actions).to(device)
                logits = logits.masked_fill(avail_mask == 0, float('-inf'))
            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = self.critic(state_tensor)
            if attention_weights is not None:
                self.attention_history.append(attention_weights.cpu().numpy())
        return action.item(), log_prob.item(), value.item()

    def store_transition(self, obs, state, action, log_prob, reward, done, avail_actions):
        self.buffer['obs'].append(obs)
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['log_probs'].append(log_prob)
        self.buffer['rewards'].append(reward)
        self.buffer['dones'].append(done)
        self.buffer['avail_actions'].append(avail_actions)

    def compute_gae(self, values, rewards, dones, last_value, gamma=0.99, lam=0.95):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            next_value = last_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        return advantages

    def update(self, ppo_epochs=4, batch_size=64):
        if len(self.buffer['obs']) == 0:
            return 0.0, 0.0, 0.0
        obs = torch.FloatTensor(np.array(self.buffer['obs'])).to(device)
        states = torch.FloatTensor(np.array(self.buffer['states'])).to(device)
        actions = torch.LongTensor(np.array(self.buffer['actions'])).to(device)
        old_log_probs = torch.FloatTensor(np.array(self.buffer['log_probs'])).to(device)
        rewards = self.buffer['rewards']
        dones = self.buffer['dones']
        avail_actions_list = self.buffer['avail_actions']

        with torch.no_grad():
            values = []
            for i in range(len(states)):
                self.critic.reset_hidden_states()
                value = self.critic(states[i].unsqueeze(0))
                values.append(value.item())
            self.critic.reset_hidden_states()
            last_value = self.critic(states[-1].unsqueeze(0)).item()
            advantages = self.compute_gae(values, rewards, dones, last_value)
            advantages = torch.FloatTensor(advantages).to(device)
            returns = advantages + torch.FloatTensor(values).to(device)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0
        total_critic_loss = 0
        total_adapter_loss = 0
        num_updates = 0

        for _ in range(ppo_epochs):
            indices = torch.randperm(len(obs))
            for start_idx in range(0, len(obs), batch_size):
                end_idx = min(start_idx + batch_size, len(obs))
                batch_indices = indices[start_idx:end_idx]
                batch_obs = obs[batch_indices]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                self.actor.reset_hidden_states(batch_size=len(batch_indices))
                self.critic.reset_hidden_states(batch_size=len(batch_indices))

                logits = self.actor(batch_obs)
                for i, idx in enumerate(batch_indices):
                    if avail_actions_list[idx] is not None:
                        avail_mask = torch.FloatTensor(avail_actions_list[idx]).to(device)
                        logits[i] = logits[i].masked_fill(avail_mask == 0, float('-inf'))

                probs = torch.softmax(logits, dim=-1)
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                batch_values = self.critic(batch_states).squeeze()
                critic_loss = nn.MSELoss()(batch_values, batch_returns)

                self.optimizer_actor.zero_grad()
                policy_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                self.optimizer_actor.step()

                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.optimizer_critic.step()

                adapter_loss = 0.0
                if self.optimizer_adapter is not None:
                    self.actor.reset_hidden_states(batch_size=len(batch_indices))
                    logits_adapter = self.actor(batch_obs)
                    for i, idx in enumerate(batch_indices):
                        if avail_actions_list[idx] is not None:
                            avail_mask = torch.FloatTensor(avail_actions_list[idx]).to(device)
                            logits_adapter[i] = logits_adapter[i].masked_fill(avail_mask == 0, float('-inf'))
                    probs_adapter = torch.softmax(logits_adapter, dim=-1)
                    dist_adapter = Categorical(probs_adapter)
                    new_log_probs_adapter = dist_adapter.log_prob(batch_actions)
                    ratio_adapter = torch.exp(new_log_probs_adapter - batch_old_log_probs)
                    surr1_adapter = ratio_adapter * batch_advantages
                    surr2_adapter = torch.clamp(ratio_adapter, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                    adapter_loss = -torch.min(surr1_adapter, surr2_adapter).mean()
                    self.optimizer_adapter.zero_grad()
                    adapter_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.get_adapter_parameters(), max_norm=0.5)
                    self.optimizer_adapter.step()
                    adapter_loss = adapter_loss.item()

                total_policy_loss += policy_loss.item()
                total_critic_loss += critic_loss.item()
                total_adapter_loss += adapter_loss
                num_updates += 1

        self.buffer = {
            'obs': [],
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'dones': [],
            'avail_actions': []
        }
        avg_policy_loss = total_policy_loss / num_updates if num_updates > 0 else 0
        avg_critic_loss = total_critic_loss / num_updates if num_updates > 0 else 0
        avg_adapter_loss = total_adapter_loss / num_updates if num_updates > 0 else 0
        return avg_policy_loss, avg_critic_loss, avg_adapter_loss

    def save(self, filepath):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_actor_state_dict': self.optimizer_actor.state_dict(),
            'optimizer_critic_state_dict': self.optimizer_critic.state_dict(),
            'optimizer_adapter_state_dict': self.optimizer_adapter.state_dict() if self.optimizer_adapter else None,
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])
        if self.optimizer_adapter and checkpoint.get('optimizer_adapter_state_dict'):
            self.optimizer_adapter.load_state_dict(checkpoint['optimizer_adapter_state_dict'])

class MALTTrainer:
    def __init__(self, map_name='3m', source_map='8m', seed=42, hidden_size=256,
                 source_model_path=None, policy_assignments_path=None,
                 transformer_adapter_path=None, adapter_lr=1e-3,
                 # NEW:
                 train_transformer=False, transformer_lr=None,
                 transformer_model_dim=128, transformer_n_heads=4, transformer_n_layers=2,
                 transformer_token_size=16, transformer_dropout=0.1):

        self.map_name = map_name
        self.source_map = source_map
        self.seed = seed
        self.hidden_size = hidden_size
        self.transformer_adapter_path = transformer_adapter_path

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Target env
        self.env = StarCraft2Env(map_name=self.map_name, seed=self.seed)
        env_info = self.env.get_env_info()
        self.n_agents = env_info['n_agents']
        self.obs_dim = env_info['obs_shape']
        self.act_dim = env_info['n_actions']
        # SMAC also has a global state:
        try:
            self.state_dim = env_info['state_shape']
        except KeyError:
            # Fallback if not present
            self.state_dim = self.n_agents * self.obs_dim

        # Source policies containers
        self.source_policies = []
        self.source_obs_dims = []
        self.source_act_dims = []

        # Policy assignments
        self.policy_assignments = {}
        if policy_assignments_path and os.path.exists(policy_assignments_path):
            with open(policy_assignments_path, 'r') as f:
                self.policy_assignments = json.load(f)
        else:
            # default: assign empty or simple round-robin
            self.policy_assignments = {str(i): [] for i in range(self.n_agents)}

        # Optionally load source policies
        self.hidden_size = hidden_size
        if source_model_path:
            self._load_source_policies(source_model_path)

        # Build agents
        self.agents = []


        for i in range(self.n_agents):
            assigned_indices = self.policy_assignments.get(str(i), [])
            agent = MALTAgent(
                agent_id=i, obs_dim=self.obs_dim, act_dim=self.act_dim, state_dim=self.state_dim,
                hidden_size=hidden_size, 
                source_policies=self.source_policies, assigned_policy_indices=assigned_indices,
                source_obs_dims=self.source_obs_dims, source_act_dims=self.source_act_dims,
                transformer_adapter_path=transformer_adapter_path,
                lr_adapter=adapter_lr,
                transformer_trainable=train_transformer,
                transformer_lr=transformer_lr,
                transformer_model_dim=transformer_model_dim,
                transformer_n_heads=transformer_n_heads,
                transformer_n_layers=transformer_n_layers,
                transformer_token_size=transformer_token_size,
                transformer_dropout=transformer_dropout
            )
            self.agents.append(agent)
            print(f"Agent {i}: assigned source policies {assigned_indices}")

        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps = []
        self.win_rates = []
        self.policy_losses = []
        self.critic_losses = []
        self.adapter_losses = []

    def _load_source_policies(self, model_path):
        print(f"Loading source policies from: {model_path}")
        source_files = glob.glob(f"{model_path}_agent_*.pth")
        if not source_files:
            print(f"Warning: No source model files found at {model_path}")
            return
        source_env = StarCraft2Env(map_name=self.source_map, seed=self.seed)
        source_env_info = source_env.get_env_info()
        source_obs_dim = source_env_info['obs_shape']
        source_act_dim = source_env_info['n_actions']
        source_env.close()
        print(f"Source Environment: {self.source_map}")
        print(f" Observation dim: {source_obs_dim}")
        print(f" Action dim: {source_act_dim}")
        for source_file in sorted(source_files):
            try:
                checkpoint = torch.load(source_file, map_location=device)
                source_actor = MALTActor(
                    obs_dim=source_obs_dim,
                    act_dim=source_act_dim,
                    hidden_size=self.hidden_size
                ).to(device)
                source_actor.load_state_dict(checkpoint['actor_state_dict'])
                source_actor.eval()
                for param in source_actor.parameters():
                    param.requires_grad = False
                self.source_policies.append(source_actor)
                self.source_obs_dims.append(source_obs_dim)
                self.source_act_dims.append(source_act_dim)
                print(f"Loaded source policy from: {source_file}")
            except Exception as e:
                print(f"Error loading {source_file}: {e}")
        print(f"Total source policies loaded: {len(self.source_policies)}")

    def train(self, max_episodes=1500, max_timesteps=500000, episodes_per_update=25,
              log_frequency=5000, save_frequency=50000):
        adapter_type = "Transformer" if self.transformer_adapter_path else "Linear"
        print(f"Starting MALT-{adapter_type} training...")
        print(f"Transfer: {self.source_map} -> {self.map_name}")
        print(f"Max episodes: {max_episodes}")
        print(f"Max timesteps: {max_timesteps}")
        episode = 0
        total_timesteps = 0
        recent_rewards = deque(maxlen=100)
        recent_wins = deque(maxlen=100)
        while episode < max_episodes and total_timesteps < max_timesteps:
            self.env.reset()
            for agent in self.agents:
                agent.reset_hidden_states()
            episode_reward = 0
            episode_timesteps = 0
            while True:
                obs_list = self.env.get_obs()
                global_state = self.env.get_state()
                avail_actions = self.env.get_avail_actions()
                actions = []
                log_probs = []
                values = []
                for i, agent in enumerate(self.agents):
                    avail_mask = avail_actions[i] if avail_actions else None
                    action, log_prob, value = agent.select_action(obs_list[i], global_state, avail_mask)
                    if avail_mask is not None and isinstance(avail_mask, (list, np.ndarray)):
                        if len(avail_mask) > action and avail_mask[action] == 0:
                            valid_actions = [idx for idx, val in enumerate(avail_mask) if val == 1]
                            action = valid_actions[0] if valid_actions else 0
                    actions.append(action)
                    log_probs.append(log_prob)
                    values.append(value)
                reward, done, info = self.env.step(actions)
                for i, agent in enumerate(self.agents):
                    avail_mask = avail_actions[i] if avail_actions else None
                    agent.store_transition(
                        obs_list[i], global_state, actions[i],
                        log_probs[i], reward, done, avail_mask
                    )
                episode_reward += reward
                episode_timesteps += 1
                total_timesteps += 1
                if done:
                    break
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_timesteps)
            recent_rewards.append(episode_reward)
            battle_won = info.get('battle_won', False)
            recent_wins.append(1 if battle_won else 0)
            episode += 1
            if episode % episodes_per_update == 0:
                for i, agent in enumerate(self.agents):
                    policy_loss, critic_loss, adapter_loss = agent.update()
                    if i == 0:
                        self.policy_losses.append(policy_loss)
                        self.critic_losses.append(critic_loss)
                        self.adapter_losses.append(adapter_loss)
            if total_timesteps % log_frequency < episode_timesteps or episode % 50 == 0:
                avg_reward = np.mean(recent_rewards) if len(recent_rewards) > 0 else 0.0
                win_rate = np.mean(recent_wins) if len(recent_wins) > 0 else 0.0
                self.timesteps.append(total_timesteps)
                self.win_rates.append(win_rate)
                print(
                    f"Episode {episode} \\n Timesteps: {total_timesteps} \\n "
                    f"Reward: {episode_reward:.2f} \\n Avg: {avg_reward:.2f} \\n "
                    f"Win Rate: {win_rate:.2%} \\n Length: {episode_timesteps}"
                )
            if total_timesteps % save_frequency < episode_timesteps:
                self.save_models(f"malt_transformer_{self.map_name}_from_{self.source_map}_{total_timesteps}")
                print(f"Models saved at timestep {total_timesteps}")
        self.env.close()
        print(f"Training completed! Total timesteps: {total_timesteps}")

    def save_models(self, filename_prefix):
        for i, agent in enumerate(self.agents):
            filepath = f"{filename_prefix}_agent_{i}.pth"
            agent.save(filepath)

    def load_models(self, filename_prefix):
        for i, agent in enumerate(self.agents):
            filepath = f"{filename_prefix}_agent_{i}.pth"
            if os.path.exists(filepath):
                agent.load(filepath)
                print(f"Loaded agent {i} from: {filepath}")
            else:
                print(f"Warning: Model file not found: {filepath}")

    def plot_training_curves(self):
        adapter_type = "Transformer" if self.transformer_adapter_path else "Linear"
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'MALT-{adapter_type} Training Curves: {self.source_map} -> {self.map_name}', fontsize=16)

        if self.episode_rewards:
            axes[0, 0].plot(self.episode_rewards, alpha=0.6, label='Episode Reward')
            if len(self.episode_rewards) >= 100:
                moving_avg = np.convolve(self.episode_rewards, np.ones(100)/100, mode='valid')
                axes[0, 0].plot(range(99, len(self.episode_rewards)), moving_avg, color='red', label='Moving Average (100 eps)')
            axes[0, 0].set_title(f'Episode Rewards (MALT-{adapter_type})')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].legend()
            axes[0, 0].grid(True)

        if self.timesteps and self.win_rates:
            axes[0, 1].plot(self.timesteps, self.win_rates, label=f'MALT-{adapter_type} {self.map_name}', color='green')
            axes[0, 1].set_title(f'Win Rate vs Timesteps\n(MALT-{adapter_type}: {self.source_map}→{self.map_name})')
            axes[0, 1].set_xlabel('Timesteps')
            axes[0, 1].set_ylabel('Win Rate')
            axes[0, 1].legend()
            axes[0, 1].grid(True)

        if self.policy_losses:
            axes[0, 2].plot(self.policy_losses, color='orange')
            axes[0, 2].set_title(f'Policy Loss vs Updates (MALT-{adapter_type})')
            axes[0, 2].set_xlabel('Update Steps')
            axes[0, 2].set_ylabel('Policy Loss')
            axes[0, 2].grid(True)

        if self.critic_losses:
            axes[1, 0].plot(self.critic_losses, color='red')
            axes[1, 0].set_title(f'Critic Loss vs Updates (MALT-{adapter_type})')
            axes[1, 0].set_xlabel('Update Steps')
            axes[1, 0].set_ylabel('Critic Loss')
            axes[1, 0].grid(True)

        if self.adapter_losses:
            axes[1, 1].plot(self.adapter_losses, color='purple')
            axes[1, 1].set_title(f'Adapter Loss vs Updates (MALT-{adapter_type})')
            axes[1, 1].set_xlabel('Update Steps')
            axes[1, 1].set_ylabel('Adapter Loss')
            axes[1, 1].grid(True)

        if self.source_policies:
            status_text = f'Transfer Learning\nEnabled\n{len(self.source_policies)} Source Policies\n{adapter_type} Adapters'
            color = "lightgreen" if adapter_type == "Transformer" else "lightblue"
            axes[1, 2].text(0.5, 0.5, status_text,
                            ha='center', va='center', transform=axes[1, 2].transAxes, fontsize=12,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor=color))
            axes[1, 2].set_title(f'Transfer Learning Status ({adapter_type})')
            axes[1, 2].set_xticks([])
            axes[1, 2].set_yticks([])

        plt.tight_layout()
        plt.savefig(f'malt_transformer_{self.map_name}_from_{self.source_map}_training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

    def evaluate(self, num_episodes=100):
        total_rewards = []
        wins = 0
        total_eval_timesteps = 0
        for episode in range(num_episodes):
            self.env.reset()
            for agent in self.agents:
                agent.reset_hidden_states()
            episode_reward = 0
            episode_timesteps = 0
            while True:
                obs_list = self.env.get_obs()
                global_state = self.env.get_state()
                avail_actions = self.env.get_avail_actions()
                actions = []
                for i, agent in enumerate(self.agents):
                    avail_mask = avail_actions[i] if avail_actions else None
                    action, _, _ = agent.select_action(obs_list[i], global_state, avail_mask)
                    if avail_mask is not None and isinstance(avail_mask, (list, np.ndarray)):
                        if len(avail_mask) > action and avail_mask[action] == 0:
                            valid_actions = [idx for idx, val in enumerate(avail_mask) if val == 1]
                            action = valid_actions[0] if valid_actions else 0
                    actions.append(action)
                reward, done, info = self.env.step(actions)
                episode_reward += reward
                episode_timesteps += 1
                total_eval_timesteps += 1
                if done:
                    break
            total_rewards.append(episode_reward)
            if info.get('battle_won', False):
                wins += 1
        avg_reward = np.mean(total_rewards) if len(total_rewards) > 0 else 0.0
        win_rate = wins / num_episodes
        avg_episode_length = total_eval_timesteps / num_episodes
        adapter_type = "Transformer" if self.transformer_adapter_path else "Linear"
        print(f"MALT-{adapter_type} Evaluation Results ({self.source_map}→{self.map_name}):")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Win Rate: {win_rate:.2f}")
        print(f"Average Episode Length: {avg_episode_length:.1f} timesteps")
        print(f"Total Evaluation Timesteps: {total_eval_timesteps}")
        print(f"Source policies used: {len(self.source_policies)}")
        print(f"Adapter type: {adapter_type}")
        if self.source_policies:
            print("Policy assignments:")
            for agent_id, assigned in self.policy_assignments.items():
                print(f" Agent {agent_id}: policies {assigned}")
        return avg_reward, win_rate


def main():
    parser = argparse.ArgumentParser(description='MALT Training with Transformer-based Observation Adapters')
    parser.add_argument('--map', type=str, default='3m', help='Target SMAC map name')
    parser.add_argument('--source_map', type=str, default='8m', help='Source SMAC map name')
    parser.add_argument('--source_model_path', type=str, help='Path to source model files (without agent suffix)')
    parser.add_argument('--policy_assignments', type=str, help='Path to policy assignments JSON file (optional)')
    parser.add_argument('--transformer_adapter', type=str, help='Path to trained Transformer adapter (.pth file)')
    parser.add_argument('--episodes', type=int, default=1500, help='Number of training episodes')
    parser.add_argument('--timesteps', type=int, default=500000, help='Number of training timesteps')
    parser.add_argument('--episodes_per_update', type=int, default=25, help='Episodes per update')
    parser.add_argument('--log_freq', type=int, default=5000, help='Logging frequency')
    parser.add_argument('--save_freq', type=int, default=50000, help='Save frequency')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size')
    parser.add_argument('--adapter_lr', type=float, default=1e-3, help='Learning rate for trainable adapters')
    parser.add_argument('--eval_only', action='store_true', help='Only evaluate')
    parser.add_argument('--load_model', type=str, help='Load MALT model filename')
    parser.add_argument('--eval_episodes', type=int, default=100, help='Evaluation episodes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')    
    parser.add_argument('--train_transformer', action='store_true', help='Train Transformer adapter jointly with PPO')
    parser.add_argument('--transformer_lr', type=float, help='LR for Transformer adapter (overrides --adapter_lr)')
    parser.add_argument('--transformer_model_dim', type=int, default=128)
    parser.add_argument('--transformer_n_heads', type=int, default=4)
    parser.add_argument('--transformer_n_layers', type=int, default=2)
    parser.add_argument('--transformer_token_size', type=int, default=16)
    parser.add_argument('--transformer_dropout', type=float, default=0.1)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("MALT TRAINING WITH TRANSFORMER-BASED OBSERVATION ADAPTERS")
    print("="*60)
    if args.transformer_adapter:
        print(f"Transformer Adapter: {args.transformer_adapter}")
        print("Using trained Transformer for observation space adaptation")
    else:
        print("No Transformer adapter specified - using linear adapters")
    print("="*60)

    
    trainer = MALTTrainer(
        map_name=args.map,
        source_map=args.source_map,
        seed=args.seed,
        hidden_size=args.hidden_size,
        source_model_path=args.source_model_path,
        policy_assignments_path=args.policy_assignments,
        transformer_adapter_path=args.transformer_adapter,
        adapter_lr=args.adapter_lr,
        # NEW:
        train_transformer=args.train_transformer,
        transformer_lr=args.transformer_lr,
        transformer_model_dim=args.transformer_model_dim,
        transformer_n_heads=args.transformer_n_heads,
        transformer_n_layers=args.transformer_n_layers,
        transformer_token_size=args.transformer_token_size,
        transformer_dropout=args.transformer_dropout
    )


    if args.eval_only:
        if args.load_model:
            trainer.load_models(args.load_model)
            trainer.evaluate(num_episodes=args.eval_episodes)
    else:
        if args.load_model:
            trainer.load_models(args.load_model)
        trainer.train(
            max_episodes=args.episodes,
            max_timesteps=args.timesteps,
            episodes_per_update=args.episodes_per_update,
            log_frequency=args.log_freq,
            save_frequency=args.save_freq
        )
        trainer.plot_training_curves()
        print(f"\nFinal MALT-Transformer Evaluation ({args.source_map}→{args.map}):")
        trainer.evaluate(num_episodes=args.eval_episodes)


if __name__ == "__main__":
    main()
