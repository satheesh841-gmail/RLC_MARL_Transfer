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

# ==========================================
# GAN GENERATOR FOR OBSERVATION ADAPTATION
# ==========================================

class GANGenerator(nn.Module):
    """GAN Generator: target_obs → source_obs (from trained GAN)"""
    def __init__(self, target_dim, source_dim, hidden_size=128):
        super(GANGenerator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(target_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, source_dim)
        )
        
    def forward(self, target_obs):
        return self.net(target_obs)

class GANObservationAdapter(nn.Module):
    """GAN-based observation adapter for source policies"""
    def __init__(self, target_obs_dim, source_obs_dim, gan_adapter_path=None):
        super(GANObservationAdapter, self).__init__()
        self.target_obs_dim = target_obs_dim
        self.source_obs_dim = source_obs_dim
        self.gan_adapter_path = gan_adapter_path
        
        if target_obs_dim != source_obs_dim and gan_adapter_path and os.path.exists(gan_adapter_path):
            # Load trained GAN generator
            print(f"Loading GAN adapter from: {gan_adapter_path}")
            try:
                checkpoint = torch.load(gan_adapter_path, map_location=device)
                
                # Extract GAN generator
                self.generator = GANGenerator(target_obs_dim, source_obs_dim)
                self.generator.load_state_dict(checkpoint['generator'])
                self.generator.eval()
                
                # Freeze GAN parameters during MALT training
                for param in self.generator.parameters():
                    param.requires_grad = False
                
                self.needs_adaptation = True
                self.adapter_type = "GAN"
                print(f"GAN adapter loaded: {target_obs_dim} → {source_obs_dim}")
                
            except Exception as e:
                print(f"Error loading GAN adapter: {e}")
                print("Falling back to linear adapter...")
                self._create_linear_adapter()
                
        elif target_obs_dim != source_obs_dim:
            # Fallback to linear adapter
            print(f"Using linear adapter: {target_obs_dim} → {source_obs_dim}")
            self._create_linear_adapter()
        else:
            # No adaptation needed
            self.adapter = nn.Identity()
            self.needs_adaptation = False
            self.adapter_type = "identity"
            print(f"No adaptation needed: {target_obs_dim} = {source_obs_dim}")
    
    def _create_linear_adapter(self):
        """Create linear adapter as fallback"""
        hidden_size = 2 * max(self.target_obs_dim, self.source_obs_dim)
        self.adapter = nn.Sequential(
            nn.Linear(self.target_obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.source_obs_dim)
        )
        
        # Xavier initialization
        nn.init.xavier_uniform_(self.adapter[0].weight)
        nn.init.xavier_uniform_(self.adapter[2].weight)
        nn.init.zeros_(self.adapter[0].bias)
        nn.init.zeros_(self.adapter[2].bias)
        
        self.needs_adaptation = True
        self.adapter_type = "linear"
    
    def forward(self, target_obs):
        """
        Args:
            target_obs: [batch_size, target_obs_dim]
        Returns:
            adapted_obs: [batch_size, source_obs_dim]
        """
        if not self.needs_adaptation:
            return target_obs
        
        if self.adapter_type == "GAN":
            with torch.no_grad():  # GAN is frozen
                return self.generator(target_obs)
        else:
            return self.adapter(target_obs)
    
    def get_adapter_parameters(self):
        """Get trainable parameters (only for linear adapters)"""
        if self.adapter_type == "linear":
            return list(self.adapter.parameters())
        else:
            return []  # GAN adapters are frozen

class AttentionModule(nn.Module):
    """Soft attention module for weighting transferred features"""
    def __init__(self, input_dim, num_policies):
        super(AttentionModule, self).__init__()
        self.num_policies = num_policies
        self.projection = nn.Linear(input_dim * num_policies, 128)
        self.attention_weights = nn.Linear(128, num_policies)
        
    def forward(self, policy_outputs):
        """
        Args:
            policy_outputs: List of outputs from assigned policies [batch_size, feature_dim]
        Returns:
            attention_weights: [batch_size, num_policies]
        """
        # Concatenate all policy outputs
        concat_outputs = torch.cat(policy_outputs, dim=-1)  # [batch_size, feature_dim * num_policies]
        
        # Project and compute attention weights
        projected = torch.relu(self.projection(concat_outputs))
        weights = torch.softmax(self.attention_weights(projected), dim=-1)
        
        return weights

class MALTActor(nn.Module):
    """MALT-enhanced Actor with lateral connections and GAN-based adapters"""
    def __init__(self, obs_dim, act_dim, hidden_size=256, gru_layers=1, 
                 source_policies=None, assigned_policy_indices=None, num_assigned_policies=3,
                 source_obs_dims=None, source_act_dims=None, gan_adapter_path=None):
        super(MALTActor, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.gru_layers = gru_layers
        self.num_assigned_policies = num_assigned_policies
        self.gan_adapter_path = gan_adapter_path
        
        # Store source policies (frozen)
        self.source_policies = source_policies if source_policies else []
        self.assigned_policy_indices = assigned_policy_indices if assigned_policy_indices else []
        
        # Store source dimensions for adapters
        self.source_obs_dims = source_obs_dims if source_obs_dims else []
        self.source_act_dims = source_act_dims if source_act_dims else []
        
        # Target agent's own networks (same architecture as source)
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
        
        # GAN-based adapters for source policies
        if source_policies and len(assigned_policy_indices) > 0:
            self.obs_adapters = nn.ModuleList()
            
            print(f"Creating GAN adapters for {len(assigned_policy_indices)} assigned policies...")
            
            for idx in assigned_policy_indices:
                if idx < len(source_obs_dims):
                    source_obs_dim = source_obs_dims[idx]
                    
                    # Create GAN adapter for this source policy
                    adapter = GANObservationAdapter(
                        target_obs_dim=obs_dim,
                        source_obs_dim=source_obs_dim,
                        gan_adapter_path=gan_adapter_path
                    )
                    
                    self.obs_adapters.append(adapter)
                    print(f"  Policy {idx}: {adapter.adapter_type} adapter ({obs_dim} -> {source_obs_dim})")
            
            # Freeze source policies
            for policy in self.source_policies:
                if policy is not None:
                    for param in policy.parameters():
                        param.requires_grad = False
            
            # Attention module for policy weighting
            self.attention = AttentionModule(hidden_size, len(assigned_policy_indices))
            
        else:
            self.obs_adapters = nn.ModuleList()
            self.attention = None
        
        # Initialize hidden state
        self.hidden = None
    
    def reset_hidden_states(self, batch_size=1):
        """Reset GRU hidden states"""
        self.hidden = torch.zeros(self.gru_layers, batch_size, self.hidden_size).to(device)
    
    def forward(self, obs, return_features=False, return_attention=False):
        """
        Forward pass with lateral transfer from assigned source policies
        
        Args:
            obs: [batch_size, obs_dim]
            return_features: Return intermediate features
            return_attention: Return attention weights
        """
        batch_size = obs.size(0)
        
        # Initialize hidden states if needed
        if self.hidden is None or self.hidden.size(1) != batch_size:
            self.reset_hidden_states(batch_size)
        
        # Extract target features
        target_features = self.feature_net(obs)
        
        # GRU processing for temporal dependencies
        gru_out, self.hidden = self.gru(target_features.unsqueeze(1), self.hidden)
        gru_out = gru_out.squeeze(1)
        
        # Lateral transfer from assigned source policies
        if len(self.source_policies) > 0 and len(self.obs_adapters) > 0:
            policy_features = []
            
            # Get features from each assigned source policy
            for adapter, policy_idx in zip(self.obs_adapters, self.assigned_policy_indices):
                if policy_idx < len(self.source_policies):
                    source_policy = self.source_policies[policy_idx]
                    
                    if source_policy is not None:
                        # Adapt observation using GAN or linear adapter
                        adapted_obs = adapter(obs)
                        
                        # Get features from source policy
                        with torch.no_grad():
                            if hasattr(source_policy, 'feature_net'):
                                source_features = source_policy.feature_net(adapted_obs)
                            else:
                                source_features = source_policy(adapted_obs, return_features=True)
                            
                            policy_features.append(source_features)
            
            # Apply attention-weighted fusion if we have source features
            if policy_features:
                attention_weights = self.attention(policy_features)
                
                # Weighted combination of source features
                weighted_features = sum(
                    w.unsqueeze(-1) * f 
                    for w, f in zip(attention_weights.unbind(1), policy_features)
                )
                
                # Combine with target features
                combined_features = gru_out + 0.3 * weighted_features
            else:
                combined_features = gru_out
                attention_weights = None
        else:
            combined_features = gru_out
            attention_weights = None
        
        # Policy head
        logits = self.policy_head(combined_features)
        
        # Return based on flags
        if return_features and return_attention:
            return logits, combined_features, attention_weights
        elif return_features:
            return logits, combined_features
        elif return_attention:
            return logits, attention_weights
        else:
            return logits
    
    def get_adapter_parameters(self):
        """Get trainable adapter parameters (only for linear adapters)"""
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
        
        # State processing networks
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
        
        # Initialize hidden state
        self.hidden = None
    
    def reset_hidden_states(self, batch_size=1):
        """Reset GRU hidden states"""
        self.hidden = torch.zeros(self.gru_layers, batch_size, self.hidden_size).to(device)
    
    def forward(self, state):
        """
        Forward pass
        
        Args:
            state: [batch_size, state_dim]
        Returns:
            value: [batch_size, 1]
        """
        batch_size = state.size(0)
        
        # Initialize hidden states if needed
        if self.hidden is None or self.hidden.size(1) != batch_size:
            self.reset_hidden_states(batch_size)
        
        # Extract features
        features = self.feature_net(state)
        
        # GRU processing
        gru_out, self.hidden = self.gru(features.unsqueeze(1), self.hidden)
        gru_out = gru_out.squeeze(1)
        
        # Value estimation
        value = self.value_head(gru_out)
        
        return value

class MALTAgent:
    """Individual MALT agent with PPO training and GAN adapters"""
    def __init__(self, agent_id, obs_dim, act_dim, state_dim, hidden_size=256, 
                 lr_actor=3e-4, lr_critic=1e-3, lr_adapter=1e-3, gamma=0.99, eps_clip=0.2, 
                 entropy_coef=0.01, source_policies=None, assigned_policy_indices=None,
                 source_obs_dims=None, source_act_dims=None, gan_adapter_path=None):
        
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.entropy_coef = entropy_coef
        
        # MALT Actor with GAN adapters
        self.actor = MALTActor(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_size=hidden_size,
            source_policies=source_policies,
            assigned_policy_indices=assigned_policy_indices,
            source_obs_dims=source_obs_dims,
            source_act_dims=source_act_dims,
            gan_adapter_path=gan_adapter_path
        ).to(device)
        
        # Centralized Critic
        self.critic = MALTCritic(
            state_dim=state_dim,
            hidden_size=hidden_size
        ).to(device)
        
        # Separate optimizers for actor, critic, and adapters
        self.optimizer_actor = optim.Adam(
            [p for p in self.actor.parameters() if p.requires_grad and 
             p not in self.actor.get_adapter_parameters()],
            lr=lr_actor
        )
        
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Optimizer for trainable adapters only (linear adapters)
        adapter_params = self.actor.get_adapter_parameters()
        if adapter_params:
            self.optimizer_adapter = optim.Adam(adapter_params, lr=lr_adapter)
            print(f"Agent {agent_id}: Created adapter optimizer with {len(adapter_params)} parameters")
        else:
            self.optimizer_adapter = None
            print(f"Agent {agent_id}: No trainable adapters (using GAN or identity)")
        
        # Replay buffer
        self.buffer = {
            'obs': [],
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'dones': [],
            'avail_actions': []
        }
        
        # Track attention weights for analysis
        self.attention_history = []
    
    def reset_hidden_states(self):
        """Reset hidden states for both actor and critic"""
        self.actor.reset_hidden_states(batch_size=1)
        self.critic.reset_hidden_states(batch_size=1)
    
    def select_action(self, obs, state, avail_actions=None):
        """
        Select action using MALT actor
        
        Args:
            obs: Agent observation
            state: Global state
            avail_actions: Available actions mask
        Returns:
            action, log_prob, value
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Get action logits and attention weights
            logits, attention_weights = self.actor(obs_tensor, return_attention=True)
            
            # Apply action mask if available
            if avail_actions is not None:
                avail_mask = torch.FloatTensor(avail_actions).to(device)
                logits = logits.masked_fill(avail_mask == 0, float('-inf'))
            
            # Sample action
            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # Get value estimate
            value = self.critic(state_tensor)
            
            # Track attention weights
            if attention_weights is not None:
                self.attention_history.append(attention_weights.cpu().numpy())
        
        return action.item(), log_prob.item(), value.item()
    
    def store_transition(self, obs, state, action, log_prob, reward, done, avail_actions):
        """Store transition in buffer"""
        self.buffer['obs'].append(obs)
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['log_probs'].append(log_prob)
        self.buffer['rewards'].append(reward)
        self.buffer['dones'].append(done)
        self.buffer['avail_actions'].append(avail_actions)
    
    def compute_gae(self, values, rewards, dones, last_value, gamma=0.99, lam=0.95):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        return advantages
    
    def update(self, ppo_epochs=4, batch_size=64):
        """
        Update actor, critic, and adapters using PPO
        
        Returns:
            policy_loss, critic_loss, adapter_loss
        """
        if len(self.buffer['obs']) == 0:
            return 0.0, 0.0, 0.0
        
        # Convert buffer to tensors
        obs = torch.FloatTensor(np.array(self.buffer['obs'])).to(device)
        states = torch.FloatTensor(np.array(self.buffer['states'])).to(device)
        actions = torch.LongTensor(np.array(self.buffer['actions'])).to(device)
        old_log_probs = torch.FloatTensor(np.array(self.buffer['log_probs'])).to(device)
        rewards = self.buffer['rewards']
        dones = self.buffer['dones']
        avail_actions_list = self.buffer['avail_actions']
        
        # Compute values and advantages
        with torch.no_grad():
            values = []
            for i in range(len(states)):
                self.critic.reset_hidden_states()
                value = self.critic(states[i].unsqueeze(0))
                values.append(value.item())
            
            # Get last value for GAE
            self.critic.reset_hidden_states()
            last_value = self.critic(states[-1].unsqueeze(0)).item()
            
            advantages = self.compute_gae(values, rewards, dones, last_value)
            advantages = torch.FloatTensor(advantages).to(device)
            returns = advantages + torch.FloatTensor(values).to(device)
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_policy_loss = 0
        total_critic_loss = 0
        total_adapter_loss = 0
        num_updates = 0
        
        for _ in range(ppo_epochs):
            # Create mini-batches
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
                
                # Reset hidden states for batch
                self.actor.reset_hidden_states(batch_size=len(batch_indices))
                self.critic.reset_hidden_states(batch_size=len(batch_indices))
                
                # Forward pass through actor
                logits = self.actor(batch_obs)
                
                # Apply action masks
                for i, idx in enumerate(batch_indices):
                    if avail_actions_list[idx] is not None:
                        avail_mask = torch.FloatTensor(avail_actions_list[idx]).to(device)
                        logits[i] = logits[i].masked_fill(avail_mask == 0, float('-inf'))
                
                # Compute new log probabilities and entropy
                probs = torch.softmax(logits, dim=-1)
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # PPO clipped objective
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                
                # Critic loss
                batch_values = self.critic(batch_states).squeeze()
                critic_loss = nn.MSELoss()(batch_values, batch_returns)
                
                # Update actor
                self.optimizer_actor.zero_grad()
                policy_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                self.optimizer_actor.step()
                
                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.optimizer_critic.step()
                
                # Update adapters (only for linear adapters)
                adapter_loss = 0.0
                if self.optimizer_adapter is not None:
                    # Recompute for adapter loss
                    self.actor.reset_hidden_states(batch_size=len(batch_indices))
                    logits_adapter = self.actor(batch_obs)
                    
                    # Apply masks again
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
        
        # Clear buffer
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
        """Save agent models"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_actor_state_dict': self.optimizer_actor.state_dict(),
            'optimizer_critic_state_dict': self.optimizer_critic.state_dict(),
            'optimizer_adapter_state_dict': self.optimizer_adapter.state_dict() if self.optimizer_adapter else None,
        }, filepath)
    
    def load(self, filepath):
        """Load agent models"""
        checkpoint = torch.load(filepath, map_location=device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])
        if self.optimizer_adapter and checkpoint.get('optimizer_adapter_state_dict'):
            self.optimizer_adapter.load_state_dict(checkpoint['optimizer_adapter_state_dict'])

class MALTTrainer:
    """MALT Trainer with GAN-based observation adapters"""
    def __init__(self, map_name='3m', source_map='8m', seed=42, hidden_size=256, 
                 source_model_path=None, policy_assignments_path=None, 
                 gan_adapter_path=None, adapter_lr=1e-3):
        
        self.map_name = map_name
        self.source_map = source_map
        self.seed = seed
        self.hidden_size = hidden_size
        self.gan_adapter_path = gan_adapter_path
        
        # Initialize target environment
        self.env = StarCraft2Env(map_name=map_name, seed=seed)
        self.env_info = self.env.get_env_info()
        
        self.n_agents = self.env_info['n_agents']
        self.obs_dim = self.env_info['obs_shape']
        self.state_dim = self.env_info['state_shape']
        self.act_dim = self.env_info['n_actions']
        
        print(f"Target Environment: {map_name}")
        print(f"  Agents: {self.n_agents}")
        print(f"  Observation dim: {self.obs_dim}")
        print(f"  State dim: {self.state_dim}")
        print(f"  Action dim: {self.act_dim}")
        
        # Load source policies if provided
        self.source_policies = []
        self.source_obs_dims = []
        self.source_act_dims = []
        self.policy_assignments = {}
        
        if source_model_path:
            self._load_source_policies(source_model_path)
        
        # Load policy assignments if provided
        if policy_assignments_path and os.path.exists(policy_assignments_path):
            with open(policy_assignments_path, 'r') as f:
                self.policy_assignments = json.load(f)
            print(f"Loaded policy assignments from: {policy_assignments_path}")
        else:
            # Default: all agents use all source policies
            if self.source_policies:
                self.policy_assignments = {
                    str(i): list(range(len(self.source_policies)))
                    for i in range(self.n_agents)
                }
                print("Using default policy assignments (all agents use all policies)")
        
        # Initialize MALT agents with GAN adapters
        self.agents = []
        for i in range(self.n_agents):
            assigned_indices = self.policy_assignments.get(str(i), [])
            
            agent = MALTAgent(
                agent_id=i,
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                state_dim=self.state_dim,
                hidden_size=hidden_size,
                source_policies=self.source_policies,
                assigned_policy_indices=assigned_indices,
                source_obs_dims=self.source_obs_dims,
                source_act_dims=self.source_act_dims,
                gan_adapter_path=gan_adapter_path,
                lr_adapter=adapter_lr
            )
            
            self.agents.append(agent)
            print(f"Agent {i}: assigned source policies {assigned_indices}")
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps = []
        self.win_rates = []
        self.policy_losses = []
        self.critic_losses = []
        self.adapter_losses = []
    
    def _load_source_policies(self, model_path):
        """Load source policies from saved models"""
        print(f"Loading source policies from: {model_path}")
        
        # Find source model files
        source_files = glob.glob(f"{model_path}_agent_*.pth")
        
        if not source_files:
            print(f"Warning: No source model files found at {model_path}")
            return
        
        # Load source environment info
        source_env = StarCraft2Env(map_name=self.source_map, seed=self.seed)
        source_env_info = source_env.get_env_info()
        source_obs_dim = source_env_info['obs_shape']
        source_act_dim = source_env_info['n_actions']
        source_env.close()
        
        print(f"Source Environment: {self.source_map}")
        print(f"  Observation dim: {source_obs_dim}")
        print(f"  Action dim: {source_act_dim}")
        
        # Load each source policy
        for source_file in sorted(source_files):
            try:
                checkpoint = torch.load(source_file, map_location=device)
                
                # Create source actor
                source_actor = MALTActor(
                    obs_dim=source_obs_dim,
                    act_dim=source_act_dim,
                    hidden_size=self.hidden_size
                ).to(device)
                
                source_actor.load_state_dict(checkpoint['actor_state_dict'])
                source_actor.eval()
                
                # Freeze source policy
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
        """Train MALT agents with GAN-based adapters"""
        
        adapter_type = "GAN" if self.gan_adapter_path else "Linear"
        print(f"Starting MALT-{adapter_type} training...")
        print(f"Transfer: {self.source_map} -> {self.map_name}")
        print(f"Max episodes: {max_episodes}")
        print(f"Max timesteps: {max_timesteps}")
        
        episode = 0
        total_timesteps = 0
        episode_buffer = []
        recent_rewards = deque(maxlen=100)
        recent_wins = deque(maxlen=100)
        
        while episode < max_episodes and total_timesteps < max_timesteps:
            self.env.reset()
            
            # Reset hidden states
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
                
                # Select actions for all agents
                for i, agent in enumerate(self.agents):
                    avail_mask = avail_actions[i] if avail_actions else None
                    action, log_prob, value = agent.select_action(obs_list[i], global_state, avail_mask)
                    
                    # Validate action
                    if avail_mask is not None and isinstance(avail_mask, (list, np.ndarray)):
                        if len(avail_mask) > action and avail_mask[action] == 0:
                            valid_actions = [idx for idx, val in enumerate(avail_mask) if val == 1]
                            if valid_actions:
                                action = valid_actions[0]
                            else:
                                action = 0
                    
                    actions.append(action)
                    log_probs.append(log_prob)
                    values.append(value)
                
                # Step environment
                reward, done, info = self.env.step(actions)
                
                # Store transitions
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
            
            # Track metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_timesteps)
            recent_rewards.append(episode_reward)
            
            battle_won = info.get('battle_won', False)
            recent_wins.append(1 if battle_won else 0)
            
            episode += 1
            
            # Update agents periodically
            if episode % episodes_per_update == 0:
                for i, agent in enumerate(self.agents):
                    policy_loss, critic_loss, adapter_loss = agent.update()
                    
                    if i == 0:  # Track metrics from first agent
                        self.policy_losses.append(policy_loss)
                        self.critic_losses.append(critic_loss)
                        self.adapter_losses.append(adapter_loss)
            
            # Logging
            if total_timesteps % log_frequency < episode_timesteps or episode % 50 == 0:
                avg_reward = np.mean(recent_rewards)
                win_rate = np.mean(recent_wins)
                
                self.timesteps.append(total_timesteps)
                self.win_rates.append(win_rate)
                
                print(f"Episode {episode} | Timesteps: {total_timesteps} | "
                      f"Reward: {episode_reward:.2f} | Avg: {avg_reward:.2f} | "
                      f"Win Rate: {win_rate:.2%} | Length: {episode_timesteps}")
            
            # Save models
            if total_timesteps % save_frequency < episode_timesteps:
                self.save_models(f"malt_gan_{self.map_name}_from_{self.source_map}_{total_timesteps}")
                print(f"Models saved at timestep {total_timesteps}")
        
        self.env.close()
        print(f"Training completed! Total timesteps: {total_timesteps}")
    
    def save_models(self, filename_prefix):
        """Save all agent models"""
        for i, agent in enumerate(self.agents):
            filepath = f"{filename_prefix}_agent_{i}.pth"
            agent.save(filepath)
    
    def load_models(self, filename_prefix):
        """Load all agent models"""
        for i, agent in enumerate(self.agents):
            filepath = f"{filename_prefix}_agent_{i}.pth"
            if os.path.exists(filepath):
                agent.load(filepath)
                print(f"Loaded agent {i} from: {filepath}")
            else:
                print(f"Warning: Model file not found: {filepath}")
    
    def plot_training_curves(self):
       """Plot training curves with GAN adapter info"""
       adapter_type = "GAN" if self.gan_adapter_path else "Linear"
       
       fig, axes = plt.subplots(2, 3, figsize=(18, 10))
       fig.suptitle(f'MALT-{adapter_type} Training Curves: {self.source_map} -> {self.map_name}', fontsize=16)
       
       # Episode rewards
       if self.episode_rewards:
           axes[0, 0].plot(self.episode_rewards, alpha=0.6, label='Episode Reward')
           if len(self.episode_rewards) >= 100:
               moving_avg = np.convolve(self.episode_rewards, np.ones(100)/100, mode='valid')
               axes[0, 0].plot(range(99, len(self.episode_rewards)), moving_avg, 
                             color='red', label='Moving Average (100 eps)')
           axes[0, 0].set_title(f'Episode Rewards (MALT-{adapter_type})')
           axes[0, 0].set_xlabel('Episode')
           axes[0, 0].set_ylabel('Reward')
           axes[0, 0].legend()
           axes[0, 0].grid(True)
       
       # Win rate over timesteps
       if self.timesteps and self.win_rates:
           axes[0, 1].plot(self.timesteps, self.win_rates, label=f'MALT-{adapter_type} {self.map_name}', color='green')
           axes[0, 1].set_title(f'Win Rate vs Timesteps\n(MALT-{adapter_type}: {self.source_map}→{self.map_name})')
           axes[0, 1].set_xlabel('Timesteps')
           axes[0, 1].set_ylabel('Win Rate')
           axes[0, 1].legend()
           axes[0, 1].grid(True)
       
       # Policy losses
       if self.policy_losses:
           axes[0, 2].plot(self.policy_losses, color='orange')
           axes[0, 2].set_title(f'Policy Loss vs Updates (MALT-{adapter_type})')
           axes[0, 2].set_xlabel('Update Steps')
           axes[0, 2].set_ylabel('Policy Loss')
           axes[0, 2].grid(True)
       
       # Critic losses
       if self.critic_losses:
           axes[1, 0].plot(self.critic_losses, color='red')
           axes[1, 0].set_title(f'Critic Loss vs Updates (MALT-{adapter_type})')
           axes[1, 0].set_xlabel('Update Steps')
           axes[1, 0].set_ylabel('Critic Loss')
           axes[1, 0].grid(True)
       
       # Adapter losses
       if self.adapter_losses:
           axes[1, 1].plot(self.adapter_losses, color='purple')
           axes[1, 1].set_title(f'Adapter Loss vs Updates (MALT-{adapter_type})')
           axes[1, 1].set_xlabel('Update Steps')
           axes[1, 1].set_ylabel('Adapter Loss')
           axes[1, 1].grid(True)
       
       # Transfer learning status with GAN info
       if self.source_policies:
           status_text = f'Transfer Learning\nEnabled\n{len(self.source_policies)} Source Policies\n{adapter_type} Adapters'
           color = "lightgreen" if adapter_type == "GAN" else "lightblue"
           axes[1, 2].text(0.5, 0.5, status_text, 
                          ha='center', va='center', transform=axes[1, 2].transAxes, fontsize=12, 
                          bbox=dict(boxstyle="round,pad=0.3", facecolor=color))
           axes[1, 2].set_title(f'Transfer Learning Status ({adapter_type})')
           axes[1, 2].set_xticks([])
           axes[1, 2].set_yticks([])
       
       plt.tight_layout()
       plt.savefig(f'malt_gan_{self.map_name}_from_{self.source_map}_training_curves.png', dpi=300, bbox_inches='tight')
       plt.show()
   
    def evaluate(self, num_episodes=100):
       """Evaluate trained MALT agents with GAN adapters"""
       total_rewards = []
       wins = 0
       total_eval_timesteps = 0
       
       for episode in range(num_episodes):
           self.env.reset()
           
           # Reset hidden states
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
                   
                   # Validate action
                   if avail_mask is not None and isinstance(avail_mask, (list, np.ndarray)):
                       if len(avail_mask) > action and avail_mask[action] == 0:
                           valid_actions = [idx for idx, val in enumerate(avail_mask) if val == 1]
                           if valid_actions:
                               action = valid_actions[0]
                           else:
                               action = 0
                   
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
       
       avg_reward = np.mean(total_rewards)
       win_rate = wins / num_episodes
       avg_episode_length = total_eval_timesteps / num_episodes
       
       adapter_type = "GAN" if self.gan_adapter_path else "Linear"
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
               print(f"  Agent {agent_id}: policies {assigned}")
       
       return avg_reward, win_rate

def main():
   parser = argparse.ArgumentParser(description='MALT Training with GAN-based Observation Adapters')
   parser.add_argument('--map', type=str, default='3m', help='Target SMAC map name')
   parser.add_argument('--source_map', type=str, default='8m', help='Source SMAC map name')
   parser.add_argument('--source_model_path', type=str, help='Path to source model files (without agent suffix)')
   parser.add_argument('--policy_assignments', type=str, help='Path to policy assignments JSON file (optional)')
   parser.add_argument('--gan_adapter', type=str, help='Path to trained GAN adapter (.pth file)')

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
   
   args = parser.parse_args()
   
   # Set random seeds
   torch.manual_seed(args.seed)
   np.random.seed(args.seed)
   
   print("MALT TRAINING WITH GAN-BASED OBSERVATION ADAPTERS")
   print("="*60)
   if args.gan_adapter:
       print(f"GAN Adapter: {args.gan_adapter}")
       print("Using trained GAN for observation space adaptation")
   else:
       print("No GAN adapter specified - using linear adapters")
   print("="*60)
   
   # Initialize MALT trainer with GAN adapters
   trainer = MALTTrainer(
       map_name=args.map,
       source_map=args.source_map,
       seed=args.seed,
       hidden_size=args.hidden_size,
       source_model_path=args.source_model_path,
       policy_assignments_path=args.policy_assignments,
       gan_adapter_path=args.gan_adapter,
       adapter_lr=args.adapter_lr,
       
   )
   
   if args.eval_only:
       if args.load_model:
           trainer.load_models(args.load_model)
       trainer.evaluate(num_episodes=args.eval_episodes)
   else:
       # Load model if specified
       if args.load_model:
           trainer.load_models(args.load_model)
       
       # Train agents with GAN adapters

       # Train agents with GAN adapters
       trainer.train(
           max_episodes=args.episodes,
           max_timesteps=args.timesteps,
           episodes_per_update=args.episodes_per_update,
           log_frequency=args.log_freq,
           save_frequency=args.save_freq
       )
       
       # Plot training curves
       trainer.plot_training_curves()
       
       # Final evaluation
       print(f"\nFinal MALT-GAN Evaluation ({args.source_map}→{args.map}):")
       trainer.evaluate(num_episodes=args.eval_episodes)

if __name__ == "__main__":
   main()