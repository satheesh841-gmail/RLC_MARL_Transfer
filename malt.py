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

class ObservationAdapter(nn.Module):
   """Learnable observation adapter for source policies"""
   def __init__(self, target_obs_dim, source_obs_dim):
       super(ObservationAdapter, self).__init__()
       self.target_obs_dim = target_obs_dim
       self.source_obs_dim = source_obs_dim
       
       if target_obs_dim != source_obs_dim:
           # Need adaptation with single hidden layer
           hidden_size = 2 * max(target_obs_dim, source_obs_dim)
           self.adapter = nn.Sequential(
               nn.Linear(target_obs_dim, hidden_size),
               nn.ReLU(),
               nn.Linear(hidden_size, source_obs_dim)
           )
           
           # Xavier initialization
           nn.init.xavier_uniform_(self.adapter[0].weight)
           nn.init.xavier_uniform_(self.adapter[2].weight)
           nn.init.zeros_(self.adapter[0].bias)
           nn.init.zeros_(self.adapter[2].bias)
           
           self.needs_adaptation = True
       else:
           # No adaptation needed
           self.adapter = nn.Identity()
           self.needs_adaptation = False
   
   def forward(self, target_obs):
       """
       Args:
           target_obs: [batch_size, target_obs_dim]
       Returns:
           adapted_obs: [batch_size, source_obs_dim]
       """
       return self.adapter(target_obs)

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
   """MALT-enhanced Actor with lateral connections and adapters"""
   def __init__(self, obs_dim, act_dim, hidden_size=256, gru_layers=1, 
                source_policies=None, assigned_policy_indices=None, num_assigned_policies=3,
                source_obs_dims=None, source_act_dims=None):
       super(MALTActor, self).__init__()
       self.obs_dim = obs_dim
       self.act_dim = act_dim
       self.hidden_size = hidden_size
       self.gru_layers = gru_layers
       self.num_assigned_policies = num_assigned_policies
       
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
       
       # Learnable adapters for source policies
       if source_policies and len(assigned_policy_indices) > 0:
           self.obs_adapters = nn.ModuleList()
           
           for policy_idx in assigned_policy_indices:
               if policy_idx < len(source_policies):
                   # Observation adapter: target_obs -> source_obs
                   source_obs_dim = self.source_obs_dims[policy_idx] if policy_idx < len(self.source_obs_dims) else obs_dim
                   obs_adapter = ObservationAdapter(obs_dim, source_obs_dim)
                   self.obs_adapters.append(obs_adapter)
           
           # Attention modules for each lateral connection point
           self.attention_feature = AttentionModule(hidden_size, len(assigned_policy_indices))
           self.attention_gru = AttentionModule(hidden_size, len(assigned_policy_indices))
           self.attention_policy = AttentionModule(hidden_size, len(assigned_policy_indices))
           
           # Linear layers for integrating lateral connections
           num_assigned = len(assigned_policy_indices)
           self.lateral_feature_integration = nn.Linear(hidden_size * num_assigned, hidden_size)
           self.lateral_gru_integration = nn.Linear(hidden_size * num_assigned, hidden_size)
           self.lateral_policy_integration = nn.Linear(hidden_size * num_assigned, hidden_size)
       
       # Freeze source policies but keep adapters trainable
       if self.source_policies:
           for policy in self.source_policies:
               for param in policy.parameters():
                   param.requires_grad = False
   
   def get_adapted_source_features(self, obs):
       """Get features from source policies with input adaptation"""
       source_features = []
       source_gru_outputs = []
       source_policy_hiddens = []
       
       for i, policy_idx in enumerate(self.assigned_policy_indices):
           if policy_idx < len(self.source_policies) and i < len(self.obs_adapters):
               with torch.no_grad():
                   # Adapt observation for source policy
                   adapted_obs = self.obs_adapters[i](obs)
                   
                   # Get source policy features
                   source_policy = self.source_policies[policy_idx]
                   source_feature = source_policy.feature_net(adapted_obs)
                   
                   # Get GRU output
                   batch_size = adapted_obs.shape[0]
                   source_feature_reshaped = source_feature.view(batch_size, 1, -1)
                   source_gru_out, _ = source_policy.gru(source_feature_reshaped)
                   source_gru_output = source_gru_out.squeeze(1)
                   
                   # Get penultimate layer features
                   penultimate_features = source_policy.policy_head[1](source_policy.policy_head[0](source_gru_output))
               
               # Store all levels of features
               source_features.append(source_feature.detach())
               source_gru_outputs.append(source_gru_output.detach())
               source_policy_hiddens.append(penultimate_features.detach())
       
       return source_features, source_gru_outputs, source_policy_hiddens
   
   def forward(self, obs, hidden_state=None, seq_len=1):
       batch_size = obs.shape[0] // seq_len
       
       # Extract features from target agent's network
       target_features = self.feature_net(obs)
       
       # Get lateral features from source policies
       if self.source_policies and len(self.assigned_policy_indices) > 0:
           source_features, source_gru_outputs, source_policy_hiddens = self.get_adapted_source_features(obs)
           
           if source_features:
               # Apply attention weights to source features
               attention_weights = self.attention_feature(source_features)
               
               # Weight each source feature by its attention
               weighted_source_features = []
               for i, feat in enumerate(source_features):
                   weighted_feat = attention_weights[:, i:i+1] * feat
                   weighted_source_features.append(weighted_feat)
               
               # Concatenate and integrate
               concat_weighted_features = torch.cat(weighted_source_features, dim=-1)
               lateral_features = self.lateral_feature_integration(concat_weighted_features)
               
               # Add to target features
               combined_features = target_features + lateral_features
           else:
               combined_features = target_features
               source_gru_outputs = []
               source_policy_hiddens = []
       else:
           combined_features = target_features
           source_gru_outputs = []
           source_policy_hiddens = []
       
       # Reshape for GRU
       if seq_len > 1:
           combined_features = combined_features.view(batch_size, seq_len, -1)
       else:
           combined_features = combined_features.view(batch_size, 1, -1)
       
       # Pass through GRU
       gru_out, new_hidden = self.gru(combined_features, hidden_state)
       
       # Get GRU output for policy head
       if seq_len > 1:
           gru_output = gru_out[:, -1, :]
       else:
           gru_output = gru_out.squeeze(1)
       
       # Apply lateral connections at GRU level
       if self.source_policies and len(self.assigned_policy_indices) > 0:
           if source_gru_outputs:
               attention_weights_gru = self.attention_gru(source_gru_outputs)
               
               weighted_source_gru = []
               for i, output in enumerate(source_gru_outputs):
                   weighted_output = attention_weights_gru[:, i:i+1] * output
                   weighted_source_gru.append(weighted_output)
               
               concat_weighted_gru = torch.cat(weighted_source_gru, dim=-1)
               lateral_gru = self.lateral_gru_integration(concat_weighted_gru)
               
               updated_gru_output = gru_output + lateral_gru
           else:
               updated_gru_output = gru_output
       else:
           updated_gru_output = gru_output
       
       # Generate target penultimate features
       target_penultimate = self.policy_head[1](self.policy_head[0](updated_gru_output))
       
       # Apply lateral connections at penultimate layer
       if self.source_policies and len(self.assigned_policy_indices) > 0 and source_policy_hiddens:
           attention_weights_policy = self.attention_policy(source_policy_hiddens)
           
           weighted_source_policy = []
           for i, hidden in enumerate(source_policy_hiddens):
               weighted_hidden = attention_weights_policy[:, i:i+1] * hidden
               weighted_source_policy.append(weighted_hidden)
           
           concat_weighted_policy = torch.cat(weighted_source_policy, dim=-1)
           lateral_policy = self.lateral_policy_integration(concat_weighted_policy)
           
           final_penultimate_features = target_penultimate + lateral_policy
       else:
           final_penultimate_features = target_penultimate
       
       # Generate final action logits
       target_logits = self.policy_head[2](final_penultimate_features)
       final_action_probs = torch.softmax(target_logits, dim=-1)
       
       return final_action_probs, new_hidden
   
   def get_adapter_parameters(self):
       """Get parameters of all adapters for separate optimization"""
       adapter_params = []
       
       if hasattr(self, 'obs_adapters'):
           for adapter in self.obs_adapters:
               adapter_params.extend(list(adapter.parameters()))
       
       return adapter_params
   
   def init_hidden(self, batch_size=1):
       return torch.zeros(self.gru_layers, batch_size, self.hidden_size).to(device)

class CentralizedCritic(nn.Module):
   """Centralized Critic Network - same as MAPPO"""
   def __init__(self, global_state_dim, n_agents, hidden_size=256, gru_layers=1):
       super(CentralizedCritic, self).__init__()
       self.global_state_dim = global_state_dim
       self.n_agents = n_agents
       self.hidden_size = hidden_size
       self.gru_layers = gru_layers
       
       self.feature_net = nn.Sequential(
           nn.Linear(global_state_dim, hidden_size),
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
           nn.Linear(hidden_size, n_agents)
       )
       
   def forward(self, global_state, hidden_state=None, seq_len=1):
       batch_size = global_state.shape[0] // seq_len
       
       features = self.feature_net(global_state)
       
       if seq_len > 1:
           features = features.view(batch_size, seq_len, -1)
       else:
           features = features.view(batch_size, 1, -1)
       
       gru_out, new_hidden = self.gru(features, hidden_state)
       
       if seq_len > 1:
           value_input = gru_out[:, -1, :]
       else:
           value_input = gru_out.squeeze(1)
       
       values = self.value_head(value_input)
       
       return values, new_hidden
   
   def init_hidden(self, batch_size=1):
       return torch.zeros(self.gru_layers, batch_size, self.hidden_size).to(device)

class MALTAgent:
   """MALT Agent with transfer learning capabilities and learnable adapters"""
   def __init__(self, agent_id, obs_dim, global_state_dim, act_dim, n_agents,
                source_policies=None, assigned_policy_indices=None,
                source_obs_dims=None, source_act_dims=None,
                lr=3e-4, adapter_lr=1e-3, gamma=0.99, eps_clip=0.2, k_epochs=4, 
                entropy_coeff=0.01, value_coeff=0.5, hidden_size=256):
       self.agent_id = agent_id
       self.n_agents = n_agents
       self.gamma = gamma
       self.eps_clip = eps_clip
       self.k_epochs = k_epochs
       self.entropy_coeff = entropy_coeff
       self.value_coeff = value_coeff
       self.hidden_size = hidden_size
       
       # MALT policy network with lateral connections and adapters
       self.policy_net = MALTActor(
           obs_dim, act_dim, hidden_size,
           source_policies=source_policies,
           assigned_policy_indices=assigned_policy_indices,
           source_obs_dims=source_obs_dims,
           source_act_dims=source_act_dims
       ).to(device)
       
       # Standard centralized critic
       self.critic_net = CentralizedCritic(global_state_dim, n_agents, hidden_size).to(device)
       
       # Separate optimizer for main policy parameters (excluding adapters)
       main_params = [param for name, param in self.policy_net.named_parameters() 
                     if not any(adapter_name in name for adapter_name in ['obs_adapters'])]
       
       self.policy_optimizer = optim.Adam(main_params, lr=lr)
       self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=lr)
       
       # Separate optimizer for adapters
       adapter_params = self.policy_net.get_adapter_parameters()
       if adapter_params:
           self.adapter_optimizer = optim.Adam(adapter_params, lr=adapter_lr)
       else:
           self.adapter_optimizer = None
       
       # Hidden states
       self.policy_hidden = None
       self.critic_hidden = None
   
   def reset_hidden_states(self):
       self.policy_hidden = self.policy_net.init_hidden(1)
       self.critic_hidden = self.critic_net.init_hidden(1)
   
   def select_action(self, obs, global_state, avail_actions_mask=None):
       obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
       global_state_tensor = torch.FloatTensor(global_state).unsqueeze(0).to(device)
       
       with torch.no_grad():
           # Get action probabilities from MALT policy
           action_probs, self.policy_hidden = self.policy_net(obs_tensor, self.policy_hidden)
           
           # Get value from centralized critic
           all_values, self.critic_hidden = self.critic_net(global_state_tensor, self.critic_hidden)
           agent_value = all_values[0, self.agent_id]
       
       # Apply action masking
       if avail_actions_mask is not None:
           if not isinstance(avail_actions_mask, torch.Tensor):
               avail_actions_mask = torch.FloatTensor(avail_actions_mask).unsqueeze(0).to(device)
           else:
               avail_actions_mask = avail_actions_mask.to(device)
           
           if avail_actions_mask.shape != action_probs.shape:
               avail_actions_mask = avail_actions_mask.view_as(action_probs)
           
           masked_action_probs = action_probs * avail_actions_mask + 1e-10
           masked_action_probs = masked_action_probs / masked_action_probs.sum(dim=-1, keepdim=True)
           dist = Categorical(masked_action_probs)
       else:
           dist = Categorical(action_probs)
       
       action = dist.sample()
       log_prob = dist.log_prob(action)
       
       return action.item(), log_prob.item(), agent_value.item()

class MALTTrainer:
   """MALT Trainer with transfer learning from source policies and learnable adapters"""
   def __init__(self, map_name="3m", source_map="8m", seed=42, hidden_size=256,
                source_model_path=None, policy_assignments_path=None, adapter_lr=1e-3,
                auto_assign_policies=True):
       self.map_name = map_name
       self.source_map = source_map
       self.seed = seed
       self.hidden_size = hidden_size
       self.adapter_lr = adapter_lr
       self.auto_assign_policies = auto_assign_policies
       
       # Initialize target environment
       self.env = StarCraft2Env(map_name=map_name, seed=seed)
       self.env_info = self.env.get_env_info()
       
       self.n_agents = self.env_info["n_agents"]
       self.n_actions = self.env_info["n_actions"]
       self.obs_dim = self.env_info["obs_shape"]
       self.global_state_dim = self.env_info["state_shape"]
       
       print(f"Target Environment: {map_name}")
       print(f"Number of agents: {self.n_agents}")
       print(f"Observation dimension: {self.obs_dim}")
       print(f"Global state dimension: {self.global_state_dim}")
       print(f"Number of actions: {self.n_actions}")
       
       # Load source policies and their dimensions
       self.source_policies = []
       self.source_obs_dims = []
       self.source_act_dims = []
       
       if source_model_path:
           self.source_policies, self.source_obs_dims, self.source_act_dims = self.load_source_policies(source_model_path)
           print(f"Loaded {len(self.source_policies)} source policies from {source_map}")
           print(f"Source obs dims: {self.source_obs_dims}")
           print(f"Source act dims: {self.source_act_dims}")
       
       # Handle policy assignments
       self.policy_assignments = {}
       
       if policy_assignments_path and os.path.exists(policy_assignments_path):
           # Load from file
           try:
               with open(policy_assignments_path, 'r') as f:
                   assignment_data = json.load(f)
               
               self.policy_assignments = {
                   int(k): v for k, v in assignment_data['policy_assignments'].items()
               }
               
               print(f"Loaded policy assignments from {policy_assignments_path}")
               print("Policy assignments:")
               for agent_id, assigned in self.policy_assignments.items():
                   print(f"  Agent {agent_id}: policies {assigned}")
                   
           except Exception as e:
               print(f"Error loading policy assignments: {e}")
               self.policy_assignments = {}
       
       elif self.auto_assign_policies and self.source_policies:
           # Auto-generate assignments using PolicyAssignment
           print("Auto-generating policy assignments using MALT algorithm...")
           
           try:
               policy_assigner = PolicyAssignment(
                   n_target_agents=self.n_agents,
                   n_policies_per_agent=3,
                   n_value_dimensions=5,
                   random_seed=seed
               )
               
               # Run the full MALT policy assignment
               assignments, _, _ = policy_assigner.run_sequential_cluster_assignment(
                   source_model_path, source_map, map_name
               )
               
               self.policy_assignments = assignments
               
               print("Auto-generated policy assignments:")
               for agent_id, assigned in self.policy_assignments.items():
                   print(f"  Agent {agent_id}: policies {assigned}")
               
           except Exception as e:
               print(f"Error in auto policy assignment: {e}")
               print("Falling back to random assignment...")
               self.policy_assignments = {}
       
       # Fallback: random assignment if needed
       if not self.policy_assignments and self.source_policies:
           print("Using random fallback assignment...")
           for i in range(self.n_agents):
               available_policies = list(range(len(self.source_policies)))
               if len(available_policies) >= 3:
                   assigned = np.random.choice(available_policies, size=3, replace=False).tolist()
               else:
                   assigned = (available_policies * 3)[:3]
               
               self.policy_assignments[i] = assigned
               print(f"  Agent {i}: policies {assigned} (random)")
       
       # Initialize MALT agents
       self.agents = []
       for i in range(self.n_agents):
           assigned_indices = self.policy_assignments.get(i, [])
           agent = MALTAgent(
               agent_id=i,
               obs_dim=self.obs_dim,
               global_state_dim=self.global_state_dim,
               act_dim=self.n_actions,
               n_agents=self.n_agents,
               source_policies=self.source_policies,
               assigned_policy_indices=assigned_indices,
               source_obs_dims=self.source_obs_dims,
               source_act_dims=self.source_act_dims,
               adapter_lr=self.adapter_lr,
               hidden_size=hidden_size
           )
           self.agents.append(agent)
       
       # Training components
       self.buffer_size = 1024
       self.clear_buffer()
       
       # Training metrics
       self.timesteps = []
       self.episode_rewards = []
       self.win_rates = []
       self.policy_losses = []
       self.critic_losses = []
       self.adapter_losses = []
       self.total_timesteps = 0
   
   def load_source_policies(self, source_model_path):
       """Load source policies from saved MAPPO models with dimension info"""
       source_policies = []
       source_obs_dims = []
       source_act_dims = []
       
       # Find all agent model files
       model_files = glob.glob(f"{source_model_path}*_agent_*.pth")
       model_files.sort()
       
       print(f"Found {len(model_files)} source model files")
       
       for model_file in model_files:
           try:
               checkpoint = torch.load(model_file, map_location=device)
               
               # Get source dimensions
               source_obs_dim = checkpoint.get('obs_dim', self.obs_dim)
               source_act_dim = checkpoint.get('n_actions', self.n_actions)
               
               # Create actor network with source dimensions
               from mappo_baseline_script import Actor
               
               source_actor = Actor(source_obs_dim, source_act_dim, self.hidden_size).to(device)
               source_actor.load_state_dict(checkpoint['policy_state_dict'])
               source_actor.eval()
               
               source_policies.append(source_actor)
               source_obs_dims.append(source_obs_dim)
               source_act_dims.append(source_act_dim)
               
               print(f"Loaded source policy from {model_file}")
               print(f"  - Obs dim: {source_obs_dim}, Act dim: {source_act_dim}")
               
           except Exception as e:
               print(f"Error loading {model_file}: {e}")
       
       return source_policies, source_obs_dims, source_act_dims
   
   def clear_buffer(self):
       """Clear experience buffer"""
       self.buffer = {
           'obs': [],
           'global_states': [],
           'actions': [],
           'rewards': [],
           'dones': [],
           'log_probs': [],
           'values': [],
           'avail_actions': []
       }
   
   def add_experience(self, obs_list, global_state, actions, reward, done, 
                     log_probs, values, avail_actions):
       """Add experience to buffer"""
       self.buffer['obs'].append(obs_list)
       self.buffer['global_states'].append(global_state)
       self.buffer['actions'].append(actions)
       self.buffer['rewards'].append(reward)
       self.buffer['dones'].append(done)
       self.buffer['log_probs'].append(log_probs)
       self.buffer['values'].append(values)
       self.buffer['avail_actions'].append(avail_actions)
   
   def compute_gae_returns(self):
       """Compute GAE advantages and returns"""
       if len(self.buffer['rewards']) == 0:
           return None, None
       
       rewards = np.array(self.buffer['rewards'])
       values = np.array(self.buffer['values'])
       dones = np.array(self.buffer['dones'])
       
       gamma = 0.99
       gae_lambda = 0.95
       
       advantages = np.zeros_like(values)
       returns = np.zeros_like(values)
       
       for agent_id in range(self.n_agents):
           agent_values = values[:, agent_id]
           gae = 0
           
           for t in reversed(range(len(rewards))):
               if t == len(rewards) - 1:
                   next_value = 0
               else:
                   next_value = agent_values[t + 1]
               
               delta = rewards[t] + gamma * next_value * (1 - dones[t]) - agent_values[t]
               gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
               
               advantages[t, agent_id] = gae
               returns[t, agent_id] = gae + agent_values[t]
       
       return advantages, returns
   
   def collect_episode(self):
       """Collect one episode of experience"""
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
               action, log_prob, value = agent.select_action(
                   obs_list[i], global_state, avail_mask
               )
               
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
           
           reward, done, info = self.env.step(actions)
           episode_reward += reward
           episode_timesteps += 1
           self.total_timesteps += 1
           
           self.add_experience(
               obs_list, global_state, actions, reward, done,
               log_probs, values, avail_actions
           )
           
           if done:
               break
       
       return episode_reward, info.get('battle_won', False), episode_timesteps
   
   def update_policy(self):
       """Update policies using MALT approach - FIXED VERSION"""
       if len(self.buffer['rewards']) == 0:
           return 0.0, 0.0, 0.0
       
       advantages, returns = self.compute_gae_returns()
       if advantages is None:
           return 0.0, 0.0, 0.0
       
       advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
       
       total_policy_loss = 0
       total_critic_loss = 0
       total_adapter_loss = 0
       
       for agent_id, agent in enumerate(self.agents):
           # Prepare data
           agent_obs = [step_obs[agent_id] for step_obs in self.buffer['obs']]
           agent_actions = [step_actions[agent_id] for step_actions in self.buffer['actions']]	
           agent_old_log_probs = [step_log_probs[agent_id] for step_log_probs in self.buffer['log_probs']]
           agent_advantages = advantages[:, agent_id]
           agent_returns = returns[:, agent_id]
           agent_avail_actions = [step_avail[agent_id] for step_avail in self.buffer['avail_actions']]
           
           # PPO updates with separate forward passes
           for epoch in range(agent.k_epochs):
               seq_len = len(agent_obs)
               
               # ===========================================
               # POLICY UPDATE - Fresh forward pass
               # ===========================================
               agent.policy_optimizer.zero_grad()
               
               policy_hidden = agent.policy_net.init_hidden(1)
               all_action_probs = []
               
               for i in range(seq_len):
                   obs_i = torch.FloatTensor(agent_obs[i]).unsqueeze(0).to(device)
                   avail_i = torch.FloatTensor(agent_avail_actions[i]).unsqueeze(0).to(device)
                   
                   action_probs, policy_hidden = agent.policy_net(obs_i, policy_hidden)
                   
                   if avail_i is not None:
                       masked_probs = action_probs * avail_i + 1e-10
                       masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True)
                       all_action_probs.append(masked_probs)
                   else:
                       all_action_probs.append(action_probs)
               
               action_probs_tensor = torch.cat(all_action_probs, dim=0)
               action_probs_tensor = torch.clamp(action_probs_tensor, min=1e-8)
               
               actions_tensor = torch.LongTensor(agent_actions).to(device)
               old_log_probs_tensor = torch.FloatTensor(agent_old_log_probs).to(device)
               advantages_tensor = torch.FloatTensor(agent_advantages).to(device)
               
               dist = Categorical(action_probs_tensor)
               new_log_probs = dist.log_prob(actions_tensor)
               entropy = dist.entropy()
               
               # PPO policy loss
               ratio = torch.exp(new_log_probs - old_log_probs_tensor)
               surr1 = ratio * advantages_tensor
               surr2 = torch.clamp(ratio, 1 - agent.eps_clip, 1 + agent.eps_clip) * advantages_tensor
               policy_loss = -torch.min(surr1, surr2).mean() - agent.entropy_coeff * entropy.mean()
               
               # Update policy network
               policy_loss.backward()
               torch.nn.utils.clip_grad_norm_(agent.policy_net.parameters(), 0.5)
               agent.policy_optimizer.step()
               
               # ===========================================
               # CRITIC UPDATE - Separate forward pass
               # ===========================================
               agent.critic_optimizer.zero_grad()
               
               critic_hidden = agent.critic_net.init_hidden(1)
               all_values = []
               
               for i in range(seq_len):
                   global_state_i = torch.FloatTensor(self.buffer['global_states'][i]).unsqueeze(0).to(device)
                   values, critic_hidden = agent.critic_net(global_state_i, critic_hidden)
                   agent_value = values[:, agent_id]
                   all_values.append(agent_value)
               
               values_tensor = torch.cat(all_values, dim=0)
               returns_tensor = torch.FloatTensor(agent_returns).to(device)
               critic_loss = nn.MSELoss()(values_tensor, returns_tensor)
               
               # Update critic
               critic_loss.backward()
               torch.nn.utils.clip_grad_norm_(agent.critic_net.parameters(), 0.5)
               agent.critic_optimizer.step()
               
               # ===========================================
               # ADAPTER UPDATE - Fresh forward pass if needed
               # ===========================================
               adapter_loss = 0.0
               if agent.adapter_optimizer is not None:
                   agent.adapter_optimizer.zero_grad()
                   
                   # Completely fresh forward pass for adapters
                   policy_hidden_adapter = agent.policy_net.init_hidden(1)
                   all_action_probs_adapter = []
                   
                   for i in range(seq_len):
                       obs_i = torch.FloatTensor(agent_obs[i]).unsqueeze(0).to(device)
                       avail_i = torch.FloatTensor(agent_avail_actions[i]).unsqueeze(0).to(device)
                       
                       action_probs, policy_hidden_adapter = agent.policy_net(obs_i, policy_hidden_adapter)
                       
                       if avail_i is not None:
                           masked_probs = action_probs * avail_i + 1e-10
                           masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True)
                           all_action_probs_adapter.append(masked_probs)
                       else:
                           all_action_probs_adapter.append(action_probs)
                   
                   action_probs_adapter = torch.cat(all_action_probs_adapter, dim=0)
                   action_probs_adapter = torch.clamp(action_probs_adapter, min=1e-8)
                   
                   # Fresh tensors for adapter update
                   actions_adapter = torch.LongTensor(agent_actions).to(device)
                   old_log_probs_adapter = torch.FloatTensor(agent_old_log_probs).to(device)
                   advantages_adapter = torch.FloatTensor(agent_advantages).to(device)
                   
                   dist_adapter = Categorical(action_probs_adapter)
                   new_log_probs_adapter = dist_adapter.log_prob(actions_adapter)
                   entropy_adapter = dist_adapter.entropy()
                   
                   ratio_adapter = torch.exp(new_log_probs_adapter - old_log_probs_adapter)
                   surr1_adapter = ratio_adapter * advantages_adapter
                   surr2_adapter = torch.clamp(ratio_adapter, 1 - agent.eps_clip, 1 + agent.eps_clip) * advantages_adapter
                   adapter_loss = -torch.min(surr1_adapter, surr2_adapter).mean() - agent.entropy_coeff * entropy_adapter.mean()
                   
                   # Update adapters
                   adapter_loss.backward()
                   torch.nn.utils.clip_grad_norm_(agent.policy_net.get_adapter_parameters(), 0.5)
                   agent.adapter_optimizer.step()
               
               total_policy_loss += policy_loss.item()
               total_critic_loss += critic_loss.item()
               total_adapter_loss += adapter_loss.item() if isinstance(adapter_loss, torch.Tensor) else adapter_loss
       
       self.clear_buffer()
       
       return (total_policy_loss / (self.n_agents * self.agents[0].k_epochs), 
               total_critic_loss / (self.n_agents * self.agents[0].k_epochs),
               total_adapter_loss / (self.n_agents * self.agents[0].k_epochs))
   
   
   
   def train(self, max_episodes=2000, max_timesteps=None, episodes_per_update=25,
             log_frequency=5000, save_frequency=100000):
       """Train MALT agents"""
       episode_rewards_buffer = deque(maxlen=100)
       win_buffer = deque(maxlen=100)
       
       episode = 0
       last_log_timestep = 0
       last_save_timestep = 0
       
       print("Starting MALT training...")
       if self.source_policies:
           print(f"Using {len(self.source_policies)} source policies from {self.source_map}")
           print("Policy assignments:")
           for agent_id, assigned in self.policy_assignments.items():
               print(f"  Agent {agent_id}: policies {assigned}")
       else:
           print("No source policies loaded - training from scratch")
       
       while True:
           # Check termination conditions
           if max_timesteps and self.total_timesteps >= max_timesteps:
               print(f"Reached maximum timesteps: {max_timesteps}")
               break
           if not max_timesteps and episode >= max_episodes:
               print(f"Reached maximum episodes: {max_episodes}")
               break
           
           try:
               # Collect episodes for batch update
               batch_rewards = []
               batch_wins = []
               
               for _ in range(episodes_per_update):
                   ep_reward, won, ep_timesteps = self.collect_episode()
                   batch_rewards.append(ep_reward)
                   batch_wins.append(won)
                   episode += 1
                   
                   if max_timesteps and self.total_timesteps >= max_timesteps:
                       break
               
               episode_rewards_buffer.extend(batch_rewards)
               win_buffer.extend(batch_wins)
               
               # Update policies
               policy_loss, critic_loss, adapter_loss = self.update_policy()
               self.policy_losses.append(policy_loss)
               self.critic_losses.append(critic_loss)
               self.adapter_losses.append(adapter_loss)
               
           except Exception as e:
               print(f"Error in training batch: {e}")
               continue
           
           # Log progress
           if self.total_timesteps - last_log_timestep >= log_frequency:
               if len(episode_rewards_buffer) > 0:
                   avg_reward = np.mean(list(episode_rewards_buffer))
                   win_rate = np.mean(list(win_buffer))
                   
                   self.timesteps.append(self.total_timesteps)
                   self.episode_rewards.append(avg_reward)
                   self.win_rates.append(win_rate)
                   
                   print(f"Timestep {self.total_timesteps:7d} | "
                         f"Episode {episode:4d} | "
                         f"Avg Reward: {avg_reward:6.2f} | "
                         f"Win Rate: {win_rate:4.2f} | "
                         f"Policy Loss: {policy_loss:6.4f} | "
                         f"Critic Loss: {critic_loss:6.4f} | "
                         f"Adapter Loss: {adapter_loss:6.4f}")
                   
               
                   
                   last_log_timestep = self.total_timesteps
           
           # Save models
           if self.total_timesteps - last_save_timestep >= save_frequency:
               self.save_models(f"malt_{self.map_name}_from_{self.source_map}_timestep_{self.total_timesteps}")
               last_save_timestep = self.total_timesteps
               
               # Save best model if high win rate
               if len(win_buffer) > 0 and np.mean(list(win_buffer)) > 0.8:
                   self.save_models(f"malt_{self.map_name}_from_{self.source_map}_80%_win")
       
       self.env.close()
   
   def save_models(self, filename):
       """Save MALT models and training history"""
       save_dir = Path(f"models_{self.map_name}_malt")
       save_dir.mkdir(exist_ok=True)
       
       for i, agent in enumerate(self.agents):
           torch.save({
               'policy_state_dict': agent.policy_net.state_dict(),
               'critic_state_dict': agent.critic_net.state_dict(),
               'policy_optimizer': agent.policy_optimizer.state_dict(),
               'critic_optimizer': agent.critic_optimizer.state_dict(),
               'adapter_optimizer': agent.adapter_optimizer.state_dict() if agent.adapter_optimizer else None,
               'agent_id': i,
               'map_name': self.map_name,
               'source_map': self.source_map,
               'timesteps': self.total_timesteps,
               'n_agents': self.n_agents,
               'n_actions': self.n_actions,
               'obs_dim': self.obs_dim,
               'global_state_dim': self.global_state_dim,
               'algorithm': 'MALT',
               'assigned_policy_indices': self.policy_assignments.get(i, []),
               'has_source_policies': len(self.source_policies) > 0
           }, save_dir / f"{filename}_agent_{i}.pth")
       
       # Save training history
       training_history = {
           'timesteps': self.timesteps,
           'episode_rewards': self.episode_rewards,
           'win_rates': self.win_rates,
           'policy_losses': self.policy_losses,
           'critic_losses': self.critic_losses,
           'adapter_losses': self.adapter_losses,
           'total_timesteps': self.total_timesteps,
           'policy_assignments': self.policy_assignments
       }
       
       with open(save_dir / f"{filename}_training_history.json", 'w') as f:
           json.dump(training_history, f, indent=4)
       
       print(f"MALT models saved: {filename}")
   
   def load_models(self, filename):
       """Load MALT models and training history"""
       try:
           for i, agent in enumerate(self.agents):
               checkpoint = torch.load(f"models_{self.map_name}_malt/{filename}_agent_{i}.pth", map_location=device)
               agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
               agent.critic_net.load_state_dict(checkpoint['critic_state_dict'])
               agent.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
               agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
               
               if checkpoint.get('adapter_optimizer') and agent.adapter_optimizer:
                  agent.adapter_optimizer.load_state_dict(checkpoint['adapter_optimizer'])
               if 'timesteps' in checkpoint:
                   self.total_timesteps = checkpoint['timesteps']
           
           # Load training history
           try:
               with open(f"models_{self.map_name}_malt/{filename}_training_history.json", 'r') as f:
                   training_history = json.load(f)
               
               self.timesteps = training_history.get('timesteps', [])
               self.episode_rewards = training_history.get('episode_rewards', [])
               self.win_rates = training_history.get('win_rates', [])
               self.policy_losses = training_history.get('policy_losses', [])
               self.critic_losses = training_history.get('critic_losses', [])
               self.adapter_losses = training_history.get('adapter_losses', [])
               
               print(f"Training history loaded: {len(self.timesteps)} data points")
               
           except FileNotFoundError:
               print("Training history file not found - starting fresh metrics")
           
           print(f"MALT models loaded: {filename}")
           print(f"Resuming from timestep: {self.total_timesteps}")
           
       except Exception as e:
           print(f"Error loading MALT models: {e}")
           print("Starting training from scratch...")
           self.total_timesteps = 0
   
   def plot_training_curves(self):
       """Plot MALT training progress"""
       fig, axes = plt.subplots(2, 3, figsize=(18, 10))
       
       # Episode rewards vs timesteps
       if self.timesteps and self.episode_rewards:
           axes[0, 0].plot(self.timesteps, self.episode_rewards, label=f'MALT {self.map_name}', color='blue')
           axes[0, 0].set_title(f'Average Episode Rewards vs Timesteps\n(MALT: {self.source_map}→{self.map_name})')
           axes[0, 0].set_xlabel('Timesteps')
           axes[0, 0].set_ylabel('Reward')
           axes[0, 0].legend()
           axes[0, 0].grid(True)
       
       # Win rates vs timesteps
       if self.timesteps and self.win_rates:
           axes[0, 1].plot(self.timesteps, self.win_rates, label=f'MALT {self.map_name}', color='green')
           axes[0, 1].set_title(f'Win Rate vs Timesteps\n(MALT: {self.source_map}→{self.map_name})')
           axes[0, 1].set_xlabel('Timesteps')
           axes[0, 1].set_ylabel('Win Rate')
           axes[0, 1].legend()
           axes[0, 1].grid(True)
       
       # Policy losses
       if self.policy_losses:
           axes[0, 2].plot(self.policy_losses, color='orange')
           axes[0, 2].set_title('Policy Loss vs Updates (MALT)')
           axes[0, 2].set_xlabel('Update Steps')
           axes[0, 2].set_ylabel('Policy Loss')
           axes[0, 2].grid(True)
       
       # Critic losses
       if self.critic_losses:
           axes[1, 0].plot(self.critic_losses, color='red')
           axes[1, 0].set_title('Critic Loss vs Updates (MALT)')
           axes[1, 0].set_xlabel('Update Steps')
           axes[1, 0].set_ylabel('Critic Loss')
           axes[1, 0].grid(True)
       
       # Adapter losses
       if self.adapter_losses:
           axes[1, 1].plot(self.adapter_losses, color='purple')
           axes[1, 1].set_title('Adapter Loss vs Updates (MALT)')
           axes[1, 1].set_xlabel('Update Steps')
           axes[1, 1].set_ylabel('Adapter Loss')
           axes[1, 1].grid(True)
       
       # Transfer learning status
       if self.source_policies:
           axes[1, 2].text(0.5, 0.5, f'Transfer Learning\nEnabled\n{len(self.source_policies)} Source Policies', 
                          ha='center', va='center', transform=axes[1, 2].transAxes, fontsize=14, 
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
           axes[1, 2].set_title('Transfer Learning Status')
           axes[1, 2].set_xticks([])
           axes[1, 2].set_yticks([])
       
       plt.tight_layout()
       plt.savefig(f'malt_{self.map_name}_from_{self.source_map}_training_curves.png', dpi=300, bbox_inches='tight')
       plt.show()
   
   def evaluate(self, num_episodes=100):
       """Evaluate trained MALT agents"""
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
       
       print(f"MALT Evaluation Results ({self.source_map}→{self.map_name}):")
       print(f"Average Reward: {avg_reward:.2f}")
       print(f"Win Rate: {win_rate:.2f}")
       print(f"Average Episode Length: {avg_episode_length:.1f} timesteps")
       print(f"Total Evaluation Timesteps: {total_eval_timesteps}")
       print(f"Source policies used: {len(self.source_policies)}")
       
       if self.source_policies:
           print("Policy assignments:")
           for agent_id, assigned in self.policy_assignments.items():
               print(f"  Agent {agent_id}: policies {assigned}")
       
       return avg_reward, win_rate

def main():
   parser = argparse.ArgumentParser(description='MALT Training for SMAC with Integrated Policy Assignment')
   parser.add_argument('--map', type=str, default='3m', help='Target SMAC map name')
   parser.add_argument('--source_map', type=str, default='8m', help='Source SMAC map name')
   parser.add_argument('--source_model_path', type=str, help='Path to source model files (without agent suffix)')
   parser.add_argument('--policy_assignments', type=str, help='Path to policy assignments JSON file (optional)')
   parser.add_argument('--auto_assign', action='store_true', default=True, help='Auto-generate policy assignments')
   parser.add_argument('--episodes', type=int, default=1500, help='Number of training episodes')
   parser.add_argument('--timesteps', type=int, default=500000, help='Number of training timesteps')
   parser.add_argument('--episodes_per_update', type=int, default=25, help='Episodes per update')
   parser.add_argument('--log_freq', type=int, default=5000, help='Logging frequency')
   parser.add_argument('--save_freq', type=int, default=50000, help='Save frequency')
   parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size')
   parser.add_argument('--adapter_lr', type=float, default=1e-3, help='Learning rate for adapters')
   parser.add_argument('--eval_only', action='store_true', help='Only evaluate')
   parser.add_argument('--load_model', type=str, help='Load MALT model filename')
   parser.add_argument('--eval_episodes', type=int, default=100, help='Evaluation episodes')
   parser.add_argument('--seed', type=int, default=42, help='Random seed')
   
   args = parser.parse_args()
   
   # Set random seeds
   torch.manual_seed(args.seed)
   np.random.seed(args.seed)
   
   # Initialize MALT trainer with integrated policy assignment
   trainer = MALTTrainer(
       map_name=args.map,
       source_map=args.source_map,
       seed=args.seed,
       hidden_size=args.hidden_size,
       source_model_path=args.source_model_path,
       policy_assignments_path=args.policy_assignments,
       adapter_lr=args.adapter_lr,
       auto_assign_policies=args.auto_assign
   )
   
   if args.eval_only:
       if args.load_model:
           trainer.load_models(args.load_model)
       trainer.evaluate(num_episodes=args.eval_episodes)
   else:
       # Load model if specified
       if args.load_model:
           trainer.load_models(args.load_model)
       
       # Train agents
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
       print(f"\nFinal MALT Evaluation ({args.source_map}→{args.map}):")
       trainer.evaluate(num_episodes=args.eval_episodes)

if __name__ == "__main__":
   main()
