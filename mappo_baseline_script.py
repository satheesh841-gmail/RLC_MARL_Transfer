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

# GPU Device Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class Actor(nn.Module):
    """Decentralized Policy Network (same as IPPO)"""
    def __init__(self, obs_dim, act_dim, hidden_size=256, gru_layers=1):
        super(Actor, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.gru_layers = gru_layers
        
        # Feature extraction layers
        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # GRU for handling partial observability
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=gru_layers,
            batch_first=True
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim)
        )
        
    def forward(self, obs, hidden_state=None, seq_len=1):
        batch_size = obs.shape[0] // seq_len
        
        # Extract features
        features = self.feature_net(obs)
        
        # Reshape for GRU (batch_size, seq_len, hidden_size)
        if seq_len > 1:
            features = features.view(batch_size, seq_len, -1)
        else:
            features = features.view(batch_size, 1, -1)
        
        # Pass through GRU
        gru_out, new_hidden = self.gru(features, hidden_state)
        
        # Use last output for policy
        if seq_len > 1:
            policy_input = gru_out[:, -1, :]  # Take last timestep
        else:
            policy_input = gru_out.squeeze(1)
        
        # Generate action logits
        logits = self.policy_head(policy_input)
        action_probs = torch.softmax(logits, dim=-1)
        
        return action_probs, new_hidden
    
    def init_hidden(self, batch_size=1):
        return torch.zeros(self.gru_layers, batch_size, self.hidden_size).to(device)

class CentralizedCritic(nn.Module):
    """Centralized Critic Network - takes global state"""
    def __init__(self, global_state_dim, n_agents, hidden_size=256, gru_layers=1):
        super(CentralizedCritic, self).__init__()
        self.global_state_dim = global_state_dim
        self.n_agents = n_agents
        self.hidden_size = hidden_size
        self.gru_layers = gru_layers
        
        # Feature extraction layers for global state
        self.feature_net = nn.Sequential(
            nn.Linear(global_state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # GRU for handling temporal dependencies
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=gru_layers,
            batch_first=True
        )
        
        # Value head - outputs value for each agent
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_agents)  # One value per agent
        )
        
    def forward(self, global_state, hidden_state=None, seq_len=1):
        batch_size = global_state.shape[0] // seq_len
        
        # Extract features
        features = self.feature_net(global_state)
        
        # Reshape for GRU (batch_size, seq_len, hidden_size)
        if seq_len > 1:
            features = features.view(batch_size, seq_len, -1)
        else:
            features = features.view(batch_size, 1, -1)
        
        # Pass through GRU
        gru_out, new_hidden = self.gru(features, hidden_state)
        
        # Use last output for value estimation
        if seq_len > 1:
            value_input = gru_out[:, -1, :]  # Take last timestep
        else:
            value_input = gru_out.squeeze(1)
        
        # Generate value estimates for all agents
        values = self.value_head(value_input)  # [batch_size, n_agents]
        
        return values, new_hidden
    
    def init_hidden(self, batch_size=1):
        return torch.zeros(self.gru_layers, batch_size, self.hidden_size).to(device)

class MAPPOAgent:
    """MAPPO Agent with decentralized policy and centralized critic"""
    def __init__(self, agent_id, obs_dim, global_state_dim, act_dim, n_agents, 
                 lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4, 
                 entropy_coeff=0.01, value_coeff=0.5, hidden_size=256):
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.hidden_size = hidden_size
        
        # Decentralized policy network (agent-specific)
        self.policy_net = Actor(obs_dim, act_dim, hidden_size).to(device)
        
        # Centralized critic network (shared knowledge)
        self.critic_net = CentralizedCritic(global_state_dim, n_agents, hidden_size).to(device)
        
        # Separate optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=lr)
        
        # Hidden states for GRU
        self.policy_hidden = None
        self.critic_hidden = None
    
    def reset_hidden_states(self):
        """Reset hidden states at the beginning of each episode"""
        self.policy_hidden = self.policy_net.init_hidden(1)
        self.critic_hidden = self.critic_net.init_hidden(1)
    
    def select_action(self, obs, global_state, avail_actions_mask=None):
        """Select action using local observation and get value using global state"""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        global_state_tensor = torch.FloatTensor(global_state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Get action probabilities from decentralized policy
            action_probs, self.policy_hidden = self.policy_net(obs_tensor, self.policy_hidden)
            
            # Get value estimates from centralized critic
            all_values, self.critic_hidden = self.critic_net(global_state_tensor, self.critic_hidden)
            agent_value = all_values[0, self.agent_id]  # Get this agent's value
        
        # Apply action masking if available
        if avail_actions_mask is not None:
            if not isinstance(avail_actions_mask, torch.Tensor):
                avail_actions_mask = torch.FloatTensor(avail_actions_mask).unsqueeze(0).to(device)
            else:
                avail_actions_mask = avail_actions_mask.to(device)
            
            if avail_actions_mask.shape != action_probs.shape:
                avail_actions_mask = avail_actions_mask.view_as(action_probs)
            
            # Mask unavailable actions
            masked_action_probs = action_probs * avail_actions_mask + 1e-10
            masked_action_probs = masked_action_probs / masked_action_probs.sum(dim=-1, keepdim=True)
            dist = Categorical(masked_action_probs)
        else:
            dist = Categorical(action_probs)
        
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), agent_value.item()

class MAPPOTrainer:
    """MAPPO Trainer with centralized experience collection"""
    def __init__(self, map_name="8m", seed=42, hidden_size=256):
        self.map_name = map_name
        self.seed = seed
        self.hidden_size = hidden_size
        
        # Initialize SMAC environment
        self.env = StarCraft2Env(map_name=map_name, seed=seed)
        self.env_info = self.env.get_env_info()
        
        self.n_agents = self.env_info["n_agents"]
        self.n_actions = self.env_info["n_actions"]
        self.obs_dim = self.env_info["obs_shape"]
        self.global_state_dim = self.env_info["state_shape"]
        
        print(f"Environment: {map_name}")
        print(f"Number of agents: {self.n_agents}")
        print(f"Observation dimension: {self.obs_dim}")
        print(f"Global state dimension: {self.global_state_dim}")
        print(f"Number of actions: {self.n_actions}")
        print(f"Hidden size: {hidden_size}")
        
        # Initialize MAPPO agents
        self.agents = [MAPPOAgent(
            agent_id=i,
            obs_dim=self.obs_dim,
            global_state_dim=self.global_state_dim,
            act_dim=self.n_actions,
            n_agents=self.n_agents,
            hidden_size=hidden_size
        ) for i in range(self.n_agents)]
        
        # Shared experience buffer for MAPPO
        self.buffer_size = 1024
        self.clear_buffer()
        
        # Training metrics
        self.timesteps = []
        self.episode_rewards = []
        self.win_rates = []
        self.policy_losses = []
        self.critic_losses = []
        self.total_timesteps = 0
    
    def clear_buffer(self):
        """Clear centralized experience buffer"""
        self.buffer = {
            'obs': [],              # [timestep][agent_id] -> observation
            'global_states': [],    # [timestep] -> global state
            'actions': [],          # [timestep][agent_id] -> action
            'rewards': [],          # [timestep] -> shared reward
            'dones': [],            # [timestep] -> done flag
            'log_probs': [],        # [timestep][agent_id] -> log probability
            'values': [],           # [timestep][agent_id] -> value estimate
            'avail_actions': []     # [timestep][agent_id] -> available actions
        }
    
    def add_experience(self, obs_list, global_state, actions, reward, done, 
                      log_probs, values, avail_actions):
        """Add experience to centralized buffer"""
        self.buffer['obs'].append(obs_list)
        self.buffer['global_states'].append(global_state)
        self.buffer['actions'].append(actions)
        self.buffer['rewards'].append(reward)
        self.buffer['dones'].append(done)
        self.buffer['log_probs'].append(log_probs)
        self.buffer['values'].append(values)
        self.buffer['avail_actions'].append(avail_actions)
    
    def compute_gae_returns(self):
        """Compute GAE advantages and returns for all agents"""
        if len(self.buffer['rewards']) == 0:
            return None, None
        
        rewards = np.array(self.buffer['rewards'])
        values = np.array(self.buffer['values'])  # [timestep, agent]
        dones = np.array(self.buffer['dones'])
        
        # GAE parameters
        gamma = 0.99
        gae_lambda = 0.95
        
        # Compute advantages for each agent
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
        
        # Reset hidden states for all agents
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
            
            # Get actions from all agents
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
            
            # Execute actions in environment
            reward, done, info = self.env.step(actions)
            episode_reward += reward
            episode_timesteps += 1
            self.total_timesteps += 1
            
            # Store experience in centralized buffer
            self.add_experience(
                obs_list, global_state, actions, reward, done,
                log_probs, values, avail_actions
            )
            
            if done:
                break
        
        return episode_reward, info.get('battle_won', False), episode_timesteps
    
    def update_policy(self):
        """Update policies using collected experience (MAPPO style)"""
        if len(self.buffer['rewards']) == 0:
            return 0.0, 0.0
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae_returns()
        if advantages is None:
            return 0.0, 0.0
        
        # Normalize advantages across all agents and timesteps
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_policy_loss = 0
        total_critic_loss = 0
        
        # Update each agent
        for agent_id, agent in enumerate(self.agents):
            # Prepare data for this agent
            agent_obs = [step_obs[agent_id] for step_obs in self.buffer['obs']]
            agent_actions = [step_actions[agent_id] for step_actions in self.buffer['actions']]
            agent_old_log_probs = [step_log_probs[agent_id] for step_log_probs in self.buffer['log_probs']]
            agent_advantages = advantages[:, agent_id]
            agent_returns = returns[:, agent_id]
            agent_avail_actions = [step_avail[agent_id] for step_avail in self.buffer['avail_actions']]
            
            # Convert to tensors
            obs_tensor = torch.FloatTensor(np.array(agent_obs)).to(device)
            global_states_tensor = torch.FloatTensor(np.array(self.buffer['global_states'])).to(device)
            actions_tensor = torch.LongTensor(agent_actions).to(device)
            old_log_probs_tensor = torch.FloatTensor(agent_old_log_probs).to(device)
            advantages_tensor = torch.FloatTensor(agent_advantages).to(device)
            returns_tensor = torch.FloatTensor(agent_returns).to(device)
            avail_tensor = torch.FloatTensor(np.array(agent_avail_actions)).to(device)
            
            # PPO updates for multiple epochs
            for epoch in range(agent.k_epochs):
                seq_len = len(agent_obs)
                
                # Policy network forward pass
                policy_hidden = agent.policy_net.init_hidden(1)
                all_action_probs = []
                
                for i in range(seq_len):
                    obs_i = obs_tensor[i:i+1]
                    avail_i = avail_tensor[i:i+1]
                    
                    action_probs, policy_hidden = agent.policy_net(obs_i, policy_hidden)
                    
                    # Apply action masking
                    if avail_i is not None:
                        masked_probs = action_probs * avail_i + 1e-10
                        masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True)
                        all_action_probs.append(masked_probs)
                    else:
                        all_action_probs.append(action_probs)
                
                action_probs_tensor = torch.cat(all_action_probs, dim=0)
                action_probs_tensor = torch.clamp(action_probs_tensor, min=1e-8)
                
                dist = Categorical(action_probs_tensor)
                new_log_probs = dist.log_prob(actions_tensor)
                entropy = dist.entropy()
                
                # Critic network forward pass
                critic_hidden = agent.critic_net.init_hidden(1)
                all_values = []
                
                for i in range(seq_len):
                    global_state_i = global_states_tensor[i:i+1]
                    values, critic_hidden = agent.critic_net(global_state_i, critic_hidden)
                    agent_value = values[:, agent_id]  # Get this agent's value
                    all_values.append(agent_value)
                
                values_tensor = torch.cat(all_values, dim=0)
                
                # PPO policy loss
                ratio = torch.exp(new_log_probs - old_log_probs_tensor)
                surr1 = ratio * advantages_tensor
                surr2 = torch.clamp(ratio, 1 - agent.eps_clip, 1 + agent.eps_clip) * advantages_tensor
                policy_loss = -torch.min(surr1, surr2).mean() - agent.entropy_coeff * entropy.mean()
                
                # Critic loss
                critic_loss = nn.MSELoss()(values_tensor, returns_tensor)
                
                # Update policy
                agent.policy_optimizer.zero_grad()
                policy_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(agent.policy_net.parameters(), 0.5)
                agent.policy_optimizer.step()
                
                # Update critic
                agent.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.critic_net.parameters(), 0.5)
                agent.critic_optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_critic_loss += critic_loss.item()
        
        # Clear buffer after update
        self.clear_buffer()
        
        return total_policy_loss / (self.n_agents * self.agents[0].k_epochs), \
               total_critic_loss / (self.n_agents * self.agents[0].k_epochs)
    
    def train(self, max_episodes=2000, max_timesteps=None, episodes_per_update=25, 
              log_frequency=5000, save_frequency= 100000):
        """Train MAPPO agents"""
        episode_rewards_buffer = deque(maxlen=100)
        win_buffer = deque(maxlen=100)
        
        episode = 0
        last_log_timestep = 0
        last_save_timestep = 0
        
        print("Starting MAPPO training...")
        
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
                
                # Add to buffers
                episode_rewards_buffer.extend(batch_rewards)
                win_buffer.extend(batch_wins)
                
                # Update policies
                policy_loss, critic_loss = self.update_policy()
                self.policy_losses.append(policy_loss)
                self.critic_losses.append(critic_loss)
                
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
                          f"Critic Loss: {critic_loss:6.4f}")
                    
                    last_log_timestep = self.total_timesteps
            
            # Save models
            if self.total_timesteps - last_save_timestep >= save_frequency:
                self.save_models(f"mappo_{self.map_name}_timestep_{self.total_timesteps}")
                last_save_timestep = self.total_timesteps
                
                # Save best model if high win rate
                if len(win_buffer) > 0 and np.mean(list(win_buffer)) > 0.8:
                    self.save_models(f"mappo_{self.map_name}_80%_win")
        
        self.env.close()
    
    def save_models(self, filename):
        """Save all agent models and training history"""
        save_dir = Path(f"models_{self.map_name}")
        save_dir.mkdir(exist_ok=True)
        
        for i, agent in enumerate(self.agents):
            torch.save({
                'policy_state_dict': agent.policy_net.state_dict(),
                'critic_state_dict': agent.critic_net.state_dict(),
                'policy_optimizer': agent.policy_optimizer.state_dict(),
                'critic_optimizer': agent.critic_optimizer.state_dict(),
                'agent_id': i,
                'map_name': self.map_name,
                'timesteps': self.total_timesteps,
                'n_agents': self.n_agents,
                'n_actions': self.n_actions,
                'obs_dim': self.obs_dim,
                'global_state_dim': self.global_state_dim,
                'algorithm': 'MAPPO'
            }, save_dir / f"{filename}_agent_{i}.pth")
        
        # Save training history for complete plots
        training_history = {
            'timesteps': self.timesteps,
            'episode_rewards': self.episode_rewards,
            'win_rates': self.win_rates,
            'policy_losses': self.policy_losses,
            'critic_losses': self.critic_losses,
            'total_timesteps': self.total_timesteps
        }
        
        with open(save_dir / f"{filename}_training_history.json", 'w') as f:
            json.dump(training_history, f, indent=4)
        
        # Save training summary
        summary = {
            'map_name': self.map_name,
            'algorithm': 'MAPPO',
            'total_timesteps': self.total_timesteps,
            'n_agents': self.n_agents,
            'n_actions': self.n_actions,
            'obs_dim': self.obs_dim,
            'global_state_dim': self.global_state_dim,
            'final_win_rate': np.mean(list(self.win_rates[-10:])) if self.win_rates else 0.0,
            'final_reward': np.mean(list(self.episode_rewards[-10:])) if self.episode_rewards else 0.0
        }
        
        with open(save_dir / f"{filename}_summary.json", 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"Models saved: {filename}")
        print(f"Training history saved: {filename}_training_history.json")
    
    def load_models(self, filename):
        """Load all agent models and training history"""
        save_dir = Path(f"models_{self.map_name}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            # Load agent models
            for i, agent in enumerate(self.agents):
                checkpoint = torch.load(save_dir/f"{filename}_agent_{i}.pth", map_location=device)
                agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
                agent.critic_net.load_state_dict(checkpoint['critic_state_dict'])
                agent.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
                agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
                if 'timesteps' in checkpoint:
                    self.total_timesteps = checkpoint['timesteps']
            
            # Load training history for complete plots
            try:
                with open(save_dir/ f"{filename}_training_history.json", 'r') as f:
                    training_history = json.load(f)
                
                self.timesteps = training_history.get('timesteps', [])
                self.episode_rewards = training_history.get('episode_rewards', [])
                self.win_rates = training_history.get('win_rates', [])
                self.policy_losses = training_history.get('policy_losses', [])
                self.critic_losses = training_history.get('critic_losses', [])
                
                print(f"Training history loaded: {len(self.timesteps)} data points")
                
            except FileNotFoundError:
                print("Training history file not found - starting fresh metrics")
                self.timesteps = []
                self.episode_rewards = []
                self.win_rates = []
                self.policy_losses = []
                self.critic_losses = []
            
            print(f"Models loaded: {filename}")
            print(f"Resuming from timestep: {self.total_timesteps}")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Starting training from scratch...")
            self.total_timesteps = 0
    
    def plot_training_curves(self):
        """Plot training progress"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards vs timesteps
        if self.timesteps and self.episode_rewards:
            axes[0, 0].plot(self.timesteps, self.episode_rewards)
            axes[0, 0].set_title('Average Episode Rewards vs Timesteps (MAPPO)')
            axes[0, 0].set_xlabel('Timesteps')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True)
        
        # Win rates vs timesteps
        if self.timesteps and self.win_rates:
            axes[0, 1].plot(self.timesteps, self.win_rates)
            axes[0, 1].set_title('Win Rate vs Timesteps (MAPPO)')
            axes[0, 1].set_xlabel('Timesteps')
            axes[0, 1].set_ylabel('Win Rate')
            axes[0, 1].grid(True)
        
        # Policy losses
        if self.policy_losses:
            axes[1, 0].plot(self.policy_losses)
            axes[1, 0].set_title('Policy Loss vs Updates (MAPPO)')
            axes[1, 0].set_xlabel('Update Steps')
            axes[1, 0].set_ylabel('Policy Loss')
            axes[1, 0].grid(True)
        
        # Critic losses
        if self.critic_losses:
            axes[1, 1].plot(self.critic_losses)
            axes[1, 1].set_title('Critic Loss vs Updates (MAPPO)')
            axes[1, 1].set_xlabel('Update Steps')
            axes[1, 1].set_ylabel('Critic Loss')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'mappo_{self.map_name}_training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate(self, num_episodes=100):
        """Evaluate trained agents"""
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
        
        print(f"MAPPO Evaluation Results:")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Win Rate: {win_rate:.2f}")
        print(f"Average Episode Length: {avg_episode_length:.1f} timesteps")
        print(f"Total Evaluation Timesteps: {total_eval_timesteps}")
        
        return avg_reward, win_rate

def main():
    parser = argparse.ArgumentParser(description='MAPPO Training for SMAC')
    parser.add_argument('--map', type=str, default='8m', help='SMAC map name')
    parser.add_argument('--episodes', type=int, default=1500, help='Number of training episodes (ignored if --timesteps is set)')
    parser.add_argument('--timesteps', type=int, default=1000000, help='Number of training timesteps (overrides --episodes if set)')
    parser.add_argument('--episodes_per_update', type=int, default=25, help='Episodes collected before each update')
    parser.add_argument('--log_freq', type=int, default=5000, help='Logging frequency (in timesteps)')
    parser.add_argument('--save_freq', type=int, default= 50000, help='Model save frequency (in timesteps)')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size for GRU networks')
    parser.add_argument('--eval_only', action='store_true', help='Only evaluate, don\'t train')
    parser.add_argument('--load_model', type=str, help='Load model filename')
    parser.add_argument('--eval_episodes', type=int, default=100, help='Number of episodes for evaluation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize MAPPO trainer
    trainer = MAPPOTrainer(map_name=args.map, seed=args.seed, hidden_size=args.hidden_size)
    
    if args.eval_only:
        if args.load_model:
            trainer.load_models(args.load_model)
        trainer.evaluate(num_episodes=args.eval_episodes)
    else:
        # Load model if specified (for resuming training) - FIXED
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
        print("\nFinal MAPPO Evaluation:")
        trainer.evaluate(num_episodes=args.eval_episodes)

if __name__ == "__main__":
    main()
