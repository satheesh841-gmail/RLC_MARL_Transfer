#!/usr/bin/env python3
"""
Complete GAN Trainer for MALT Observation Adaptation
Loads pretrained target and source policies, collects observations, trains GAN adapters
"""

import torch
import torch.nn as nn
import numpy as np
from smac.env import StarCraft2Env
import argparse
import glob
from pathlib import Path
import json
import time
import matplotlib.pyplot as plt
from collections import defaultdict

# Import your existing modules
from mappo_baseline_script import Actor, CentralizedCritic

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Generator(nn.Module):
    """Generator: target_obs → source_obs"""
    def __init__(self, target_dim, source_dim, hidden_size=128):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(target_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, source_dim)
            
        )
        # Small weights for stability
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.3)
                nn.init.zeros_(layer.bias)
    
    def forward(self, target_obs):
        return self.net(target_obs)

class Discriminator(nn.Module):
    """Discriminator: real vs fake source observations"""
    def __init__(self, source_dim, hidden_size=64):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(source_dim, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.3)
                nn.init.zeros_(layer.bias)
    
    def forward(self, source_obs):
        return self.net(source_obs)

class ObsGAN:
    """GAN for observation adaptation with plotting capabilities"""
    def __init__(self, target_dim, source_dim, lr=1e-3):
        self.target_dim = target_dim
        self.source_dim = source_dim
        
        self.gen = Generator(target_dim, source_dim).to(device)
        self.disc = Discriminator(source_dim).to(device)
        
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=1e-4, weight_decay=1e-4)
        self.disc_opt = torch.optim.Adam(self.disc.parameters(), lr=5e-5, weight_decay=1e-4)
        
        self.bce_loss = nn.BCELoss()
        self.history = {'gen': [], 'disc': [], 'epochs': []}
        
        print(f"ObsGAN: {target_dim} → {source_dim}")
    
    def train_step(self, real_source, target_obs, batch_size=32):
        """Single training step"""
        n = min(len(real_source), len(target_obs))
        batch_size = min(batch_size, n)
        
        real_batch = torch.FloatTensor(real_source[np.random.choice(len(real_source), batch_size)]).to(device)
        target_batch = torch.FloatTensor(target_obs[np.random.choice(len(target_obs), batch_size)]).to(device)
        
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        # Train Discriminator
        self.disc_opt.zero_grad()
        real_pred = self.disc(real_batch)
        real_loss = self.bce_loss(real_pred, real_labels)
        
        with torch.no_grad():
            fake_batch = self.gen(target_batch)
        fake_pred = self.disc(fake_batch.detach())
        fake_loss = self.bce_loss(fake_pred, fake_labels)
        
        disc_loss = (real_loss + fake_loss) / 2
        disc_loss.backward()
        self.disc_opt.step()
        
        # Train Generator
        self.gen_opt.zero_grad()
        fake_batch = self.gen(target_batch)
        fake_pred = self.disc(fake_batch)
        gen_loss = self.bce_loss(fake_pred, real_labels)
        gen_loss.backward()
        self.gen_opt.step()
        
        return gen_loss.item(), disc_loss.item()
    
    def train(self, source_obs, target_obs, epochs=300, batch_size=32, plot_every=50):
        """Train the GAN with plotting"""
        print(f"Training for {epochs} epochs...")
        
        for epoch in range(epochs):
            gen_loss, disc_loss = self.train_step(source_obs, target_obs, batch_size)
            
            # Store history
            self.history['gen'].append(gen_loss)
            self.history['disc'].append(disc_loss)
            self.history['epochs'].append(epoch + 1)
            
            if (epoch + 1) % plot_every == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | Gen: {gen_loss:.4f} | Disc: {disc_loss:.4f}")
        
        print("Training completed!")
    
    def plot_training_curves(self, save_path=None):
        """Plot GAN training curves"""
        if not self.history['epochs']:
            print("No training history to plot!")
            return
        
        plt.figure(figsize=(12, 4))
        
        # Generator and Discriminator losses
        plt.subplot(1, 2, 1)
        plt.plot(self.history['epochs'], self.history['gen'], label='Generator Loss', color='blue', alpha=0.7)
        plt.plot(self.history['epochs'], self.history['disc'], label='Discriminator Loss', color='red', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('GAN Training Losses')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Running average (smoother curves)
        plt.subplot(1, 2, 2)
        window = min(20, len(self.history['gen']) // 10)
        if window > 1:
            gen_smooth = np.convolve(self.history['gen'], np.ones(window)/window, mode='valid')
            disc_smooth = np.convolve(self.history['disc'], np.ones(window)/window, mode='valid')
            epochs_smooth = self.history['epochs'][window-1:]
            
            plt.plot(epochs_smooth, gen_smooth, label='Generator (Smoothed)', color='blue', linewidth=2)
            plt.plot(epochs_smooth, disc_smooth, label='Discriminator (Smoothed)', color='red', linewidth=2)
        else:
            plt.plot(self.history['epochs'], self.history['gen'], label='Generator', color='blue')
            plt.plot(self.history['epochs'], self.history['disc'], label='Discriminator', color='red')
            
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('GAN Training Losses (Smoothed)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved: {save_path}")
        
        plt.show()
    
    def save(self, path):
        """Save trained GAN"""
        torch.save({
            'generator': self.gen.state_dict(),
            'discriminator': self.disc.state_dict(),
            'target_dim': self.target_dim,
            'source_dim': self.source_dim,
            'history': self.history
        }, path)
        print(f"Saved: {path}")

class PolicyAgent:
    """Simplified wrapper for loaded MAPPO agents - policy only"""
    def __init__(self, agent_id, policy_net):
        self.agent_id = agent_id
        self.policy_net = policy_net
        self.policy_hidden = None
        
        # Action statistics
        self.action_stats = {
            'total_actions': 0,
            'valid_actions': 0,
            'invalid_actions': 0,
            'action_distribution': defaultdict(int)
        }
    
    def reset_hidden(self, batch_size=1):
        """Reset GRU hidden state"""
        if hasattr(self.policy_net, 'gru'):
            self.policy_hidden = torch.zeros(
                self.policy_net.gru.num_layers,
                batch_size,
                self.policy_net.gru.hidden_size
            ).to(device)
        else:
            self.policy_hidden = None
    
    def select_action(self, obs, avail_actions=None):
        """Select action using loaded policy"""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        
        with torch.no_grad():
            if self.policy_hidden is None:
                self.reset_hidden()
            
            # Get action logits
            if hasattr(self.policy_net, 'gru'):
                features = self.policy_net.feature_net(obs_tensor)
                gru_out, self.policy_hidden = self.policy_net.gru(
                    features.unsqueeze(1), self.policy_hidden
                )
                logits = self.policy_net.policy_head(gru_out.squeeze(1))
            else:
                logits = self.policy_net(obs_tensor)
            
            # Apply action mask if available
            if avail_actions is not None:
                avail_mask = torch.FloatTensor(avail_actions).to(device)
                logits = logits.masked_fill(avail_mask == 0, float('-inf'))
            
            # Sample action
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
            
            # Track action statistics
            self.action_stats['total_actions'] += 1
            self.action_stats['action_distribution'][action] += 1
            
            # Check if action is valid
            if avail_actions is not None:
                if avail_actions[action] == 1:
                    self.action_stats['valid_actions'] += 1
                else:
                    self.action_stats['invalid_actions'] += 1
                    # Fallback to valid action
                    valid_actions = [i for i, a in enumerate(avail_actions) if a == 1]
                    if valid_actions:
                        action = valid_actions[0]
            
            return action
    
    def get_action_stats(self):
        """Get action statistics"""
        total = self.action_stats['total_actions']
        if total == 0:
            return "No actions taken yet"
        
        valid_pct = (self.action_stats['valid_actions'] / total) * 100
        invalid_pct = (self.action_stats['invalid_actions'] / total) * 100
        
        stats_str = f"Agent {self.agent_id} Action Stats:\n"
        stats_str += f"  Total actions: {total}\n"
        stats_str += f"  Valid: {self.action_stats['valid_actions']} ({valid_pct:.1f}%)\n"
        stats_str += f"  Invalid: {self.action_stats['invalid_actions']} ({invalid_pct:.1f}%)\n"
        stats_str += f"  Action distribution: {dict(self.action_stats['action_distribution'])}"
        
        return stats_str

class PolicyLoader:
    """Load policies from saved checkpoints"""
    def __init__(self, model_path_pattern, map_name, hidden_size=256):
        self.model_path_pattern = model_path_pattern
        self.map_name = map_name
        self.hidden_size = hidden_size
        
        # Get environment info
        env = StarCraft2Env(map_name=map_name)
        self.env_info = env.get_env_info()
        env.close()
        
        self.obs_dim = self.env_info['obs_shape']
        self.act_dim = self.env_info['n_actions']
        self.n_agents = self.env_info['n_agents']
    
    def load_agents(self):
        """Load all agent policies from checkpoint files"""
        agents = []
        
        # Find agent checkpoint files
        checkpoint_files = sorted(glob.glob(f"{self.model_path_pattern}_*.pth"))
        
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found for pattern: {self.model_path_pattern}")
        
        print(f"Found {len(checkpoint_files)} checkpoint files")
        
        for i, checkpoint_file in enumerate(checkpoint_files):
            try:
                # Load checkpoint
                checkpoint = torch.load(checkpoint_file, map_location=device)
                
                # Create Actor network
                actor = Actor(
                    obs_dim=self.obs_dim,
                    act_dim=self.act_dim,
                    hidden_size=self.hidden_size
                ).to(device)
                
                # Load actor weights
                actor.load_state_dict(checkpoint['actor_state_dict'])
                actor.eval()
                
                # Wrap in PolicyAgent
                agent = PolicyAgent(agent_id=i, policy_net=actor)
                agents.append(agent)
                
                print(f"Loaded agent {i} from: {checkpoint_file}")
                
            except Exception as e:
                print(f"Error loading {checkpoint_file}: {e}")
                continue
        
        if len(agents) == 0:
            raise RuntimeError("Failed to load any agents!")
        
        return agents

class SmartObservationCollector:
    """Smart observation collector with full team coordination"""
    def __init__(self, source_env, target_env, source_agents, target_agents):
        self.source_env = source_env
        self.target_env = target_env
        self.source_agents = source_agents
        self.target_agents = target_agents
        
        # Collection statistics
        self.collection_stats = {
            'source_episodes': 0,
            'target_episodes': 0,
            'source_timesteps': 0,
            'target_timesteps': 0,
            'source_wins': 0,
            'target_wins': 0
        }
    
    def collect_single_agent_observations(self, source_agent_idx, target_agent_idx, num_episodes):
        """
        Collect observations from single agents while running full teams
        
        Args:
            source_agent_idx: Which source agent to collect from
            target_agent_idx: Which target agent to collect from
            num_episodes: Number of episodes to collect
        """
        print(f"\nCollecting single-agent observations:")
        print(f"  Source agent {source_agent_idx} from {len(self.source_agents)}-agent team")
        print(f"  Target agent {target_agent_idx} from {len(self.target_agents)}-agent team")
        print(f"  Episodes: {num_episodes}")
        
        source_observations = []
        target_observations = []
        
        # Collect source observations
        print(f"\nCollecting from source environment...")
        for episode in range(num_episodes):
            self.source_env.reset()
            
            # Reset all agents
            for agent in self.source_agents:
                agent.reset_hidden()
            
            episode_obs = []
            
            while True:
                obs_list = self.source_env.get_obs()
                avail_actions = self.source_env.get_avail_actions()
                
                # All agents select actions (team coordination)
                actions = []
                for i, agent in enumerate(self.source_agents):
                    avail_mask = avail_actions[i] if avail_actions else None
                    action = agent.select_action(obs_list[i], avail_mask)
                    actions.append(action)
                
                # Store observation from selected agent only
                episode_obs.append(obs_list[source_agent_idx])
                
                # Step environment
                reward, done, info = self.source_env.step(actions)
                
                if done:
                    self.collection_stats['source_episodes'] += 1
                    self.collection_stats['source_timesteps'] += len(episode_obs)
                    if info.get('battle_won', False):
                        self.collection_stats['source_wins'] += 1
                    break
            
            source_observations.extend(episode_obs)
            
            if (episode + 1) % 20 == 0:
                print(f"  Source: {episode + 1}/{num_episodes} episodes, "
                      f"{len(source_observations)} observations collected")
        
        # Collect target observations
        print(f"\nCollecting from target environment...")
        for episode in range(num_episodes):
            self.target_env.reset()
            
            # Reset all agents
            for agent in self.target_agents:
                agent.reset_hidden()
            
            episode_obs = []
            
            while True:
                obs_list = self.target_env.get_obs()
                avail_actions = self.target_env.get_avail_actions()
                
                # All agents select actions (team coordination)
                actions = []
                for i, agent in enumerate(self.target_agents):
                    avail_mask = avail_actions[i] if avail_actions else None
                    action = agent.select_action(obs_list[i], avail_mask)
                    actions.append(action)
                
                # Store observation from selected agent only
                episode_obs.append(obs_list[target_agent_idx])
                
                # Step environment
                reward, done, info = self.target_env.step(actions)
                
                if done:
                    self.collection_stats['target_episodes'] += 1
                    self.collection_stats['target_timesteps'] += len(episode_obs)
                    if info.get('battle_won', False):
                        self.collection_stats['target_wins'] += 1
                    break
            
            target_observations.extend(episode_obs)
            
            if (episode + 1) % 20 == 0:
                print(f"  Target: {episode + 1}/{num_episodes} episodes, "
                      f"{len(target_observations)} observations collected")
        
        # Print collection statistics
        print(f"\nCollection Statistics:")
        print(f"  Source: {self.collection_stats['source_episodes']} episodes, "
              f"{self.collection_stats['source_timesteps']} timesteps, "
              f"{self.collection_stats['source_wins']} wins "
              f"({self.collection_stats['source_wins']/self.collection_stats['source_episodes']*100:.1f}% win rate)")
        print(f"  Target: {self.collection_stats['target_episodes']} episodes, "
              f"{self.collection_stats['target_timesteps']} timesteps, "
              f"{self.collection_stats['target_wins']} wins "
              f"({self.collection_stats['target_wins']/self.collection_stats['target_episodes']*100:.1f}% win rate)")
        
        # Print action statistics for selected agents
        print(f"\nAction Statistics:")
        print(self.source_agents[source_agent_idx].get_action_stats())
        print(self.target_agents[target_agent_idx].get_action_stats())
        
        return np.array(source_observations), np.array(target_observations)

class GANTrainerPipeline:
    """Complete pipeline for GAN training"""
    def __init__(self, source_model_path, target_model_path, source_map, target_map, seed=42):
        self.source_model_path = source_model_path
        self.target_model_path = target_model_path
        self.source_map = source_map
        self.target_map = target_map
        self.seed = seed
        
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        print(f"GAN Trainer Pipeline: {source_map} → {target_map}")
    
    def run(self, collection_episodes=100, gan_epochs=300, save_dir="gan_adapters",
            source_agent_idx=0, target_agent_idx=0):
        """Run complete GAN training pipeline"""
        
        # Create save directory
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        print("="*60)
        print("STEP 1: LOADING POLICIES")
        print("="*60)
        
        # Load source policies
        print(f"Loading source policies from: {self.source_model_path}")
        source_env = StarCraft2Env(map_name=self.source_map, seed=self.seed)
        source_loader = PolicyLoader(self.source_model_path, self.source_map)
        source_agents = source_loader.load_agents()
        source_env_info = source_env.get_env_info()
        
        # Load target policies  
        print(f"Loading target policies from: {self.target_model_path}")
        target_env = StarCraft2Env(map_name=self.target_map, seed=self.seed)
        target_loader = PolicyLoader(self.target_model_path, self.target_map)
        target_agents = target_loader.load_agents()
        target_env_info = target_env.get_env_info()
        
        print(f"Source: {self.source_map} - {len(source_agents)} agents, obs_dim={source_env_info['obs_shape']}")
        print(f"Target: {self.target_map} - {len(target_agents)} agents, obs_dim={target_env_info['obs_shape']}")
        
        print("\n" + "="*60)
        print("STEP 2: COLLECTING OBSERVATIONS")
        print("="*60)
        
        # Use SmartObservationCollector with single-agent strategy
        smart_collector = SmartObservationCollector(
            source_env, target_env, source_agents, target_agents
        )
        
        # Collect single-agent observations
        source_observations, target_observations = smart_collector.collect_single_agent_observations(
            source_agent_idx=source_agent_idx,
            target_agent_idx=target_agent_idx,
            num_episodes=collection_episodes
        )
        
        # Validate collected data
        if len(source_observations) == 0 or len(target_observations) == 0:
            raise ValueError("No observations collected! Check your models and environments.")
        
        print(f"Source obs: {source_observations.shape}")
        print(f"Target obs: {target_observations.shape}")
        
        print("\n" + "="*60)
        print("STEP 3: TRAINING UNIVERSAL GAN ADAPTER")
        print("="*60)
        
        # Train single universal GAN adapter
        print(f"Training universal GAN adapter (works for all agents)")
        print("-" * 50)
        
        # Create GAN
        gan = ObsGAN(
            target_dim=target_env_info['obs_shape'],
            source_dim=source_env_info['obs_shape']
        )
        
        # Train GAN with plotting
        start_time = time.time()
        gan.train(source_observations, target_observations, epochs=gan_epochs, plot_every=50)
        training_time = time.time() - start_time
        
        # Plot training curves
        plot_path = save_dir / f"training_curves_{self.source_map}_to_{self.target_map}.png"
        gan.plot_training_curves(save_path=str(plot_path))
        
        # Save universal adapter
        adapter_path = save_dir / f"universal_adapter_{self.source_map}_to_{self.target_map}.pth"
        gan.save(str(adapter_path))
        
        print(f"Universal adapter trained in {training_time:.1f}s")
        
        # Clean up environments
        source_env.close()
        target_env.close()
        
        print("\n" + "="*60)
        print("GAN TRAINING COMPLETED!")
        print(f"Universal adapter saved: {adapter_path}")
        print(f"Training curves saved: {plot_path}")
        print("="*60)
        
        # Save metadata
        metadata = {
            'source_map': self.source_map,
            'target_map': self.target_map,
            'source_model_path': self.source_model_path,
            'target_model_path': self.target_model_path,
            'collection_episodes': collection_episodes,
            'gan_epochs': gan_epochs,
            'source_obs_dim': source_env_info['obs_shape'],
            'target_obs_dim': target_env_info['obs_shape'],
            'num_source_policies': len(source_agents),
            'num_target_agents': len(target_agents),
            'source_agent_idx': source_agent_idx,
            'target_agent_idx': target_agent_idx,
            'adapter_type': 'universal_single_agent_gan',
            'training_time': training_time,
            'collection_stats': smart_collector.collection_stats
        }
        
        with open(save_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"Metadata saved: {save_dir}/metadata.json")
        
        return str(adapter_path)

def main():
    parser = argparse.ArgumentParser(description='Smart Single-Agent GAN Training for MALT')
    parser.add_argument('--source_model', type=str, required=True,
                       help='Source model path pattern (e.g., mappo_3m_baseline/mappo_3m_best_agent)')
    parser.add_argument('--target_model', type=str, required=True,
                       help='Target model path pattern (e.g., models_8m/mappo_8m_timestep_100260_agent)')
    parser.add_argument('--source_map', type=str, required=True,
                       help='Source map name (e.g., 3m)')
    parser.add_argument('--target_map', type=str, required=True,
                       help='Target map name (e.g., 8m)')
    parser.add_argument('--collection_episodes', type=int, default= 500,
                       help='Episodes for observation collection')
    parser.add_argument('--gan_epochs', type=int, default= 1000,
                       help='GAN training epochs')
    parser.add_argument('--save_dir', type=str, default='universal_gan_adapter',
                       help='Directory to save adapter')
    parser.add_argument('--source_agent_idx', type=int, default=0,
                       help='Which source agent to use for training (default: 0)')
    parser.add_argument('--target_agent_idx', type=int, default=0,
                       help='Which target agent to use for training (default: 0)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    print("SMART SINGLE-AGENT GAN TRAINING FOR MALT")
    print("="*50)
    print("Strategy:")
    print("  - Run full teams but collect single agent's observations")
    print("  - Train GAN on high-quality single-agent data")  
    print("  - Leverage homogeneity for universal generalization")
    print("  - One GAN works for ALL agent combinations")
    print("  - Monitor action validity and observation collection")
    print("  - Plot training curves and save statistics")
    print("="*50)
   
    # Create and run pipeline
    pipeline = GANTrainerPipeline(
       source_model_path=args.source_model,
       target_model_path=args.target_model,
       source_map=args.source_map,
       target_map=args.target_map,
       seed=args.seed
    )
   
    try:
       adapter_path = pipeline.run(
           collection_episodes=args.collection_episodes,
           gan_epochs=args.gan_epochs,
           save_dir=args.save_dir,
           source_agent_idx=args.source_agent_idx,
           target_agent_idx=args.target_agent_idx
       )
       
       print(f"\nCOMPLETED! Universal adapter ready: {adapter_path}")
       print("This single GAN will work for ALL your MALT transfer scenarios!")
       print("Check the training curves and statistics in the save directory!")
       
    except Exception as e:
       print(f"\nPIPELINE FAILED: {e}")
       import traceback
       traceback.print_exc()
       print("\nDebug suggestions:")
       print("  - Check if model files exist and are valid")
       print("  - Verify SMAC environments are properly installed")
       print("  - Ensure sufficient episodes for meaningful data collection")
       print("  - Check if agents can take valid actions in the environments")

if __name__ == "__main__":
   main()