import torch
import torch.nn as nn
import numpy as np
from sklearn.mixture import GaussianMixture
from smac.env import StarCraft2Env
import glob
import os
from pathlib import Path
import json
from scipy.stats import multivariate_normal
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PolicyAssignment:
    """GMM-based policy assignment for MALT with one datapoint per agent"""
    
    def __init__(self, n_target_agents, n_policies_per_agent=3, n_value_dimensions=5, random_seed=42):
        self.n_target_agents = n_target_agents
        self.n_policies_per_agent = n_policies_per_agent
        self.n_value_dimensions = n_value_dimensions
        self.random_seed = random_seed
        
        # GMM components
        self.gmm = None
        self.policy_assignments = {}
        self.policy_value_datapoints = None
        self.policy_value_vectors = None
        self.target_map_name = None
        
        # Source info
        self.source_env = None
        self.source_env_info = None
        self.source_actors = []
        self.source_critics = []
        
        # Environment cache
        self.envs = {}
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    
    def _get_or_create_env(self, map_name):
        """Get cached environment or create new one"""
        if map_name not in self.envs:
            self.envs[map_name] = StarCraft2Env(map_name=map_name, seed=self.random_seed)
        return self.envs[map_name]
    
    def _get_valid_action(self, avail_actions, agent_id, n_actions):
        """Unified action validation logic"""
        if avail_actions and len(avail_actions) > agent_id:
            agent_avail_actions = avail_actions[agent_id]
            valid_actions = [idx for idx, available in enumerate(agent_avail_actions) if available == 1]
            
            if valid_actions:
                return np.random.choice(valid_actions)
            else:
                return 0
        else:
            return np.random.randint(0, n_actions)
    
    def _sample_random_states_from_env(self, env, env_info, n_states, max_steps_range=(0, 50)):
        """Unified random state sampling logic"""
        sample_states = []
        
        for state_idx in range(n_states):
            env.reset()
            
            # Take random steps
            n_steps = np.random.randint(*max_steps_range)
            for _ in range(n_steps):
                avail_actions = env.get_avail_actions()
                actions = []
                
                for agent_id in range(env_info["n_agents"]):
                    action = self._get_valid_action(avail_actions, agent_id, env_info["n_actions"])
                    actions.append(action)
                
                _, done, _ = env.step(actions)
                if done:
                    break
            
            sample_states.append(env.get_state())
            
            if (state_idx + 1) % 25 == 0:
                print(f"  Sampled {state_idx + 1}/{n_states} states")
        
        return sample_states
    
    def _compute_value_functions_with_critic(self, critic, states):
        """Compute value functions using centralized critic"""
        state_tensors = [torch.FloatTensor(state).unsqueeze(0).to(device) for state in states]
        all_values = []
        
        with torch.no_grad():
            for i, state_tensor in enumerate(state_tensors):
                values, _ = critic(state_tensor)  # [1, n_agents]
                all_values.append(values[0].cpu().numpy())  # [n_agents]
                
                if (i + 1) % 25 == 0:
                    print(f"  Processed {i + 1}/{len(state_tensors)} states")
        
        return np.array(all_values)  # [n_states, n_agents]
    
    def load_source_agents(self, source_model_path, source_map_name):
        """Load source environment agents from trained models"""
        print(f"Loading source agents from {source_model_path} trained on {source_map_name}")
        
        # Initialize source environment
        self.source_env = self._get_or_create_env(source_map_name)
        self.source_env_info = self.source_env.get_env_info()
        
        n_source_agents = self.source_env_info["n_agents"]
        obs_dim = self.source_env_info["obs_shape"]
        global_state_dim = self.source_env_info["state_shape"]
        n_actions = self.source_env_info["n_actions"]
        
        print(f"Source environment info:")
        print(f"  - Map: {source_map_name}")
        print(f"  - Agents: {n_source_agents}")
        print(f"  - Obs dim: {obs_dim}")
        print(f"  - Global state dim: {global_state_dim}")
        print(f"  - Actions: {n_actions}")
        
        # Find model files
        model_files = glob.glob(f"{source_model_path}_*.pth")
        model_files.sort()
        
        if len(model_files) == 0:
            raise FileNotFoundError(f"No source model files found with pattern: {source_model_path}_*.pth")
        
        print(f"Found {len(model_files)} source agent models")
        
        # Load source actors and critics
        self.source_actors = []
        self.source_critics = []
        
        for i, model_file in enumerate(model_files):
            try:
                checkpoint = torch.load(model_file, map_location=device, weights_only=False)
                
                # Import the exact same classes from baseline
                from mappo_baseline_script import Actor, CentralizedCritic
                
                # Load actor with exact same architecture
                actor = Actor(obs_dim, n_actions, hidden_size=256).to(device)
                actor.load_state_dict(checkpoint['policy_state_dict'])
                actor.eval()
                self.source_actors.append(actor)
                
                # Load critic with exact same architecture
                critic = CentralizedCritic(global_state_dim, n_source_agents, hidden_size=256).to(device)
                critic.load_state_dict(checkpoint['critic_state_dict'])
                critic.eval()
                self.source_critics.append(critic)
                
                print(f"Loaded agent {i} from {model_file}")
                
            except Exception as e:
                print(f"Error loading {model_file}: {e}")
                continue
        
        print(f"Successfully loaded {len(self.source_actors)} source agents")
        return self.source_actors, self.source_critics
    
    def compute_source_value_functions(self):
        """Compute source policy value functions - ONE DATAPOINT PER AGENT"""
        print("Computing source policy value functions...")
        
        # Sample states
        sample_states = self._sample_random_states_from_env(
            self.source_env, self.source_env_info, self.n_value_dimensions, max_steps_range=(0, 50)
        )
        
        # Use centralized critic (all critics are the same in MAPPO)
        centralized_critic = self.source_critics[0]
        all_values = self._compute_value_functions_with_critic(centralized_critic, sample_states)
        
        print(f"All source values shape: {all_values.shape}")  # [n_states, n_agents]
        
        # Prepare ONE value vector per agent: [n_agents, n_dimensions]
        policy_value_datapoints = []
        for agent_id in range(len(self.source_actors)):
            agent_values = all_values[:, agent_id]  # [n_dimensions] values for this agent
            policy_value_datapoints.append(agent_values)
            print(f"Source Agent {agent_id}: value_vector shape={agent_values.shape}, mean={np.mean(agent_values):.4f}")
        
        self.policy_value_datapoints = np.array(policy_value_datapoints)  # [n_agents, n_dimensions]
        print(f"Final source value vectors shape: {self.policy_value_datapoints.shape}")
        
        return self.policy_value_datapoints
    
    def prepare_for_gmm_clustering(self):
        """Prepare source value vectors for GMM clustering"""
        print("Preparing source data for GMM clustering...")
        
        if self.policy_value_datapoints is None:
            raise ValueError("No value vectors available. Call compute_source_value_functions() first.")
        
        self.policy_value_vectors = self.policy_value_datapoints  # [n_agents, n_dimensions]
        print(f"Using {self.policy_value_vectors.shape[0]} agent vectors for GMM clustering")
        
        return self.policy_value_vectors
    
    def fit_gmm(self):
        """Fit GMM with n_target_agents components on source agent value vectors"""
        print(f"Fitting GMM with {self.n_target_agents} components...")
        
        if self.policy_value_vectors is None:
            raise ValueError("No policy value vectors available.")
        
        n_source_agents = self.policy_value_vectors.shape[0]
        
        # Check if we have fewer source agents than target agents
        if n_source_agents < self.n_target_agents:
            print(f"WARNING: Only {n_source_agents} source agents available for {self.n_target_agents} target agents")
            print("Will use round-robin assignment instead of GMM clustering")
            return None  # Skip GMM fitting
        
        self.gmm = GaussianMixture(
            n_components=self.n_target_agents,
            covariance_type='diag',
            max_iter=100,
            random_state=self.random_seed,
            init_params='kmeans',
            reg_covar=1e-3
        )
        
        self.gmm.fit(self.policy_value_vectors)
        
        print(f"GMM fitted successfully! Converged: {self.gmm.converged_}")
        
        # Print cluster weights
        print("Cluster weights:")
        for i in range(self.n_target_agents):
            print(f"  Cluster {i}: weight={self.gmm.weights_[i]:.4f}")
        
        return self.gmm
    
    def assign_policies_round_robin(self):
        """Assign 3 source policies to each target agent using round-robin method"""
        print("Using round-robin assignment (insufficient source agents for GMM clustering)...")
        
        n_source_agents = len(self.source_actors)
        
        for target_agent_id in range(self.n_target_agents):
            # Select 3 policies with round-robin + random selection
            if n_source_agents >= 3:
                # If we have at least 3 source agents, randomly select 3 unique ones
                selected_policies = np.random.choice(n_source_agents, size=3, replace=False).tolist()
            else:
                # If fewer than 3 source agents, use all available with repetition
                selected_policies = []
                for i in range(self.n_policies_per_agent):
                    policy_idx = i % n_source_agents  # Round-robin
                    selected_policies.append(policy_idx)
            
            self.policy_assignments[target_agent_id] = selected_policies
            print(f"Target Agent {target_agent_id} ← Source Policies {selected_policies} (round-robin)")
        
        return self.policy_assignments
    
    def assign_clusters_sequentially_to_agents(self):
        """Assign clusters to target agents sequentially (1-to-1 mapping)"""
        print("Assigning clusters to target agents sequentially...")
        
        agent_cluster_mapping = {}
        for target_agent_id in range(self.n_target_agents):
            agent_cluster_mapping[target_agent_id] = target_agent_id  # 0→0, 1→1, 2→2
            print(f"Target Agent {target_agent_id} → Source Cluster {target_agent_id}")
        
        return agent_cluster_mapping
    
    def _gmm_probability_density(self, value_vector, cluster_i):
        """Compute probability density for value vector under cluster i"""
        if cluster_i >= self.n_target_agents:
            raise IndexError(f"Cluster index {cluster_i} out of range")
        
        mu_i = self.gmm.means_[cluster_i]
        sigma_i = self.gmm.covariances_[cluster_i]  # For diagonal: [n_dims] array
        
        # Check for singular covariance matrix
        min_var = np.min(sigma_i)
        
        if min_var < 1e-10:  # Any variance too small
            # Add small regularization to diagonal elements
            sigma_i_reg = sigma_i + 1e-6
            cov_matrix = np.diag(sigma_i_reg)
        else:
            # Convert diagonal to full covariance matrix for multivariate_normal
            cov_matrix = np.diag(sigma_i)
        
        try:
            density = multivariate_normal.pdf(value_vector, mean=mu_i, cov=cov_matrix)
            return max(density, 1e-15)  # Prevent true zeros
        except Exception as e:
            print(f"  Cluster {cluster_i} error: {e}")
            return 1e-15
    
    def _select_policies_from_cluster_malt_formula(self, cluster_i):
        """Select policies from cluster using exact MALT paper formula: argmax ∏ N(y_ik | μ_i, Σ_i)"""
        print(f"  Selecting {self.n_policies_per_agent} policies for cluster {cluster_i} using MALT formula...")
        
        n_source_agents = len(self.source_actors)
        best_combination = None
        best_product = -1
        
        from itertools import combinations
        
        # Apply MALT formula: argmax ∏_{k=1}^m N(y_ik | μ_i, Σ_i)
        # where m = n_policies_per_agent
        print(f"    Testing all combinations of {self.n_policies_per_agent} unique policies from {n_source_agents} source agents...")
        
        combination_count = 0
        for policy_combination in combinations(range(n_source_agents), self.n_policies_per_agent):
            combination_count += 1
            product = 1.0
            
            # Compute product over all selected policies k
            for policy_k in policy_combination:
                # Get value vector for this source policy
                policy_value_vector = self.policy_value_vectors[policy_k]  # y_ik in the formula
                
                # Compute N(y_ik | μ_i, Σ_i) - probability density under cluster i
                prob_density = self._gmm_probability_density(policy_value_vector, cluster_i)
                product *= prob_density
            
            if product > best_product:
                best_product = product
                best_combination = policy_combination
            
            # Log progress for large searches
            if combination_count % 50 == 0:
                print(f"      Tested {combination_count} combinations so far...")
        
        print(f"    Tested {combination_count} total combinations")
        print(f"    Best combination: {list(best_combination)} with product: {best_product:.2e}")
        
        return list(best_combination)
    
    def assign_policies_with_sequential_clusters(self, agent_cluster_mapping):
        """Assign policies to target agents using sequential cluster mapping"""
        print("Assigning policies using sequential cluster mapping...")
        
        for target_agent_id, cluster_id in agent_cluster_mapping.items():
            print(f"\nProcessing Target Agent {target_agent_id} (assigned to Cluster {cluster_id}):")
            
            # Select best policies for this cluster using MALT formula
            selected_policies = self._select_policies_from_cluster_malt_formula(cluster_id)
            
            # Assign to target agent
            self.policy_assignments[target_agent_id] = selected_policies
            
            print(f"  Target Agent {target_agent_id} ← Policies {selected_policies} (from Cluster {cluster_id})")
        
        return self.policy_assignments
    
    def run_sequential_cluster_assignment(self, source_model_path, source_map_name, target_map_name):
        """Run MALT with sequential cluster-to-agent assignment (NO BOOTSTRAP TRAINING)"""
        print("=" * 80)
        print("MALT POLICY ASSIGNMENT (SEQUENTIAL CLUSTER ASSIGNMENT)")
        print("GMM Clustering + Sequential Agent-Cluster Mapping + MALT Policy Selection")
        print("NO BOOTSTRAP TRAINING REQUIRED")
        print("=" * 80)
        
        self.target_map_name = target_map_name
        
        try:
            # Step 1: Process source policies and fit GMM
            print("\n1. PROCESSING SOURCE POLICIES...")
            print("-" * 40)
            self.load_source_agents(source_model_path, source_map_name)
            self.compute_source_value_functions()
            self.prepare_for_gmm_clustering()
            self.fit_gmm()
            
            # Check if we have sufficient source agents for GMM clustering
            n_source_agents = len(self.source_actors)
            if n_source_agents < self.n_target_agents or self.gmm is None:
                # Step 2: Use round-robin assignment when insufficient source agents
                print("\n2. ROUND-ROBIN POLICY ASSIGNMENT...")
                print("-" * 40)
                self.assign_policies_round_robin()
                agent_cluster_mapping = None  # No clusters used
            else:
                # Step 2: Sequential cluster assignment (normal case)
                print("\n2. SEQUENTIAL CLUSTER ASSIGNMENT...")
                print("-" * 40)
                agent_cluster_mapping = self.assign_clusters_sequentially_to_agents()
                
                # Step 3: Select policies from clusters using exact MALT formula
                print("\n3. SELECTING POLICIES USING MALT FORMULA...")
                print("-" * 40)
                print("Using: argmax ∏_{k=1}^m N(y_ik | μ_i, Σ_i)")
                self.assign_policies_with_sequential_clusters(agent_cluster_mapping)
            
            print("\n" + "=" * 80)
            print("POLICY ASSIGNMENT COMPLETED!")
            print("=" * 80)
            
            print("\nFINAL ASSIGNMENTS:")
            print("-" * 50)
            for agent_id, policies in self.policy_assignments.items():
                if agent_cluster_mapping is not None:
                    cluster_used = agent_cluster_mapping[agent_id]
                    print(f"Target Agent {agent_id} ← Source Policies {policies} (from Cluster {cluster_used})")
                else:
                    print(f"Target Agent {agent_id} ← Source Policies {policies} (round-robin)")
            
            return self.policy_assignments, self.source_actors, agent_cluster_mapping
            
        finally:
            # Always cleanup environments
            self.cleanup()
    
    def save_assignments(self, save_path):
        """Save policy assignments and metadata to file"""
        assignment_data = {
            'policy_assignments': self.policy_assignments,
            'n_target_agents': self.n_target_agents,
            'n_policies_per_agent': self.n_policies_per_agent,
            'target_map_name': self.target_map_name,
            'algorithm': 'MALT_sequential_no_bootstrap'
        }
        
        with open(save_path, 'w') as f:
            json.dump(assignment_data, f, indent=4)
        
        print(f"Policy assignments saved to {save_path}")
    
    def load_assignments(self, load_path):
        """Load policy assignments from file"""
        with open(load_path, 'r') as f:
            assignment_data = json.load(f)
        
        self.policy_assignments = assignment_data['policy_assignments']
        self.n_target_agents = assignment_data['n_target_agents']
        self.n_policies_per_agent = assignment_data['n_policies_per_agent']
        self.target_map_name = assignment_data.get('target_map_name', None)
        
        print(f"Policy assignments loaded from {load_path}")
        return self.policy_assignments
    
    def cleanup(self):
        """Clean up cached environments"""
        try:
            for env in self.envs.values():
                try:
                    env.close()
                except:
                    pass
            self.envs.clear()
            
            if self.source_env and self.source_env not in self.envs.values():
                try:
                    self.source_env.close()
                except:
                    pass
            
            print("Environment resources cleaned up")
        except Exception as e:
            print(f"Warning during cleanup: {e}")
            # Continue anyway - cleanup errors shouldn't stop the program

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="MALT Policy Assignment - Sequential Cluster Assignment (No Bootstrap Training)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--source_model_path", 
        type=str, 
        required=True,
        help="Path to source model files (without extension, e.g., 'models_8m/mappo_8m_best_agent')"
    )
    parser.add_argument(
        "--source_map", 
        type=str, 
        required=True,
        help="Source map name (e.g., '8m', '3s5z', '2s3z')"
    )
    parser.add_argument(
        "--target_map", 
        type=str, 
        required=True,
        help="Target map name (e.g., '3m', '2s3z', '8m')"
    )
    parser.add_argument(
        "--target_agents", 
        type=int, 
        required=True,
        help="Number of target agents"
    )
    
    # Optional arguments
    parser.add_argument(
        "--policies_per_agent", 
        type=int, 
        default=3,
        help="Number of policies to assign per target agent"
    )
    parser.add_argument(
        "--value_dimensions", 
        type=int, 
        default=5,
        help="Number of value function dimensions for clustering"
    )
    parser.add_argument(
        "--random_seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="policy_assignments.json",
        help="Output file to save policy assignments"
    )
    
    return parser.parse_args()

# Example usage and testing
if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    print("Starting MALT Policy Assignment")
    print("=" * 70)
    print(f"Source: {args.source_model_path} ({args.source_map})")
    print(f"Target: {args.target_map} ({args.target_agents} agents)")
    print(f"Policies per agent: {args.policies_per_agent}")
    print(f"Value dimensions: {args.value_dimensions}")
    print(f"Random seed: {args.random_seed}")
    print(f"Output file: {args.output_file}")
    print("=" * 70)
    
    # Create policy assignment instance
    policy_assigner = PolicyAssignment(
        n_target_agents=args.target_agents,
        n_policies_per_agent=args.policies_per_agent,
        n_value_dimensions=args.value_dimensions,
        random_seed=args.random_seed
    )
    
    # Run policy assignment process
    try:
        assignments, source_actors, cluster_mapping = policy_assigner.run_sequential_cluster_assignment(
            args.source_model_path, args.source_map, args.target_map
        )
        
        # Save assignments
        policy_assigner.save_assignments(args.output_file)
        
        print("\n" + "="*70)
        print("SUCCESS: Policy assignment completed!")
        
        # Show different success messages based on assignment method
        if cluster_mapping is None:
            print("Method: Round-robin assignment (insufficient source agents)")
            print("✅ 3 source policies assigned to each target agent")
            print("✅ Random selection with round-robin fallback")
        else:
            print("Method: Sequential cluster assignment")
            print("✅ GMM clustering identifies source behavioral patterns")
            print("✅ Sequential 1-to-1 cluster-agent mapping") 
            print("✅ Exact MALT paper formula: argmax ∏ N(y_ik | μ_i, Σ_i)")
            print("✅ No duplicate policies per cluster")
        
        print(f"✅ Policy assignments saved to {args.output_file}")
        print("="*70)
            
    except Exception as e:
        print(f"Error in policy assignment: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Ensure cleanup even if error occurs
        policy_assigner.cleanup()
