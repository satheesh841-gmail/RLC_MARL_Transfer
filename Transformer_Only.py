

import os
import json
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


from malt_transformer import TransformerMapper, MALTActor
from smac.env import StarCraft2Env

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ------------------------------
# Utilities
# ------------------------------
def rbf_kernel(x, y, sigma=1.0):
    """Compute RBF kernel matrix between batches x and y."""
    # x: [N, D], y: [M, D]
    x_norm = (x**2).sum(dim=1, keepdim=True)   # [N, 1]
    y_norm = (y**2).sum(dim=1, keepdim=True)   # [M, 1]
    dist_sq = x_norm + y_norm.T - 2.0 * x @ y.T
    k = torch.exp(-dist_sq / (2.0 * sigma**2))
    return k

def mmd_loss(x, y, sigma=1.0):
    """
    Unbiased MMD^2 estimate between two batches using RBF kernel.
    x: [N, D], y: [M, D]
    """
    Kxx = rbf_kernel(x, x, sigma)
    Kyy = rbf_kernel(y, y, sigma)
    Kxy = rbf_kernel(x, y, sigma)
    # Unbiased estimates: subtract diagonals
    nx = x.size(0)
    ny = y.size(0)
    if nx > 1:
        mmd_x = (Kxx.sum() - Kxx.diag().sum()) / (nx * (nx - 1))
    else:
        mmd_x = torch.tensor(0.0, device=x.device)
    if ny > 1:
        mmd_y = (Kyy.sum() - Kyy.diag().sum()) / (ny * (ny - 1))
    else:
        mmd_y = torch.tensor(0.0, device=y.device)
    mmd_xy = 2.0 * Kxy.mean()
    return mmd_x + mmd_y - mmd_xy


# ------------------------------
# Datasets
# ------------------------------
class PairedObsDataset(torch.utils.data.Dataset):
    """Supervised dataset with aligned (target_obs, source_obs) pairs."""
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.target_obs = torch.tensor(data['target_obs'], dtype=torch.float32)
        self.source_obs = torch.tensor(data['source_obs'], dtype=torch.float32)
        assert self.target_obs.shape[0] == self.source_obs.shape[0], "Pair count mismatch"

    def __len__(self):
        return self.target_obs.shape[0]

    def __getitem__(self, idx):
        return self.target_obs[idx], self.source_obs[idx]


class UnpairedRolloutDataset(torch.utils.data.IterableDataset):
    """
    Streams batches of (target_obs_batch, source_obs_batch) by rolling out the
    target and source SMAC maps independently. The batches are *unpaired*.

    This is sufficient for distribution-level alignment via MMD (and optional KL on logits).
    """
    def __init__(self, target_map, source_map, seed=42, batch_size=128, max_steps_per_episode=200):
        super().__init__()
        self.target_map = target_map
        self.source_map = source_map
        self.seed = seed
        self.batch_size = batch_size
        self.max_steps_per_episode = max_steps_per_episode

        # Build envs (get dims, then create separate instances for stepping)
        self.target_env = StarCraft2Env(map_name=target_map, seed=seed)
        self.source_env = StarCraft2Env(map_name=source_map, seed=seed)

        self.target_info = self.target_env.get_env_info()
        self.source_info = self.source_env.get_env_info()
        self.target_dim = self.target_info['obs_shape']
        self.source_dim = self.source_info['obs_shape']

    def __iter__(self):
        # Fresh environments for iteration to avoid stale state
        target_env = StarCraft2Env(map_name=self.target_map, seed=self.seed)
        source_env = StarCraft2Env(map_name=self.source_map, seed=self.seed)

        while True:
            # Collect one target batch
            target_env.reset()
            tgt_obs_list = []
            steps = 0
            while steps < self.max_steps_per_episode and len(tgt_obs_list) < self.batch_size:
                obs_list = target_env.get_obs()       # list per agent; we use agent 0 for distribution sampling
                tgt_obs_list.append(obs_list[0])      # choose a representative agent (or stack from all agents)
                actions = [0] * target_env.get_env_info()['n_agents']  # dummy actions
                reward, done, _ = target_env.step(actions)
                steps += 1
                if done:
                    break

            # Collect one source batch
            source_env.reset()
            src_obs_list = []
            steps = 0
            while steps < self.max_steps_per_episode and len(src_obs_list) < self.batch_size:
                obs_list = source_env.get_obs()
                src_obs_list.append(obs_list[0])
                actions = [0] * source_env.get_env_info()['n_agents']
                reward, done, _ = source_env.step(actions)
                steps += 1
                if done:
                    break

            # Yield as tensors
            tgt = torch.tensor(np.array(tgt_obs_list), dtype=torch.float32)
            src = torch.tensor(np.array(src_obs_list), dtype=torch.float32)
            yield tgt, src


# ------------------------------
# Training
# ------------------------------
def train_supervised(args):
    """
    Train TransformerMapper with MSE on paired data.
    """
    dataset = PairedObsDataset(args.paired_npz)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Build mapper
    mapper = TransformerMapper(
        target_dim=dataset.target_obs.shape[1],
        source_dim=dataset.source_obs.shape[1],
        model_dim=args.model_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        token_size=args.token_size,
        dropout=args.dropout,
    ).to(device)

    opt = optim.Adam(mapper.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    best = math.inf
    for epoch in range(1, args.epochs + 1):
        mapper.train()
        running = 0.0
        for target_obs, source_obs in loader:
            target_obs = target_obs.to(device)
            source_obs = source_obs.to(device)
            pred = mapper(target_obs)
            loss = mse(pred, source_obs)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mapper.parameters(), max_norm=1.0)
            opt.step()

            running += loss.item()

        avg = running / len(loader)
        print(f"[Supervised] Epoch {epoch}/{args.epochs}  MSE: {avg:.6f}")
        if avg < best:
            best = avg
            save_transformer(mapper, args.out_path)
            print(f"  -> Saved best checkpoint to {args.out_path} (key='transformer')")

    print("Training finished.")
    return args.out_path


def build_frozen_source_actor(source_map, source_actor_path, hidden_size=256, seed=42):
    """
    Build a frozen MALTActor for the source env dims and load its checkpoint.
    We only use its feature_net (and policy_head if KL is enabled).
    """
    src_env = StarCraft2Env(map_name=source_map, seed=seed)
    src_info = src_env.get_env_info()
    src_env.close()

    src_obs_dim = src_info['obs_shape']
    src_act_dim = src_info['n_actions']

    # Create an actor with source dims
    src_actor = MALTActor(
        obs_dim=src_obs_dim,
        act_dim=src_act_dim,
        hidden_size=hidden_size,
        source_policies=None, assigned_policy_indices=[],
        source_obs_dims=None, source_act_dims=None,
        transformer_adapter_path=None
    ).to(device)

    # Load its state dict from the saved agent file (actor_state_dict key)
    ckpt = torch.load(source_actor_path, map_location=device)
    src_actor.load_state_dict(ckpt['actor_state_dict'])
    src_actor.eval()
    for p in src_actor.parameters():
        p.requires_grad = False

    return src_actor, src_obs_dim, src_act_dim


def train_unpaired_distill(args):
    """
    Train TransformerMapper without paired data.
    - Match source_actor.feature_net distributions via MMD
    - (Optional) add KL over logits from policy_head

    The dataset streams unpaired batches from source/target SMAC maps.
    """
    # Prepare dataset
    ds = UnpairedRolloutDataset(
        target_map=args.target_map,
        source_map=args.source_map,
        seed=args.seed,
        batch_size=args.batch_size,
        max_steps_per_episode=args.max_steps_per_episode
    )

    # Build frozen source actor
    src_actor, src_obs_dim, src_act_dim = build_frozen_source_actor(
        args.source_map, args.source_actor_path, hidden_size=args.hidden_size, seed=args.seed
    )

    # Build mapper
    # Get target obs dim from a temp env
    tgt_env = StarCraft2Env(map_name=args.target_map, seed=args.seed)
    tgt_info = tgt_env.get_env_info()
    tgt_env.close()
    tgt_obs_dim = tgt_info['obs_shape']

    mapper = TransformerMapper(
        target_dim=tgt_obs_dim,
        source_dim=src_obs_dim,
        model_dim=args.model_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        token_size=args.token_size,
        dropout=args.dropout,
    ).to(device)

    opt = optim.Adam(mapper.parameters(), lr=args.lr)

    # Loss weights
    w_mmd = args.w_mmd
    w_kl  = args.w_kl

    # KL for logits (optional)
    kl_div = nn.KLDivLoss(reduction='batchmean')

    # Training loop
    it = iter(ds)
    best = math.inf
    for step in range(1, args.steps + 1):
        mapper.train()

        # Fetch a batch of unpaired obs
        tgt_batch, src_batch = next(it)
        tgt_batch = tgt_batch.to(device)  # [B, tgt_dim]
        src_batch = src_batch.to(device)  # [B, src_dim]

        # Forward through mapper then source_actor.feature_net
        adapted_src_obs = mapper(tgt_batch)                          # [B, src_dim]
        src_feats_real  = src_actor.feature_net(src_batch)           # [B, hidden_size]
        src_feats_adapt = src_actor.feature_net(adapted_src_obs)     # [B, hidden_size]

        # MMD loss over features
        loss_mmd = mmd_loss(src_feats_real, src_feats_adapt, sigma=args.mmd_sigma)

        # Optional KL distillation on logits
        loss_kl = torch.tensor(0.0, device=device)
        if w_kl > 0.0:
            # Prepare logits on real vs adapted
            # We use the actor's forward path to produce logits; reset hidden states for batch
            src_actor.reset_hidden_states(batch_size=src_batch.size(0))
            logits_real = src_actor(src_batch)           # [B, src_act_dim]
            src_actor.reset_hidden_states(batch_size=adapted_src_obs.size(0))
            logits_adapt = src_actor(adapted_src_obs)    # [B, src_act_dim]

            # Compute KL( real || adapt )
            # Use log_softmax for P, softmax for Q: KL(P||Q) = sum P * (logP - logQ)
            logP = torch.log_softmax(logits_real, dim=-1)
            Q    = torch.softmax(logits_adapt, dim=-1)
            loss_kl = kl_div(logP, Q)

        loss = w_mmd * loss_mmd + w_kl * loss_kl

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mapper.parameters(), max_norm=1.0)
        opt.step()

        if step % args.log_every == 0:
            print(f"[Unpaired] Step {step}/{args.steps}  Loss: {loss.item():.6f}  (MMD {loss_mmd.item():.6f}, KL {loss_kl.item():.6f})")

        # Simple best checkpointing
        if loss.item() < best:
            best = loss.item()
            save_transformer(mapper, args.out_path)
            print(f"  -> Saved best checkpoint to {args.out_path} (key='transformer')")

    print("Training finished.")
    return args.out_path


def save_transformer(mapper: TransformerMapper, out_path: str):
    """
    Save a checkpoint with the exact key your loader expects: 'transformer'.
    This ensures TransformerObservationAdapter loads and then freezes it. [1](https://ericsson-my.sharepoint.com/personal/perepu_satheesh_kumar_ericsson_com/Documents/Microsoft%20Copilot%20Chat%20Files/malt_transformer.py)
    """
    ckpt = {'transformer': mapper.state_dict()}
    torch.save(ckpt, out_path)


# ------------------------------
# CLI
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train TransformerMapper for MALT transfer (SMAC)")
    sub = parser.add_subparsers(dest="mode", required=True)

    # Supervised mode
    p_sup = sub.add_parser("supervised", help="Train with paired (target_obs, source_obs) data")
    p_sup.add_argument("--paired_npz", type=str, required=True, help="Path to .npz with arrays target_obs, source_obs")
    p_sup.add_argument("--epochs", type=int, default=50)
    p_sup.add_argument("--batch_size", type=int, default=256)
    p_sup.add_argument("--lr", type=float, default=1e-3)
    p_sup.add_argument("--model_dim", type=int, default=128)
    p_sup.add_argument("--n_heads", type=int, default=4)
    p_sup.add_argument("--n_layers", type=int, default=2)
    p_sup.add_argument("--token_size", type=int, default=16)
    p_sup.add_argument("--dropout", type=float, default=0.1)
    p_sup.add_argument("--out_path", type=str, required=True, help="Where to save adapter .pth (with key 'transformer')")

    # Unpaired distillation mode
    p_uns = sub.add_parser("unpaired", help="Train without pairs via feature MMD, optional logit KL")
    p_uns.add_argument("--target_map", type=str, default="3m")
    p_uns.add_argument("--source_map", type=str, default="8m")
    p_uns.add_argument("--source_actor_path", type=str, required=True, help="Path to source agent checkpoint (e.g., malt_8m_agent_0.pth)")
    p_uns.add_argument("--hidden_size", type=int, default=256)
    p_uns.add_argument("--seed", type=int, default=42)
    p_uns.add_argument("--batch_size", type=int, default=128)
    p_uns.add_argument("--steps", type=int, default=50000)
    p_uns.add_argument("--max_steps_per_episode", type=int, default=200)
    p_uns.add_argument("--lr", type=float, default=1e-3)
    p_uns.add_argument("--model_dim", type=int, default=128)
    p_uns.add_argument("--n_heads", type=int, default=4)
    p_uns.add_argument("--n_layers", type=int, default=2)
    p_uns.add_argument("--token_size", type=int, default=16)
    p_uns.add_argument("--dropout", type=float, default=0.1)
    p_uns.add_argument("--w_mmd", type=float, default=1.0, help="Weight on MMD feature loss")
    p_uns.add_argument("--w_kl", type=float, default=0.0, help="Weight on KL(logits) loss; set >0 to enable")
    p_uns.add_argument("--mmd_sigma", type=float, default=1.0, help="RBF kernel sigma")
    p_uns.add_argument("--log_every", type=int, default=200)
    p_uns.add_argument("--out_path", type=str, required=True, help="Where to save adapter .pth (with key 'transformer')")

    args = parser.parse_args()
    torch.manual_seed( args.seed if hasattr(args, "seed") else 42 )
    np.random.seed(    args.seed if hasattr(args, "seed") else 42 )

    if args.mode == "supervised":
        train_supervised(args)
    elif args.mode == "unpaired":
        train_unpaired_distill(args)
    else:
        raise ValueError("Unknown mode")

if __name__ == "__main__":
    main()
