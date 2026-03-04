# MALT-SMAC

This repository provides the official implementation for reproducing the experiments presented in the paper “ASALT: Adaptive State Alignment for Lateral Transfer in Multi‑agent Reinforcement Learning.”
The framework supports two types of knowledge transfer between source and target multi‑agent reinforcement learning (MARL) tasks with mismatched observation and state dimensions.
🔄 Types of Transfer
1. Critic + Actor Transfer
File: Critic_Actor_Transfer.py
This script transfers knowledge from both actors and critics, using dedicated observation and state adapters to handle domain mismatches.
2. Actor‑Only Transfer
File: malt_transformer.py
This implementation transfers only the actor networks, combined with observation adapters, without involving critic parameters.

🧪 Baseline Training
MAPPO Baseline
File: mappo_baseline_script.py
This script trains a standard MAPPO model on the SMAC environment without any transfer learning components.
Use this to compare the performance of transfer‑based methods against a non‑transfer baseline.

