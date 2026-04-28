#!/usr/bin/env python3
"""
LLaMA + VLM Trajectory Prediction Experiment
============================================================

This script tests whether VLM textual descriptions improve LLaMA's
trajectory prediction performance.

Combines:
1. LLaMA transformer backbone
2. VisionTRAP text contrastive learning (InfoNCE loss)
3. Same training setup as best VLM run

Purpose: Determine if textual descriptions aid offline RL trajectory
prediction with LLaMA architecture.

Based on:
- trajectory_experiment_llama.py (pure LLaMA swap)
- trajectory_experiment_visiontrap.py (VLM-enhanced DT)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Use the VisionTRAP experiment as base (it has all the infrastructure)
import trajectory_experiment_visiontrap as base_experiment

# Import LLaMA trajectory model wrapper
from llama_trajectory_model import LlamaTrajectoryPredictor

# Patch the model creation in base experiment
original_create_model = base_experiment.DecisionTransformer

def create_llama_model(*args, **kwargs):
    """Replace DecisionTransformer with LlamaTrajectoryPredictor"""
    return LlamaTrajectoryPredictor(*args, **kwargs)

# Monkey-patch the model class
base_experiment.DecisionTransformer = create_llama_model

# Run the experiment with LLaMA backbone
if __name__ == '__main__':
    print("=" * 80)
    print("LLaMA + VLM Trajectory Prediction Experiment")
    print("=" * 80)
    print(f"Transformer: LlamaTrajectoryPredictor")
    print(f"VLM: Enabled (VisionTRAP)")
    print("=" * 80)
    print()
    
    # Run the base VisionTRAP experiment (now with LLaMA backbone)
    base_experiment.main()
