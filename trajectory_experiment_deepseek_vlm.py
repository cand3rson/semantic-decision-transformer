#!/usr/bin/env python3
"""
DeepSeek + VLM Trajectory Prediction Experiment
============================================================

This script tests whether VLM textual descriptions improve DeepSeek's
trajectory prediction performance.

Combines:
1. DeepSeek transformer backbone
2. VisionTRAP text contrastive learning (InfoNCE loss)
3. Same training setup as best VLM run

Purpose: Determine if textual descriptions aid offline RL trajectory
prediction with DeepSeek architecture.

Based on:
- trajectory_experiment_deepseek.py (pure DeepSeek swap)
- trajectory_experiment_visiontrap.py (VLM-enhanced DT)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Use the VisionTRAP experiment as base (it has all the infrastructure)
import trajectory_experiment_visiontrap as base_experiment

# Import DeepSeek trajectory model wrapper
from deepseek_trajectory_model import DeepSeekTrajectoryPredictor

# Patch the model creation in base experiment
original_create_model = base_experiment.DecisionTransformer

def create_deepseek_model(*args, **kwargs):
    """Replace DecisionTransformer with DeepSeekTrajectoryPredictor"""
    return DeepSeekTrajectoryPredictor(*args, **kwargs)

# Monkey-patch the model class
base_experiment.DecisionTransformer = create_deepseek_model

# Run the experiment with DeepSeek backbone
if __name__ == '__main__':
    print("=" * 80)
    print("DeepSeek + VLM Trajectory Prediction Experiment")
    print("=" * 80)
    print(f"Transformer: DeepSeekTrajectoryPredictor")
    print(f"VLM: Enabled (VisionTRAP)")
    print("=" * 80)
    print()
    
    # Run the base VisionTRAP experiment (now with DeepSeek backbone)
    base_experiment.main()
