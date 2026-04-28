#!/usr/bin/env python3
"""
GPT-2 + VLM Trajectory Prediction Experiment
============================================================

This script tests whether VLM textual descriptions improve GPT-2's
trajectory prediction performance.

Combines:
1. GPT-2 transformer backbone
2. VisionTRAP text contrastive learning (InfoNCE loss)
3. Same training setup as best VLM run

Purpose: Determine if textual descriptions aid offline RL trajectory
prediction with GPT-2 architecture.

Based on:
- trajectory_experiment_gpt2.py (pure GPT-2 swap)
- trajectory_experiment_visiontrap.py (VLM-enhanced DT)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Use the VisionTRAP experiment as base (it has all the infrastructure)
import trajectory_experiment_visiontrap as base_experiment

# Import GPT-2 trajectory model wrapper
from gpt2_trajectory_model import GPT2TrajectoryPredictor

# Patch the model creation in base experiment
original_create_model = base_experiment.DecisionTransformer

def create_gpt2_model(*args, **kwargs):
    """Replace DecisionTransformer with GPT2TrajectoryPredictor"""
    return GPT2TrajectoryPredictor(*args, **kwargs)

# Monkey-patch the model class
base_experiment.DecisionTransformer = create_gpt2_model

# Run the experiment with GPT-2 backbone
if __name__ == '__main__':
    print("=" * 80)
    print("GPT2 + VLM Trajectory Prediction Experiment")
    print("=" * 80)
    print(f"Transformer: GPT2TrajectoryPredictor")
    print(f"VLM: Enabled (VisionTRAP)")
    print("=" * 80)
    print()
    
    # Run the base VisionTRAP experiment (now with GPT-2 backbone)
    base_experiment.main()
