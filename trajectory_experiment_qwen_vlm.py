#!/usr/bin/env python3
"""
QWEN + VLM Trajectory Prediction Experiment
============================================================

This script tests whether VLM textual descriptions improve QWEN's
trajectory prediction performance.

Combines:
1. QWEN transformer backbone
2. VisionTRAP text contrastive learning (InfoNCE loss)
3. Same training setup as best VLM run (1.67m ADE)

Purpose: Determine if textual descriptions aid offline RL trajectory
prediction across different transformer architectures.

Based on:
- trajectory_experiment_qwen.py (pure QWEN swap)
- trajectory_experiment_visiontrap.py (VLM-enhanced DT)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Use the VisionTRAP experiment as base (it has all the infrastructure)
import trajectory_experiment_visiontrap as base_experiment

# Import QWEN model
from qwen_trajectory_model import QwenTrajectoryPredictor

# Patch the model creation in base experiment
original_create_model = base_experiment.DecisionTransformer

def create_qwen_model(*args, **kwargs):
    """Replace DecisionTransformer with QwenTrajectoryPredictor"""
    return QwenTrajectoryPredictor(*args, **kwargs)

# Monkey-patch the model class
base_experiment.DecisionTransformer = create_qwen_model

# Run the experiment with QWEN backbone
if __name__ == '__main__':
    print("=" * 80)
    print("QWEN + VLM Trajectory Prediction Experiment")
    print("=" * 80)
    print(f"Transformer: QwenTrajectoryPredictor")
    print(f"VLM: Enabled (VisionTRAP)")
    print("=" * 80)
    print()
    
    # Run the base VisionTRAP experiment (now with QWEN backbone)
    base_experiment.main()
