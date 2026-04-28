#!/usr/bin/env python3
"""
Qwen + LoRA + VisionTRAP Trajectory Experiment
===============================================

NEW additive experiment — does NOT modify or replace any existing files:
  - trajectory_experiment_qwen_vlm.py     (unchanged — from-scratch Qwen)
  - trajectory_experiment_visiontrap.py   (unchanged — base DT)
  - qwen_trajectory_model.py              (unchanged)

What this adds
--------------
Uses the SAME VisionTRAP pipeline and data infrastructure, but swaps the
backbone from a randomly-initialized Qwen2 to a PRE-TRAINED Qwen2-1.5B-Instruct
with LoRA adapters.  This produces the "fair LLM fine-tuning baseline" the
professor requested:

    Comparison:
        trajectory_experiment_qwen_vlm.py      → Qwen2 architecture, RANDOM WEIGHTS
        THIS script                            → Qwen2-1.5B-Instruct, PRE-TRAINED + LoRA

Usage (mirrors run_qwen_vlm_lambda_05_test.sh):
    python trajectory_experiment_qwen_lora_vlm.py \
        --dataset_path /data/nuscenes_fixed_matrices \
        --text_manifest_path /data/nuplan_text_finetuned/nuplan_text_manifest_vlm_metadata_enhanced.parquet \
        --max_files 130 \
        --max_iters 50 \
        --batch_size 16 \
        --learning_rate 1e-4 \
        --weight_decay 1e-4 \
        --warmup_steps 500 \
        --context_length 15 \
        --prediction_horizon 10 \
        --text_loss_weight 0.1 \
        --enable_visiontrap \
        --enable_oversampling \
        --max_movement_filter 10.0 \
        --movement_variance_filter 50.0 \
        --num_steps_per_iter 100

Note: --embed_dim / --n_layer / --n_head are accepted but IGNORED — the
pre-trained model's architecture (hidden=1536, 28 layers, 16 heads) is used.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0 (separate from standard Qwen on GPU 1)

# Import the full VisionTRAP pipeline as base
import trajectory_experiment_visiontrap as base_experiment

# Import the new LoRA model
from qwen_lora_trajectory_model import QwenLoRATrajectoryPredictor


def create_qwen_lora_model(*args, **kwargs):
    """Replace DecisionTransformer with QwenLoRATrajectoryPredictor."""
    return QwenLoRATrajectoryPredictor(*args, **kwargs)


# Monkey-patch — same pattern as trajectory_experiment_qwen_vlm.py
base_experiment.DecisionTransformer = create_qwen_lora_model


if __name__ == "__main__":
    print("=" * 80)
    print("QWEN + LoRA + VisionTRAP Trajectory Experiment")
    print("=" * 80)
    print("  Backbone:  Qwen2-1.5B-Instruct (PRE-TRAINED)")
    print("  Adapters:  LoRA (r=16, alpha=32) — parameter-efficient fine-tuning")
    print("  Pipeline:  Full VisionTRAP (InfoNCE semantic stream + trajectory stream)")
    print("  λ=0.1:     Same text loss weight as best Qwen baseline")
    print()
    print("  Comparison baseline:")
    print("    trajectory_experiment_qwen_vlm.py  →  Qwen2 from SCRATCH (ADE=3.36m)")
    print("    THIS script                        →  Qwen2-1.5B PRE-TRAINED + LoRA")
    print("=" * 80)
    print()

    # Run the full VisionTRAP experiment (now with LoRA-adapted pre-trained Qwen backbone)
    base_experiment.main()
