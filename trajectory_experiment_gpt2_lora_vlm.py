#!/usr/bin/env python3
"""
GPT-2 + LoRA + VisionTRAP Trajectory Experiment
================================================

NEW additive experiment — does NOT modify or replace any existing files:
  - trajectory_experiment_gpt2_vlm.py    (unchanged — from-scratch GPT-2)
  - trajectory_experiment_visiontrap.py  (unchanged — base DT)
  - trajectory_experiment_qwen_lora_vlm.py (unchanged — Qwen LoRA baseline)
  - gpt2_trajectory_model.py             (unchanged)

What this adds
--------------
Uses the SAME VisionTRAP pipeline and data infrastructure as the Qwen LoRA run,
but swaps the backbone to a PRE-TRAINED GPT-2 (124M) with LoRA adapters.

This gives a direct apples-to-apples comparison:
    trajectory_experiment_qwen_lora_vlm.py   → Qwen2-1.5B-Instruct + LoRA  (ADE=2.59m)
    THIS script                              → GPT-2 base (124M) + LoRA      (TBD)
    trajectory_experiment_gpt2_vlm.py        → GPT-2 from scratch config     (ADE=2.05m, FDE=4.44m, MR=25.5%)

Usage (mirrors run_qwen_lora_vlm_lambda_01_test.sh exactly):
    python3 trajectory_experiment_gpt2_lora_vlm.py \\
        --dataset_path /data/nuscenes_fixed_matrices \\
        --text_manifest_path /data/nuplan_text_finetuned/nuplan_text_manifest_vlm_metadata_enhanced.parquet \\
        --max_files 130 \\
        --max_iters 50 \\
        --batch_size 16 \\
        --learning_rate 1e-4 \\
        --weight_decay 1e-4 \\
        --warmup_steps 500 \\
        --context_length 15 \\
        --prediction_horizon 10 \\
        --text_loss_weight 0.1 \\
        --text_warmup_epochs 10 \\
        --text_loss_cap 0.2 \\
        --infonce_scale_factor 0.001 \\
        --text_queue_size 2048 \\
        --enable_visiontrap \\
        --enable_oversampling \\
        --target_straight_frac 0.25 \\
        --target_mild_frac 0.35 \\
        --target_strong_frac 0.40 \\
        --max_movement_filter 10.0 \\
        --movement_variance_filter 200.0 \\
        --max_objects 128 \\
        --traj_loss_type mse \\
        --lateral_weight 3.5 \\
        --time_weighting none \\
        --num_steps_per_iter 100 \\
        --lr_schedule cosine

Note: --embed_dim / --n_layer / --n_head are accepted but IGNORED — the
pre-trained GPT-2 architecture (hidden=768, 12 layers, 12 heads) is used.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Import the full VisionTRAP pipeline as base (unchanged)
import trajectory_experiment_visiontrap as base_experiment

# Import the new GPT-2 LoRA model
from gpt2_lora_trajectory_model import GPT2LoRATrajectoryPredictor


def create_gpt2_lora_model(*args, **kwargs):
    """Replace DecisionTransformer with GPT2LoRATrajectoryPredictor."""
    return GPT2LoRATrajectoryPredictor(*args, **kwargs)


# Monkey-patch — same pattern as qwen_lora_vlm
base_experiment.DecisionTransformer = create_gpt2_lora_model


if __name__ == "__main__":
    print("=" * 80)
    print("GPT-2 + LoRA + VisionTRAP Trajectory Experiment")
    print("=" * 80)
    print("  Backbone:  GPT-2 base (PRE-TRAINED, 124M params)")
    print("  Adapters:  LoRA (r=16, alpha=32) — parameter-efficient fine-tuning")
    print("  Pipeline:  Full VisionTRAP (InfoNCE semantic stream + trajectory stream)")
    print("  λ=0.1:     Same text loss weight as best Qwen LoRA run")
    print()
    print("  Comparison baselines:")
    print("    GPT-2 from-scratch (SemDT, λ=0.1): ADE=2.05m, FDE=4.44m, MR=25.5%")
    print("    Qwen+LoRA (VisionTRAP, λ=0.1):     ADE=2.59m, FDE=5.42m, MR=22.2%")
    print("    Instruction Tuning + Qwen LoRA:    ADE=1.99m, FDE=3.02m, MR=11.1%")
    print("    SemDT (DT backbone, best):          ADE=1.67m, FDE=3.27m, MR=12.8%")
    print("=" * 80)
    print()

    # Run the full VisionTRAP experiment (now with LoRA-adapted pre-trained GPT-2)
    base_experiment.main()
