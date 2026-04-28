#!/bin/bash
# ============================================================================
# PAPER Q1 / Fig. 2 — GPT2 Kinematics-Only Baseline (no semantic grounding)
# ============================================================================
# Runs the GPT2 backbone WITHOUT VisionTRAP text alignment.
# Kinematics-only baseline for Q1 comparison (Fig. 2 in paper).
# ============================================================================

cd "$(dirname "$0")"

python3 trajectory_experiment_gpt2_vlm.py \
    --dataset_path /data/nuscenes_fixed_matrices \
    --text_manifest_path /data/nuplan_text_finetuned/nuplan_text_manifest_vlm_metadata_enhanced.parquet \
    --max_files 130 \
    --max_iters 50 \
    --num_steps_per_iter 100 \
    --batch_size 16 \
    --embed_dim 256 \
    --n_layer 3 \
    --n_head 4 \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --warmup_steps 500 \
    --context_length 15 \
    --prediction_horizon 10 \
    --max_movement_filter 10.0 \
    --movement_variance_filter 50.0 \
    --max_objects 128 \
    --traj_loss_type mse \
    --lateral_weight 2.5 \
    --time_weighting none \
    --enable_oversampling \
    --target_straight_frac 0.25 \
    --target_mild_frac 0.35 \
    --target_strong_frac 0.40 \
    2>&1 | tee training_gpt2_kinematics_only_$(date +%Y%m%d_%H%M%S).log
