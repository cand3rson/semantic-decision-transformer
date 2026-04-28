#!/bin/bash

################################################################################
# Qwen VLM Lambda Sensitivity Test: λ=0.8
################################################################################
# 
# Purpose: Test Qwen + VLM with HIGH text loss weight (λ=0.8)
#
# Comparison:
# - Baseline (λ=0.1): ADE=3.36m, FDE=6.39m, Miss=65.0%
# - Lambda=0.5: ADE=7.52m, FDE=15.58m, Miss=77.8%
# - Lambda=0.8: Testing now...
#
# Context: 
# - Decision Transformer λ=0.8: ADE=2.35m (low sensitivity)
# - GPT-2 λ=0.8: ADE=6.86m (very high sensitivity)
# - LLaMA λ=0.8: ADE=7.24m (high sensitivity)
# - Qwen λ=0.8: Testing to complete architecture comparison
#
################################################################################

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="training_qwen_vlm_lambda08_${TIMESTAMP}.log"

echo "================================================================================
Qwen VLM Lambda Sensitivity Test: λ=0.8
================================================================================
Configuration:
  - Architecture: Qwen Transformer Swap
  - VLM Pipeline: BLIP-2 → GPT-2 Refinement → CLIP Embeddings
  - Text Loss Weight: 0.8 (HIGH - text-dominant)
  - Prompt Type: Detailed (best performing)
  - Training: 50 iterations
  - Baseline: λ=0.1 (ADE=3.36m)
  - Previous: λ=0.5 (ADE=7.52m)
================================================================================
"

cd /home/chris/CascadeProjects/decision-transformer-ref-for-nuscenes

python3 trajectory_experiment_qwen_vlm.py \
    --dataset_path /data/nuscenes_fixed_matrices \
    --text_manifest_path /data/nuplan_text_finetuned/nuplan_text_manifest_vlm_metadata_enhanced.parquet \
    --max_files 130 \
    --max_iters 50 \
    --batch_size 16 \
    --embed_dim 256 \
    --n_layer 3 \
    --n_head 4 \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --warmup_steps 500 \
    --context_length 15 \
    --prediction_horizon 10 \
    --text_loss_weight 0.8 \
    --text_warmup_epochs 10 \
    --text_loss_cap 0.2 \
    --infonce_scale_factor 0.001 \
    --text_queue_size 2048 \
    --enable_visiontrap \
    --enable_oversampling \
    --target_straight_frac 0.25 \
    --target_mild_frac 0.35 \
    --target_strong_frac 0.40 \
    --max_movement_filter 10.0 \
    --movement_variance_filter 50.0 \
    --max_objects 128 \
    --traj_loss_type mse \
    --lateral_weight 2.5 \
    --time_weighting none \
    --num_steps_per_iter 100 \
    --enable_plots \
    --save_plots \
    2>&1 | tee "$LOG_FILE"

echo "
================================================================================
Qwen Lambda 0.8 test completed!
Log saved to: $LOG_FILE

Next steps:
1. Compare with λ=0.1 baseline (ADE=3.36m)
2. Compare with λ=0.5 test (ADE=7.52m)
3. Compare with Decision Transformer λ=0.8 (ADE=2.35m)
4. Compare with GPT-2 λ=0.8 (ADE=6.86m)
5. Compare with LLaMA λ=0.8 (ADE=7.24m)
6. Create final comprehensive lambda sensitivity analysis across all architectures
================================================================================
"
