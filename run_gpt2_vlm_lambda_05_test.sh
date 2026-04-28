#!/bin/bash
# Lambda Sensitivity Test for GPT-2: λ = 0.5
# Testing how text loss weight affects GPT-2 transformer swap performance
# 
# Configuration: IDENTICAL to best GPT-2 VLM descriptive prompt run
# Baseline (λ=0.1): ADE=2.05m, FDE=4.44m, Lateral ADE=1.76m, Long ADE=0.71m, Miss Rate=25.5%
# ONLY CHANGE: text_loss_weight = 0.5 (increased from 0.1)
#
# Expected: Performance degradation similar to DT results

cd "$(dirname "$0")"

echo "================================================================================"
echo "GPT-2 + VLM LAMBDA SENSITIVITY TEST: λ = 0.5"
echo "================================================================================"
echo "Baseline (λ=0.1): ADE=2.05m, FDE=4.44m, Miss Rate=25.5%"
echo "Testing (λ=0.5): Text-dominant configuration"
echo "Architecture: GPT-2 Transformer Swap + VLM"
echo "================================================================================"
echo ""

python3 trajectory_experiment_gpt2_vlm.py \
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
    --text_loss_weight 0.5 \
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
    2>&1 | tee training_gpt2_vlm_lambda_05_descriptive_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "================================================================================"
echo "GPT-2 Lambda 0.5 test completed!"
echo "Check log file for results comparison with baseline λ=0.1"
echo "================================================================================"
