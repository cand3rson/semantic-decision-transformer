#!/bin/bash
# Lambda Sensitivity Test: λ = 0.8 (EXTREME Text-Dominant)
# Testing how extreme text loss weight affects trajectory prediction
# 
# Configuration: IDENTICAL to best descriptive prompt run (1.67m ADE)
# ONLY CHANGE: text_loss_weight = 0.8 (increased from 0.1)
#
# Expected: Severe degradation - text loss will completely dominate

cd /home/chris/CascadeProjects/decision-transformer-ref-for-nuscenes

echo "================================================================================"
echo "LAMBDA SENSITIVITY TEST: λ = 0.8 (EXTREME)"
echo "================================================================================"
echo "Baseline (λ=0.1): ADE=1.67m, FDE=3.27m, Miss Rate=12.8%"
echo "Previous (λ=0.5): ADE=7.39m, FDE=13.77m, Miss Rate=77.8%"
echo "Testing (λ=0.8): EXTREME text-dominant configuration"
echo "================================================================================"
echo ""

python3 trajectory_experiment_visiontrap.py \
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
    2>&1 | tee training_dt_vlm_lambda_08_descriptive_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "================================================================================"
echo "Lambda 0.8 test completed!"
echo "Check log file for results comparison with baseline λ=0.1 and λ=0.5"
echo "================================================================================"
