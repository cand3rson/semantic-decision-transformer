#!/bin/bash
# ============================================================================
# PAPER Table II — DEEPSEEK + SEMDT INTENT, λ=0.1
# ============================================================================
# Backbone swap: DEEPSEEK under identical interface and semantic supervision.
# Reproduces Table II entry for DEEPSEEK.
# ============================================================================

cd /home/chris/CascadeProjects/decision-transformer-ref-for-nuscenes

python3 trajectory_experiment_deepseek_vlm.py \
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
    --text_loss_weight 0.1 \
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
    2>&1 | tee training_deepseek_vlm_lambda01_$(date +%Y%m%d_%H%M%S).log
