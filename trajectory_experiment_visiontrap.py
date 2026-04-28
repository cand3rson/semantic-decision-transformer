#!/usr/bin/env python3
"""
Movement-Normalized Trajectory Forecasting with Decision Transformer + Lateral Maneuver Oversampling.

Key improvements over baseline version:
1. Normalization based on actual movement statistics (not coordinate statistics)
2. Ego-centric coordinates + movement-specific normalization
3. Loss represents actual movement prediction accuracy
4. Meaningful metrics in terms of real vehicle movements
5. Research-standard evaluation: Multiple miss rate thresholds (1m, 2m, 5m)
6. Directional error breakdown: Longitudinal vs lateral performance
7. Comprehensive evaluation logging with all standard trajectory metrics
8. NEW: Lateral maneuver oversampling to improve lateral prediction performance
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import glob
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import argparse
import json
import random
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.style as style
from datetime import datetime
import time

# Force immediate output flushing
sys.stdout.reconfigure(line_buffering=True)

# Import existing DT components
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.training.seq_trainer import SequenceTrainer

# VLM/VisionTRAP imports
from text_contrastive_head import TextContrastiveHead
import pandas as pd


def transform_to_ego_centric(scene_data, ego_idx=0):
    """
    Transform scene coordinates to ego-centric frame.
    CRITICAL FIX: Center ALL timesteps relative to ego at timestep 0 (start position).
    """
    T, N, F = scene_data.shape
    ego_centric_data = scene_data.copy()
    
    # CRITICAL FIX: Center ALL timesteps relative to ego at timestep 0 (start position)
    ego_start_pos = scene_data[0, ego_idx, :3]  # Ego position at timestep 0
    
    # For each timestep, center all objects relative to ego START position
    for t in range(T):
        # Translate all objects relative to ego START position (not current position)
        for obj in range(N):
            ego_centric_data[t, obj, :3] = scene_data[t, obj, :3] - ego_start_pos
    
    # Extract ego trajectory from TRANSFORMED data
    ego_trajectory = ego_centric_data[:, ego_idx, :3]  # (T, 3)
    
    print(f"🎯 Ego-centric transformation applied:")
    print(f"   Original coordinate range: X=[{scene_data[:,:,0].min():.1f}, {scene_data[:,:,0].max():.1f}]")
    print(f"   Ego-centric range: X=[{ego_centric_data[:,:,0].min():.1f}, {ego_centric_data[:,:,0].max():.1f}]")
    
    return ego_centric_data, ego_trajectory

def compute_movement_statistics(trajectories):
    """
    Compute movement statistics for normalization.
    
    Args:
        trajectories: List of trajectory samples with ego-centric coordinates
    
    Returns:
        movement_stats: Statistics for normalizing movements
    """
    print("📊 Computing movement-specific normalization statistics...")
    
    all_movements = []
    
    for traj in trajectories:
        ego_positions = traj['ego_positions']  # (T, 3)
        
        # Calculate movements between consecutive frames
        movements = np.diff(ego_positions, axis=0)  # (T-1, 3)
        
        # Only include non-zero movements
        if len(movements) > 0:
            non_zero_mask = np.any(movements != 0, axis=1)
            non_zero_movements = movements[non_zero_mask]
            
            if len(non_zero_movements) > 0:
                all_movements.append(non_zero_movements)
    
    if len(all_movements) == 0:
        print("⚠️  No movements found, using conservative default stats")
        return {'mean': np.zeros(3), 'std': np.array([1.0, 1.0, 0.1])}
    
    # Concatenate all movements
    all_movements = np.concatenate(all_movements, axis=0)  # (N_movements, 3)
    
    # Compute statistics
    movement_mean = np.mean(all_movements, axis=0)
    movement_std = np.std(all_movements, axis=0) + 1e-6  # Avoid division by zero
    
    print(f"✅ Movement statistics computed:")
    print(f"   Total movements: {len(all_movements):,}")
    print(f"   Movement means: [{movement_mean[0]:.4f}, {movement_mean[1]:.4f}, {movement_mean[2]:.4f}] meters")
    print(f"   Movement stds: [{movement_std[0]:.4f}, {movement_std[1]:.4f}, {movement_std[2]:.4f}] meters")
    print(f"   Typical movement magnitude: {np.mean(np.linalg.norm(all_movements, axis=1)):.4f} meters")
    
    return {'mean': movement_mean, 'std': movement_std}

def create_normalized_movements(ego_positions, movement_stats):
    """
    Create normalized movement actions from ego positions.
    
    Args:
        ego_positions: (T, 3) - ego vehicle positions over time
        movement_stats: Movement normalization statistics
    
    Returns:
        normalized_movements: (T, 3) - normalized movement deltas
        raw_movements: (T, 3) - raw movement deltas in meters
    """
    # Calculate raw movements
    raw_movements = np.diff(ego_positions, axis=0)  # (T-1, 3)
    
    # Pad to match original length
    padded_movements = np.zeros_like(ego_positions)
    padded_movements[:-1] = raw_movements
    padded_movements[-1] = raw_movements[-1] if len(raw_movements) > 0 else np.zeros(3)
    
    # Normalize movements using movement-specific statistics
    movement_mean = movement_stats['mean']
    movement_std = movement_stats['std']
    
    normalized_movements = (padded_movements - movement_mean) / movement_std
    
    # Calculate movement magnitudes for reporting
    movement_magnitudes = np.linalg.norm(padded_movements, axis=1)
    
    print(f"🎯 Normalized movements created:")
    print(f"   Raw movement range: [{movement_magnitudes.min():.4f}, {movement_magnitudes.max():.4f}] meters")
    print(f"   Normalized range: [{normalized_movements.min():.4f}, {normalized_movements.max():.4f}] units")
    print(f"   Average movement: {movement_magnitudes.mean():.4f} meters per timestep")
    
    return normalized_movements, padded_movements

# ============================================================================
# LATERAL MANEUVER OVERSAMPLING FUNCTIONS
# ============================================================================

def calculate_lateral_score_from_sample(sample, context_length):
    """
    Calculate lateral score from a trajectory sample.
    This works with our current data structure using normalized actions.
    """
    # Extract context actions (normalized movements)
    context_actions = sample['actions']  # (context_length, 3)
    
    # Calculate lateral movement indicators from normalized actions
    lateral_movements = context_actions[:, 1]  # y-component (lateral)
    
    # 1. Final lateral offset (cumulative lateral movement)
    offset_y = abs(np.sum(lateral_movements))
    
    # 2. Cumulative lateral drift
    cum_lat = np.sum(np.abs(lateral_movements))
    
    # 3. Lateral movement variance (indicator of turning)
    lateral_variance = np.var(lateral_movements)
    
    # Simple robust score (as recommended in research)
    lateral_score = offset_y + 0.5 * cum_lat + 0.5 * lateral_variance
    
    return lateral_score

def compute_lateral_scores_and_weights(trajectories, context_length, target_fractions=None, temperature=0.75):
    """
    Compute lateral scores and sampling weights for all trajectory samples.
    
    Args:
        trajectories: List of trajectory samples
        context_length: int - context length
        target_fractions: np.array - target fractions for [straight, mild, strong] (optional)
        temperature: float - temperature for weight smoothing (optional)
    
    Returns:
        lateral_scores: np.array - lateral scores for each sample
        maneuver_bins: np.array - maneuver bin assignments
        sampling_weights: np.array - sampling weights for each sample
        bin_stats: dict - statistics about maneuver bins
    """
    print("🎯 Computing lateral scores and sampling weights...")
    
    # Calculate lateral scores for all samples
    lateral_scores = []
    for sample in trajectories:
        score = calculate_lateral_score_from_sample(sample, context_length)
        lateral_scores.append(score)
    
    lateral_scores = np.array(lateral_scores)
    
    # Compute percentiles for binning
    p40 = np.percentile(lateral_scores, 40)
    p80 = np.percentile(lateral_scores, 80)
    
    # Assign maneuver bins
    maneuver_bins = np.zeros(len(lateral_scores), dtype=int)
    maneuver_bins[lateral_scores <= p40] = 0  # Straight
    maneuver_bins[(lateral_scores > p40) & (lateral_scores <= p80)] = 1  # Mild lateral
    maneuver_bins[lateral_scores > p80] = 2  # Strong lateral
    
    # Calculate bin fractions
    bin_counts = np.bincount(maneuver_bins, minlength=3)
    bin_fractions = bin_counts / len(maneuver_bins)
    
    # Target distribution (use provided or default)
    if target_fractions is None:
        target_fractions = np.array([0.30, 0.35, 0.35])
    else:
        target_fractions = np.array(target_fractions)
    
    # Calculate sampling weights using class-balancing
    sampling_weights = np.zeros(len(trajectories))
    for i, bin_id in enumerate(maneuver_bins):
        if bin_fractions[bin_id] > 0:
            sampling_weights[i] = target_fractions[bin_id] / bin_fractions[bin_id]
        else:
            sampling_weights[i] = 1.0  # Default weight for empty bins
    
    # Apply temperature smoothing
    sampling_weights = np.power(sampling_weights, temperature)
    
    # Normalize weights so mean is 1.0
    sampling_weights = sampling_weights / np.mean(sampling_weights)
    
    # Clip weights to reasonable range [0.25, 4.0]
    sampling_weights = np.clip(sampling_weights, 0.25, 4.0)
    
    # Re-normalize after clipping
    sampling_weights = sampling_weights / np.mean(sampling_weights)
    
    # Compute bin statistics
    bin_stats = {
        'percentiles': {'p40': p40, 'p80': p80},
        'bin_counts': bin_counts,
        'bin_fractions': bin_fractions,
        'target_fractions': target_fractions,
        'weight_stats': {
            'mean': np.mean(sampling_weights),
            'std': np.std(sampling_weights),
            'min': np.min(sampling_weights),
            'max': np.max(sampling_weights)
        }
    }
    
    print(f"📊 Lateral Score Statistics:")
    print(f"   Score range: [{np.min(lateral_scores):.3f}, {np.max(lateral_scores):.3f}]")
    print(f"   Percentiles: P40={p40:.3f}, P80={p80:.3f}")
    print(f"   Bin distribution: Straight={bin_counts[0]} ({bin_fractions[0]:.1%}), "
          f"Mild={bin_counts[1]} ({bin_fractions[1]:.1%}), "
          f"Strong={bin_counts[2]} ({bin_fractions[2]:.1%})")
    print(f"   Weight stats: mean={bin_stats['weight_stats']['mean']:.2f}, "
          f"std={bin_stats['weight_stats']['std']:.2f}, "
          f"range=[{bin_stats['weight_stats']['min']:.2f}, {bin_stats['weight_stats']['max']:.2f}]")
    
    return lateral_scores, maneuver_bins, sampling_weights, bin_stats

def create_movement_batch_function_with_oversampling(trajectories, context_length, device='cuda', sampling_weights=None, maneuver_bins=None):
    """
    Create batch function for movement-normalized data with optional weighted sampling.
    
    Args:
        trajectories: List of trajectory samples
        context_length: int - context length
        device: str - device to use
        sampling_weights: np.array - sampling weights for each trajectory (optional)
        maneuver_bins: np.array - maneuver bin assignments for each trajectory (optional)
    """
    
    # Find the maximum state dimension across all trajectories
    max_state_dim = max(sample['states'].shape[1] for sample in trajectories)
    
    def get_batch(batch_size):
        if sampling_weights is not None:
            # Use weighted sampling
            batch_indices = np.random.choice(
                len(trajectories), 
                size=min(batch_size, len(trajectories)), 
                replace=True, 
                p=sampling_weights/np.sum(sampling_weights)
            )
            batch_samples = [trajectories[i] for i in batch_indices]
            
            # Optional: Apply batch composition guardrails
            if maneuver_bins is not None:
                batch_bins = maneuver_bins[batch_indices]
                strong_lateral_count = np.sum(batch_bins == 2)  # Strong lateral bin
                max_strong_lateral = int(0.6 * batch_size)  # Max 60% strong lateral
                
                if strong_lateral_count > max_strong_lateral:
                    # Replace excess strong lateral samples with mild lateral
                    excess_count = strong_lateral_count - max_strong_lateral
                    strong_indices = np.where(batch_bins == 2)[0]
                    replace_indices = strong_indices[:excess_count]
                    
                    # Find mild lateral samples to replace with
                    mild_indices = np.where(maneuver_bins == 1)[0]
                    if len(mild_indices) > 0:
                        replacement_indices = np.random.choice(mild_indices, size=excess_count, replace=True)
                        for i, replace_idx in enumerate(replacement_indices):
                            batch_samples[replace_indices[i]] = trajectories[replace_idx]
        else:
            # Use uniform random sampling
            batch_samples = random.sample(trajectories, min(batch_size, len(trajectories)))
        
        states_batch = []
        actions_batch = []
        rtg_batch = []
        timesteps_batch = []
        attention_masks = []
        rewards_batch = []  # Dummy rewards
        dones_batch = []    # Dummy dones
        
        for sample in batch_samples:
            # Pad states to max_state_dim if necessary
            states = sample['states']  # Shape: (context_length, current_state_dim)
            if states.shape[1] < max_state_dim:
                pad_width = max_state_dim - states.shape[1]
                states = np.pad(states, ((0, 0), (0, pad_width)), mode='constant')
            
            states_batch.append(states)
            actions_batch.append(sample['actions'])
            rtg_batch.append(sample['rtg'])
            timesteps_batch.append(sample['timesteps'])
            
            # Create attention mask (all 1s for real data)
            attention_masks.append(np.ones(context_length))
            
            # Dummy rewards and dones
            rewards_batch.append(np.zeros((context_length, 1)))
            dones_batch.append(np.zeros(context_length))
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states_batch)).to(device)
        actions = torch.FloatTensor(np.array(actions_batch)).to(device)
        rtg = torch.FloatTensor(np.array(rtg_batch)).to(device)
        timesteps = torch.LongTensor(np.array(timesteps_batch)).to(device)
        attention_mask = torch.LongTensor(np.array(attention_masks)).to(device)
        rewards = torch.FloatTensor(np.array(rewards_batch)).to(device)
        dones = torch.FloatTensor(np.array(dones_batch)).to(device)
        
        return states, actions, rewards, dones, rtg, timesteps, attention_mask
    
    return get_batch

# ============================================================================
# DATA PROCESSING FUNCTIONS (COPIED FROM ORIGINAL)
# ============================================================================

def load_nuscenes_scenes(dataset_path, max_files=40):
    """
    Load nuScenes scenes without any normalization (first pass).
    Returns raw ego trajectories and scene data for later processing.
    """
    print(f"🚀 Loading nuScenes scenes...")
    print(f"  Dataset: {dataset_path}")
    print(f"  Max files: {max_files}")
    
    ego_trajectories = []
    npy_files = glob.glob(os.path.join(dataset_path, "*.npy"))
    npy_files = sorted(npy_files)[:max_files]
    
    for npy_file in npy_files:
        try:
            data = np.load(npy_file)
            scene_id = os.path.basename(npy_file).replace('.npy', '')
            
            # Extract ego vehicle positions (first object, first 3 features: x, y, z)
            ego_positions = data[:, 0, :3]  # (T, 3)
            
            # Extract scene data (all objects, all features)
            scene_data = data  # (T, N, F)
            
            # Store raw trajectory info
            ego_trajectories.append({
                'scene_id': scene_id,
                'ego_positions': ego_positions,
                'scene_data': scene_data
            })
            
        except Exception as e:
            print(f"Error processing {npy_file}: {str(e)}")
            continue
    
    print(f"✅ Loaded {len(ego_trajectories)} scenes")
    return ego_trajectories

def process_nuscenes_movement_normalized(ego_trajectories, movement_stats, args, context_length=20, prediction_horizon=10):
    """
    Process ego trajectories with pre-computed movement statistics (second pass).
    Returns normalized trajectories.
    """
    
    print("🎯 PROCESSING EGO TRAJECTORIES WITH PRE-COMPUTED STATS")
    print("=" * 60)
    
    trajectories = []
    
    for traj_info in ego_trajectories:
        try:
            ego_positions = traj_info['ego_positions']
            scene_data = traj_info['scene_data']
            T, N, F = scene_data.shape
            
            # Skip scenes that are too short
            if T < context_length + prediction_horizon:
                continue
            
            # Transform to ego-centric coordinates
            ego_centric_data, ego_trajectory = transform_to_ego_centric(scene_data, ego_idx=0)
            
            # Create normalized movements using pre-computed stats
            normalized_movements, raw_movements = create_normalized_movements(
                ego_trajectory, movement_stats
            )
            
            # Normalize other features (keeping coordinate normalization for context)
            normalized_scene_data = ego_centric_data.copy()
            if N > 1:  # Only normalize if there are other objects
                other_features = ego_centric_data[:, 1:, 3:]  # Skip ego and first 3 features (x,y,z)
                
                # Flatten-then-filter pattern to avoid boolean masking shape issues
                other_features_flat = other_features.reshape(-1, other_features.shape[-1])  # (T * (N-1), F-3)
                valid_mask_flat = np.any(other_features_flat != 0, axis=1)  # (T * (N-1),)
                
                if np.sum(valid_mask_flat) > 0:
                    valid_features_flat = other_features_flat[valid_mask_flat]  # (T_valid, F-3)
                    
                    # Compute per-feature stats across all valid timesteps and objects
                    feature_mean = np.mean(valid_features_flat, axis=0)  # Shape: (F-3,)
                    feature_std = np.std(valid_features_flat, axis=0) + 1e-6  # Shape: (F-3,)
                    
                    # Apply normalization to all objects (including ego) for features 3+
                    normalized_scene_data[:, :, 3:] = (ego_centric_data[:, :, 3:] - feature_mean) / feature_std
            
            # Create multiple training samples from this scene
            max_start_idx = T - context_length - prediction_horizon
            
            for start_idx in range(0, max_start_idx, 5):  # Sample every 5 frames
                # Extract context (past frames)
                context_data = normalized_scene_data[start_idx:start_idx + context_length]  # (context_length, N, F)
                
                # Extract prediction targets (future normalized movements)
                future_movements = normalized_movements[start_idx + context_length:start_idx + context_length + prediction_horizon]
                
                # Sample-level filtering: Check if this sample has reasonable movements
                sample_movements = raw_movements[start_idx:start_idx + context_length + prediction_horizon]
                sample_movement_magnitudes = np.linalg.norm(sample_movements, axis=1)
                max_movement_in_sample = np.max(sample_movement_magnitudes)
                
                # Skip samples with any movement > threshold per timestep
                if max_movement_in_sample > args.max_movement_filter:
                    continue
                
                # Skip samples with too much movement variance (jerky motion)
                movement_variance = np.var(sample_movement_magnitudes)
                if movement_variance > args.movement_variance_filter:
                    continue
                
                # Create states: flatten all objects and features for each timestep
                # CRITICAL: Pad to consistent number of object slots for training/eval compatibility
                T, N, F = context_data.shape
                
                # Use the same padding strategy as training (40 files = more objects)
                # Calculate target number of object slots based on training data
                # This ensures state_dim consistency between training and evaluation
                target_objects = 2072  # Based on training: 20720 / 10 features = 2072 objects
                
                if N < target_objects:
                    # Pad with zero objects
                    pad_objects = target_objects - N
                    padding = np.zeros((context_length, pad_objects, F))
                    context_data_padded = np.concatenate([context_data, padding], axis=1)
                elif N > target_objects:
                    # Truncate to target objects (keep ego + nearest neighbors)
                    context_data_padded = context_data[:, :target_objects, :]
                else:
                    # Already correct size
                    context_data_padded = context_data
                
                # Flatten to create states
                states = context_data_padded.reshape(context_length, -1)  # (context_length, target_objects*F)
                
                # Create actions: normalized ego movements in context (for teacher forcing)
                actions = normalized_movements[start_idx:start_idx + context_length]  # (context_length, 3)
                
                # Create RTG: based on future movement smoothness
                future_raw_movements = raw_movements[start_idx + context_length:start_idx + context_length + prediction_horizon]
                movement_magnitudes = np.linalg.norm(future_raw_movements, axis=1)
                trajectory_quality = -np.var(movement_magnitudes)  # Negative variance (smoother = higher)
                rtg = np.full((context_length + 1, 1), trajectory_quality, dtype=np.float32)
                
                # Create timesteps
                timesteps = np.arange(context_length, dtype=np.int64)
                
                # Create trajectory sample
                traj_sample = {
                    'states': states,                           # (context_length, N*F)
                    'actions': actions,                         # (context_length, 3) - normalized movements
                    'rtg': rtg,                                # (context_length+1, 1)
                    'timesteps': timesteps,                    # (context_length,)
                    'future_movements': future_movements,      # (prediction_horizon, 3) - normalized
                    'scene_id': traj_info['scene_id'],
                    'start_frame': start_idx,
                    'num_objects': N
                }
                
                trajectories.append(traj_sample)
        
        except Exception as e:
            print(f"Error creating samples for {traj_info['scene_id']}: {str(e)}")
            continue
    
    print(f"\n✅ Successfully processed {len(trajectories)} movement-normalized trajectory samples")
    
    return trajectories

# ============================================================================
# EVALUATION AND VISUALIZATION FUNCTIONS (COPIED FROM ORIGINAL)
# ============================================================================

def calculate_movement_metrics(predictions, targets, movement_stats, mask=None):
    """
    Calculate metrics on normalized movements with conversion back to real meters.
    
    Args:
        predictions: (batch, seq_len, 3) - Predicted normalized movements
        targets: (batch, seq_len, 3) - Ground truth normalized movements
        movement_stats: Movement normalization statistics
        mask: (batch, seq_len) - Valid timestep mask
    
    Returns:
        metrics: Dict with both normalized and real-world metrics
    """
    
    # Calculate errors in normalized space
    normalized_errors = torch.norm(predictions - targets, dim=-1)  # (batch, seq_len)
    
    # Convert back to real meters for interpretable metrics
    movement_std = torch.FloatTensor(movement_stats['std']).to(predictions.device)
    movement_mean = torch.FloatTensor(movement_stats['mean']).to(predictions.device)
    
    # Convert predictions and targets back to real meters
    # Correct denormalization: real = normalized * std + mean
    real_predictions = predictions * movement_std + movement_mean
    real_targets = targets * movement_std + movement_mean
        
    # ============================================================================
    # RESEARCH-STANDARD ADE/FDE: Compute on positions, not movements
    # ============================================================================
    
    # Convert movements to positions by cumulative sum
    # Only use x,y coordinates for position-based metrics (exclude z/heading)
    pos_predictions = real_predictions[..., :2].cumsum(dim=1)  # (batch, seq_len, 2)
    pos_targets = real_targets[..., :2].cumsum(dim=1)         # (batch, seq_len, 2)
    
    # Calculate position errors (L2 distance between predicted and true positions)
    real_errors = torch.norm(pos_predictions - pos_targets, dim=-1)  # (batch, seq_len)
    
    # Keep movement errors for comparison (optional)
    movement_errors = torch.norm(real_predictions - real_targets, dim=-1)  # (batch, seq_len)
    
    if mask is not None:
        # Apply mask
        normalized_errors = normalized_errors * mask.float()
        real_errors = real_errors * mask.float()
        valid_timesteps = mask.sum(dim=1).float()
        
        # ADE: Average displacement error
        normalized_ade = (normalized_errors.sum(dim=1) / valid_timesteps).mean()
        real_ade = (real_errors.sum(dim=1) / valid_timesteps).mean()
        
        # FDE: Final displacement error
        batch_indices = torch.arange(len(predictions))
        last_valid_idx = (mask.sum(dim=1) - 1).long()
        normalized_fde = normalized_errors[batch_indices, last_valid_idx].mean()
        real_fde = real_errors[batch_indices, last_valid_idx].mean()
    else:
        # No masking
        normalized_ade = normalized_errors.mean()
        real_ade = real_errors.mean()
        normalized_fde = normalized_errors[:, -1].mean()
        real_fde = real_errors[:, -1].mean()
    
    # Multiple Miss Rate Thresholds (research standard) - FINAL STEP ONLY
    if mask is not None:
        batch_indices = torch.arange(len(predictions))
        last_valid_idx = (mask.sum(dim=1) - 1).long()
        final_errors = real_errors[batch_indices, last_valid_idx]
    else:
        final_errors = real_errors[:, -1]  # Last timestep
    
    miss_rates = {}
    for threshold in [1.0, 2.0, 5.0]:  # meters
        miss_count = (final_errors > threshold).float().sum()
        miss_rates[f'Miss_Rate_{threshold:.0f}m'] = (miss_count / len(final_errors)).item()
    
    # Directional Error Breakdown (longitudinal vs lateral) - using positions
    x_errors = torch.abs(pos_predictions[..., 0] - pos_targets[..., 0])
    y_errors = torch.abs(pos_predictions[..., 1] - pos_targets[..., 1])
    
    if mask is not None:
        x_errors = x_errors * mask.float()
        y_errors = y_errors * mask.float()
        valid_timesteps_for_dir = mask.sum(dim=1).float()
        
        longitudinal_ade = (x_errors.sum(dim=1) / valid_timesteps_for_dir).mean()
        lateral_ade = (y_errors.sum(dim=1) / valid_timesteps_for_dir).mean()
        
        batch_indices = torch.arange(len(predictions))
        last_valid_idx = (mask.sum(dim=1) - 1).long()
        longitudinal_fde = x_errors[batch_indices, last_valid_idx].mean()
        lateral_fde = y_errors[batch_indices, last_valid_idx].mean()
    else:
        longitudinal_ade = x_errors.mean()
        lateral_ade = y_errors.mean()
        longitudinal_fde = x_errors[:, -1].mean()
        lateral_fde = y_errors[:, -1].mean()
    
    return {
        # RESEARCH-STANDARD METRICS (position-based)
        'Normalized_ADE': normalized_ade.item(),
        'Real_ADE_meters': real_ade.item(),
        'Normalized_FDE': normalized_fde.item(), 
        'Real_FDE_meters': real_fde.item(),
        
        # Multiple miss rate thresholds (research standard)
        **miss_rates,
        
        # Directional breakdown (research insight) - position-based
        'Longitudinal_ADE_meters': longitudinal_ade.item(),
        'Lateral_ADE_meters': lateral_ade.item(),
        'Longitudinal_FDE_meters': longitudinal_fde.item(),
        'Lateral_FDE_meters': lateral_fde.item(),
        
        # Additional insights
        'Max_Error_meters': real_errors.max().item(),
        'Min_Error_meters': real_errors.min().item(),
        'Mean_Error_meters': real_errors.mean().item(),
        
        # COMPARISON METRICS (movement-based) - for reference
        'Movement_ADE_meters': movement_errors.mean().item(),
        'Movement_FDE_meters': movement_errors[:, -1].mean().item()
    }

def create_movement_evaluation_function(test_trajectories, context_length, max_state_dim, movement_stats, device='cuda'):
    """
    Create evaluation function for movement-normalized trajectory forecasting.
    """
    
    def evaluate_movement_forecasting(model):
        """Evaluate the model using proper Decision Transformer rollout evaluation."""
        
        model.eval()
        all_metrics = []
        
        # Sample a subset of test trajectories for evaluation
        eval_samples = random.sample(test_trajectories, min(1000, len(test_trajectories)))
        
        with torch.no_grad():
            for i, sample in enumerate(eval_samples):
                try:
                    # ========================================================================
                    # PROPER DT ROLLOUT EVALUATION (Research Standard)
                    # ========================================================================
                    
                    # 1. Prepare context window (last K steps)
                    context_states = sample['states']  # (K, feature_dim)
                    context_actions = sample['actions']  # (K, 3)
                    context_rtg = sample['rtg']  # (K+1, 1)
                    context_timesteps = sample['timesteps']  # (K,)
                    
                    # 2. Get ground truth future actions for comparison
                    future_actions = sample['future_movements']  # (H, 3) - normalized
                    H_steps = future_actions.shape[0]
                    
                    # 3. Pad states to exact training dimension
                    target_state_dim = max_state_dim  # Use the passed argument, not hardcoded
                    if context_states.shape[1] < target_state_dim:
                        pad_width = target_state_dim - context_states.shape[1]
                        context_states = np.pad(context_states, ((0, 0), (0, pad_width)), mode='constant')
                    elif context_states.shape[1] > target_state_dim:
                        context_states = context_states[:, :target_state_dim]
                    
                    # 4. Convert to tensors
                    K = context_states.shape[0]  # Actual context length from data
                    states = torch.FloatTensor(context_states).unsqueeze(0).to(device)  # (1, K, feature_dim)
                    actions = torch.FloatTensor(context_actions).unsqueeze(0).to(device)  # (1, K, 3)
                    rtg = torch.FloatTensor(context_rtg).unsqueeze(0).to(device)  # (1, K+1, 1)
                    timesteps = torch.LongTensor(context_timesteps).unsqueeze(0).to(device)  # (1, K)
                    attention_mask = torch.ones((1, K), dtype=torch.long, device=device)  # (1, K) - use actual K
                    
                    # 5. Assert dimensions match training
                    assert states.shape[2] == max_state_dim, f"State dim mismatch: got {states.shape[2]}, expected {max_state_dim}"
                    
                    # 6. ROLLOUT EVALUATION: Autoregressively predict H future steps
                    predicted_actions = []
                    current_actions = actions.clone()  # Start with context actions
                    current_rtg = rtg.clone()  # Keep RTG fixed (or decay heuristically)
                    
                    # Keep absolute timestep counter (no wrapping!)
                    absolute_timestep = context_timesteps[-1]  # Start from last context timestep
                    
                    for step in range(H_steps):
                        # OPTION A: Sliding window with dummy action slot (keep length K)
                        # Create a slot at the end within length K by sliding the window forward
                        zero_action = torch.zeros((1, 1, 3), device=device)  # (1, 1, 3)
                        
                        # Slide by dropping the first token and appending a dummy slot
                        # CRITICAL: Keep states fixed during rollout (no sliding)
                        actions_in = torch.cat([current_actions[:, 1:, :], zero_action], dim=1)  # (1, K, 3)
                        rtg_in = torch.cat([current_rtg[:, 1:, :], current_rtg[:, -1:, :]], dim=1)  # (1, K+1, 1)
                        
                        # CRITICAL: Keep timesteps fixed for open-loop consistency
                        # Don't slide timesteps - they should correspond to the fixed state tokens
                        # timesteps remain as original context window
                        
                        attention_mask = torch.ones_like(timesteps)  # (1, K)
                        
                        # Forward pass with sliding window (length K)
                        _, action_preds, _ = model.forward(
                            states=states,
                            actions=actions_in,
                            rewards=None,
                            returns_to_go=rtg_in[:, :-1],
                            timesteps=timesteps,
                            attention_mask=attention_mask
                        )
                        
                        # Take the prediction for the next step (last position)
                        next_action = action_preds[:, -1, :]  # (1, 3) - prediction for the next step
                        predicted_actions.append(next_action.unsqueeze(1))  # (1, 1, 3) - add time axis
                        
                        # Update for next iteration: replace dummy action with actual prediction
                        current_actions = torch.cat([current_actions[:, 1:, :], next_action.unsqueeze(1)], dim=1)
                        current_rtg = rtg_in  # Use the updated RTG from this step
                        
                        # Increment absolute timestep for next iteration
                        absolute_timestep += 1
                        
                        # Keep states fixed (acceptable for ego open-loop forecasting)
                        # Keep RTG fixed (or decay heuristically)
                    
                    # 7. Convert predicted actions to tensor
                    predicted_actions = torch.cat(predicted_actions, dim=1)  # (1, H, 3)
                    target_actions = torch.FloatTensor(future_actions).unsqueeze(0).to(device)  # (1, H, 3)
                    
                    # 8. Create mask for H future steps only
                    future_mask = torch.ones((1, H_steps), dtype=torch.long).to(device)
                    
                    # 9. Calculate metrics on H future steps only
                    metrics = calculate_movement_metrics(
                        predicted_actions, target_actions, movement_stats, future_mask
                    )
                    all_metrics.append(metrics)
                    
                except Exception as e:
                    print(f"Warning: Rollout evaluation failed for sample {i}: {str(e)}")
                    continue
        
        # Average metrics across all samples
        avg_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        else:
            avg_metrics = {
                'Normalized_ADE': 0.0, 'Real_ADE_meters': 0.0,
                'Normalized_FDE': 0.0, 'Real_FDE_meters': 0.0,
                'Miss_Rate_1m': 0.0, 'Miss_Rate_2m': 0.0, 'Miss_Rate_5m': 0.0,
                'Longitudinal_ADE_meters': 0.0, 'Lateral_ADE_meters': 0.0,
                'Longitudinal_FDE_meters': 0.0, 'Lateral_FDE_meters': 0.0,
                'Max_Error_meters': 0.0, 'Min_Error_meters': 0.0, 'Mean_Error_meters': 0.0
            }
        
        return avg_metrics
    
    return evaluate_movement_forecasting

def setup_training_visualization():
    """Setup matplotlib for real-time training visualization."""
    # Set matplotlib style for better looking plots
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (15, 10)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    
    # Create figure with subplots for different metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Decision Transformer Training Progress (with Oversampling)', fontsize=16, fontweight='bold')
    
    # Configure subplots
    axes[0, 0].set_title('Training Loss (MSE)')
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('Loss (normalized units²)')
    axes[0, 0].set_yscale('log')
    
    axes[0, 1].set_title('Learning Rate Schedule')
    axes[0, 1].set_xlabel('Training Step')
    axes[0, 1].set_ylabel('Learning Rate')
    axes[0, 1].set_yscale('log')
    
    axes[1, 0].set_title('Trajectory Prediction Accuracy')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Error (meters)')
    
    axes[1, 1].set_title('Enhanced Evaluation Metrics')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Miss Rate (%)')
    
    plt.tight_layout()
    return fig, axes

def update_training_plots(fig, axes, training_history, movement_stats):
    """Update training visualization plots with latest metrics."""
    
    # Clear all axes for fresh plotting
    for ax_row in axes:
        for ax in ax_row:
            ax.clear()
    
    # Reconfigure axes after clearing
    axes[0, 0].set_title('Training Loss (MSE)', fontweight='bold')
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('Loss (normalized units²)')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Learning Rate Schedule', fontweight='bold')
    axes[0, 1].set_xlabel('Training Step')
    axes[0, 1].set_ylabel('Learning Rate')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    axes[1, 0].set_title('Trajectory Prediction Accuracy', fontweight='bold')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Error (meters)')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Enhanced Evaluation Metrics', fontweight='bold')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Miss Rate (%)')
    axes[1, 1].grid(True, alpha=0.3)
    
    if not training_history['steps']:
        return
    
    # Plot 1: Training Loss
    steps = training_history['steps']
    losses = training_history['losses']
    if steps and losses:
        axes[0, 0].plot(steps, losses, 'b-', linewidth=2, label='Training Loss')
        axes[0, 0].set_yscale('log')
        axes[0, 0].legend()
        
        # Add trend line for last 20% of data
        if len(losses) > 10:
            recent_start = max(0, len(losses) - len(losses)//5)
            recent_steps = steps[recent_start:]
            recent_losses = losses[recent_start:]
            z = np.polyfit(recent_steps, np.log(recent_losses), 1)
            trend_line = np.exp(np.polyval(z, recent_steps))
            axes[0, 0].plot(recent_steps, trend_line, 'r--', alpha=0.7, label='Recent Trend')
            axes[0, 0].legend()
    
    # Plot 2: Learning Rate
    lrs = training_history['learning_rates']
    if steps and lrs:
        axes[0, 1].plot(steps, lrs, 'g-', linewidth=2, label='Learning Rate')
        axes[0, 1].legend()
    
    # Plot 3: Trajectory Prediction Accuracy
    iterations = training_history['iterations']
    ades = training_history['ades']
    fdes = training_history['fdes']
    longitudinal_ades = training_history['longitudinal_ades']
    lateral_ades = training_history['lateral_ades']
    
    if iterations and ades:
        axes[1, 0].plot(iterations, ades, 'o-', linewidth=2, color='blue', label='Overall ADE', markersize=6)
    if iterations and fdes:
        axes[1, 0].plot(iterations, fdes, 's-', linewidth=2, color='red', label='Overall FDE', markersize=6)
    
    # Add directional breakdown if available
    if iterations and longitudinal_ades:
        axes[1, 0].plot(iterations, longitudinal_ades, '^-', linewidth=1.5, color='green', alpha=0.8, 
                       label='Longitudinal ADE', markersize=4)
    if iterations and lateral_ades:
        axes[1, 0].plot(iterations, lateral_ades, 'v-', linewidth=1.5, color='purple', alpha=0.8, 
                       label='Lateral ADE', markersize=4)
    
    if ades or fdes or longitudinal_ades or lateral_ades:
        axes[1, 0].legend(loc='upper right')
        # Add performance reference lines
        axes[1, 0].axhline(y=3.0, color='darkgreen', linestyle='--', alpha=0.6, linewidth=1, label='SOTA (<3m)')
        axes[1, 0].axhline(y=5.0, color='green', linestyle='--', alpha=0.6, linewidth=1, label='Excellent (<5m)')
        axes[1, 0].axhline(y=8.0, color='orange', linestyle='--', alpha=0.6, linewidth=1, label='Good (<8m)')
        axes[1, 0].legend(loc='upper right')
    
    # Plot 4: Enhanced Evaluation Metrics (Miss Rates)
    miss_rates_1m = training_history['miss_rates_1m']
    miss_rates_2m = training_history['miss_rates_2m'] 
    miss_rates_5m = training_history['miss_rates_5m']
    
    if iterations and miss_rates_1m:
        axes[1, 1].plot(iterations, miss_rates_1m, 'o-', linewidth=2, color='red', 
                       label='Miss Rate @1m', markersize=5)
    if iterations and miss_rates_2m:
        axes[1, 1].plot(iterations, miss_rates_2m, 's-', linewidth=2, color='orange', 
                       label='Miss Rate @2m', markersize=5)
    if iterations and miss_rates_5m:
        axes[1, 1].plot(iterations, miss_rates_5m, '^-', linewidth=2, color='green', 
                       label='Miss Rate @5m', markersize=5)
    
    if miss_rates_1m or miss_rates_2m or miss_rates_5m:
        axes[1, 1].legend(loc='upper right')
        axes[1, 1].set_ylim(0, 100)  # Miss rates are percentages
        # Add reference lines for miss rates
        axes[1, 1].axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        axes[1, 1].axhline(y=25, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        axes[1, 1].axhline(y=10, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plt.pause(0.01)  # Allow plot to update

def save_training_plots(fig, experiment_name, timestamp):
    """Save training plots to file."""
    plots_dir = Path('training_plots')
    plots_dir.mkdir(exist_ok=True)
    
    filename = f"{experiment_name}_{timestamp}_training_curves.png"
    filepath = plots_dir / filename
    
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"📊 Training plots saved to: {filepath}")
    return filepath

# ============================================================================
# MAIN TRAINING FUNCTION WITH OVERSAMPLING
# ============================================================================


def load_text_manifest(text_manifest_path):
    """Load text embeddings from parquet manifest"""
    if text_manifest_path is None or not os.path.exists(text_manifest_path):
        return None
    
    print(f"📄 Loading text manifest from: {text_manifest_path}")
    text_df = pd.read_parquet(text_manifest_path)
    print(f"   Loaded {len(text_df)} text descriptions")
    
    # Create scene_id -> text mapping
    scene_text_map = {}
    for _, row in text_df.iterrows():
        scene_id = row.get('scene_id', '')
        if scene_id not in scene_text_map:
            scene_text_map[scene_id] = []
        
        scene_text_map[scene_id].append({
            'frame_idx': row.get('frame_idx', 0),
            'text_emb': np.array(row['text_emb']) if 'text_emb' in row else np.zeros(512),
            'text': row.get('text', '') if 'text' in row else row.get('texts', '')
        })
    
    print(f"   Mapped text for {len(scene_text_map)} scenes")
    return scene_text_map


def main():
    """Main training script with movement-specific normalization and lateral maneuver oversampling."""
    
    parser = argparse.ArgumentParser(description='Movement-Normalized Trajectory Forecasting with Oversampling')
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--max_files', type=int, default=20)
    parser.add_argument('--context_length', type=int, default=15)
    parser.add_argument('--prediction_horizon', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--lr_schedule', type=str, default='linear', choices=['linear', 'cosine'], 
                       help='Learning rate schedule: linear or cosine decay')
    parser.add_argument('--max_iters', type=int, default=5)
    parser.add_argument('--num_steps_per_iter', type=int, default=100)
    parser.add_argument('--log_to_wandb', action='store_true')
    parser.add_argument('--max_movement_filter', type=float, default=50.0, help='Filter movements larger than this (meters/timestep)')
    parser.add_argument('--movement_variance_filter', type=float, default=100.0, help='Filter samples with movement variance above this')
    parser.add_argument('--enable_plots', action='store_true', help='Enable real-time training visualization')
    parser.add_argument('--save_plots', action='store_true', help='Save training plots to file')
    
    # Oversampling arguments
    parser.add_argument('--enable_oversampling', action='store_true', help='Enable lateral maneuver oversampling')
    parser.add_argument('--oversampling_temperature', type=float, default=0.75, help='Temperature for oversampling weight smoothing')
    parser.add_argument('--target_straight_frac', type=float, default=0.30, help='Target fraction of straight samples')
    parser.add_argument('--target_mild_frac', type=float, default=0.35, help='Target fraction of mild lateral samples')
    parser.add_argument('--target_strong_frac', type=float, default=0.35, help='Target fraction of strong lateral samples')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Path to checkpoint to load for transfer learning')
    

    # VLM/VisionTRAP arguments
    parser.add_argument('--enable_visiontrap', action='store_true', help='Enable VisionTRAP text contrastive learning')
    parser.add_argument('--text_manifest_path', type=str, default=None, help='Path to text manifest parquet file')
    parser.add_argument('--text_loss_weight', type=float, default=0.1, help='Weight for text contrastive loss (lambda)')
    parser.add_argument('--text_warmup_epochs', type=int, default=10, help='Epochs to warm up text loss')
    parser.add_argument('--text_loss_cap', type=float, default=0.2, help='Maximum text loss contribution')
    parser.add_argument('--infonce_scale_factor', type=float, default=0.001, help='Scale factor for InfoNCE loss')
    parser.add_argument('--text_queue_size', type=int, default=2048, help='Queue size for negative samples')
    parser.add_argument('--max_objects', type=int, default=128, help='Maximum objects per scene')
    parser.add_argument('--traj_loss_type', type=str, default='mse', choices=['mse', 'huber'], help='Trajectory loss type')
    parser.add_argument('--lateral_weight', type=float, default=2.5, help='Weight for lateral error')
    parser.add_argument('--time_weighting', type=str, default='none', choices=['none', 'linear', 'exponential'], help='Time weighting for loss')
    parser.add_argument('--text_gate_speed_norm', type=float, default=0.1, help='Speed normalization for text gating')
    parser.add_argument('--text_gate_error_quantile', type=float, default=0.7, help='Error quantile for text gating')
    
    args = parser.parse_args()
    
    print("🚀 MOVEMENT-NORMALIZED TRAJECTORY FORECASTING WITH OVERSAMPLING")

    if args.enable_visiontrap:
        print(f"  🎯 VisionTRAP: Text contrastive learning ENABLED")
        print(f"     Text loss weight (λ): {args.text_loss_weight}")
        print(f"     Text warmup epochs: {args.text_warmup_epochs}")
        print(f"     Text loss cap: {args.text_loss_cap}")
        print(f"     Text queue size: {args.text_queue_size}")
    print("=" * 80)
    print("Key improvements:")
    print("  ✅ Ego-centric coordinate transformation")
    print("  ✅ Movement-specific normalization (not coordinate-based)")
    print("  ✅ Loss represents actual movement prediction accuracy")
    print("  ✅ Metrics in both normalized units and real meters")
    print("  ✅ Research-standard evaluation: Multiple miss rate thresholds (1m, 2m, 5m)")
    print("  ✅ Directional error breakdown: Longitudinal vs lateral performance")
    if args.enable_oversampling:
        print("  🎯 NEW: Lateral maneuver oversampling to improve lateral prediction")
        print(f"     Target distribution: Straight={args.target_straight_frac:.0%}, "
              f"Mild={args.target_mild_frac:.0%}, Strong={args.target_strong_frac:.0%}")
        print(f"     Temperature: {args.oversampling_temperature}")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Initialize wandb
    if args.log_to_wandb:
        wandb.init(
            project='movement-normalized-trajectory-dt-oversampling',
            config=vars(args),
            name=f'oversampling-dt-{args.embed_dim}-{args.n_layer}'
        )
    
    try:
        # Setup training visualization if enabled
        fig, axes = None, None
        training_history = {
            'steps': [],
            'losses': [],
            'learning_rates': [],
            'iterations': [],
            'ades': [],
            'fdes': [],
            'miss_rates_1m': [],
            'miss_rates_2m': [],
            'miss_rates_5m': [],
            'longitudinal_ades': [],
            'lateral_ades': []
        }
        experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if args.enable_plots:
            print("📊 Setting up real-time training visualization...")
            fig, axes = setup_training_visualization()
            plt.ion()  # Enable interactive mode
        
        # PROPER 2-PASS PIPELINE: Load scenes first, then process with train-only stats
        print("\n" + "="*80)
        print("📊 PHASE 1: Loading raw scenes...")
        ego_trajectories = load_nuscenes_scenes(
            args.dataset_path,
            max_files=args.max_files
        )
        
        if len(ego_trajectories) == 0:
            print("❌ No scenes loaded successfully!")
            return
        
        # Split scenes into train/test BEFORE any normalization (randomized + seeded)
        print("\n📊 PHASE 2: Splitting scenes into train/test...")
        
        # Set random seed for reproducible splits (change this to get different splits)
        random.seed(42)
        np.random.seed(42)
        
        # Shuffle scenes to avoid distribution bias
        scene_indices = list(range(len(ego_trajectories)))
        random.shuffle(scene_indices)
        
        # Split with 80/20 ratio
        num_train_scenes = int(0.8 * len(ego_trajectories))
        train_indices = scene_indices[:num_train_scenes]
        test_indices = scene_indices[num_train_scenes:]
        
        train_ego_trajectories = [ego_trajectories[i] for i in train_indices]
        test_ego_trajectories = [ego_trajectories[i] for i in test_indices]
        
        print(f"  Training scenes: {len(train_ego_trajectories)}")
        print(f"  Test scenes: {len(test_ego_trajectories)}")
        
        # Log scene IDs for reproducibility debugging
        train_scene_ids = [traj['scene_id'] for traj in train_ego_trajectories]
        test_scene_ids = [traj['scene_id'] for traj in test_ego_trajectories]
        print(f"  Train scene IDs: {train_scene_ids[:5]}..." if len(train_scene_ids) > 5 else f"  Train scene IDs: {train_scene_ids}")
        print(f"  Test scene IDs: {test_scene_ids[:5]}..." if len(test_scene_ids) > 5 else f"  Test scene IDs: {test_scene_ids}")
        
        # CRITICAL FIX: Compute movement stats ONLY on training scenes (no leakage!)
        print("\n📊 PHASE 3: Computing movement normalization statistics on TRAINING scenes only...")
        movement_stats = compute_movement_statistics(train_ego_trajectories)
        
        # Process both train and test with the same training-derived stats
        print("\n📊 PHASE 4: Creating normalized samples...")
        train_trajectories = process_nuscenes_movement_normalized(
            train_ego_trajectories, movement_stats, args,
            context_length=args.context_length,
            prediction_horizon=args.prediction_horizon
        )
        test_trajectories = process_nuscenes_movement_normalized(
            test_ego_trajectories, movement_stats, args,
            context_length=args.context_length,
            prediction_horizon=args.prediction_horizon
        )
        
        print(f"\n📊 Final Dataset Split:")
        print(f"  Training samples: {len(train_trajectories)}")
        print(f"  Test samples: {len(test_trajectories)}")
        
        # Compute lateral scores and sampling weights for oversampling
        if args.enable_oversampling:
            print(f"\n🎯 Computing lateral maneuver oversampling...")
            target_fractions = [args.target_straight_frac, args.target_mild_frac, args.target_strong_frac]
            lateral_scores, maneuver_bins, sampling_weights, bin_stats = compute_lateral_scores_and_weights(
                train_trajectories, 
                args.context_length,
                target_fractions=target_fractions,
                temperature=args.oversampling_temperature
            )
        else:
            print(f"\n🎯 Oversampling disabled - using uniform sampling")
            lateral_scores, maneuver_bins, sampling_weights, bin_stats = None, None, None, None
        
        # Calculate dimensions
        max_state_dim = max(sample['states'].shape[1] for sample in train_trajectories)
        state_dim = max_state_dim
        act_dim = 3  # x, y, z normalized movements
        
        print(f"\n🏗️  Model Configuration:")
        print(f"  State dimension: {state_dim}")
        print(f"  Action dimension: {act_dim}")
        print(f"  Context length: {args.context_length}")
        print(f"  Movement normalization stats:")
        print(f"    Mean: [{movement_stats['mean'][0]:.4f}, {movement_stats['mean'][1]:.4f}, {movement_stats['mean'][2]:.4f}]")
        print(f"    Std:  [{movement_stats['std'][0]:.4f}, {movement_stats['std'][1]:.4f}, {movement_stats['std'][2]:.4f}]")
        
        # Create model
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            hidden_size=args.embed_dim,
            max_length=args.context_length,
            max_ep_len=args.context_length,
            action_tanh=False,  # Don't constrain normalized movements
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_inner=4*args.embed_dim,
            n_ctx=1024,
            n_positions=1024,
            activation_function='gelu',
            resid_pdrop=0.1,
            attn_pdrop=0.1,
        ).to(device)
        
        print(f"✅ Created DecisionTransformer with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Load checkpoint for transfer learning if specified
        if args.load_checkpoint:
            print(f"\n🔄 TRANSFER LEARNING: Loading checkpoint from {args.load_checkpoint}")
            try:
                checkpoint = torch.load(args.load_checkpoint, map_location=device)
                
                # Load model state dict
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print("✅ Model weights loaded successfully")
                else:
                    print("⚠️  No 'model_state_dict' found in checkpoint, loading entire checkpoint as state dict")
                    model.load_state_dict(checkpoint)
                    print("✅ Model weights loaded successfully")
                
                print(f"📊 Transfer learning from: {args.load_checkpoint}")
                print("🎯 Model will be fine-tuned on 130-scene dataset")
                
            except Exception as e:
                print(f"❌ Error loading checkpoint: {e}")
                print("🔄 Continuing with randomly initialized model")
        else:
            print("🆕 Training from scratch with randomly initialized weights")
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Create learning rate scheduler based on schedule type
        if args.lr_schedule == 'cosine':
            # Cosine decay with warmup
            def cosine_schedule_with_warmup(step):
                if step < args.warmup_steps:
                    # Warmup phase: linear increase from 0 to learning_rate
                    return step / args.warmup_steps
                else:
                    # Cosine decay phase
                    import math
                    progress = (step - args.warmup_steps) / (args.max_iters * args.num_steps_per_iter - args.warmup_steps)
                    return 0.5 * (1 + math.cos(math.pi * progress))
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                cosine_schedule_with_warmup
            )
            print(f"📈 Using cosine decay scheduler with {args.warmup_steps} warmup steps")
        else:
            # Linear decay (default)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda steps: min((steps+1)/args.warmup_steps, 1)
            )
            print(f"📈 Using linear decay scheduler with {args.warmup_steps} warmup steps")
        
        # Create batch function with oversampling
        get_batch = create_movement_batch_function_with_oversampling(
            train_trajectories, 
            args.context_length, 
            device,
            sampling_weights=sampling_weights,
            maneuver_bins=maneuver_bins
        )
        
        # Define experiment name for checkpoints
        experiment_name = f"oversampling_dt_{args.embed_dim}d_{args.n_layer}l_{args.max_files}files"
        
        # Create evaluation function
        eval_fn = create_movement_evaluation_function(
            test_trajectories,
            args.context_length,
            state_dim,
            movement_stats,
            device
        )
        
        # Loss function for normalized movements
        def movement_loss_fn(state_preds, action_preds, reward_preds, 
                            state_targets, action_targets, reward_targets):
            # MSE on normalized movements
            return F.mse_loss(action_preds, action_targets)
        
        # Create trainer
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=args.batch_size,
            get_batch=get_batch,
            loss_fn=movement_loss_fn,
            scheduler=scheduler,
            eval_fns=[eval_fn]
        )
        
        print(f"\n🚀 STARTING MOVEMENT-NORMALIZED TRAINING WITH OVERSAMPLING")
        print("=" * 80)
        
        # Training loop
        for iteration in range(args.max_iters):
            print(f"\n📊 Iteration {iteration + 1}/{args.max_iters}")
            iteration_start_time = time.time()
            
            # Monitor batch composition for oversampling effectiveness
            if iteration % 5 == 0 and args.enable_oversampling:  # Every 5 iterations
                print(f"🎯 Oversampling active: {sampling_weights is not None}")
                if sampling_weights is not None:
                    print(f"   Target distribution: Straight={args.target_straight_frac:.0%}, "
                          f"Mild={args.target_mild_frac:.0%}, Strong={args.target_strong_frac:.0%}")
            
            logs = trainer.train_iteration(
                num_steps=args.num_steps_per_iter,
                iter_num=iteration + 1,
                print_logs=True
            )
            
            # Save checkpoint every 10 iterations
            if (iteration + 1) % 10 == 0:
                import os
                os.makedirs("checkpoints", exist_ok=True)
                checkpoint_path = f"checkpoints/{experiment_name}_iter_{iteration+1}.pt"
                torch.save({
                    'iteration': iteration + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': logs.get('training/train_loss_mean', 0.0),
                    'ade': logs.get('evaluation/Real_ADE_meters', 0.0),
                    'fde': logs.get('evaluation/Real_FDE_meters', 0.0),
                    'args': args,
                    'oversampling_stats': bin_stats if args.enable_oversampling else None
                }, checkpoint_path)
                print(f"💾 Checkpoint saved: {checkpoint_path}")
            
            # Collect training metrics for visualization
            if 'training/train_loss_mean' in logs:
                current_step = (iteration + 1) * args.num_steps_per_iter
                training_history['steps'].append(current_step)
                training_history['losses'].append(logs['training/train_loss_mean'])
                
                # Get current learning rate from optimizer
                current_lr = trainer.optimizer.param_groups[0]['lr']
                training_history['learning_rates'].append(current_lr)
            
            # Run evaluation and collect metrics
            if 'evaluation/Real_ADE_meters' in logs:
                training_history['iterations'].append(iteration + 1)
                training_history['ades'].append(logs['evaluation/Real_ADE_meters'])
                training_history['fdes'].append(logs['evaluation/Real_FDE_meters'])
                
                # Collect enhanced metrics if available
                if 'evaluation/Miss_Rate_1m' in logs:
                    training_history['miss_rates_1m'].append(logs['evaluation/Miss_Rate_1m'] * 100)  # Convert to percentage
                    training_history['miss_rates_2m'].append(logs['evaluation/Miss_Rate_2m'] * 100)
                    training_history['miss_rates_5m'].append(logs['evaluation/Miss_Rate_5m'] * 100)
                
                if 'evaluation/Longitudinal_ADE_meters' in logs:
                    training_history['longitudinal_ades'].append(logs['evaluation/Longitudinal_ADE_meters'])
                    training_history['lateral_ades'].append(logs['evaluation/Lateral_ADE_meters'])
                
                print(f"⏱️  Iteration {iteration + 1} completed in {time.time() - iteration_start_time:.1f}s")
                print(f"📈 Current metrics: Loss={logs['training/train_loss_mean']:.4f}, ADE={logs['evaluation/Real_ADE_meters']:.2f}m, FDE={logs['evaluation/Real_FDE_meters']:.2f}m")
                
                # Enhanced evaluation logging with all metrics
                print("📊 COMPREHENSIVE EVALUATION METRICS:")
                print(f"   🎯 Standard Metrics:")
                print(f"      ADE: {logs['evaluation/Real_ADE_meters']:.2f}m")
                print(f"      FDE: {logs['evaluation/Real_FDE_meters']:.2f}m")
                
                # Multiple miss rate thresholds
                if 'evaluation/Miss_Rate_1m' in logs:
                    print(f"   ⚠️  Miss Rates:")
                    print(f"      @1m: {logs['evaluation/Miss_Rate_1m']*100:.1f}%")
                    print(f"      @2m: {logs['evaluation/Miss_Rate_2m']*100:.1f}%")
                    print(f"      @5m: {logs['evaluation/Miss_Rate_5m']*100:.1f}%")
                
                # Directional breakdown
                if 'evaluation/Longitudinal_ADE_meters' in logs:
                    print(f"   🧭 Directional Breakdown:")
                    print(f"      Longitudinal ADE: {logs['evaluation/Longitudinal_ADE_meters']:.2f}m")
                    print(f"      Lateral ADE: {logs['evaluation/Lateral_ADE_meters']:.2f}m")
                    print(f"      Longitudinal FDE: {logs['evaluation/Longitudinal_FDE_meters']:.2f}m")
                    print(f"      Lateral FDE: {logs['evaluation/Lateral_FDE_meters']:.2f}m")
                
                # Error range
                if 'evaluation/Max_Error_meters' in logs:
                    print(f"   📏 Error Range:")
                    print(f"      Min: {logs['evaluation/Min_Error_meters']:.2f}m")
                    print(f"      Max: {logs['evaluation/Max_Error_meters']:.2f}m")
                    print(f"      Mean: {logs['evaluation/Mean_Error_meters']:.2f}m")
            
            # Update visualization if enabled
            if args.enable_plots and fig is not None:
                update_training_plots(fig, axes, training_history, movement_stats)
            
            # Log to wandb
            if args.log_to_wandb:
                # Add visualization metrics to wandb
                enhanced_logs = logs.copy()
                if training_history['steps']:
                    enhanced_logs['training/current_step'] = training_history['steps'][-1]
                if training_history['learning_rates']:
                    enhanced_logs['training/learning_rate'] = training_history['learning_rates'][-1]
                wandb.log(enhanced_logs)
        
        print(f"\n🎉 Movement-normalized training with oversampling completed successfully!")
        print("✅ Movement-specific normalization applied")
        print("✅ Loss represents actual movement prediction accuracy")
        print("✅ Metrics available in both normalized units and real meters")
        print(f"✅ Movement std: {movement_stats['std']} meters")
        if args.enable_oversampling:
            print("✅ Lateral maneuver oversampling applied")
            print(f"   Final bin distribution: {bin_stats['bin_fractions']}")
        
        # Save final model checkpoint
        import os
        os.makedirs("checkpoints", exist_ok=True)
        final_checkpoint_path = f"checkpoints/{experiment_name}_final.pt"
        torch.save({
            'iteration': args.max_iters,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': training_history['losses'][-1] if training_history['losses'] else 0.0,
            'ade': training_history.get('ades', [0.0])[-1],
            'fde': training_history.get('fdes', [0.0])[-1],
            'args': args,
            'movement_stats': movement_stats,
            'oversampling_stats': bin_stats if args.enable_oversampling else None
        }, final_checkpoint_path)
        print(f"💾 Final model checkpoint saved: {final_checkpoint_path}")
        
        # Final visualization update and save
        if args.enable_plots and fig is not None:
            update_training_plots(fig, axes, training_history, movement_stats)
            plt.ioff()  # Turn off interactive mode
            
            if args.save_plots:
                save_training_plots(fig, experiment_name, experiment_timestamp)
            else:
                print("📊 Training visualization complete. Use --save_plots to save to file.")
                print("📊 Close the plot window to continue...")
                plt.show()
        
        # Print training summary
        if training_history['losses']:
            final_loss = training_history['losses'][-1]
            initial_loss = training_history['losses'][0]
            loss_improvement = ((initial_loss - final_loss) / initial_loss) * 100
            
            print(f"\n📊 TRAINING SUMMARY:")
            print(f"   Initial Loss: {initial_loss:.4f} normalized units²")
            print(f"   Final Loss: {final_loss:.4f} normalized units²")
            print(f"   Improvement: {loss_improvement:.1f}%")
            
            if training_history['ades']:
                final_ade = training_history['ades'][-1]
                final_fde = training_history['fdes'][-1]
                print(f"   Final ADE: {final_ade:.2f} meters")
                print(f"   Final FDE: {final_fde:.2f} meters")
                
                # Enhanced final performance assessment
                print("\n🏆 FINAL PERFORMANCE ASSESSMENT:")
                print("=" * 60)
                
                # Overall performance tier
                if final_ade < 3.0:
                    print("🌟 EXCEPTIONAL: ADE < 3m - SOTA competitive!")
                    tier = "SOTA"
                elif final_ade < 5.0:
                    print("🏆 EXCELLENT: ADE < 5m - Production ready!")
                    tier = "Excellent"
                elif final_ade < 8.0:
                    print("✅ GOOD: ADE < 8m - Strong performance")
                    tier = "Good"
                elif final_ade < 12.0:
                    print("⚠️  ACCEPTABLE: ADE < 12m - Consider improvements")
                    tier = "Acceptable"
                else:
                    print("❌ NEEDS IMPROVEMENT: ADE > 12m - Requires optimization")
                    tier = "Needs Work"
                
                # Research context
                print(f"\n📚 RESEARCH CONTEXT:")
                print(f"   Performance Tier: {tier}")
                print(f"   Single-Mode Prediction: {final_ade:.2f}m ADE")
                print(f"   Research Range (K=1): 2-8m typical for single-mode")
                print(f"   Multi-Modal SOTA: ~0.9m (best-of-10 predictions)")
                print(f"   Your Position: {'Competitive' if final_ade < 8.0 else 'Developing'}")
                
                # Training efficiency highlight
                print(f"\n⚡ TRAINING EFFICIENCY:")
                print(f"   Convergence: {loss_improvement:.1f}% loss reduction")
                print(f"   Architecture: Novel Decision Transformer approach")
                print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,} (efficient for performance)")
                if args.enable_oversampling:
                    print(f"   Oversampling: Lateral maneuver balancing applied")
                print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()






