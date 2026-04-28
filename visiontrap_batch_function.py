#!/usr/bin/env python3
"""
VisionTRAP-enhanced batch function that loads text embeddings alongside trajectory data.
Based on the baseline create_movement_batch_function_with_oversampling.
"""

import numpy as np
import torch
import random
import pandas as pd
from pathlib import Path


def create_visiontrap_batch_function(trajectories, context_length, device='cuda', 
                                     sampling_weights=None, maneuver_bins=None,
                                     text_manifest_path=None):
    """
    Create batch function for VisionTRAP training with text embeddings.
    
    Args:
        trajectories: List of trajectory samples (from baseline processing)
        context_length: int - context length
        device: str - device to use
        sampling_weights: np.array - sampling weights for each trajectory (optional)
        maneuver_bins: np.array - maneuver bin assignments for each trajectory (optional)
        text_manifest_path: str - path to nuPlan-Text Parquet manifest
    
    Returns:
        get_batch: function that returns (states, actions, rewards, dones, rtg, timesteps, attention_mask, text_emb)
    """
    
    # Find the maximum state dimension across all trajectories
    max_state_dim = max(sample['states'].shape[1] for sample in trajectories)
    
    # Load text manifest if VisionTRAP is enabled
    text_manifest_df = None
    if text_manifest_path is not None:
        print(f"📄 Loading text manifest: {text_manifest_path}")
        text_manifest_df = pd.read_parquet(text_manifest_path)
        print(f"   Loaded {len(text_manifest_df)} text embeddings")
        
        # Pre-process text embeddings for faster lookup
        # Create a dictionary: scene_id -> list of text data
        text_lookup = {}
        for _, row in text_manifest_df.iterrows():
            scene_id = row['scene_id']
            if scene_id not in text_lookup:
                text_lookup[scene_id] = []
            text_lookup[scene_id].append({
                'frame_idx': row['frame_idx'],
                'object_idx': row['object_idx'],
                'text_emb': np.array(row['text_emb']),
                'text': row.get('texts', row.get('text', ''))  # Support both 'texts' and 'text' columns
            })
        print(f"   Indexed {len(text_lookup)} unique scenes")
    else:
        text_lookup = None
    
    def get_batch(batch_size):
        """
        Get a batch of trajectories with text embeddings.
        
        Returns:
            If text_manifest_path is None:
                (states, actions, rewards, dones, rtg, timesteps, attention_mask)
            If text_manifest_path is provided:
                (states, actions, rewards, dones, rtg, timesteps, attention_mask, text_emb)
        """
        
        # ====================================================================
        # STEP 1: Sample trajectories (same as baseline)
        # ====================================================================
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
        
        # ====================================================================
        # STEP 2: Build trajectory batches (same as baseline)
        # ====================================================================
        states_batch = []
        actions_batch = []
        rtg_batch = []
        timesteps_batch = []
        attention_masks = []
        rewards_batch = []  # Dummy rewards
        dones_batch = []    # Dummy dones
        
        # NEW: Text embeddings batch
        text_emb_batch = []
        
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
            
            # ====================================================================
            # STEP 3: Load text embeddings for this sample (NEW for VisionTRAP)
            # ====================================================================
            if text_lookup is not None and 'scene_id' in sample:
                scene_id = sample['scene_id']
                # Align text to the last observed context frame for this sample
                # Prefer ego captions (object_idx == 0) when available
                target_frame = None
                if 'start_frame' in sample:
                    target_frame = int(sample['start_frame']) + (context_length - 1)
                # Fallback: if start_frame is missing, align to context_length-1 (relative index)
                if target_frame is None:
                    target_frame = context_length - 1
                
                # Get text data for this scene
                scene_text_data = text_lookup.get(scene_id, [])
                
                if len(scene_text_data) > 0:
                    # Prefer ego entries if available
                    ego_entries = [d for d in scene_text_data if int(d.get('object_idx', -1)) == 0]
                    pool_source = ego_entries if len(ego_entries) > 0 else scene_text_data
                    
                    # Collect a small temporal window around target_frame (±2 frames)
                    window = [d for d in pool_source if abs(int(d.get('frame_idx', target_frame)) - target_frame) <= 2]
                    if len(window) == 0:
                        # Fallback: nearest single embedding
                        closest = min(
                            pool_source, 
                            key=lambda d: abs(int(d.get('frame_idx', target_frame)) - target_frame)
                        )
                        text_emb = closest['text_emb']
                    else:
                        # Average embeddings in the window, then L2-normalize
                        embs = np.stack([np.array(d['text_emb']) for d in window], axis=0)  # [K, D]
                        mean_emb = embs.mean(axis=0)
                        norm = np.linalg.norm(mean_emb) + 1e-8
                        text_emb = mean_emb / norm
                else:
                    # No text data for this scene, use zero embedding
                    text_emb = np.zeros(512)  # CLIP embedding dimension
            else:
                # VisionTRAP not enabled or no scene_id, use zero embedding
                text_emb = np.zeros(512)
            
            text_emb_batch.append(text_emb)
        
        # ====================================================================
        # STEP 4: Convert to tensors
        # ====================================================================
        states = torch.FloatTensor(np.array(states_batch)).to(device)
        actions = torch.FloatTensor(np.array(actions_batch)).to(device)
        rtg = torch.FloatTensor(np.array(rtg_batch)).to(device)
        timesteps = torch.LongTensor(np.array(timesteps_batch)).to(device)
        attention_mask = torch.LongTensor(np.array(attention_masks)).to(device)
        rewards = torch.FloatTensor(np.array(rewards_batch)).to(device)
        dones = torch.FloatTensor(np.array(dones_batch)).to(device)
        
        # NEW: Convert text embeddings to tensor
        if text_lookup is not None:
            text_emb = torch.FloatTensor(np.array(text_emb_batch)).to(device)
            return states, actions, rewards, dones, rtg, timesteps, attention_mask, text_emb
        else:
            # Return without text embeddings (backward compatible)
            return states, actions, rewards, dones, rtg, timesteps, attention_mask
    
    return get_batch


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == '__main__':
    print("Testing VisionTRAP batch function...")
    
    # Create dummy trajectory samples
    dummy_trajectories = [
        {
            'states': np.random.randn(10, 100),
            'actions': np.random.randn(10, 3),
            'rtg': np.random.randn(10, 1),
            'timesteps': np.arange(10),
            'scene_id': '2021.05.12.23.36.44_veh-35_00712_00774'
        }
        for _ in range(20)
    ]
    
    # Create batch function WITHOUT text (baseline mode)
    print("\n1. Testing baseline mode (no text)...")
    get_batch_baseline = create_visiontrap_batch_function(
        dummy_trajectories,
        context_length=10,
        device='cpu',
        text_manifest_path=None
    )
    
    batch_baseline = get_batch_baseline(batch_size=4)
    print(f"   Baseline batch length: {len(batch_baseline)} (should be 7)")
    
    # Create batch function WITH text (VisionTRAP mode)
    print("\n2. Testing VisionTRAP mode (with text)...")
    get_batch_visiontrap = create_visiontrap_batch_function(
        dummy_trajectories,
        context_length=10,
        device='cpu',
        text_manifest_path='/data/nuplan_text_finetuned/nuplan_text_manifest_fixed.parquet'
    )
    
    batch_visiontrap = get_batch_visiontrap(batch_size=4)
    print(f"   VisionTRAP batch length: {len(batch_visiontrap)} (should be 8)")
    print(f"   Text embeddings shape: {batch_visiontrap[7].shape}")
    
    print("\n✅ VisionTRAP batch function is working!")

