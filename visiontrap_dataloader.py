#!/usr/bin/env python3
"""
VisionTRAP-enhanced dataloader for Decision Transformer training
Loads text embeddings and false-negative masks for contrastive learning
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class VisionTRAPDataset(Dataset):
    """
    Dataset that loads nuScenes data with text embeddings for VisionTRAP training
    """
    
    def __init__(self, dataset_path, text_manifest_path, context_length=15, prediction_horizon=5):
        """
        Args:
            dataset_path: Path to nuScenes .npy files
            text_manifest_path: Path to Parquet file with text embeddings
            context_length: Number of context timesteps
            prediction_horizon: Number of prediction timesteps
        """
        self.dataset_path = dataset_path
        self.context_length = context_length
        self.prediction_horizon = prediction_horizon
        
        # Load text manifest
        logger.info(f"Loading text manifest from {text_manifest_path}")
        self.text_df = pd.read_parquet(text_manifest_path)
        logger.info(f"Loaded {len(self.text_df)} text descriptions")
        
        # Create mapping from scene_id to text data
        self.scene_text_map = {}
        for _, row in self.text_df.iterrows():
            scene_id = row['scene_id']
            if scene_id not in self.scene_text_map:
                self.scene_text_map[scene_id] = []
            
            self.scene_text_map[scene_id].append({
                'frame_idx': row['frame_idx'],
                'object_idx': row['object_idx'],
                't_2hz_idx': row['t_2hz_idx'],
                'text_emb': row['text_emb'],
                'texts': row['texts']
            })
        
        # Load scene files
        self.scene_files = sorted(glob.glob(os.path.join(dataset_path, "*.npy")))
        logger.info(f"Found {len(self.scene_files)} scene files")
        
        # Filter scenes that have text data
        self.valid_scenes = []
        for scene_file in self.scene_files:
            scene_id = os.path.basename(scene_file).replace('.npy', '')
            if scene_id in self.scene_text_map:
                self.valid_scenes.append(scene_file)
        
        logger.info(f"Found {len(self.valid_scenes)} scenes with text data")
        
        # Precompute false-negative masks for each scene
        self.scene_fn_masks = {}
        for scene_id, text_data in self.scene_text_map.items():
            if len(text_data) > 1:  # Need at least 2 objects for contrastive learning
                embeddings = np.array([item['text_emb'] for item in text_data])
                fn_mask = self._compute_false_negative_mask(embeddings)
                self.scene_fn_masks[scene_id] = fn_mask
    
    def _compute_false_negative_mask(self, embeddings, threshold=0.8):
        """Compute false-negative mask based on text similarities"""
        similarity_matrix = np.dot(embeddings, embeddings.T)
        mask = (similarity_matrix > threshold) & (similarity_matrix < 1.0)
        return mask
    
    def __len__(self):
        return len(self.valid_scenes)
    
    def __getitem__(self, idx):
        scene_file = self.valid_scenes[idx]
        scene_id = os.path.basename(scene_file).replace('.npy', '')
        
        # Load scene data
        scene_data = np.load(scene_file)  # (T, N, F)
        T, N, F = scene_data.shape
        
        # Get text data for this scene
        text_data = self.scene_text_map[scene_id]
        
        # Sample a random timestep for training
        max_start = T - self.context_length - self.prediction_horizon
        if max_start <= 0:
            # Scene too short, pad or skip
            start_idx = 0
        else:
            start_idx = np.random.randint(0, max_start)
        
        end_idx = start_idx + self.context_length + self.prediction_horizon
        
        # Extract trajectory data
        trajectory = scene_data[start_idx:end_idx]  # (T_total, N, F)
        
        # Extract ego trajectory (first object)
        ego_trajectory = trajectory[:, 0, :3]  # (T_total, 3) - x, y, z
        
        # Compute movement vectors (actions)
        ego_movements = np.diff(ego_trajectory, axis=0)  # (T_total-1, 3)
        
        # Split into context and prediction
        context_movements = ego_movements[:self.context_length-1]  # (context_length-1, 3)
        target_movements = ego_movements[self.context_length-1:]  # (prediction_horizon, 3)
        
        # Get text embeddings for this timestep
        # Find text data that corresponds to our context timesteps
        context_text_embs = []
        context_text_ids = []
        
        for t in range(self.context_length):
            # Map timestep to 2Hz index
            t_2hz_idx = t // 2  # Approximate mapping
            
            # Find matching text data
            for text_item in text_data:
                if text_item['t_2hz_idx'] == t_2hz_idx:
                    context_text_embs.append(text_item['text_emb'])
                    context_text_ids.append(text_item['object_idx'])
                    break
        
        # Pad with zeros if we don't have enough text data
        while len(context_text_embs) < self.context_length:
            context_text_embs.append(np.zeros(512))  # CLIP embedding dimension
            context_text_ids.append("")
        
        # Truncate if we have too many
        context_text_embs = context_text_embs[:self.context_length]
        context_text_ids = context_text_ids[:self.context_length]
        
        # Get false-negative mask for this scene
        fn_mask = self.scene_fn_masks.get(scene_id, np.zeros((len(text_data), len(text_data)), dtype=bool))
        
        return {
            'state_tokens': torch.FloatTensor(trajectory[:self.context_length]),  # (context_length, N, F)
            'action_targets': torch.FloatTensor(target_movements),  # (prediction_horizon, 3)
            'valid_mask': torch.ones(self.context_length, dtype=torch.bool),  # (context_length,)
            'agent_vec': torch.FloatTensor(ego_trajectory[:self.context_length].mean(axis=0)),  # (3,) - pooled agent embedding
            'text_emb': torch.FloatTensor(np.array(context_text_embs)),  # (context_length, 384)
            'fn_mask': torch.BoolTensor(fn_mask),  # (N_text, N_text)
            'scene_id': scene_id,
            'text_ids': context_text_ids
        }

def create_visiontrap_dataloader(dataset_path, text_manifest_path, batch_size=16, 
                                context_length=15, prediction_horizon=5, num_workers=4):
    """
    Create VisionTRAP dataloader with text embeddings
    """
    dataset = VisionTRAPDataset(
        dataset_path=dataset_path,
        text_manifest_path=text_manifest_path,
        context_length=context_length,
        prediction_horizon=prediction_horizon
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=visiontrap_collate_fn
    )
    
    return dataloader

def visiontrap_collate_fn(batch):
    """
    Custom collate function for VisionTRAP batches
    """
    # Stack tensors
    state_tokens = torch.stack([item['state_tokens'] for item in batch])
    action_targets = torch.stack([item['action_targets'] for item in batch])
    valid_mask = torch.stack([item['valid_mask'] for item in batch])
    agent_vec = torch.stack([item['agent_vec'] for item in batch])
    text_emb = torch.stack([item['text_emb'] for item in batch])
    
    # Handle variable-size false-negative masks
    fn_masks = [item['fn_mask'] for item in batch]
    scene_ids = [item['scene_id'] for item in batch]
    text_ids = [item['text_ids'] for item in batch]
    
    return {
        'state_tokens': state_tokens,  # (B, T, N, F)
        'action_targets': action_targets,  # (B, H, 3)
        'valid_mask': valid_mask,  # (B, T)
        'agent_vec': agent_vec,  # (B, 3)
        'text_emb': text_emb,  # (B, T, 384)
        'fn_mask': fn_masks,  # List of (N_text, N_text) masks
        'scene_ids': scene_ids,  # List of scene IDs
        'text_ids': text_ids  # List of text IDs
    }

# Test the dataloader
if __name__ == "__main__":
    # Test with our generated data
    dataset_path = "/data/nuscenes_fixed_matrices"
    text_manifest_path = "/data/nuplan_text/2021.10.08.15.06.38_veh-28_01228_01310_with_embeddings.parquet"
    
    logger.info("Testing VisionTRAP dataloader...")
    
    try:
        dataloader = create_visiontrap_dataloader(
            dataset_path=dataset_path,
            text_manifest_path=text_manifest_path,
            batch_size=4,
            context_length=15,
            prediction_horizon=5
        )
        
        logger.info(f"Created dataloader with {len(dataloader.dataset)} samples")
        
        # Test one batch
        for batch in dataloader:
            logger.info(f"Batch shapes:")
            logger.info(f"  state_tokens: {batch['state_tokens'].shape}")
            logger.info(f"  action_targets: {batch['action_targets'].shape}")
            logger.info(f"  agent_vec: {batch['agent_vec'].shape}")
            logger.info(f"  text_emb: {batch['text_emb'].shape}")
            logger.info(f"  fn_mask count: {len(batch['fn_mask'])}")
            
            break
        
        logger.info("✅ VisionTRAP dataloader test passed!")
        
    except Exception as e:
        logger.error(f"❌ VisionTRAP dataloader test failed: {e}")
        import traceback
        traceback.print_exc()
