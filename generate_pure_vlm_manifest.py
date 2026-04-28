#!/usr/bin/env python3
"""
Generate a pure VLM manifest with simple, short descriptions.
No metadata enhancement - just the raw VLM visual descriptions.
SAFE: Does not overwrite best run files.
"""

import pandas as pd
import numpy as np
import os
import re
import glob
from pathlib import Path
from tqdm import tqdm
from transformers import CLIPModel, CLIPTokenizer
import torch

def clean_vlm_text(text: str) -> str:
    """
    Clean VLM text to make it simpler and shorter.
    Remove metadata patterns, keep only visual description.
    """
    if pd.isna(text) or not text:
        return "A vehicle on the road."
    
    text = str(text).strip()
    
    # Remove speed patterns like "(4.5 m/s)" or "at 4.5 m/s"
    text = re.sub(r'\([\d.]+\s*m/s\)', '', text)
    text = re.sub(r'at\s+[\d.]+\s*m/s', '', text)
    text = re.sub(r'~[\d.]+\s*m/s', '', text)
    
    # Remove metadata patterns
    text = re.sub(r'\[speed:[^\]]+\]', '', text)
    text = re.sub(r'\[direction:[^\]]+\]', '', text)
    text = re.sub(r',\s*continuing\s+straight[^.]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r',\s*steady[^.]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r',\s*Stopped\s+in\s+lane[^.]*', '', text, flags=re.IGNORECASE)
    
    # Clean up extra spaces and punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\s*,\s*\.', '.', text)  # Fix ", ." to "."
    text = re.sub(r'\.+', '.', text)  # Fix multiple periods
    
    # Ensure it ends with period
    if not text.endswith('.'):
        text = text.rstrip('.') + '.'
    
    # If text is too short or empty after cleaning, use fallback
    if len(text) < 10:
        return "A vehicle on the road."
    
    return text

def generate_pure_vlm_manifest(
    vlm_manifest_path: str,
    dataset_path: str,
    output_path: str,
    device: str = "cuda",
    max_scenes: int = None
):
    """
    Generate a manifest with pure VLM descriptions (no metadata enhancement).
    SAFE: Checks to prevent overwriting best run files.
    """
    # Safety check: Don't overwrite best run files
    BEST_RUN_FILES = [
        "/data/nuplan_text_finetuned/nuplan_text_manifest_vlm_first_130scenes.parquet",  # 1.88m ADE
        "/data/nuplan_text_finetuned/nuplan_text_manifest_vlm_metadata_enhanced.parquet",  # 1.67m ADE
    ]
    
    if output_path in BEST_RUN_FILES:
        print(f"⚠️  WARNING: Attempting to overwrite best run file!")
        print(f"   Best run files are protected:")
        for f in BEST_RUN_FILES:
            print(f"     - {f}")
        print(f"   Please use a different output path.")
        raise ValueError(f"Cannot overwrite best run file: {output_path}")
    
    print("="*80)
    print("GENERATING PURE VLM MANIFEST")
    print("="*80)
    print()
    print(f"Output: {output_path}")
    print("✅ Safe - will not overwrite best run files")
    print()
    
    # Load VLM manifest
    print(f"Loading VLM manifest: {vlm_manifest_path}")
    vlm_df = pd.read_parquet(vlm_manifest_path)
    print(f"Loaded {len(vlm_df)} VLM descriptions")
    
    # Create lookup: (scene_id, frame_idx) -> vlm_text
    vlm_dict = {}
    for _, row in vlm_df.iterrows():
        scene_id = str(row['scene_id'])
        frame_idx = int(row['frame_idx'])
        vlm_text = str(row['text'])
        vlm_dict[(scene_id, frame_idx)] = vlm_text
    print(f"Created VLM lookup with {len(vlm_dict)} entries")
    print()
    
    # Load trajectory data to get scene list
    print(f"Loading trajectory data: {dataset_path}")
    npy_files = sorted(glob.glob(os.path.join(dataset_path, "*.npy")))
    if max_scenes:
        npy_files = npy_files[:max_scenes]
    
    scene_ids = []
    for npy_file in npy_files:
        scene_id = os.path.basename(npy_file).replace('.npy', '')
        scene_ids.append(scene_id)
    print(f"Found {len(scene_ids)} scenes")
    print()
    
    # Load CLIP model
    print(f"Loading CLIP model on {device}...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    print("CLIP model loaded")
    print()
    
    # Generate manifest with pure VLM text
    print("Generating pure VLM manifest...")
    print("  Strategy: Use VLM visual descriptions only (cleaned, no metadata)")
    print("  Mode: Simple and short descriptions")
    print()
    
    rows = []
    for scene_id in tqdm(sorted(scene_ids), desc="Processing scenes"):
        # Process all 50 frames
        for frame_idx in range(50):
            # Get VLM text
            vlm_text = vlm_dict.get((scene_id, frame_idx), None)
            
            if vlm_text:
                # Clean VLM text to make it simpler
                final_text = clean_vlm_text(vlm_text)
            else:
                # Fallback if VLM text not available
                final_text = "A vehicle on the road."
            
            rows.append({
                "scene_id": scene_id,
                "frame_idx": int(frame_idx),
                "object_idx": 0,  # Ego vehicle
                "text": final_text,
            })
    
    df = pd.DataFrame(rows)
    print(f"\nGenerated manifest:")
    print(f"  Total rows: {len(df):,}")
    print(f"  Unique scenes: {df['scene_id'].nunique()}")
    print(f"  Unique texts: {df['text'].nunique():,}")
    print(f"  Text diversity: {df['text'].nunique()/len(df)*100:.1f}%")
    print()
    
    # Show sample texts
    print("Sample texts:")
    for i, text in enumerate(df['text'].head(10)):
        print(f"  {i+1}. {text} ({len(text)} chars)")
    print()
    
    # Generate CLIP embeddings
    print("Generating CLIP embeddings...")
    unique_texts = df['text'].unique()
    text_embeddings = {}
    
    batch_size = 100
    with torch.no_grad():
        for i in tqdm(range(0, len(unique_texts), batch_size), desc="Encoding CLIP"):
            batch_texts = unique_texts[i:i+batch_size]
            inputs = clip_tokenizer(
                batch_texts.tolist(),
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt"
            ).to(device)
            
            feats = clip_model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=1, keepdim=True)  # Normalize
            
            for j, text in enumerate(batch_texts):
                text_embeddings[text] = feats[j].cpu().numpy()
    
    # Add embeddings to dataframe
    df['text_emb'] = df['text'].map(text_embeddings).apply(lambda x: x.tolist() if x is not None else None)
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"✅ Saved pure VLM manifest: {output_path}")
    print(f"   Total rows: {len(df):,}")
    print(f"   Unique texts: {df['text'].nunique():,}")
    print(f"   Diversity: {df['text'].nunique()/len(df)*100:.1f}%")
    print()
    
    # Final verification
    print("="*80)
    print("FINAL VERIFICATION")
    print("="*80)
    print(f"✅ Total rows: {len(df):,}")
    print(f"✅ Scenes: {df['scene_id'].nunique()}")
    print(f"✅ Expected: {df['scene_id'].nunique()} × 50 = {df['scene_id'].nunique() * 50}")
    print(f"✅ Match: {len(df) == df['scene_id'].nunique() * 50}")
    print(f"✅ Text diversity: {df['text'].nunique()/len(df)*100:.1f}%")
    print()
    
    # Check text length distribution
    lengths = df['text'].str.len()
    print(f"Text Length Statistics:")
    print(f"  Min: {lengths.min()} chars")
    print(f"  Max: {lengths.max()} chars")
    print(f"  Mean: {lengths.mean():.1f} chars")
    print(f"  Median: {lengths.median():.1f} chars")
    print()
    print("✅ Best run files protected - this is a test manifest")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm_manifest", type=str,
                       default="/data/nuplan_text_finetuned/nuplan_text_manifest_enhanced.parquet",
                       help="Path to VLM manifest")
    parser.add_argument("--dataset_path", type=str,
                       default="/data/nuscenes_fixed_matrices")
    parser.add_argument("--output_path", type=str,
                       default="/data/nuplan_text_finetuned/nuplan_text_manifest_pure_vlm_test.parquet",
                       help="Output path for pure VLM manifest (TEST - safe)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_scenes", type=int, default=130,
                       help="Maximum number of scenes to process")
    
    args = parser.parse_args()
    generate_pure_vlm_manifest(
        args.vlm_manifest,
        args.dataset_path,
        args.output_path,
        args.device,
        args.max_scenes
    )
