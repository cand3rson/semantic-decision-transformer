#!/usr/bin/env python3
"""
Generate metadata-driven, prompt-style nuPlan-Text with normalized tags.
No external API calls; captions are deterministic from scene metadata.
Outputs a Parquet manifest with CLIP text embeddings.
"""

import os
import glob
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPTokenizer

ALLOWED_TAGS = [
    "turn_left", "turn_right", "go_straight", "lane_change_left", "lane_change_right",
    "merge", "overtake", "u_turn", "stop", "yield", "brake", "accelerate", "hazard_lights", "park"
]

def compute_motion_signals(scene_data: np.ndarray, frame_idx: int, obj_idx: int, k_hist: int = 5) -> Dict:
    """
    Compute motion cues using up to k_hist past frames (if available).
    This dataset doesn't store velocity, so we compute speed from position differences.
    scene_data: [T, N, F], with positions in [:, :, :3]
    """
    T, N, F = scene_data.shape
    f0 = max(0, frame_idx - k_hist + 1)
    window = scene_data[f0:frame_idx+1, obj_idx, :]  # [W, F]
    W = window.shape[0]
    pos = window[:, :3]  # x, y, z
    
    if W < 2:
        return {"speed": 0.0, "accel": 0.0, "dyaw_deg": 0.0, "dx": 0.0, "dy": 0.0}
    
    # Compute speed from position differences
    movements = np.diff(pos, axis=0)  # [W-1, 3]
    speeds = np.linalg.norm(movements[:, :2], axis=1)  # [W-1] - XY speed in m/s
    speed = float(speeds[-1]) if len(speeds) > 0 else 0.0
    accel = float(speeds[-1] - speeds[-2]) if len(speeds) >= 2 else 0.0
    
    # Compute yaw change from position differences
    if W >= 2:
        dx = float(movements[-1, 0])
        dy = float(movements[-1, 1])
        if W >= 3:
            dx_prev = float(movements[-2, 0])
            dy_prev = float(movements[-2, 1])
            yaw_curr = np.arctan2(dy, dx)
            yaw_prev = np.arctan2(dy_prev, dx_prev)
            dyaw = float(np.rad2deg(yaw_curr - yaw_prev))
        else:
            dyaw = 0.0
    else:
        dyaw = 0.0
    
    # XY deltas
    dx = float(pos[-1, 0] - pos[-2, 0]) if W >= 2 else 0.0
    dy = float(pos[-1, 1] - pos[-2, 1]) if W >= 2 else 0.0
    
    return {
        "speed": speed,
        "accel": accel,
        "dyaw_deg": dyaw,
        "dx": dx,
        "dy": dy,
    }

def choose_tags(m: Dict) -> List[str]:
    """Map motion signals to a small, consistent tag subset."""
    tags: List[str] = []
    speed, accel, dyaw = m["speed"], m["accel"], m["dyaw_deg"]
    # Base behavior
    if speed < 0.3:
        tags.append("stop")
    else:
        # Turning threshold ~5 deg per step (heuristic)
        if dyaw > 5.0:
            tags.append("turn_left")
        elif dyaw < -5.0:
            tags.append("turn_right")
        else:
            tags.append("go_straight")
        # Accel / brake cues
        if accel > 0.2:
            tags.append("accelerate")
        elif accel < -0.2:
            tags.append("brake")
    # De-duplicate and ensure allowed
    tags = [t for t in dict.fromkeys(tags) if t in ALLOWED_TAGS]
    return tags

def make_caption(m: Dict) -> str:
    """Compose a concise, unit-consistent, single-sentence caption."""
    speed = m["speed"]
    accel = m["accel"]
    dyaw = m["dyaw_deg"]
    if speed < 0.3:
        return "Stopped in lane."
    # Direction phrase
    if dyaw > 5.0:
        direction = "turning left"
    elif dyaw < -5.0:
        direction = "turning right"
    else:
        direction = "continuing straight"
    # Speed phrase
    sp = f"at ~{speed:.1f} m/s"
    # Accel phrase
    if accel > 0.2:
        ap = "accelerating"
    elif accel < -0.2:
        ap = "slowing"
    else:
        ap = "steady"
    return f"{direction} {sp}, {ap}."

def compute_clip_embeddings(clip_model, clip_tokenizer, device, texts: List[str]) -> np.ndarray:
    inputs = clip_tokenizer(texts, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = clip_model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()

def generate_manifest(data_dir: str,
                      output_path: str,
                      ego_only: bool = True,
                      max_scenes: int = None,
                      device: str = "cuda:1"):
    data_dir = Path(data_dir)
    scene_files = sorted(list(data_dir.glob("*.npy")))
    if max_scenes is not None:
        scene_files = scene_files[:max_scenes]

    # Load CLIP once
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()

    rows: List[Dict] = []
    for scene_path in tqdm(scene_files, desc="Generating captions"):
        try:
            scene = np.load(scene_path)
            scene_id = scene_path.stem
            T, N, F = scene.shape
            obj_indices = [0] if ego_only else list(range(N))

            for frame_idx in range(T):
                for obj_idx in obj_indices:
                    m = compute_motion_signals(scene, frame_idx, obj_idx, k_hist=5)
                    caption = make_caption(m)
                    tags = choose_tags(m)
                    rows.append({
                        "scene_id": scene_id,
                        "frame_idx": int(frame_idx),
                        "object_idx": int(obj_idx),
                        "text": caption,
                        "tags": json.dumps(tags),  # store as JSON string
                    })
        except Exception as e:
            print(f"⚠️  Skipping {scene_path.name}: {e}")
            continue

    df = pd.DataFrame(rows)
    # Compute CLIP embeddings in chunks to save memory
    texts = df["text"].tolist()
    embs = []
    batch = 1024
    for i in tqdm(range(0, len(texts), batch), desc="Encoding CLIP"):
        batch_texts = texts[i:i+batch]
        feats = compute_clip_embeddings(clip_model, clip_tokenizer, device, batch_texts)
        embs.append(feats)
    embs = np.vstack(embs)
    # Convert to lists for Parquet
    df["text_emb"] = [embs[i].astype(np.float32) for i in range(embs.shape[0])]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"💾 Saved prompted manifest: {output_path} ({len(df)} rows)")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/data/nuscenes_fixed_matrices")
    parser.add_argument("--output_path", type=str, default="/data/nuplan_text_finetuned/nuplan_text_manifest_prompted.parquet")
    parser.add_argument("--ego_only", action="store_true")
    parser.add_argument("--max_scenes", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:1")
    args = parser.parse_args()
    generate_manifest(args.data_dir, args.output_path, ego_only=args.ego_only, max_scenes=args.max_scenes, device=args.device)

if __name__ == "__main__":
    main()


