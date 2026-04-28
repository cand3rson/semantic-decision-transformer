#!/usr/bin/env python3
"""
Generate manifest using GPT-2 LLM (like best run) with VLM enhancement.
Strategy: GPT-2 generates natural language from metadata, VLM adds agent context.
This matches the best run's approach while incorporating VLM.
"""

import os
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    CLIPModel, 
    CLIPTokenizer
)

# Import motion signal computation (fixed version)
from generate_prompted_metadata_captions import compute_motion_signals

def categorize_speed(speed: float) -> str:
    """Categorize speed into labels (matching best run format)."""
    if speed < 0.1:
        return 'stopped'
    elif speed < 1.5:
        return 'very slow'
    elif speed < 3.5:
        return 'slow'
    elif speed < 6.5:
        return 'moderate'
    elif speed < 10.0:
        return 'fast'
    else:
        return 'very fast'

def infer_direction_from_motion(motion: Dict) -> str:
    """
    Infer direction from motion signals (matching best run logic).
    Uses dy (position change) like the best run, not dyaw_deg.
    """
    speed = motion["speed"]
    dx = motion.get("dx", 0.0)
    dy = motion.get("dy", 0.0)
    
    if speed < 0.5:
        return 'stopped'
    
    # Best run logic: use dy directly (not dyaw_deg)
    if abs(dy) < 0.3:  # Mostly straight
        return 'straight'
    elif dy > 0.3:  # Turning left
        return 'left_turn'
    elif dy < -0.3:  # Turning right
        return 'right_turn'
    else:
        return 'straight'

def generate_gpt2_text(
    gpt2_model, 
    gpt2_tokenizer, 
    category: str, 
    speed: float, 
    direction: str, 
    device: str
) -> str:
    """
    Generate text using fine-tuned GPT-2 (like best run).
    Prompt format: [category|speed:X|direction:Y]
    """
    prompt = f"[{category}|speed:{speed:.1f}|direction:{direction}]"
    
    inputs = gpt2_tokenizer(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = gpt2_model.generate(
            inputs['input_ids'],
            max_length=50,
            num_return_sequences=1,
            temperature=0.7,  # Same as best run
            do_sample=True,
            top_p=0.9,
            pad_token_id=gpt2_tokenizer.eos_token_id,
            eos_token_id=gpt2_tokenizer.eos_token_id
        )
    
    generated_text = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the description part (after the prompt)
    if prompt in generated_text:
        description = generated_text.replace(prompt, '').strip()
        # Find the end of the first complete sentence
        if '. ' in description:
            description = description.split('. ')[0] + '.'
        elif description and not description.endswith('.'):
            description = description + '.'
        return description
    else:
        return generated_text.strip()

def categorize_acceleration(accel: float) -> str:
    """Categorize acceleration into descriptive labels."""
    if accel < -0.5:
        return 'braking hard'
    elif accel < -0.2:
        return 'braking'
    elif accel < 0.2:
        return 'maintaining constant speed'
    elif accel < 0.5:
        return 'accelerating'
    else:
        return 'accelerating rapidly'

def categorize_yaw_intensity(dyaw_deg: float) -> str:
    """Categorize yaw change into turning intensity descriptions."""
    abs_dyaw = abs(dyaw_deg)
    if abs_dyaw < 2.0:
        return None  # No significant turn
    elif abs_dyaw < 5.0:
        return 'slight'
    elif abs_dyaw < 10.0:
        return 'moderate'
    elif abs_dyaw < 20.0:
        return 'sharp'
    else:
        return 'very sharp'

def enhance_vlm_with_metadata(
    vlm_text: str,
    motion: Dict,
    category: str,
    include_detailed_motion: bool = True,
    include_acceleration: bool = True,
    include_turning_intensity: bool = True,
    include_movement_details: bool = True
) -> str:
    """
    Enhance VLM text with rich metadata details.
    Strategy: VLM provides visual context as BASE, metadata adds explicit detailed instructions.
    
    Args:
        vlm_text: Base text from VLM (visual description - KEEP AS BASE)
        motion: Motion signals dict with speed, accel, dyaw_deg, dx, dy
        category: Object category (vehicle, pedestrian, etc.)
        include_detailed_motion: Include detailed motion descriptions
        include_acceleration: Include acceleration state details
        include_turning_intensity: Include turning intensity details
        include_movement_details: Include movement magnitude and direction details
    
    Returns:
        Enhanced description: VLM base + detailed metadata instructions
    """
    if not vlm_text or pd.isna(vlm_text):
        # Fallback: generate from metadata only
        return generate_metadata_fallback(motion, category)
    
    # Start with VLM text as base (preserve visual context)
    # Clean VLM text: remove any existing metadata patterns to avoid duplication
    vlm_base = str(vlm_text).strip()
    
    # Remove common metadata patterns that might already be in VLM text
    import re
    # Remove speed patterns like "(X.X m/s)" or "at X.X m/s"
    vlm_base = re.sub(r'\([\d.]+\s*m/s\)', '', vlm_base)
    vlm_base = re.sub(r'at\s+[\d.]+\s*m/s', '', vlm_base)
    # Remove direction patterns that might conflict
    vlm_base = re.sub(r'\[speed:[\d.]+\s*m/s\)', '', vlm_base)
    vlm_base = re.sub(r'\[direction:[^\]]+\]', '', vlm_base)
    # Clean up extra spaces and punctuation
    vlm_base = re.sub(r'\s+', ' ', vlm_base).strip()
    vlm_base = re.sub(r'\s*,\s*\.', '.', vlm_base)  # Fix ", ." to "."
    vlm_base = re.sub(r'\.+', '.', vlm_base)  # Fix multiple periods
    if not vlm_base.endswith('.'):
        vlm_base = vlm_base.rstrip('.') + '.'
    
    # Extract metadata
    speed = motion["speed"]
    accel = motion.get("accel", 0.0)
    dyaw_deg = motion.get("dyaw_deg", 0.0)
    dx = motion.get("dx", 0.0)
    dy = motion.get("dy", 0.0)
    movement_magnitude = np.sqrt(dx**2 + dy**2)
    
    # Build enhancement parts (detailed metadata instructions)
    enhancement_parts = []
    
    # 1. Speed details (always include)
    speed_category = categorize_speed(speed)
    if include_detailed_motion:
        enhancement_parts.append(f"moving at {speed_category} speed ({speed:.1f} m/s)")
    else:
        enhancement_parts.append(f"at {speed:.1f} m/s")
    
    # 2. Acceleration state (if significant)
    if include_acceleration and abs(accel) > 0.1:
        accel_state = categorize_acceleration(accel)
        if accel_state != 'maintaining constant speed':
            enhancement_parts.append(accel_state)
    
    # 3. Direction and turning intensity
    direction = infer_direction_from_motion(motion)
    if direction == 'stopped':
        if include_detailed_motion:
            enhancement_parts.append("currently stopped")
    elif direction == 'left_turn':
        turn_intensity = categorize_yaw_intensity(dyaw_deg) if include_turning_intensity else None
        if turn_intensity:
            enhancement_parts.append(f"executing a {turn_intensity} left turn")
        else:
            enhancement_parts.append("turning left")
    elif direction == 'right_turn':
        turn_intensity = categorize_yaw_intensity(dyaw_deg) if include_turning_intensity else None
        if turn_intensity:
            enhancement_parts.append(f"executing a {turn_intensity} right turn")
        else:
            enhancement_parts.append("turning right")
    else:  # straight
        if include_detailed_motion:
            enhancement_parts.append("traveling straight ahead")
    
    # 4. Movement magnitude and direction details
    if include_movement_details and movement_magnitude > 0.3:
        # Determine movement direction
        if abs(dy) > abs(dx):
            if dy > 0:
                movement_dir = "laterally left"
            else:
                movement_dir = "laterally right"
        else:
            if dx > 0:
                movement_dir = "forward"
            else:
                movement_dir = "backward"
        
        enhancement_parts.append(f"with {movement_magnitude:.1f}m {movement_dir} displacement")
    
    # 5. Yaw change details (if significant and not already covered)
    if include_turning_intensity and abs(dyaw_deg) > 2.0 and direction == 'straight':
        turn_intensity = categorize_yaw_intensity(dyaw_deg)
        if turn_intensity:
            enhancement_parts.append(f"with {turn_intensity} yaw change ({dyaw_deg:.1f}°)")
    
    # Combine: VLM base + detailed metadata enhancements
    if enhancement_parts:
        # Add metadata as explicit instructions after VLM base
        metadata_instruction = ", ".join(enhancement_parts)
        enhanced = f"{vlm_base} The vehicle is {metadata_instruction}."
    else:
        # Just add basic speed if no other details
        enhanced = f"{vlm_base} Moving at {speed:.1f} m/s."
    
    # Truncate if too long for CLIP (max ~77 tokens, ~300 chars to be safe)
    # CLIP uses max_position_embeddings=77, so we need to stay under that
    MAX_LENGTH = 250  # Conservative limit to stay under CLIP's 77 token limit
    if len(enhanced) > MAX_LENGTH:
        # Try to truncate intelligently - keep VLM base, truncate metadata
        if len(vlm_base) < MAX_LENGTH - 50:  # If base is short enough
            # Keep base + essential metadata only
            essential_parts = []
            if enhancement_parts:
                essential_parts.append(enhancement_parts[0])  # Speed (most important)
                if len(enhancement_parts) > 1:
                    essential_parts.append(enhancement_parts[1])  # Direction/action
            if essential_parts:
                metadata_short = ", ".join(essential_parts)
                enhanced = f"{vlm_base} The vehicle is {metadata_short}."
            else:
                enhanced = f"{vlm_base} Moving at {speed:.1f} m/s."
        else:
            # VLM base itself is too long, just truncate it
            enhanced = vlm_base[:MAX_LENGTH-3] + "..."
    
    return enhanced

def generate_metadata_fallback(motion: Dict, category: str) -> str:
    """Generate description from metadata only (fallback when VLM missing)."""
    speed = motion["speed"]
    accel = motion.get("accel", 0.0)
    direction = infer_direction_from_motion(motion)
    
    parts = [f"A {category.lower()}"]
    
    if direction == 'stopped':
        parts.append("stopped")
    elif direction == 'left_turn':
        parts.append("turning left")
    elif direction == 'right_turn':
        parts.append("turning right")
    else:
        parts.append("going straight")
    
    speed_category = categorize_speed(speed)
    parts.append(f"at {speed_category} speed ({speed:.1f} m/s)")
    
    if abs(accel) > 0.1:
        accel_state = categorize_acceleration(accel)
        if accel_state != 'maintaining constant speed':
            parts.append(accel_state)
    
    return " ".join(parts) + "."

def enhance_gpt2_with_vlm(gpt2_text: str, vlm_text: Optional[str], category: str) -> str:
    """
    Enhance GPT-2 generated text with VLM agent type if available.
    Strategy: GPT-2 provides natural language, VLM adds agent specificity.
    Only enhances if VLM agent type matches our category (e.g., "vehicle" for ego).
    """
    if vlm_text is None or pd.isna(vlm_text):
        return gpt2_text
    
    vlm_lower = str(vlm_text).lower()
    category_lower = category.lower()
    
    # Map VLM agent types to our category
    vehicle_keywords = ["vehicle", "car", "truck", "bus", "motorcycle", "automobile"]
    pedestrian_keywords = ["pedestrian", "person", "walker"]
    cyclist_keywords = ["cyclist", "bicycle", "bike"]
    
    # Check if VLM mentions an agent type that matches our category
    vlm_matches_category = False
    if category_lower == "vehicle":
        vlm_matches_category = any(kw in vlm_lower for kw in vehicle_keywords)
    elif category_lower == "pedestrian":
        vlm_matches_category = any(kw in vlm_lower for kw in pedestrian_keywords)
    elif category_lower == "cyclist":
        vlm_matches_category = any(kw in vlm_lower for kw in cyclist_keywords)
    
    # Only enhance if VLM agent type matches our category
    # This prevents VLM from incorrectly changing "vehicle" to "pedestrian" etc.
    if not vlm_matches_category:
        return gpt2_text  # VLM doesn't match, use pure GPT-2
    
    # Extract specific agent type from VLM for enhancement
    # IMPORTANT: Only extract agent types that match our category
    # This prevents extracting "pedestrian" when category is "vehicle"
    if category_lower == "vehicle":
        agent_types = ["vehicle", "car", "truck", "bus", "motorcycle", "automobile"]
    elif category_lower == "pedestrian":
        agent_types = ["pedestrian", "person", "walker"]
    elif category_lower == "cyclist":
        agent_types = ["cyclist", "bicycle", "bike"]
    else:
        agent_types = ["vehicle"]  # Default to vehicle
    
    agent_mention = None
    for agent in agent_types:
        if agent in vlm_lower:
            # Prefer "vehicle" over "car" for consistency
            if agent == "car" and "vehicle" in vlm_lower:
                agent_mention = "vehicle"
                break
            agent_mention = agent
            break
    
    # If no matching agent type found, use category as fallback
    if agent_mention is None:
        agent_mention = category_lower
    
    # If we found a matching agent type, enhance the text
    if agent_mention:
        # Try to integrate naturally
        if gpt2_text.lower().startswith("the "):
            # Replace "The" with "A [agent]"
            enhanced = f"A {agent_mention} " + gpt2_text[4:]  # Skip "The "
        elif gpt2_text.lower().startswith("a "):
            # Already has "A", just replace with agent
            words = gpt2_text.split()
            if len(words) > 1:
                enhanced = f"A {agent_mention} " + " ".join(words[2:])  # Skip "A [old_agent]"
            else:
                enhanced = gpt2_text
        else:
            # Prepend agent
            enhanced = f"A {agent_mention} {gpt2_text}"
        
        return enhanced
    
    # Otherwise, keep GPT-2 text as-is
    return gpt2_text

def enhance_vlm_with_gpt2(vlm_text: str, gpt2_text: str, category: str, speed: float, direction: str) -> str:
    """
    Enhance VLM text with GPT-2 motion information.
    Strategy: VLM provides visual context and agent type, GPT-2 adds precise motion details.
    
    Args:
        vlm_text: Base text from VLM (visual description)
        gpt2_text: GPT-2 generated text (contains precise motion info)
        category: Object category (vehicle, pedestrian, etc.)
        speed: Current speed in m/s
        direction: Direction label (straight, left_turn, right_turn, stopped)
    """
    if not vlm_text or pd.isna(vlm_text):
        return gpt2_text  # Fallback to GPT-2 if VLM missing
    
    import re
    vlm_lower = str(vlm_text).lower()
    gpt2_lower = str(gpt2_text).lower()
    
    # Extract motion information from GPT-2 text
    # GPT-2 typically has format: "A vehicle [action] at [speed_label] speed (X.X m/s)."
    
    # Extract speed value from GPT-2 (format: "X.X m/s")
    speed_match = re.search(r'\(([\d.]+)\s*m/s\)', gpt2_text)
    speed_value = speed_match.group(1) if speed_match else f"{speed:.1f}"
    
    # Extract speed label from GPT-2
    speed_labels = ['stopped', 'very slow', 'slow', 'moderate', 'fast', 'very fast']
    speed_label = None
    for label in speed_labels:
        if label in gpt2_lower:
            speed_label = label
            break
    
    # Extract action/direction from GPT-2
    if 'stopped' in gpt2_lower or 'in place' in gpt2_lower:
        action_phrase = "stopped in place"
    elif 'left turn' in gpt2_lower or 'turning left' in gpt2_lower:
        action_phrase = "making a left turn"
    elif 'right turn' in gpt2_lower or 'turning right' in gpt2_lower:
        action_phrase = "making a right turn"
    elif 'straight' in gpt2_lower:
        action_phrase = "going straight"
    else:
        # Fallback to direction from metadata
        if direction == 'stopped':
            action_phrase = "stopped in place"
        elif direction == 'left_turn':
            action_phrase = "making a left turn"
        elif direction == 'right_turn':
            action_phrase = "making a right turn"
        else:
            action_phrase = "going straight"
    
    # Extract motion state from GPT-2 (if present)
    motion_state = None
    for state in ['accelerating', 'steady', 'slowing', 'decelerating']:
        if state in gpt2_lower:
            motion_state = state
            break
    
    # Determine agent type from VLM (if matches category) or use category
    vehicle_keywords = ["vehicle", "car", "truck", "bus", "motorcycle", "automobile"]
    pedestrian_keywords = ["pedestrian", "person", "walker"]
    cyclist_keywords = ["cyclist", "bicycle", "bike"]
    
    agent_mention = None
    if category.lower() == "vehicle":
        for kw in vehicle_keywords:
            if kw in vlm_lower:
                agent_mention = "vehicle" if kw == "car" and "vehicle" in vlm_lower else kw
                break
    elif category.lower() == "pedestrian":
        for kw in pedestrian_keywords:
            if kw in vlm_lower:
                agent_mention = kw
                break
    elif category.lower() == "cyclist":
        for kw in cyclist_keywords:
            if kw in vlm_lower:
                agent_mention = kw
                break
    
    if not agent_mention:
        agent_mention = category.lower()
    
    # Build enhanced text: Agent + Action + Speed + State
    parts = [f"A {agent_mention}", action_phrase]
    
    if speed_label:
        parts.append(f"at {speed_label} speed ({speed_value} m/s)")
    else:
        parts.append(f"at {speed_value} m/s")
    
    if motion_state:
        parts.append(motion_state)
    
    enhanced = " ".join(parts) + "."
    
    return enhanced

def generate_gpt2_vlm_hybrid_manifest(
    vlm_manifest_path: Optional[str],
    dataset_path: str,
    output_path: str,
    gpt2_model_path: str,
    device: str = "cuda",
    max_scenes: int = 130,
    vlm_usage_ratio: float = 1.0,  # Use VLM for X% of frames (when VLM-first mode)
    category: str = "vehicle",  # Default category (ego vehicle)
    vlm_first: bool = False  # If True: VLM primary + GPT-2 enhancer, If False: GPT-2 primary + VLM enhancer
):
    """
    Generate manifest using hybrid GPT-2 + VLM approach.
    
    Two modes:
    1. GPT-2-first (default, vlm_first=False):
       - Primary: GPT-2 generates natural language from metadata (like best run)
       - Secondary: VLM enhances with agent type for subset of frames
       - Ensures high diversity through LLM generation
    
    2. VLM-first (vlm_first=True):
       - Primary: VLM provides visual context and agent type from images
       - Secondary: GPT-2 enhances with precise motion information (speed, direction, state)
       - Leverages visual understanding while adding accurate motion details
    """
    print("="*70)
    print("GENERATING GPT-2 + VLM HYBRID MANIFEST")
    print("="*70)
    print()
    
    # Load GPT-2 model (like best run)
    print(f"Loading fine-tuned GPT-2 model: {gpt2_model_path}")
    gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_path).to(device)
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_path)
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    gpt2_model.eval()
    print("✅ GPT-2 model loaded")
    print()
    
    # Load VLM manifest if provided
    vlm_dict = {}
    if vlm_manifest_path and os.path.exists(vlm_manifest_path):
        print(f"Loading VLM manifest: {vlm_manifest_path}")
        df_vlm = pd.read_parquet(vlm_manifest_path)
        print(f"Loaded {len(df_vlm):,} VLM descriptions")
        
        # Get text column name
        text_col = 'texts' if 'texts' in df_vlm.columns else 'text'
        if text_col not in df_vlm.columns:
            text_col = [c for c in df_vlm.columns if 'text' in c.lower() or 'caption' in c.lower()][0]
        
        # Create lookup: (scene_id, frame_idx) -> vlm_text
        for idx, row in df_vlm.iterrows():
            scene_id = str(row['scene_id'])
            frame_idx = int(row['frame_idx'])
            vlm_text = str(row[text_col])
            vlm_dict[(scene_id, frame_idx)] = vlm_text
        
        print(f"Created VLM lookup with {len(vlm_dict):,} entries")
    else:
        print("No VLM manifest provided, using pure GPT-2")
    print()
    
    # Load trajectory data
    print(f"Loading trajectory data: {dataset_path}")
    npy_files = sorted(glob.glob(os.path.join(dataset_path, "*.npy")))
    if max_scenes:
        npy_files = npy_files[:max_scenes]
    
    scene_data_dict = {}
    for npy_file in tqdm(npy_files, desc="Loading scenes"):
        scene_id = os.path.basename(npy_file).replace('.npy', '')
        try:
            scene_data_dict[scene_id] = np.load(npy_file)
        except Exception as e:
            print(f"  Warning: Error loading {scene_id}: {e}")
    print(f"Loaded {len(scene_data_dict)} scenes")
    print()
    
    # Load CLIP model
    print(f"Loading CLIP model on {device}...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    print("CLIP model loaded")
    print()
    
    # Generate hybrid manifest
    print("Generating GPT-2 + VLM hybrid manifest...")
    if vlm_first:
        print(f"  Mode: VLM-FIRST (VLM primary, metadata enhancer)")
        print(f"  Strategy: VLM provides visual context as BASE for {vlm_usage_ratio*100:.0f}% of frames")
        print(f"           Rich metadata adds explicit detailed motion instructions")
    else:
        print(f"  Mode: GPT-2-FIRST (GPT-2 primary, VLM enhancer)")
        print(f"  Strategy: GPT-2 LLM for all frames (like best run)")
        print(f"           VLM enhancement for {vlm_usage_ratio*100:.0f}% of frames")
    print()
    
    rows: List[Dict] = []
    np.random.seed(42)  # For reproducible VLM selection
    
    for scene_id in tqdm(sorted(scene_data_dict.keys()), desc="Processing scenes"):
        scene_data = scene_data_dict[scene_id]
        T, N, F = scene_data.shape
        
        # Process all 50 frames (or scene length)
        for frame_idx in range(min(50, T)):
            obj_idx = 0  # Ego vehicle
            
            # Compute motion signals
            motion = compute_motion_signals(scene_data, frame_idx, obj_idx, k_hist=5)
            
            # Prepare metadata for GPT-2 (like best run)
            speed = motion["speed"]
            speed_category = categorize_speed(speed)
            direction = infer_direction_from_motion(motion)
            
            # Generate GPT-2 text (needed for both modes)
            gpt2_text = generate_gpt2_text(
                gpt2_model, 
                gpt2_tokenizer, 
                category, 
                speed, 
                direction, 
                device
            )
            
            # Get VLM text if available
            use_vlm = np.random.random() < vlm_usage_ratio
            vlm_text = vlm_dict.get((scene_id, frame_idx), None)
            
            # Choose generation strategy based on mode
            if vlm_first:
                # VLM-FIRST MODE: VLM is primary, metadata enhances with detailed instructions
                if use_vlm and vlm_text:
                    # Enhance VLM text with rich metadata (direct, not from GPT-2)
                    final_text = enhance_vlm_with_metadata(
                        vlm_text,
                        motion,
                        category,
                        include_detailed_motion=True,
                        include_acceleration=True,
                        include_turning_intensity=True,
                        include_movement_details=True
                    )
                else:
                    # Fallback to GPT-2 if VLM not available
                    final_text = gpt2_text
            else:
                # GPT-2-FIRST MODE: GPT-2 is primary, VLM enhances
                if use_vlm and vlm_text:
                    # Enhance GPT-2 text with VLM agent type
                    final_text = enhance_gpt2_with_vlm(gpt2_text, vlm_text, category)
                else:
                    # Use pure GPT-2 text (like best run)
                    final_text = gpt2_text
            
            rows.append({
                "scene_id": scene_id,
                "frame_idx": int(frame_idx),
                "object_idx": int(obj_idx),
                "text": final_text,
            })
    
    df = pd.DataFrame(rows)
    print(f"\nGenerated manifest:")
    print(f"  Total rows: {len(df):,}")
    print(f"  Unique scenes: {df['scene_id'].nunique()}")
    print(f"  Unique texts: {df['text'].nunique():,}")
    print(f"  Text diversity: {df['text'].nunique()/len(df)*100:.1f}%")
    print()
    
    # Analyze text types
    try:
        pure_gpt2 = df[~df['text'].str.lower().str.startswith('a ', na=False)]
        vlm_enhanced = df[df['text'].str.lower().str.startswith('a ', na=False)]
        
        print(f"Text type breakdown:")
        print(f"  Pure GPT-2 (like best run): {len(pure_gpt2):,} ({len(pure_gpt2)/len(df)*100:.1f}%)")
        print(f"  GPT-2 + VLM enhanced: {len(vlm_enhanced):,} ({len(vlm_enhanced)/len(df)*100:.1f}%)")
        print()
        
        # Show sample texts
        print("Sample texts:")
        print("  Pure GPT-2 examples:")
        for text in pure_gpt2['text'].head(3):
            print(f"    - {text}")
        if len(vlm_enhanced) > 0:
            print("  VLM-enhanced examples:")
            for text in vlm_enhanced['text'].head(3):
                print(f"    - {text}")
        print()
    except Exception as e:
        print(f"  Note: Could not analyze text types: {e}")
        print()
    
    # Generate CLIP embeddings
    print("Generating CLIP embeddings...")
    texts = df["text"].tolist()
    embs = []
    batch_size = 1024
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding CLIP"):
        batch_texts = texts[i:i+batch_size]
        inputs = clip_tokenizer(batch_texts, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = clip_model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        embs.append(feats.cpu().numpy())
    embs = np.vstack(embs)
    df["text_emb"] = [embs[i].astype(np.float32) for i in range(embs.shape[0])]
    print()
    
    # Safety check: Don't overwrite best run file
    BEST_RUN_FILE = "/data/nuplan_text_finetuned/nuplan_text_manifest_vlm_first_130scenes.parquet"
    if output_path == BEST_RUN_FILE:
        print(f"⚠️  WARNING: Attempting to overwrite best run file!")
        print(f"   Best run: {BEST_RUN_FILE}")
        print(f"   Please use a different output path to preserve the best run.")
        raise ValueError(f"Cannot overwrite best run file: {BEST_RUN_FILE}")
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"✅ Saved GPT-2 + VLM hybrid manifest: {output_path}")
    print(f"   Total rows: {len(df):,}")
    print(f"   Unique texts: {df['text'].nunique():,}")
    print(f"   Diversity: {df['text'].nunique()/len(df)*100:.1f}%")
    print()
    
    # Final verification
    print("="*70)
    print("FINAL VERIFICATION")
    print("="*70)
    print(f"✅ Total rows: {len(df):,}")
    print(f"✅ Scenes: {df['scene_id'].nunique()}")
    print(f"✅ Expected: {df['scene_id'].nunique()} × 50 = {df['scene_id'].nunique() * 50}")
    print(f"✅ Match: {len(df) == df['scene_id'].nunique() * 50}")
    print(f"✅ Text diversity: {df['text'].nunique()/len(df)*100:.1f}%")
    print()
    
    # Check frame distribution
    frames_per_scene = df.groupby('scene_id')['frame_idx'].nunique()
    print(f"Frames per scene:")
    print(f"  Min: {frames_per_scene.min()}")
    print(f"  Max: {frames_per_scene.max()}")
    print(f"  Mean: {frames_per_scene.mean():.1f}")
    print(f"  All scenes have 50 frames: {(frames_per_scene == 50).all()}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm_manifest", type=str, default=None,
                       help="Path to VLM manifest (optional)")
    parser.add_argument("--dataset_path", type=str,
                       default="/data/nuscenes_fixed_matrices")
    parser.add_argument("--output_path", type=str,
                       default="/data/nuplan_text_finetuned/nuplan_text_manifest_vlm_metadata_enhanced.parquet",
                       help="Output path for enhanced manifest (default: vlm_metadata_enhanced.parquet - does NOT overwrite best run)")
    parser.add_argument("--gpt2_model_path", type=str,
                       default=str(Path(__file__).parent / "gpt2_finetuned_model_improved"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_scenes", type=int, default=130)
    parser.add_argument("--vlm_usage_ratio", type=float, default=1.0,
                       help="Fraction of frames to use VLM (0.0-1.0, default=1.0 for 100%)")
    parser.add_argument("--category", type=str, default="vehicle",
                       help="Object category (vehicle, pedestrian, etc.)")
    parser.add_argument("--vlm_first", action="store_true", default=False,
                       help="Use VLM-first mode: VLM primary + metadata enhancer (default: GPT-2-first)")
    parser.add_argument("--include_detailed_motion", action="store_true", default=True,
                       help="Include detailed motion descriptions in metadata enhancement")
    parser.add_argument("--include_acceleration", action="store_true", default=True,
                       help="Include acceleration state in metadata enhancement")
    parser.add_argument("--include_turning_intensity", action="store_true", default=True,
                       help="Include turning intensity details in metadata enhancement")
    parser.add_argument("--include_movement_details", action="store_true", default=True,
                       help="Include movement magnitude and direction in metadata enhancement")
    args = parser.parse_args()
    
    generate_gpt2_vlm_hybrid_manifest(
        args.vlm_manifest,
        args.dataset_path,
        args.output_path,
        args.gpt2_model_path,
        args.device,
        args.max_scenes,
        args.vlm_usage_ratio,
        args.category,
        args.vlm_first
    )
    
    # Note: Metadata enhancement options (include_detailed_motion, etc.) are used
    # when vlm_first=True and enhance_vlm_with_metadata() is called

