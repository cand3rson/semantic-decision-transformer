#!/usr/bin/env python3
"""
Qwen + LoRA + Instruction-Tuned SEMDT Trajectory Experiment
===========================================================

This script adds a NEW experimental baseline:

    QwenInstructionTunedTrajectoryLLM
    - Qwen backbone with LoRA adapters
    - Natural language INTENT instructions as prompts
    - Continuous kinematic history as "soft tokens"
    - Semantic + trajectory streams combined with λ = 0.1

It reuses:
    - Movement-normalized DT data pipeline (ego-centric, movement stats)
    - nuPlan-style context/prediction horizons
    - Text manifest with motion-aware captions (for INTENT guidance)

IMPORTANT:
    - This DOES NOT modify or replace existing DT / Qwen experiments.
    - It is an additive, fairer LLM baseline for comparison.
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from qwen_instruction_lora_model import QwenInstructionTunedTrajectoryLLM

# Reuse movement-normalized pipeline + text manifest loader
from trajectory_experiment_visiontrap import (
    load_nuscenes_scenes,
    compute_movement_statistics,
    process_nuscenes_movement_normalized,
    load_text_manifest,
    compute_lateral_scores_and_weights,
    calculate_movement_metrics,
)


def build_instruction_from_intent(intent_text: str, horizon: int) -> str:
    """Wrap raw INTENT caption into an instruction prompt."""
    base = (
        "You are an autonomous driving planner operating in urban traffic. "
        f"The current maneuver intent is: {intent_text}. "
        f"Based on the following kinematic history, predict the next {horizon} "
        "steps of the ego vehicle trajectory as physically plausible movements."
    )
    return base


def match_text_for_sample(
    scene_text_map: Dict[str, List[Dict]],
    scene_id: str,
    target_frame: int,
) -> Tuple[str, np.ndarray]:
    """
    Find the closest text entry for a given (scene_id, frame_idx).
    Falls back to a generic caption if none are found.
    """
    entries = scene_text_map.get(scene_id, [])
    if not entries:
        # Fallback generic intent if we have no caption
        return (
            "The ego vehicle continues in traffic with cautious, "
            "intent-aware urban driving behavior.",
            np.zeros(512, dtype=np.float32),
        )

    # Choose entry with minimal |frame_idx - target_frame|
    best = min(
        entries,
        key=lambda e: abs(int(e.get("frame_idx", 0)) - int(target_frame)),
    )
    text = best.get("text", "") or best.get("texts", "")
    text_emb = np.array(best.get("text_emb", np.zeros(512)), dtype=np.float32)
    if text == "":
        text = (
            "The ego vehicle continues in traffic with cautious, "
            "intent-aware urban driving behavior."
        )
    return text, text_emb


class MovementSemanticDataset(Dataset):
    """
    Dataset for movement-normalized trajectories with semantic INTENT captions.

    Each item provides:
        - states: (K, state_dim)
        - actions: (K, 3) - context actions (normalized ego movements)
        - returns_to_go: (K, 1) - context RTG values aligned to DT-style tokens
        - timesteps: (K,) - timestep ids for DT-style embeddings
        - future_movements: (H, 3)
        - instruction: string prompt using INTENT caption
        - text_emb: (512,) semantic embedding for InfoNCE
    """

    def __init__(
        self,
        trajectories: List[Dict],
        scene_text_map: Dict[str, List[Dict]],
        context_length: int,
        prediction_horizon: int,
    ):
        self.trajectories = trajectories
        self.scene_text_map = scene_text_map
        self.context_length = context_length
        self.prediction_horizon = prediction_horizon

        # All states share the same flattened dimension by construction
        if len(trajectories) == 0:
            raise ValueError("Empty trajectory list passed to MovementSemanticDataset")
        self.state_dim = trajectories[0]["states"].shape[1]

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.trajectories[idx]

        states = sample["states"]  # (K, state_dim)
        actions = sample["actions"]  # (K, 3)
        rtg = sample["rtg"]  # (K+1, 1)
        timesteps = sample["timesteps"]  # (K,)
        future_movements = sample["future_movements"]  # (H, 3)
        scene_id = sample["scene_id"]
        start_frame = sample["start_frame"]

        # Match INTENT caption near the last context frame
        target_frame = int(start_frame + self.context_length - 1)
        intent_text, text_emb = match_text_for_sample(
            self.scene_text_map, scene_id, target_frame
        )
        instruction = build_instruction_from_intent(
            intent_text, self.prediction_horizon
        )

        return {
            "states": torch.from_numpy(states).float(),  # (K, state_dim)
            "actions": torch.from_numpy(actions).float(),  # (K, 3)
            # Use first K RTG entries to align with DT tokens over context length.
            "returns_to_go": torch.from_numpy(rtg[: self.context_length]).float(),  # (K, 1)
            "timesteps": torch.from_numpy(timesteps).long(),  # (K,)
            "future_movements": torch.from_numpy(future_movements).float(),  # (H, 3)
            "instruction": instruction,
            "text_emb": torch.from_numpy(text_emb).float(),  # (512,)
        }


def collate_movement_semantic(batch: List[Dict]) -> Dict:
    """Simple collate function (no padding needed; dims are fixed)."""
    states = torch.stack([b["states"] for b in batch], dim=0)
    actions = torch.stack([b["actions"] for b in batch], dim=0)
    returns_to_go = torch.stack([b["returns_to_go"] for b in batch], dim=0)
    timesteps = torch.stack([b["timesteps"] for b in batch], dim=0)
    future_movements = torch.stack([b["future_movements"] for b in batch], dim=0)
    instructions = [b["instruction"] for b in batch]
    text_embs = torch.stack([b["text_emb"] for b in batch], dim=0)
    return {
        "states": states,  # (B, K, state_dim)
        "actions": actions,  # (B, K, 3)
        "returns_to_go": returns_to_go,  # (B, K, 1)
        "timesteps": timesteps,  # (B, K)
        "future_movements": future_movements,  # (B, H, 3)
        "instructions": instructions,  # list of len B
        "text_embs": text_embs,  # (B, 512)
    }


def filter_trajectories_by_rtg_quality(
    trajectories: List[Dict],
    context_length: int,
    keep_top_fraction: float,
    description: str,
) -> List[Dict]:
    """
    Drop the lowest-quality samples according to the RTG proxy stored in each sample.

    In this codebase, RTG is constructed from a negative motion-variance quantity:
      trajectory_quality = -var(||movement||)
    so higher RTG values correspond to smoother / lower-variance trajectories.
    """
    if keep_top_fraction is None or keep_top_fraction >= 1.0:
        return trajectories
    if keep_top_fraction <= 0.0:
        raise ValueError("keep_top_fraction must be > 0.0")
    if len(trajectories) == 0:
        return trajectories

    scores = []
    for t in trajectories:
        rtg = t.get("rtg", None)
        if rtg is None:
            scores.append(-1e9)
            continue
        # rtg shape is typically (context_length+1, 1)
        rtg_arr = np.asarray(rtg)
        q = rtg_arr[:context_length].mean()
        scores.append(float(q))
    scores_np = np.asarray(scores, dtype=np.float32)

    drop_fraction = 1.0 - keep_top_fraction
    threshold = float(np.quantile(scores_np, drop_fraction))
    keep_mask = scores_np >= threshold
    kept = [t for t, m in zip(trajectories, keep_mask.tolist()) if m]

    # Safety: don't accidentally drop everything due to numerical oddities
    if len(kept) == 0:
        idx = int(np.argmax(scores_np))
        kept = [trajectories[idx]]

    print(
        f"\n🔧 RTG quality filtering ({description}): keep_top_fraction={keep_top_fraction:.3f}, "
        f"kept {len(kept)}/{len(trajectories)} samples, threshold={threshold:.6f}, "
        f"score_range=[{float(scores_np.min()):.6f},{float(scores_np.max()):.6f}]"
    )
    return kept


def info_nce_loss(
    agent_z: torch.Tensor,
    text_z: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Standard InfoNCE contrastive loss for semantic grounding.

    Args:
        agent_z: (B, D) normalized embeddings
        text_z: (B, D) normalized embeddings
    """
    logits = agent_z @ text_z.t() / temperature  # (B, B)
    labels = torch.arange(agent_z.size(0), device=agent_z.device)
    return F.cross_entropy(logits, labels)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Qwen + LoRA + Instruction-Tuned SEMDT Trajectory Experiment"
    )
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--text_manifest_path", type=str, required=True)
    parser.add_argument("--max_files", type=int, default=20)
    parser.add_argument("--context_length", type=int, default=15)
    parser.add_argument("--prediction_horizon", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument(
        "--text_loss_weight",
        type=float,
        default=0.1,
        help="λ: weight for text contrastive loss",
    )
    parser.add_argument(
        "--text_warmup_epochs",
        type=int,
        default=5,
        help="Warmup epochs for the text loss weight (set 0 to disable)",
    )
    parser.add_argument(
        "--text_loss_cap",
        type=float,
        default=0.2,
        help="Cap (after InfoNCE scaling) for text contrastive loss contribution",
    )
    parser.add_argument(
        "--infonce_scale_factor",
        type=float,
        default=0.001,
        help="Scale InfoNCE loss to match trajectory loss magnitude",
    )
    parser.add_argument(
        "--enable_oversampling",
        action="store_true",
        help="Enable lateral maneuver oversampling (matches DT best-run behavior)",
    )
    parser.add_argument(
        "--oversampling_temperature",
        type=float,
        default=0.75,
        help="Temperature for oversampling weight smoothing",
    )
    parser.add_argument(
        "--target_straight_frac",
        type=float,
        default=0.25,
        help="Target fraction for straight samples",
    )
    parser.add_argument(
        "--target_mild_frac",
        type=float,
        default=0.35,
        help="Target fraction for mild lateral samples",
    )
    parser.add_argument(
        "--target_strong_frac",
        type=float,
        default=0.40,
        help="Target fraction for strong lateral samples",
    )
    parser.add_argument(
        "--max_movement_filter",
        type=float,
        default=50.0,
        help="Filter movements larger than this (meters/timestep)",
    )
    parser.add_argument(
        "--movement_variance_filter",
        type=float,
        default=100.0,
        help="Filter samples with movement variance above this",
    )
    parser.add_argument(
        "--rtg_keep_top_fraction",
        type=float,
        default=1.0,
        help="Drop low-RTG (low-quality) samples based on the RTG proxy stored in the dataset (train split only).",
    )
    parser.add_argument(
        "--llm_model_name",
        type=str,
        default="Qwen/Qwen2-1.5B-Instruct",
        help="Base Qwen model name",
    )
    parser.add_argument(
        "--no_lora",
        action="store_true",
        help="Disable LoRA adapters — fine-tune ALL Qwen parameters instead. "
             "Produces a pure instruction-tuning baseline without LoRA.",
    )
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("QWEN + LoRA + INSTRUCTION-TUNED SEMDT TRAJECTORY EXPERIMENT")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Text manifest: {args.text_manifest_path}")
    print(f"Context length (K): {args.context_length}")
    print(f"Prediction horizon (H): {args.prediction_horizon}")
    print(f"λ (text_loss_weight): {args.text_loss_weight}  [FIXED NEAR 0.1]")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Data loading & preprocessing (reuse movement-normalized pipeline)
    # ------------------------------------------------------------------
    print("\n📊 PHASE 1: Loading raw scenes...")
    ego_trajectories = load_nuscenes_scenes(
        args.dataset_path,
        max_files=args.max_files,
    )
    if len(ego_trajectories) == 0:
        raise RuntimeError("No scenes loaded from dataset_path")

    # Train/test split at scene level
    print("\n📊 PHASE 2: Splitting scenes into train/test...")
    random.seed(42)
    np.random.seed(42)
    scene_indices = list(range(len(ego_trajectories)))
    random.shuffle(scene_indices)

    num_train_scenes = int(0.8 * len(ego_trajectories))
    train_indices = scene_indices[:num_train_scenes]
    test_indices = scene_indices[num_train_scenes:]

    train_ego_trajectories = [ego_trajectories[i] for i in train_indices]
    test_ego_trajectories = [ego_trajectories[i] for i in test_indices]

    print(f"  Training scenes: {len(train_ego_trajectories)}")
    print(f"  Test scenes: {len(test_ego_trajectories)}")

    # Movement statistics from TRAIN scenes only
    print("\n📊 PHASE 3: Computing movement normalization statistics on TRAIN scenes only...")
    movement_stats = compute_movement_statistics(train_ego_trajectories)

    # Process trajectories with fixed stats
    print("\n📊 PHASE 4: Creating normalized samples...")
    train_trajectories = process_nuscenes_movement_normalized(
        train_ego_trajectories,
        movement_stats,
        args,
        context_length=args.context_length,
        prediction_horizon=args.prediction_horizon,
    )
    test_trajectories = process_nuscenes_movement_normalized(
        test_ego_trajectories,
        movement_stats,
        args,
        context_length=args.context_length,
        prediction_horizon=args.prediction_horizon,
    )

    # Optional: drop low-quality (likely high-ADE) training samples
    train_trajectories = filter_trajectories_by_rtg_quality(
        train_trajectories,
        context_length=args.context_length,
        keep_top_fraction=args.rtg_keep_top_fraction,
        description="TRAIN",
    )

    print(f"\n📊 Final Dataset Split:")
    print(f"  Training samples: {len(train_trajectories)}")
    print(f"  Test samples: {len(test_trajectories)}")

    # Load text manifest with INTENT captions + embeddings
    scene_text_map = load_text_manifest(args.text_manifest_path)
    if scene_text_map is None:
        raise RuntimeError(
            "Text manifest is required for instruction-tuned SEMDT experiment"
        )

    # Build datasets / loaders
    train_dataset = MovementSemanticDataset(
        train_trajectories,
        scene_text_map,
        context_length=args.context_length,
        prediction_horizon=args.prediction_horizon,
    )
    test_dataset = MovementSemanticDataset(
        test_trajectories,
        scene_text_map,
        context_length=args.context_length,
        prediction_horizon=args.prediction_horizon,
    )

    # ------------------------------------------------------------------
    # Optional lateral maneuver oversampling (matches DT best-run behavior)
    # ------------------------------------------------------------------
    sampler = None
    if args.enable_oversampling:
        print("\n🎯 Computing lateral maneuver oversampling...")
        target_fractions = [
            args.target_straight_frac,
            args.target_mild_frac,
            args.target_strong_frac,
        ]
        _, _, sampling_weights, bin_stats = compute_lateral_scores_and_weights(
            train_trajectories,
            args.context_length,
            target_fractions=target_fractions,
            temperature=args.oversampling_temperature,
        )
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.as_tensor(sampling_weights, dtype=torch.double),
            num_samples=len(sampling_weights),
            replacement=True,
        )
        print(f"   Oversampling bin counts: {bin_stats.get('bin_counts', None)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate_movement_semantic,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_movement_semantic,
    )

    # ------------------------------------------------------------------
    # Model: Qwen + LoRA + instruction tuning + SEMDT semantic stream
    # ------------------------------------------------------------------
    state_dim = train_dataset.state_dim
    act_dim = 3  # normalized movements (x, y, z/heading)

    print("\n🏗️  MODEL CONFIGURATION")
    print(f"  State dim: {state_dim}")
    print(f"  Act dim: {act_dim}")
    print(f"  Context length K: {args.context_length}")
    print(f"  Prediction horizon H: {args.prediction_horizon}")
    print(f"  Base Qwen model: {args.llm_model_name}")

    use_lora = not args.no_lora
    print(f"  LoRA enabled: {use_lora}")
    model = QwenInstructionTunedTrajectoryLLM(
        state_dim=state_dim,
        act_dim=act_dim,
        context_length=args.context_length,
        prediction_horizon=args.prediction_horizon,
        llm_model_name=args.llm_model_name,
        use_lora=use_lora,
        lora_r=16,
        lora_alpha=32,
        text_embedding_dim=512,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # ------------------------------------------------------------------
    # Training loop (trajectory loss + λ * text InfoNCE)
    # ------------------------------------------------------------------
    lambda_text = args.text_loss_weight  # we expect this to be ~0.1
    print("\n🚀 STARTING TRAINING (SEMDT-style text scaling/cap + optional oversampling)")

    for epoch in range(args.num_epochs):
        model.train()
        total_traj_loss = 0.0
        total_text_loss = 0.0
        total_batches = 0

        # Optional warmup for the text loss weight
        if args.text_warmup_epochs and args.text_warmup_epochs > 0:
            current_lambda = lambda_text * min((epoch + 1) / args.text_warmup_epochs, 1.0)
        else:
            current_lambda = lambda_text

        for batch in train_loader:
            states = batch["states"].to(device)  # (B, K, state_dim)
            actions = batch["actions"].to(device)  # (B, K, 3)
            returns_to_go = batch["returns_to_go"].to(device)  # (B, K, 1)
            timesteps = batch["timesteps"].to(device)  # (B, K)
            future_movements = batch["future_movements"].to(device)  # (B, H, 3)
            instructions = batch["instructions"]
            text_embs = batch["text_embs"].to(device)  # (B, 512)

            optimizer.zero_grad()

            # Instruction-tuned Qwen forward
            action_preds, agent_emb = model(
                states,
                actions,
                returns_to_go,
                timesteps,
                instructions,
                return_agent_emb=True,
            )

            # Trajectory loss (movement-normalized regression)
            traj_loss = F.mse_loss(action_preds, future_movements)

            # Semantic contrastive loss (InfoNCE) using shared projection
            agent_z, text_z = model.project_for_contrastive(agent_emb, text_embs)
            text_loss = info_nce_loss(agent_z, text_z)

            # SEMDT-style scaling/capping so the text term doesn't dominate
            text_loss_scaled = text_loss * args.infonce_scale_factor
            if args.text_loss_cap is not None:
                text_loss_scaled = torch.clamp(text_loss_scaled, max=args.text_loss_cap)

            loss = (1.0 - current_lambda) * traj_loss + current_lambda * text_loss_scaled

            loss.backward()
            optimizer.step()

            total_traj_loss += traj_loss.item()
            total_text_loss += text_loss_scaled.item()
            total_batches += 1

        avg_traj_loss = total_traj_loss / max(total_batches, 1)
        avg_text_loss = total_text_loss / max(total_batches, 1)

        print(
            f"Epoch {epoch+1}/{args.num_epochs} "
            f"- L_traj: {avg_traj_loss:.4f}, L_text: {avg_text_loss:.4f}"
        )

        # --------------------------------------------------------------
        # Simple evaluation on test set using ADE/FDE-style metrics
        # --------------------------------------------------------------
        model.eval()
        all_metrics = []
        with torch.no_grad():
            for batch in test_loader:
                states = batch["states"].to(device)
                actions = batch["actions"].to(device)
                returns_to_go = batch["returns_to_go"].to(device)
                timesteps = batch["timesteps"].to(device)
                future_movements = batch["future_movements"].to(device)
                instructions = batch["instructions"]

                action_preds, _ = model(
                    states,
                    actions,
                    returns_to_go,
                    timesteps,
                    instructions,
                    return_agent_emb=False,
                )  # (B, H, 3)

                # Build fake mask for H future steps only
                B, H, _ = action_preds.shape
                mask = torch.ones((B, H), dtype=torch.long, device=device)

                # Use movement-normalized metric helper to convert to ADE/FDE in meters
                metrics = calculate_movement_metrics(
                    predictions=action_preds,
                    targets=future_movements,
                    movement_stats=movement_stats,
                    mask=mask,
                )
                all_metrics.append(metrics)

        if all_metrics:
            avg_metrics = {
                key: float(np.mean([m[key] for m in all_metrics]))
                for key in all_metrics[0].keys()
            }
        else:
            avg_metrics = {}

        if avg_metrics:
            print(
                f"  Eval: ADE={avg_metrics['Real_ADE_meters']:.2f}m, "
                f"FDE={avg_metrics['Real_FDE_meters']:.2f}m, "
                f"MR@5m={avg_metrics['Miss_Rate_5m']*100:.1f}%"
            )

    # ------------------------------------------------------------------
    # Save final model
    # ------------------------------------------------------------------
    out_dir = Path("checkpoints")
    out_dir.mkdir(exist_ok=True)
    ckpt_path = out_dir / "qwen_instruction_lora_semdt_final.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "movement_stats": movement_stats,
            "args": vars(args),
        },
        ckpt_path,
    )
    print(f"\n💾 Final instruction-tuned Qwen+LoRA SEMDT model saved to: {ckpt_path}")


if __name__ == "__main__":
    main()

