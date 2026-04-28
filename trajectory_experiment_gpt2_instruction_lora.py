#!/usr/bin/env python3
"""
GPT-2 + LoRA + Instruction-Tuned SEMDT Trajectory Experiment
=============================================================

Direct GPT-2 equivalent of trajectory_experiment_qwen_instruction_lora.py.

Uses the identical pipeline, dataset, training loop, losses, and hyperparameters
as the best Qwen Instruction Tuning + LoRA run (ADE=1.99m, FDE=3.02m, MR=11.1%),
but with the GPT-2 base backbone (124M params, hidden=768) instead of Qwen2-1.5B.

This DOES NOT modify or replace any existing experiment files.

Comparison context:
    SemDT (DT backbone, best):          ADE=1.67m, FDE=3.27m,  MR=12.8%
    Instruction Tuning + Qwen LoRA:     ADE=1.99m, FDE=3.02m,  MR=11.1%
    GPT-2 + LoRA only (VisionTRAP):     ADE=2.24m, FDE=4.20m,  MR=22.2%
    GPT-2 from-scratch (SemDT):         ADE=2.05m, FDE=4.44m,  MR=25.5%
    Qwen LoRA-only (VisionTRAP):        ADE=2.59m, FDE=5.42m,  MR=22.2%
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

from gpt2_instruction_lora_model import GPT2InstructionTunedTrajectoryLLM

# Reuse movement-normalized pipeline + text manifest loader (unchanged)
from trajectory_experiment_visiontrap import (
    load_nuscenes_scenes,
    compute_movement_statistics,
    process_nuscenes_movement_normalized,
    load_text_manifest,
    compute_lateral_scores_and_weights,
    calculate_movement_metrics,
)


# --------------------------------------------------------------------------- #
# Shared helpers (identical to Qwen instruction experiment)                    #
# --------------------------------------------------------------------------- #

def build_instruction_from_intent(intent_text: str, horizon: int) -> str:
    """Wrap raw INTENT caption into an instruction prompt."""
    return (
        "You are an autonomous driving planner operating in urban traffic. "
        f"The current maneuver intent is: {intent_text}. "
        f"Based on the following kinematic history, predict the next {horizon} "
        "steps of the ego vehicle trajectory as physically plausible movements."
    )


def match_text_for_sample(
    scene_text_map: Dict[str, List[Dict]],
    scene_id: str,
    target_frame: int,
) -> Tuple[str, np.ndarray]:
    """Find the closest text entry for a given (scene_id, frame_idx)."""
    entries = scene_text_map.get(scene_id, [])
    if not entries:
        return (
            "The ego vehicle continues in traffic with cautious, "
            "intent-aware urban driving behavior.",
            np.zeros(512, dtype=np.float32),
        )
    best = min(entries, key=lambda e: abs(int(e.get("frame_idx", 0)) - int(target_frame)))
    text = best.get("text", "") or best.get("texts", "")
    text_emb = np.array(best.get("text_emb", np.zeros(512)), dtype=np.float32)
    if text == "":
        text = (
            "The ego vehicle continues in traffic with cautious, "
            "intent-aware urban driving behavior."
        )
    return text, text_emb


class MovementSemanticDataset(Dataset):
    """Movement-normalized trajectory dataset with INTENT caption lookup."""

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
        if len(trajectories) == 0:
            raise ValueError("Empty trajectory list passed to MovementSemanticDataset")
        self.state_dim = trajectories[0]["states"].shape[1]

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.trajectories[idx]
        states = sample["states"]
        actions = sample["actions"]
        rtg = sample["rtg"]
        timesteps = sample["timesteps"]
        future_movements = sample["future_movements"]
        scene_id = sample["scene_id"]
        start_frame = sample["start_frame"]

        target_frame = int(start_frame + self.context_length - 1)
        intent_text, text_emb = match_text_for_sample(
            self.scene_text_map, scene_id, target_frame
        )
        instruction = build_instruction_from_intent(intent_text, self.prediction_horizon)

        return {
            "states": torch.from_numpy(states).float(),
            "actions": torch.from_numpy(actions).float(),
            "returns_to_go": torch.from_numpy(rtg[: self.context_length]).float(),
            "timesteps": torch.from_numpy(timesteps).long(),
            "future_movements": torch.from_numpy(future_movements).float(),
            "instruction": instruction,
            "text_emb": torch.from_numpy(text_emb).float(),
        }


def collate_movement_semantic(batch: List[Dict]) -> Dict:
    return {
        "states": torch.stack([b["states"] for b in batch], dim=0),
        "actions": torch.stack([b["actions"] for b in batch], dim=0),
        "returns_to_go": torch.stack([b["returns_to_go"] for b in batch], dim=0),
        "timesteps": torch.stack([b["timesteps"] for b in batch], dim=0),
        "future_movements": torch.stack([b["future_movements"] for b in batch], dim=0),
        "instructions": [b["instruction"] for b in batch],
        "text_embs": torch.stack([b["text_emb"] for b in batch], dim=0),
    }


def filter_trajectories_by_rtg_quality(
    trajectories: List[Dict],
    context_length: int,
    keep_top_fraction: float,
    description: str,
) -> List[Dict]:
    """Drop the lowest-quality samples by mean RTG (higher RTG = smoother trajectory)."""
    if keep_top_fraction is None or keep_top_fraction >= 1.0:
        return trajectories
    if keep_top_fraction <= 0.0:
        raise ValueError("keep_top_fraction must be > 0.0")
    if len(trajectories) == 0:
        return trajectories

    scores = []
    for t in trajectories:
        rtg = t.get("rtg", None)
        scores.append(float(np.asarray(rtg)[:context_length].mean()) if rtg is not None else -1e9)
    scores_np = np.asarray(scores, dtype=np.float32)

    threshold = float(np.quantile(scores_np, 1.0 - keep_top_fraction))
    kept = [t for t, m in zip(trajectories, (scores_np >= threshold).tolist()) if m]
    if len(kept) == 0:
        kept = [trajectories[int(np.argmax(scores_np))]]

    print(
        f"\nRTG quality filtering ({description}): keep_top_fraction={keep_top_fraction:.3f}, "
        f"kept {len(kept)}/{len(trajectories)} samples, threshold={threshold:.6f}"
    )
    return kept


def info_nce_loss(agent_z: torch.Tensor, text_z: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    logits = agent_z @ text_z.t() / temperature
    labels = torch.arange(agent_z.size(0), device=agent_z.device)
    return F.cross_entropy(logits, labels)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="GPT-2 + LoRA + Instruction-Tuned SEMDT Trajectory Experiment"
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
    parser.add_argument("--text_loss_weight", type=float, default=0.1)
    parser.add_argument("--text_warmup_epochs", type=int, default=5)
    parser.add_argument("--text_loss_cap", type=float, default=0.2)
    parser.add_argument("--infonce_scale_factor", type=float, default=0.001)
    parser.add_argument("--enable_oversampling", action="store_true")
    parser.add_argument("--oversampling_temperature", type=float, default=0.75)
    parser.add_argument("--target_straight_frac", type=float, default=0.25)
    parser.add_argument("--target_mild_frac", type=float, default=0.35)
    parser.add_argument("--target_strong_frac", type=float, default=0.40)
    parser.add_argument("--max_movement_filter", type=float, default=50.0)
    parser.add_argument("--movement_variance_filter", type=float, default=100.0)
    parser.add_argument(
        "--rtg_keep_top_fraction", type=float, default=1.0,
        help="Drop low-RTG training samples (1.0 = keep all)",
    )
    parser.add_argument(
        "--gpt2_model_name", type=str, default="gpt2",
        help="Pre-trained GPT-2 checkpoint to use",
    )
    parser.add_argument(
        "--no_lora", action="store_true",
        help="Disable LoRA — fine-tune ALL GPT-2 parameters instead.",
    )
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("GPT-2 + LoRA + INSTRUCTION-TUNED SEMDT TRAJECTORY EXPERIMENT")
    print("=" * 80)
    print(f"Device:             {device}")
    print(f"Backbone:           {args.gpt2_model_name} (pre-trained GPT-2, hidden=768)")
    print(f"LoRA enabled:       {not args.no_lora}")
    print(f"Dataset:            {args.dataset_path}")
    print(f"Context length K:   {args.context_length}")
    print(f"Prediction H:       {args.prediction_horizon}")
    print(f"λ (text_loss):      {args.text_loss_weight}")
    print()
    print("Comparison baselines:")
    print("  SemDT (DT, best):              ADE=1.67m, FDE=3.27m, MR=12.8%")
    print("  Instruction Tuning + Qwen LoRA: ADE=1.99m, FDE=3.02m, MR=11.1%")
    print("  GPT-2 + LoRA only:             ADE=2.24m, FDE=4.20m, MR=22.2%")
    print("  GPT-2 from-scratch SemDT:      ADE=2.05m, FDE=4.44m, MR=25.5%")
    print("  Qwen LoRA-only:                ADE=2.59m, FDE=5.42m, MR=22.2%")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Data loading & preprocessing (identical pipeline as Qwen run)
    # ------------------------------------------------------------------
    print("\nPHASE 1: Loading raw scenes...")
    ego_trajectories = load_nuscenes_scenes(args.dataset_path, max_files=args.max_files)
    if len(ego_trajectories) == 0:
        raise RuntimeError("No scenes loaded from dataset_path")

    print("\nPHASE 2: Splitting scenes into train/test...")
    random.seed(42)
    np.random.seed(42)
    scene_indices = list(range(len(ego_trajectories)))
    random.shuffle(scene_indices)
    num_train_scenes = int(0.8 * len(ego_trajectories))
    train_ego = [ego_trajectories[i] for i in scene_indices[:num_train_scenes]]
    test_ego  = [ego_trajectories[i] for i in scene_indices[num_train_scenes:]]
    print(f"  Training scenes: {len(train_ego)}, Test scenes: {len(test_ego)}")

    print("\nPHASE 3: Computing movement normalization statistics...")
    movement_stats = compute_movement_statistics(train_ego)

    print("\nPHASE 4: Creating normalized samples...")
    train_trajectories = process_nuscenes_movement_normalized(
        train_ego, movement_stats, args,
        context_length=args.context_length,
        prediction_horizon=args.prediction_horizon,
    )
    test_trajectories = process_nuscenes_movement_normalized(
        test_ego, movement_stats, args,
        context_length=args.context_length,
        prediction_horizon=args.prediction_horizon,
    )

    train_trajectories = filter_trajectories_by_rtg_quality(
        train_trajectories, args.context_length, args.rtg_keep_top_fraction, "TRAIN"
    )

    print(f"\nFinal Dataset Split:")
    print(f"  Training samples: {len(train_trajectories)}")
    print(f"  Test samples:     {len(test_trajectories)}")

    scene_text_map = load_text_manifest(args.text_manifest_path)
    if scene_text_map is None:
        raise RuntimeError("Text manifest is required for instruction-tuned experiment")

    train_dataset = MovementSemanticDataset(
        train_trajectories, scene_text_map,
        context_length=args.context_length, prediction_horizon=args.prediction_horizon,
    )
    test_dataset = MovementSemanticDataset(
        test_trajectories, scene_text_map,
        context_length=args.context_length, prediction_horizon=args.prediction_horizon,
    )

    # ------------------------------------------------------------------
    # Optional lateral maneuver oversampling (matches DT best-run)
    # ------------------------------------------------------------------
    sampler = None
    if args.enable_oversampling:
        print("\nComputing lateral maneuver oversampling...")
        _, _, sampling_weights, bin_stats = compute_lateral_scores_and_weights(
            train_trajectories, args.context_length,
            target_fractions=[args.target_straight_frac, args.target_mild_frac, args.target_strong_frac],
            temperature=args.oversampling_temperature,
        )
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.as_tensor(sampling_weights, dtype=torch.double),
            num_samples=len(sampling_weights),
            replacement=True,
        )
        print(f"  Oversampling bin counts: {bin_stats.get('bin_counts', None)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(sampler is None), sampler=sampler,
        collate_fn=collate_movement_semantic,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_movement_semantic,
    )

    # ------------------------------------------------------------------
    # Model: GPT-2 + LoRA + instruction tuning + SEMDT semantic stream
    # ------------------------------------------------------------------
    state_dim = train_dataset.state_dim
    act_dim = 3

    print(f"\nMODEL CONFIGURATION")
    print(f"  State dim: {state_dim}, Act dim: {act_dim}")
    print(f"  Context K: {args.context_length}, Horizon H: {args.prediction_horizon}")
    print(f"  GPT-2 model: {args.gpt2_model_name}")
    print(f"  LoRA enabled: {not args.no_lora}")

    model = GPT2InstructionTunedTrajectoryLLM(
        state_dim=state_dim,
        act_dim=act_dim,
        context_length=args.context_length,
        prediction_horizon=args.prediction_horizon,
        gpt2_model_name=args.gpt2_model_name,
        use_lora=not args.no_lora,
        lora_r=16,
        lora_alpha=32,
        text_embedding_dim=512,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    # ------------------------------------------------------------------
    # Training loop (trajectory MSE + λ * InfoNCE — identical to Qwen run)
    # ------------------------------------------------------------------
    lambda_text = args.text_loss_weight
    print(f"\nSTARTING TRAINING — {args.num_epochs} epochs")

    for epoch in range(args.num_epochs):
        model.train()
        total_traj_loss = 0.0
        total_text_loss = 0.0
        total_batches = 0

        if args.text_warmup_epochs and args.text_warmup_epochs > 0:
            current_lambda = lambda_text * min((epoch + 1) / args.text_warmup_epochs, 1.0)
        else:
            current_lambda = lambda_text

        for batch in train_loader:
            states        = batch["states"].to(device)
            actions       = batch["actions"].to(device)
            returns_to_go = batch["returns_to_go"].to(device)
            timesteps     = batch["timesteps"].to(device)
            future_movements = batch["future_movements"].to(device)
            instructions  = batch["instructions"]
            text_embs     = batch["text_embs"].to(device)

            optimizer.zero_grad()

            action_preds, agent_emb = model(
                states, actions, returns_to_go, timesteps, instructions,
                return_agent_emb=True,
            )

            traj_loss = F.mse_loss(action_preds, future_movements)

            agent_z, text_z = model.project_for_contrastive(agent_emb, text_embs)
            text_loss = info_nce_loss(agent_z, text_z)
            text_loss_scaled = text_loss * args.infonce_scale_factor
            if args.text_loss_cap is not None:
                text_loss_scaled = torch.clamp(text_loss_scaled, max=args.text_loss_cap)

            loss = (1.0 - current_lambda) * traj_loss + current_lambda * text_loss_scaled
            loss.backward()
            optimizer.step()

            total_traj_loss += traj_loss.item()
            total_text_loss += text_loss_scaled.item()
            total_batches += 1

        avg_traj = total_traj_loss / max(total_batches, 1)
        avg_text = total_text_loss / max(total_batches, 1)
        print(
            f"Epoch {epoch+1}/{args.num_epochs} "
            f"- L_traj: {avg_traj:.4f}, L_text: {avg_text:.4f}"
        )

        # Evaluation
        model.eval()
        all_metrics = []
        with torch.no_grad():
            for batch in test_loader:
                states        = batch["states"].to(device)
                actions       = batch["actions"].to(device)
                returns_to_go = batch["returns_to_go"].to(device)
                timesteps     = batch["timesteps"].to(device)
                future_movements = batch["future_movements"].to(device)
                instructions  = batch["instructions"]

                action_preds, _ = model(
                    states, actions, returns_to_go, timesteps, instructions,
                    return_agent_emb=False,
                )

                B, H, _ = action_preds.shape
                mask = torch.ones((B, H), dtype=torch.long, device=device)
                metrics = calculate_movement_metrics(
                    predictions=action_preds,
                    targets=future_movements,
                    movement_stats=movement_stats,
                    mask=mask,
                )
                all_metrics.append(metrics)

        if all_metrics:
            avg_metrics = {
                k: float(np.mean([m[k] for m in all_metrics]))
                for k in all_metrics[0].keys()
            }
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
    ckpt_path = out_dir / "gpt2_instruction_lora_semdt_final.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "movement_stats": movement_stats,
            "args": vars(args),
        },
        ckpt_path,
    )
    print(f"\nFinal GPT-2 instruction+LoRA SEMDT model saved to: {ckpt_path}")


if __name__ == "__main__":
    main()
