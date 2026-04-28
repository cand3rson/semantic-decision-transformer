#!/usr/bin/env python3
"""
Qwen + LoRA Trajectory Predictor
=================================

NEW additive file — does NOT modify or replace any existing files:
  - qwen_trajectory_model.py          (unchanged — trains Qwen2 from scratch)
  - trajectory_experiment_qwen_vlm.py (unchanged)
  - trajectory_experiment_visiontrap.py (unchanged)

Key difference from qwen_trajectory_model.py:
  - qwen_trajectory_model.py   → Qwen2 ARCHITECTURE with RANDOM weights (from-scratch)
  - THIS file                  → Qwen2-1.5B-Instruct PRE-TRAINED weights + LoRA adapters

This creates the "fair LLM benchmark" the professor asked for:
  - The LLM actually brings pre-trained language/reasoning knowledge
  - LoRA keeps the vast majority of those weights frozen (parameter-efficient)
  - Only a small set of low-rank adapters are trained on trajectory data
  - The VisionTRAP semantic stream (InfoNCE loss) provides instruction-style grounding

Interface:  QwenLoRATrajectoryPredictor is a drop-in replacement for
            QwenTrajectoryPredictor within the VisionTRAP pipeline.
"""

import torch
import torch.nn as nn
from transformers import Qwen2Model
from peft import LoraConfig, get_peft_model, TaskType

from decision_transformer.models.model import TrajectoryModel

# Pre-trained model to load (same model used in qwen_instruction_lora_model.py)
PRETRAINED_MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"


class QwenLoRATrajectoryPredictor(TrajectoryModel):
    """
    Qwen2-1.5B-Instruct + LoRA trajectory predictor with Decision Transformer interface.

    Uses the SAME forward pass structure as QwenTrajectoryPredictor:
      RTG tokens | State tokens | Action tokens  →  Qwen2 backbone  →  prediction heads

    The difference is the backbone:
      - QwenTrajectoryPredictor:     Qwen2 architecture, trained from SCRATCH
      - QwenLoRATrajectoryPredictor: Qwen2-1.5B-Instruct PRE-TRAINED + LoRA fine-tuning

    Parameters
    ----------
    state_dim, act_dim     : passed from VisionTRAP pipeline (data-derived)
    hidden_size            : IGNORED — overridden by the pre-trained model's hidden_size (1536)
    max_length, max_ep_len : context window for the DT sequence
    action_tanh            : whether to apply Tanh to action predictions
    n_layer, n_head        : IGNORED — pre-trained model architecture is fixed
    lora_r                 : LoRA rank  (default 16)
    lora_alpha             : LoRA scaling (default 32)
    lora_dropout           : LoRA dropout (default 0.05)
    **kwargs               : absorbs GPT-specific kwargs from visiontrap (n_inner, n_ctx, etc.)
    """

    def __init__(
        self,
        state_dim,
        act_dim,
        hidden_size,        # overridden by pretrained config — kept for interface compat
        max_length=50,
        max_ep_len=50,
        action_tanh=True,
        n_layer=3,          # overridden by pretrained config — kept for interface compat
        n_head=4,           # overridden by pretrained config — kept for interface compat
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        **kwargs,
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        # ------------------------------------------------------------------
        # 1. Load pre-trained Qwen2 base model (transformer only, no LM head)
        # ------------------------------------------------------------------
        print(f"  [QwenLoRA] Loading pre-trained base model: {PRETRAINED_MODEL_NAME}")
        base_transformer = Qwen2Model.from_pretrained(PRETRAINED_MODEL_NAME)

        # ------------------------------------------------------------------
        # 2. Apply LoRA adapters to attention + MLP projection layers
        #    All other parameters are FROZEN — only LoRA adapters are trained
        # ------------------------------------------------------------------
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.transformer = get_peft_model(base_transformer, lora_config)
        self.transformer.print_trainable_parameters()

        # Use the actual pre-trained hidden size (1536 for Qwen2-1.5B)
        self.hidden_size = self.transformer.config.hidden_size

        # ------------------------------------------------------------------
        # 3. Decision Transformer embedding layers
        #    Sized to pre-trained hidden_size — trained from scratch on traj data
        # ------------------------------------------------------------------
        self.embed_timestep = nn.Embedding(max_ep_len, self.hidden_size)
        self.embed_return = nn.Linear(1, self.hidden_size)
        self.embed_state = nn.Linear(self.state_dim, self.hidden_size)
        self.embed_action = nn.Linear(self.act_dim, self.hidden_size)
        self.embed_ln = nn.LayerNorm(self.hidden_size)

        # ------------------------------------------------------------------
        # 4. Prediction heads (same structure as QwenTrajectoryPredictor)
        # ------------------------------------------------------------------
        self.predict_state = nn.Linear(self.hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(self.hidden_size, self.act_dim)]
              + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = nn.Linear(self.hidden_size, 1)

    # ------------------------------------------------------------------
    # Forward — identical logic to QwenTrajectoryPredictor
    # ------------------------------------------------------------------
    def forward(
        self,
        states,
        actions,
        rewards,
        returns_to_go,
        timesteps,
        attention_mask=None,
        return_agent_emb=False,
    ):
        """
        Forward pass — same interface as QwenTrajectoryPredictor / DecisionTransformer.

        Args:
            states:        (B, T, state_dim)
            actions:       (B, T, act_dim)
            rewards:       (B, T, 1)  — unused (follows DT interface)
            returns_to_go: (B, T, 1)
            timesteps:     (B, T)
            attention_mask:(B, T)  optional
            return_agent_emb: bool — if True, also return agent embedding for VLM

        Returns:
            state_preds:   (B, T, state_dim)
            action_preds:  (B, T, act_dim)
            return_preds:  (B, T, 1)
            agent_emb:     (B, hidden_size)  — only when return_agent_emb=True
        """
        device = states.device
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.long, device=device
            )

        # Embed each modality and add timestep positional encoding
        state_embeddings = self.embed_state(states) + self.embed_timestep(timesteps)
        action_embeddings = self.embed_action(actions) + self.embed_timestep(timesteps)
        returns_embeddings = (
            self.embed_return(returns_to_go) + self.embed_timestep(timesteps)
        )

        # Interleave: [RTG_1, S_1, A_1, RTG_2, S_2, A_2, ...]
        stacked_inputs = (
            torch.stack(
                (returns_embeddings, state_embeddings, action_embeddings), dim=1
            )
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_length, self.hidden_size)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        stacked_attention_mask = (
            torch.stack(
                (attention_mask, attention_mask, attention_mask), dim=1
            )
            .permute(0, 2, 1)
            .reshape(batch_size, 3 * seq_length)
        )

        # Pass through LoRA-adapted Qwen2 backbone
        # Qwen2Model (base) returns BaseModelOutputWithPast → has .last_hidden_state
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            use_cache=False,
        )
        x = transformer_outputs.last_hidden_state  # (B, 3*T, hidden)

        # Reshape back to (B, T, 3, hidden) then permute → (B, 3, T, hidden)
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # Predictions from each token type
        return_preds = self.predict_return(x[:, 2])  # from action tokens
        state_preds = self.predict_state(x[:, 2])    # from action tokens
        action_preds = self.predict_action(x[:, 1])  # from state tokens

        if return_agent_emb:
            # Agent embedding for InfoNCE contrastive loss (last state token)
            agent_emb = x[:, 1, -1, :]  # (B, hidden_size)
            return state_preds, action_preds, return_preds, agent_emb

        return state_preds, action_preds, return_preds

    # ------------------------------------------------------------------
    # get_action — used during evaluation (identical to QwenTrajectoryPredictor)
    # ------------------------------------------------------------------
    def get_action(
        self, states, actions, rewards, returns_to_go, timesteps, **kwargs
    ):
        """Get single action prediction for evaluation — same as base version."""
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:, -self.max_length :]
            actions = actions[:, -self.max_length :]
            returns_to_go = returns_to_go[:, -self.max_length :]
            timesteps = timesteps[:, -self.max_length :]

            pad = self.max_length - states.shape[1]
            attention_mask = torch.cat(
                [torch.zeros(pad), torch.ones(states.shape[1])]
            ).to(dtype=torch.long, device=states.device).reshape(1, -1)

            states = torch.cat(
                [torch.zeros((1, pad, self.state_dim), device=states.device), states],
                dim=1,
            ).float()
            actions = torch.cat(
                [torch.zeros((1, pad, self.act_dim), device=actions.device), actions],
                dim=1,
            ).float()
            returns_to_go = torch.cat(
                [
                    torch.zeros((1, pad, 1), device=returns_to_go.device),
                    returns_to_go,
                ],
                dim=1,
            ).float()
            timesteps = torch.cat(
                [torch.zeros((1, pad), device=timesteps.device), timesteps], dim=1
            ).long()
        else:
            attention_mask = None

        _, action_preds, _ = self.forward(
            states,
            actions,
            None,
            returns_to_go,
            timesteps,
            attention_mask=attention_mask,
            **kwargs,
        )
        return action_preds[0, -1]
