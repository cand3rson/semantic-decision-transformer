#!/usr/bin/env python3
"""
Qwen Instruction-Tuned Trajectory LLM with LoRA
===============================================

This module defines a Qwen-based instruction-tuned trajectory model that:
- Uses LoRA for parameter-efficient fine-tuning
- Consumes both natural language instructions (INTENT captions) and
  continuous kinematic state sequences
- Predicts a horizon of future movement actions
- Exposes an agent embedding suitable for semantic contrastive alignment

This DOES NOT replace any existing models; it is an additive baseline
for fair comparison.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class QwenInstructionTunedTrajectoryLLM(nn.Module):
    """
    Instruction-tuned Qwen trajectory model with LoRA.

    Inputs:
        - states: (batch, K, state_dim) continuous kinematic history
        - instruction_texts: list[str], INTENT-aware instructions

    Outputs:
        - action_preds: (batch, H, act_dim) future normalized movements
        - agent_emb (optional): (batch, hidden) for semantic contrastive loss

    Key design:
        - Text prompt + "soft kinematic tokens" are concatenated and passed
          through Qwen
        - Last kinematic token representation is decoded into H-step actions
        - LoRA adapters fine-tune only a small subset of Qwen parameters
    """

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        context_length: int,
        prediction_horizon: int,
        llm_model_name: str = "Qwen/Qwen2-1.5B-Instruct",
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        text_embedding_dim: int = 512,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.context_length = context_length
        self.prediction_horizon = prediction_horizon
        self.text_embedding_dim = text_embedding_dim

        # Lazy-import heavy deps so importing this module is cheap
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model

        # Tokenizer and base Qwen model
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

        # Load base model WITHOUT 8-bit quantization to avoid bitsandbytes dependency
        # (LoRA still provides parameter-efficient fine-tuning)
        base_model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
        )

        if use_lora:
            # Attach LoRA adapters to attention + MLP projections
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
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llm = get_peft_model(base_model, lora_config)
        else:
            self.llm = base_model

        self.llm_hidden_size = self.llm.config.hidden_size

        # Encode each timestep state vector into Qwen hidden space
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.llm_hidden_size),
        )

        # DT-style token embedding components for context:
        # RTG/state/action are embedded in the same hidden space, with
        # shared timestep embeddings across modalities.
        max_ep_len = max(1, context_length)
        self.embed_timestep = nn.Embedding(max_ep_len, self.llm_hidden_size)
        self.embed_return = nn.Linear(1, self.llm_hidden_size)
        self.embed_action = nn.Linear(self.act_dim, self.llm_hidden_size)
        self.embed_ln = nn.LayerNorm(self.llm_hidden_size)

        # Decode single latent into H-step action sequence
        self.action_decoder = nn.Sequential(
            nn.Linear(self.llm_hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, prediction_horizon * act_dim),
        )

        # Projections for semantic contrastive alignment (agent ↔ text)
        proj_dim = 256
        self.agent_proj = nn.Linear(self.llm_hidden_size, proj_dim)
        self.text_proj = nn.Linear(self.text_embedding_dim, proj_dim)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        instruction_texts: List[str],
        attention_mask: Optional[torch.Tensor] = None,
        return_agent_emb: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with DT-style interleaving and instruction prefix conditioning.

        The Qwen backbone sees a concatenation of:
          [instruction tokens] + [RTG_1, state_1, action_1, ..., RTG_K, state_K, action_K]

        The last state token in the DT interleaving is decoded into an H-step action sequence.
        """
        device = states.device
        batch_size, K, state_dim = states.shape
        assert K == self.context_length, f"Expected context_length={self.context_length}, got {K}"
        assert state_dim == self.state_dim, f"Expected state_dim={self.state_dim}, got {state_dim}"
        assert actions.shape == (batch_size, K, self.act_dim), f"Expected actions shape (B,K,{self.act_dim})"
        assert returns_to_go.shape == (batch_size, K, 1), "Expected returns_to_go shape (B,K,1)"
        assert timesteps.shape == (batch_size, K), f"Expected timesteps shape (B,K), got {timesteps.shape}"

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, K), dtype=torch.long, device=device)

        # Encode states per timestep into Qwen hidden dimension + timestep embedding
        flat_states = states.view(batch_size * K, self.state_dim)
        flat_state_embeddings = self.state_encoder(flat_states)  # (B*K, hidden)
        state_embeddings = flat_state_embeddings.view(batch_size, K, self.llm_hidden_size)
        state_embeddings = state_embeddings + self.embed_timestep(timesteps)

        # DT-style embeddings for actions and RTG
        action_embeddings = self.embed_action(actions) + self.embed_timestep(timesteps)
        returns_embeddings = self.embed_return(returns_to_go) + self.embed_timestep(timesteps)

        # Interleave: [RTG_t, state_t, action_t]
        stacked_inputs = (
            torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * K, self.llm_hidden_size)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        # Tokenize instructions
        tokenized = self.tokenizer(
            instruction_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        # Get instruction embeddings from Qwen token embedding layer
        input_embeds_module = self.llm.get_input_embeddings()
        instruction_embeddings = input_embeds_module(tokenized["input_ids"])  # (B, L, hidden)

        # Respect tokenizer padding in attention mask.
        instruction_attention_mask = tokenized.get("attention_mask", None)
        if instruction_attention_mask is None:
            instruction_attention_mask = torch.ones(
                tokenized["input_ids"].shape, dtype=torch.long, device=device
            )

        # Stack attention mask for [RTG,state,action] tokens.
        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 3 * K)
        )

        combined_embeddings = torch.cat([instruction_embeddings, stacked_inputs], dim=1)
        combined_attention_mask = torch.cat(
            [instruction_attention_mask, stacked_attention_mask.to(instruction_attention_mask.dtype)],
            dim=1,
        )

        llm_outputs = self.llm(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention_mask,
            output_hidden_states=True,
        )

        last_hidden = llm_outputs.hidden_states[-1]  # (B, L+3K, hidden)

        # Extract DT-interleaved portion and take the last state token as latent.
        dt_hidden = last_hidden[:, -3 * K :, :]  # (B, 3K, hidden)
        x = dt_hidden.reshape(batch_size, K, 3, self.llm_hidden_size).permute(0, 2, 1, 3)  # (B,3,K,hidden)
        agent_emb = x[:, 1, -1, :]  # (B, hidden) - state token at last context timestep

        # Decode to H-step action sequence
        action_flat = self.action_decoder(agent_emb)  # (B, H * act_dim)
        action_preds = action_flat.view(batch_size, self.prediction_horizon, self.act_dim)  # (B, H, act_dim)

        if return_agent_emb:
            return action_preds, agent_emb
        return action_preds, None

    def project_for_contrastive(
        self, agent_emb: torch.Tensor, text_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project agent and text embeddings into a shared space for InfoNCE.

        Args:
            agent_emb: (batch, hidden_size)
            text_emb: (batch, text_embedding_dim)

        Returns:
            agent_z, text_z: (batch, proj_dim), L2-normalized
        """
        agent_z = F.normalize(self.agent_proj(agent_emb), dim=-1)
        text_z = F.normalize(self.text_proj(text_emb), dim=-1)
        return agent_z, text_z

