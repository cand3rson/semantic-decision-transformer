#!/usr/bin/env python3
"""
GPT-2 Instruction-Tuned Trajectory LLM with LoRA
=================================================

Direct GPT-2 equivalent of qwen_instruction_lora_model.py.

Architecture is identical in every structural sense:
  - INTENT caption tokenized as an instruction prefix
  - DT-style [RTG, state, action] interleaved kinematic tokens appended after prefix
  - LoRA adapters for parameter-efficient fine-tuning (~1% trainable params)
  - InfoNCE projection heads for semantic contrastive alignment

The only difference is the backbone:
  qwen_instruction_lora_model.py  → Qwen2-1.5B-Instruct (hidden=1536)
  THIS file                       → GPT-2 base pre-trained (hidden=768)

This DOES NOT replace or modify any existing model files.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GPT2InstructionTunedTrajectoryLLM(nn.Module):
    """
    Instruction-tuned GPT-2 trajectory model with LoRA.

    Inputs:
        - states: (batch, K, state_dim) continuous kinematic history
        - actions: (batch, K, act_dim) context actions
        - returns_to_go: (batch, K, 1) DT-style RTG values
        - timesteps: (batch, K) integer timestep ids
        - instruction_texts: list[str], INTENT-aware instructions

    Outputs:
        - action_preds: (batch, H, act_dim) future normalized movements
        - agent_emb (optional): (batch, hidden) for semantic contrastive loss

    The backbone sees:
        [instruction tokens] + [RTG_1, state_1, action_1, ..., RTG_K, state_K, action_K]

    The last state token in the DT interleaving is decoded into H-step actions.
    """

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        context_length: int,
        prediction_horizon: int,
        gpt2_model_name: str = "gpt2",
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

        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        from peft import LoraConfig, get_peft_model

        # GPT-2 tokenizer — has no pad token by default; reuse eos
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # Load pre-trained GPT-2 (CausalLM wrapper so hidden_states are accessible)
        base_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)

        if use_lora:
            # GPT-2 uses Conv1D layers — target c_attn (QKV fused) and c_proj
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["c_attn", "c_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llm = get_peft_model(base_model, lora_config)
            self.llm.print_trainable_parameters()
        else:
            self.llm = base_model

        self.llm_hidden_size = self.llm.config.hidden_size  # 768 for GPT-2 base

        # State encoder: project state_dim → GPT-2 hidden space
        # Intermediate layer scaled to match hidden size (512 for 768-dim GPT-2)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.llm_hidden_size),
        )

        # DT-style token embedding components (same logic as Qwen instruction model)
        max_ep_len = max(1, context_length)
        self.embed_timestep = nn.Embedding(max_ep_len, self.llm_hidden_size)
        self.embed_return = nn.Linear(1, self.llm_hidden_size)
        self.embed_action = nn.Linear(self.act_dim, self.llm_hidden_size)
        self.embed_ln = nn.LayerNorm(self.llm_hidden_size)

        # Action decoder: single decision latent → H-step action sequence
        self.action_decoder = nn.Sequential(
            nn.Linear(self.llm_hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, prediction_horizon * act_dim),
        )

        # Projection heads for InfoNCE semantic contrastive alignment
        proj_dim = 256
        self.agent_proj = nn.Linear(self.llm_hidden_size, proj_dim)
        self.text_proj = nn.Linear(text_embedding_dim, proj_dim)

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

        The GPT-2 backbone sees:
          [instruction tokens] + [RTG_1, state_1, action_1, ..., RTG_K, state_K, action_K]

        The last state token in the DT interleaving is decoded into H-step actions.
        """
        device = states.device
        batch_size, K, state_dim = states.shape
        assert K == self.context_length, f"Expected context_length={self.context_length}, got {K}"
        assert state_dim == self.state_dim, f"Expected state_dim={self.state_dim}, got {state_dim}"
        assert actions.shape == (batch_size, K, self.act_dim), \
            f"Expected actions shape (B,K,{self.act_dim}), got {actions.shape}"
        assert returns_to_go.shape == (batch_size, K, 1), \
            f"Expected returns_to_go shape (B,K,1), got {returns_to_go.shape}"
        assert timesteps.shape == (batch_size, K), \
            f"Expected timesteps shape (B,K), got {timesteps.shape}"

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, K), dtype=torch.long, device=device)

        # ------------------------------------------------------------------ #
        # 1. Build DT-style interleaved kinematic embeddings                  #
        # ------------------------------------------------------------------ #
        flat_states = states.view(batch_size * K, self.state_dim)
        state_embeddings = self.state_encoder(flat_states).view(
            batch_size, K, self.llm_hidden_size
        )
        state_embeddings = state_embeddings + self.embed_timestep(timesteps)

        action_embeddings = self.embed_action(actions) + self.embed_timestep(timesteps)
        returns_embeddings = self.embed_return(returns_to_go) + self.embed_timestep(timesteps)

        # Interleave: [RTG_t, state_t, action_t] for t=1..K → (B, 3K, hidden)
        stacked_inputs = (
            torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * K, self.llm_hidden_size)
        )
        stacked_inputs = self.embed_ln(stacked_inputs)

        # Expand attention mask for the 3K kinematic tokens
        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 3 * K)
        )

        # ------------------------------------------------------------------ #
        # 2. Tokenize instruction prefix and embed through GPT-2 emb layer    #
        # ------------------------------------------------------------------ #
        tokenized = self.tokenizer(
            instruction_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        input_embeds_module = self.llm.get_input_embeddings()
        instruction_embeddings = input_embeds_module(tokenized["input_ids"])  # (B, L, hidden)

        instruction_attention_mask = tokenized.get("attention_mask", None)
        if instruction_attention_mask is None:
            instruction_attention_mask = torch.ones(
                tokenized["input_ids"].shape, dtype=torch.long, device=device
            )

        # ------------------------------------------------------------------ #
        # 3. Concatenate prefix + kinematic tokens and run through GPT-2      #
        # ------------------------------------------------------------------ #
        combined_embeddings = torch.cat([instruction_embeddings, stacked_inputs], dim=1)
        combined_attention_mask = torch.cat(
            [instruction_attention_mask,
             stacked_attention_mask.to(instruction_attention_mask.dtype)],
            dim=1,
        )

        llm_outputs = self.llm(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention_mask,
            output_hidden_states=True,
        )

        # GPT2LMHeadModel with output_hidden_states=True returns a tuple of all
        # layer hidden states; the last entry is the final transformer layer output.
        last_hidden = llm_outputs.hidden_states[-1]  # (B, L+3K, hidden)

        # ------------------------------------------------------------------ #
        # 4. Extract the last state token from the DT interleaved portion      #
        # ------------------------------------------------------------------ #
        dt_hidden = last_hidden[:, -3 * K:, :]  # (B, 3K, hidden)
        x = dt_hidden.reshape(batch_size, K, 3, self.llm_hidden_size).permute(0, 2, 1, 3)
        agent_emb = x[:, 1, -1, :]  # (B, hidden) — state token at last context step

        # ------------------------------------------------------------------ #
        # 5. Decode to H-step action sequence                                 #
        # ------------------------------------------------------------------ #
        action_flat = self.action_decoder(agent_emb)  # (B, H * act_dim)
        action_preds = action_flat.view(
            batch_size, self.prediction_horizon, self.act_dim
        )  # (B, H, act_dim)

        if return_agent_emb:
            return action_preds, agent_emb
        return action_preds, None

    def project_for_contrastive(
        self, agent_emb: torch.Tensor, text_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project agent and text embeddings into shared space for InfoNCE.

        Returns L2-normalized (agent_z, text_z) of shape (B, proj_dim).
        """
        agent_z = F.normalize(self.agent_proj(agent_emb), dim=-1)
        text_z = F.normalize(self.text_proj(text_emb), dim=-1)
        return agent_z, text_z
