#!/usr/bin/env python3
"""
DeepSeek Trajectory Predictor Wrapper
===================================

This wrapper makes DeepSeek compatible with the Decision Transformer interface
while swapping ONLY the transformer component.

Architecture:
- Same embedding layers as Decision Transformer
- Same prediction heads as Decision Transformer  
- DeepSeek transformer backbone (instead of custom GPT)
- Same forward pass logic

This ensures a fair comparison where ONLY the transformer differs.
"""

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from decision_transformer.models.model import TrajectoryModel


class DeepSeekTrajectoryPredictor(TrajectoryModel):
    """
    DeepSeek based trajectory predictor with Decision Transformer interface.
    
    This model uses DeepSeek to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    exactly like Decision Transformer, but with DeepSeek as the backbone.
    """
    
    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=50,
            max_ep_len=50,
            action_tanh=True,
            n_layer=3,
            n_head=4,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)
        
        self.hidden_size = hidden_size
        
        # Create DeepSeek config
        # DeepSeek uses similar architecture to LLaMA with some modifications
        config = AutoConfig.from_pretrained(
            "deepseek-ai/deepseek-llm-7b-base",
                hidden_size=hidden_size,
                num_hidden_layers=n_layer,
                num_attention_heads=n_head,
            num_key_value_heads=n_head,  # Use standard MHA instead of GQA
            intermediate_size=hidden_size * 4,
            max_position_embeddings=max_length * 3,  # RTG, state, action per timestep
            vocab_size=1,  # Not used for embeddings
            torch_dtype=torch.float32,  # Force float32 to match our inputs
            trust_remote_code=True,
                **kwargs
            )
            
        # Use DeepSeek transformer
        self.transformer = AutoModel.from_config(config, trust_remote_code=True)
        
        # Ensure all transformer parameters are in float32
        self.transformer = self.transformer.to(dtype=torch.float32)
        
        # Same embedding layers as Decision Transformer
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)
        
        self.embed_ln = nn.LayerNorm(hidden_size)
        
        # Same prediction heads as Decision Transformer
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = torch.nn.Linear(hidden_size, 1)
    
    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None, return_agent_emb=False):
        """
        Forward pass - identical to Decision Transformer.
        
        Args:
            states: (batch_size, seq_length, state_dim)
            actions: (batch_size, seq_length, act_dim)
            rewards: (batch_size, seq_length, 1)
            returns_to_go: (batch_size, seq_length, 1)
            timesteps: (batch_size, seq_length)
            attention_mask: (batch_size, seq_length) - optional
            return_agent_emb: bool - if True, return agent embeddings for VLM
            
        Returns:
            state_preds: (batch_size, seq_length, state_dim)
            action_preds: (batch_size, seq_length, act_dim)
            return_preds: (batch_size, seq_length, 1)
            agent_emb: (batch_size, hidden_size) - only if return_agent_emb=True
        """
        batch_size, seq_length = states.shape[0], states.shape[1]
        
        if attention_mask is None:
            # attention mask for DeepSeek: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
        
        # Embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)
        
        # Time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        
        # Stack RTG, state, action in sequence
        # (batch_size, seq_length, 3, hidden_size)
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        
        stacked_inputs = self.embed_ln(stacked_inputs)
        
        # Expand attention mask to match stacked sequence
        # (batch_size, 3*seq_length)
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)
        
        # Pass through DeepSeek transformer
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs.last_hidden_state
        
        # Reshape to (batch_size, seq_length, 3, hidden_size)
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)
        
        # Get predictions from each position
        return_preds = self.predict_return(x[:,2])  # predict next return from action
        state_preds = self.predict_state(x[:,2])    # predict next state from action
        action_preds = self.predict_action(x[:,1])  # predict action from state

        if return_agent_emb:
            # Extract agent embedding for VLM (from state tokens)
            # Use the last state token's representation
            agent_emb = x[:, 1, -1, :]  # (batch_size, hidden_size)
            return state_preds, action_preds, return_preds, agent_emb
        
        return state_preds, action_preds, return_preds

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        """
        Get action prediction - used during evaluation.
        Identical to Decision Transformer.
        """
        # Reshape to batch format
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # Pad to max_length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, return_preds = self.forward(
            states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)
    
        return action_preds[0,-1]
