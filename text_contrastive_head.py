#!/usr/bin/env python3
"""
TextContrastiveHead for VisionTRAP-style text guidance
Implements InfoNCE loss with false-negative masking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TextContrastiveHead(nn.Module):
    """
    Text contrastive head for VisionTRAP-style training
    
    Projects agent embeddings to text embedding space and computes InfoNCE loss
    with false-negative masking to avoid punishing similar text descriptions.
    """
    
    def __init__(self, d_agent, d_text, learnable_temp=True, init_temp=0.07, queue_size: int = 0):
        """
        Args:
            d_agent: Dimension of agent embeddings
            d_text: Dimension of text embeddings  
            learnable_temp: Whether to use learnable temperature
            init_temp: Initial temperature value
        """
        super().__init__()
        
        self.d_agent = d_agent
        self.d_text = d_text
        self.queue_size = int(queue_size) if queue_size is not None else 0
        # Simple memory for additional text negatives (not registered as a buffer to allow easy resizing)
        self.queue_text = None
        
        # Projection layer (no bias for better normalization)
        self.proj = nn.Linear(d_agent, d_text, bias=False)
        
        # Initialize projection layer with Xavier/Glorot initialization
        # This ensures better initial alignment and positive similarity
        nn.init.xavier_uniform_(self.proj.weight, gain=0.1)  # Small gain for conservative start
        
        # Learnable temperature parameter
        if learnable_temp:
            self.log_tau = nn.Parameter(torch.log(torch.tensor(init_temp)))
        else:
            self.register_buffer('log_tau', torch.log(torch.tensor(init_temp)))
    
    def forward(self, agent_vec):
        """
        Project agent embeddings to text space
        
        Args:
            agent_vec: Agent embeddings [B, d_agent]
            
        Returns:
            z: Projected embeddings [B, d_text] (unit-normalized)
            tau: Temperature scalar
        """
        # Project to text space
        z = self.proj(agent_vec)
        
        # Unit normalize
        z = F.normalize(z, dim=-1)
        
        # Get temperature
        tau = self.log_tau.exp().clamp(1e-3, 1.0)
        
        return z, tau
    
    def compute_contrastive_loss(self, agent_emb, text_emb, fn_mask=None, update_queue: bool = True):
        """
        Compute InfoNCE loss with optional false-negative masking
        
        Args:
            agent_emb: Agent embeddings [B, d_text] (unit-normalized)
            text_emb: Text embeddings [B, d_text] (unit-normalized)
            fn_mask: False-negative mask [B, B] (True = mask out)
            
        Returns:
            loss: InfoNCE loss scalar
            metrics: Dictionary with contrastive metrics
        """
        B = agent_emb.shape[0]
        
        # Compute similarity matrix (batch positives/negatives)
        tau = self.log_tau.exp()
        logits = torch.mm(agent_emb, text_emb.T) / tau
        
        # Append memory-bank negatives if available
        has_queue = (self.queue_size > 0) and (self.queue_text is not None) and (self.queue_text.numel() > 0)
        if has_queue:
            # Ensure queue on same device
            queue_text = self.queue_text.to(agent_emb.device)
            # Only negatives; no positives in queue
            queue_logits = torch.mm(agent_emb, queue_text.T) / tau  # [B, Q]
            logits_all = torch.cat([logits, queue_logits], dim=1)  # [B, B+Q]
        else:
            logits_all = logits
        
        # Labels are diagonal (positive pairs)
        labels = torch.arange(B, device=agent_emb.device)
        
        # Apply false-negative mask if provided, but NEVER mask the diagonal
        if fn_mask is not None:
            # Ensure diagonal is never masked (positive pairs must remain)
            diagonal_mask = torch.eye(B, device=agent_emb.device, dtype=bool)
            fn_mask = fn_mask & ~diagonal_mask
            # Extend mask with zeros for queue negatives (unknown relations)
            if has_queue:
                zeros_ext = torch.zeros((B, logits_all.shape[1] - B), device=agent_emb.device, dtype=torch.bool)
                fn_mask_ext = torch.cat([fn_mask, zeros_ext], dim=1)
            else:
                fn_mask_ext = fn_mask
            # Mask out false negatives (set to -inf)
            logits_all = logits_all.masked_fill(fn_mask_ext, float('-inf'))
        
        # Compute cross-entropy loss with numerical stability check
        loss = F.cross_entropy(logits_all, labels)
        
        # Check for NaN and handle gracefully
        if torch.isnan(loss):
            logger.warning("NaN detected in InfoNCE loss! Using zero loss.")
            loss = torch.tensor(0.0, device=agent_emb.device)
        
        # Compute metrics for logging
        with torch.no_grad():
            # Positive similarities (diagonal)
            pos_sim = torch.diag(torch.mm(agent_emb, text_emb.T))
            pos_sim_mean = pos_sim.mean().item()
            
            # Negative similarities (batch off-diagonal + queue)
            sim_batch = torch.mm(agent_emb, text_emb.T)
            neg_vals = []
            # Off-diagonal batch
            neg_mask = ~torch.eye(B, device=agent_emb.device, dtype=bool)
            if fn_mask is not None:
                neg_mask = neg_mask & ~fn_mask
            if neg_mask.any():
                neg_vals.append(sim_batch[neg_mask])
            # Queue
            if has_queue:
                sim_queue = torch.mm(agent_emb, queue_text.T)
                neg_vals.append(sim_queue.reshape(-1))
            if len(neg_vals) > 0:
                neg_sim_mean = torch.cat(neg_vals).mean().item()
            else:
                neg_sim_mean = 0.0
            
            # Masked percentage (guard against B <= 1)
            if fn_mask is not None:
                denom = B * (B - 1)
                if denom <= 0:
                    masked_pct = 0.0
                else:
                    masked_pct = (fn_mask.sum().item() / denom) * 100
            else:
                masked_pct = 0.0
        
        metrics = {
            'pos_sim': pos_sim_mean,
            'neg_sim': neg_sim_mean,
            'masked_pct': masked_pct,
            'temp': self.log_tau.exp().item()
        }
        
        # Update memory queue with current batch text embeddings (detach)
        if self.training and update_queue and self.queue_size > 0:
            with torch.no_grad():
                cur = text_emb.detach()
                # Initialize
                if (self.queue_text is None) or (self.queue_text.numel() == 0):
                    self.queue_text = cur[-self.queue_size:].clone()
                else:
                    new_q = torch.cat([self.queue_text.to(cur.device), cur], dim=0)
                    if new_q.shape[0] > self.queue_size:
                        new_q = new_q[-self.queue_size:]
                    self.queue_text = new_q.clone()
        
        return loss, metrics

def compute_false_negative_mask(text_embeddings, threshold=0.8):
    """
    Compute false-negative mask based on text-text similarities
    
    Args:
        text_embeddings: Text embeddings [N, D] (unit-normalized)
        threshold: Cosine similarity threshold for false negatives
        
    Returns:
        mask: Boolean mask [N, N] where True indicates false negative
    """
    # Compute pairwise cosine similarities
    similarity_matrix = torch.mm(text_embeddings, text_embeddings.T)
    
    # Create mask: True where similarity > threshold (excluding diagonal)
    mask = (similarity_matrix > threshold) & (similarity_matrix < 1.0)
    
    return mask

def lambda_text_schedule(epoch, target=0.1, warmup_epochs=2, schedule_type='cosine'):
    """
    Compute text loss weight with warmup schedule
    
    Args:
        epoch: Current epoch (0-indexed)
        target: Target weight after warmup
        warmup_epochs: Number of warmup epochs
        schedule_type: 'linear', 'cosine', or 'exponential'
        
    Returns:
        weight: Current text loss weight
    """
    if epoch < warmup_epochs:
        if schedule_type == 'cosine':
            # Smooth cosine warmup for gradual increase
            progress = (epoch + 1) / warmup_epochs
            return target * (1 - np.cos(np.pi * progress / 2))  # Cosine from 0 to target
        elif schedule_type == 'exponential':
            # Exponential warmup (slower start, faster end)
            progress = (epoch + 1) / warmup_epochs
            return target * (np.exp(progress * 2) - 1) / (np.exp(2) - 1)
        else:  # linear (default)
            return (epoch + 1) / warmup_epochs * target
    else:
        return target

# Example usage and testing
if __name__ == "__main__":
    # Test the contrastive head
    B, d_agent, d_text = 8, 256, 384
    
    # Create mock data
    agent_emb = torch.randn(B, d_agent)
    text_emb = torch.randn(B, d_text)
    text_emb = F.normalize(text_emb, dim=-1)  # Unit normalize
    
    # Create contrastive head
    head = TextContrastiveHead(d_agent, d_text)
    
    # Forward pass
    projected_emb, tau = head(agent_emb)
    print(f"Projected embedding shape: {projected_emb.shape}")
    print(f"Temperature: {tau.item():.4f}")
    
    # Compute false-negative mask
    fn_mask = compute_false_negative_mask(text_emb)
    print(f"False-negative mask shape: {fn_mask.shape}")
    print(f"Masked pairs: {fn_mask.sum().item()}")
    
    # Compute contrastive loss
    loss, metrics = head.compute_contrastive_loss(projected_emb, text_emb, fn_mask)
    print(f"InfoNCE loss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")
    
    # Test warmup schedule
    for epoch in range(5):
        weight = lambda_text_schedule(epoch)
        print(f"Epoch {epoch}: λ_text = {weight:.3f}")
    
    print("✅ TextContrastiveHead test passed!")
