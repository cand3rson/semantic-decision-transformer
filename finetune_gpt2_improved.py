#!/usr/bin/env python3
"""
Fine-tune GPT-2 on IMPROVED trajectory descriptions
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrajectoryDescriptionDataset(Dataset):
    """Dataset for trajectory descriptions"""
    
    def __init__(self, jsonl_path, tokenizer, max_length=64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load examples
        self.examples = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.examples.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.examples)} examples from {jsonl_path}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        
        # Combine prompt + completion
        full_text = ex['prompt'] + ex['completion']
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # For GPT-2, labels are same as input_ids (shifted internally)
        labels = input_ids.clone()
        
        # Mask padding tokens in labels
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    progress = tqdm(dataloader, desc="Training")
    for batch in progress:
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    """Evaluate on validation set"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
    
    return total_loss / len(dataloader)

def generate_samples(model, tokenizer, device, n_samples=5):
    """Generate sample descriptions to check quality"""
    model.eval()
    
    test_prompts = [
        '[vehicle|speed:8.0|direction:straight]',
        '[pedestrian|speed:1.5|direction:left_turn]',
        '[cyclist|speed:6.0|direction:right_turn]',
        '[vehicle|speed:0.0|direction:stopped]',
        '[pedestrian|speed:4.0|direction:straight]'
    ]
    
    logger.info("\n" + "="*60)
    logger.info("Sample Generations:")
    logger.info("="*60)
    
    for prompt in test_prompts[:n_samples]:
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_length=50,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        logger.info(f"\nPrompt: {prompt}")
        logger.info(f"Output: {generated_text}")

def main():
    # Config - use the improved training data
    data_dir = Path('/home/chris/CascadeProjects/decision-transformer-ref-for-nuscenes')
    output_dir = Path('/home/chris/CascadeProjects/decision-transformer-ref-for-nuscenes/gpt2_finetuned_model_improved')
    output_dir.mkdir(exist_ok=True)
    
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    batch_size = 16
    epochs = 15  # More epochs for larger dataset
    learning_rate = 5e-5
    
    logger.info(f"🚀 Fine-tuning GPT-2 on IMPROVED trajectory descriptions")
    logger.info(f"   Device: {device}")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   Epochs: {epochs}")
    logger.info(f"   Learning rate: {learning_rate}")
    
    # Load tokenizer and model
    logger.info("\n📥 Loading GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    
    # Create datasets - use improved files
    train_dataset = TrajectoryDescriptionDataset(
        data_dir / 'trajectory_descriptions_train_improved.jsonl', 
        tokenizer
    )
    val_dataset = TrajectoryDescriptionDataset(
        data_dir / 'trajectory_descriptions_val_improved.jsonl', 
        tokenizer
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )
    
    # Training loop
    logger.info(f"\n🏋️ Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        logger.info(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        logger.info(f"  Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = evaluate(model, val_loader, device)
        logger.info(f"  Val Loss:   {val_loss:.4f}")
        
        # Generate samples
        if (epoch + 1) % 3 == 0:
            generate_samples(model, tokenizer, device)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info(f"  💾 New best model! Saving...")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
    
    logger.info(f"\n✅ Fine-tuning complete!")
    logger.info(f"   Best val loss: {best_val_loss:.4f}")
    logger.info(f"   Model saved to: {output_dir}")
    
    # Final generation test
    logger.info("\n" + "="*60)
    logger.info("Final Model Test:")
    logger.info("="*60)
    generate_samples(model, tokenizer, device, n_samples=5)

if __name__ == '__main__':
    main()

