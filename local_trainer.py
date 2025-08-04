#!/usr/bin/env python3
"""
Local Amharic Model Trainer - Works Offline
"""

import torch
from transformers import (
    GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
    TextDataset, DataCollatorForLanguageModeling,
    Trainer, TrainingArguments
)
from pathlib import Path
import os

class LocalAmharicTrainer:
    def __init__(self):
        self.model_dir = Path("models/amharic-gpt2-local")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Force CPU usage for local development
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        self.device = torch.device('cpu')
        
    def create_small_model(self):
        """Create a small GPT-2 model for local testing"""
        print("Creating small Amharic GPT-2 model...")
        
        # Create a simple tokenizer first
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        # Small configuration for CPU training with correct vocab size
        config = GPT2Config(
            vocab_size=tokenizer.vocab_size,  # Match tokenizer vocab size
            n_positions=256,  # Short sequences
            n_embd=256,       # Small embedding
            n_layer=4,        # Few layers
            n_head=4,         # Few attention heads
        )
        
        model = GPT2LMHeadModel(config)
        model.to(self.device)
        
        # Save model and tokenizer
        model.save_pretrained(self.model_dir)
        tokenizer.save_pretrained(self.model_dir)
        
        print(f"✅ Model saved to {self.model_dir}")
        return model, tokenizer
    
    def train_local(self):
        """Train model locally on CPU"""
        print("Starting local training...")
        
        # Load or create model
        if (self.model_dir / "config.json").exists():
            print("Loading existing model...")
            model = GPT2LMHeadModel.from_pretrained(self.model_dir)
            tokenizer = GPT2Tokenizer.from_pretrained(self.model_dir)
        else:
            model, tokenizer = self.create_small_model()
        
        # Training arguments for CPU
        training_args = TrainingArguments(
            output_dir="./results",
            overwrite_output_dir=True,
            num_train_epochs=1,  # Quick training
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            warmup_steps=100,
            prediction_loss_only=True,
            logging_dir='./logs',
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            use_cpu=True,  # Force CPU
        )
        
        # Load dataset
        train_file = Path("data/processed/train.txt")
        if train_file.exists():
            dataset = TextDataset(
                tokenizer=tokenizer,
                file_path=str(train_file),
                block_size=128
            )
            
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=dataset,
            )
            
            # Train
            print("Training model (this will take a few minutes on CPU)...")
            trainer.train()
            
            # Save final model
            trainer.save_model(self.model_dir)
            print(f"✅ Training complete! Model saved to {self.model_dir}")
        else:
            print("❌ No training data found. Run local_data_collector.py first!")
            
if __name__ == "__main__":
    trainer = LocalAmharicTrainer()
    trainer.train_local()
