#!/usr/bin/env python3
"""
Advanced Training Script for Transformer H-Net
Enhanced training with learning rate scheduling, gradient clipping, and monitoring
"""

import sys
import os
from pathlib import Path
import time
import json
import yaml
from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from amharichnet.models.transformer_hnet import TransformerHNet, TransformerHNetConfig
from amharichnet.data.loader import make_dataloader
from amharichnet.evaluation import AmharicTextEvaluator


class Config:
    """Configuration loader."""
    def __init__(self, config_dict):
        for k, v in config_dict.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            else:
                setattr(self, k, v)


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Config(config_dict)


class TransformerTrainer:
    """Advanced trainer for Transformer H-Net."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set random seeds for reproducibility
        torch.manual_seed(config.infrastructure.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.infrastructure.seed)
        
        # Initialize model
        self.model = self._create_model()
        print(f"✅ Model created: {self.model.get_num_params():,} parameters")
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize evaluator
        self.evaluator = AmharicTextEvaluator()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = []
        
        # Create output directories
        self._create_directories()
        
    def _create_model(self) -> TransformerHNet:
        """Create Transformer H-Net model."""
        model_config = TransformerHNetConfig(
            vocab_size=self.config.model.vocab_size,
            hidden_dim=self.config.model.hidden_dim,
            num_layers=self.config.model.num_layers,
            num_heads=self.config.model.num_heads,
            dropout=self.config.model.dropout,
            max_seq_len=self.config.model.max_seq_len,
            intermediate_size=self.config.model.intermediate_size,
            activation_function=self.config.model.activation_function,
            use_cache=self.config.model.use_cache
        )
        return TransformerHNet(model_config)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with weight decay."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if any(nd in name for nd in ["bias", "LayerNorm", "layer_norm"]):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": self.config.training.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]
        
        return optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.training.learning_rate,
            betas=(self.config.training.adam_beta1, self.config.training.adam_beta2),
            eps=self.config.training.adam_epsilon
        )
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.training.scheduler == "cosine_with_warmup":
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.training.warmup_steps,
                eta_min=self.config.training.min_learning_rate
            )
        else:
            return LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=self.config.training.warmup_steps
            )
    
    def _create_directories(self):
        """Create necessary directories."""
        Path(self.config.paths.model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.paths.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.paths.log_dir).mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'training_history': self.training_history
        }
        
        # Save latest checkpoint
        checkpoint_path = Path(self.config.paths.checkpoint_dir) / f"checkpoint_step_{self.global_step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = Path(self.config.paths.model_dir) / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"💾 Best model saved at step {self.global_step}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint.get('training_history', [])
        
        print(f"✅ Checkpoint loaded from {checkpoint_path}")
    
    def train_step(self, batch) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        
        # Move batch to device
        input_ids = batch.input_ids.to(self.device)
        attention_mask = getattr(batch, 'attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids  # For language modeling
        )
        
        loss = outputs["loss"]
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if self.config.training.max_grad_norm > 0:
            clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        self.global_step += 1
        
        return {
            "loss": loss.item(),
            "learning_rate": self.scheduler.get_last_lr()[0],
            "step": self.global_step
        }
    
    def validate_step(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validation step."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch.input_ids.to(self.device)
                attention_mask = getattr(batch, 'attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                
                total_loss += outputs["loss"].item()
                num_batches += 1
                
                # Limit validation batches for speed
                if num_batches >= 50:
                    break
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            "val_loss": avg_loss,
            "val_perplexity": perplexity
        }
    
    def generate_samples(self, num_samples: int = 5) -> List[str]:
        """Generate sample texts for evaluation."""
        self.model.eval()
        samples = []
        
        prompts = ["ኢትዮጵያ", "አዲስ አበባ", "ትምህርት", "ሰላም", ""]
        
        with torch.no_grad():
            for i, prompt in enumerate(prompts[:num_samples]):
                # Simple generation (can be improved with advanced_generator)
                if prompt:
                    # Encode prompt (basic tokenization)
                    input_ids = torch.tensor(
                        [[hash(token) % self.config.model.vocab_size for token in prompt.split()]], 
                        device=self.device
                    )
                else:
                    input_ids = torch.tensor([[1]], device=self.device)  # Start token
                
                # Generate
                generated = self.model.generate(
                    input_ids,
                    max_length=50,
                    temperature=0.8,
                    top_k=40,
                    top_p=0.9,
                    do_sample=True
                )
                
                # Decode (placeholder)
                sample_text = f"Generated sample {i+1} from prompt '{prompt}'"
                samples.append(sample_text)
        
        return samples
    
    def train(self):
        """Main training loop."""
        print(f"🚀 Starting training with {self.model.get_num_params():,} parameters")
        print(f"📊 Device: {self.device}")
        print(f"📈 Epochs: {self.config.training.num_epochs}")
        print(f"🎯 Learning Rate: {self.config.training.learning_rate}")
        print("=" * 60)
        
        # Load data
        train_loader = make_dataloader(
            self.config.data.train_path,
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
            tokenizer_type=self.config.data.tokenizer_type,
            max_len=self.config.data.max_length
        )
        
        val_loader = make_dataloader(
            self.config.data.val_path,
            batch_size=self.config.evaluation.eval_batch_size,
            num_workers=self.config.data.num_workers,
            tokenizer_type=self.config.data.tokenizer_type,
            max_len=self.config.data.max_length
        )
        
        print(f"📚 Training samples: {len(train_loader.dataset)}")
        print(f"📚 Validation samples: {len(val_loader.dataset)}")
        
        # Training loop
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(self.config.training.num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            num_batches = 0
            
            print(f"\n📅 Epoch {epoch + 1}/{self.config.training.num_epochs}")
            
            # Training
            for batch_idx, batch in enumerate(train_loader):
                step_metrics = self.train_step(batch)
                epoch_loss += step_metrics["loss"]
                num_batches += 1
                
                # Logging
                if self.global_step % self.config.training.logging_steps == 0:
                    print(f"Step {self.global_step}: Loss={step_metrics['loss']:.4f}, "
                          f"LR={step_metrics['learning_rate']:.2e}")
                
                # Validation
                if self.global_step % self.config.training.eval_steps == 0:
                    val_metrics = self.validate_step(val_loader)
                    print(f"🔍 Validation - Loss: {val_metrics['val_loss']:.4f}, "
                          f"Perplexity: {val_metrics['val_perplexity']:.2f}")
                    
                    # Save checkpoint if best
                    is_best = val_metrics['val_loss'] < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_metrics['val_loss']
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    # Save checkpoint
                    if self.global_step % self.config.training.save_steps == 0:
                        self.save_checkpoint(is_best)
                    
                    # Record metrics
                    self.training_history.append({
                        "epoch": epoch,
                        "step": self.global_step,
                        "train_loss": step_metrics["loss"],
                        "val_loss": val_metrics['val_loss'],
                        "val_perplexity": val_metrics['val_perplexity'],
                        "learning_rate": step_metrics['learning_rate']
                    })
                
                # Early stopping
                if (self.config.training.early_stopping and 
                    patience_counter >= self.config.training.patience):
                    print(f"⏹️  Early stopping triggered at step {self.global_step}")
                    break
            
            # Epoch summary
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            print(f"📊 Epoch {epoch + 1} complete - Avg Loss: {avg_epoch_loss:.4f}")
            
            # Generate samples
            if epoch % 5 == 0:
                print("🎨 Generating sample texts...")
                samples = self.generate_samples()
                for i, sample in enumerate(samples):
                    print(f"   Sample {i+1}: {sample}")
            
            # Early stopping check
            if (self.config.training.early_stopping and 
                patience_counter >= self.config.training.patience):
                break
        
        # Training complete
        training_time = time.time() - start_time
        print(f"\n✅ Training completed in {training_time/3600:.2f} hours")
        print(f"🏆 Best validation loss: {self.best_val_loss:.4f}")
        
        # Save final metrics
        final_metrics = {
            "training_time_hours": training_time / 3600,
            "total_steps": self.global_step,
            "final_epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "model_parameters": self.model.get_num_params(),
            "training_history": self.training_history
        }
        
        metrics_path = Path(self.config.paths.metrics_path)
        with open(metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        print(f"📄 Training metrics saved to {metrics_path}")


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Transformer H-Net")
    parser.add_argument("--config", default="configs/transformer_hnet.yaml", 
                       help="Path to config file")
    parser.add_argument("--resume", help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create trainer
    trainer = TransformerTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        trainer.save_checkpoint()
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        trainer.save_checkpoint()
        raise


if __name__ == "__main__":
    main()