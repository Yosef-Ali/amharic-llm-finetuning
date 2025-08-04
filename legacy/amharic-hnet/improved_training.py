import os
import math
import time
import json
import logging
import argparse
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from improved_model import HNetTransformer
from hybrid_tokenizer import HybridAmharicTokenizer

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class AmharicDataset(Dataset):
    """Dataset for Amharic text data with improved preprocessing."""
    
    def __init__(self, 
                 file_paths: List[str], 
                 tokenizer: HybridAmharicTokenizer,
                 max_length: int = 512,
                 stride: int = 256,
                 use_augmentation: bool = False):
        """Initialize dataset.
        
        Args:
            file_paths: List of file paths containing Amharic text
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
            stride: Stride for sliding window
            use_augmentation: Whether to use data augmentation
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.use_augmentation = use_augmentation
        
        # Load and preprocess data
        self.examples = self._load_and_preprocess_data(file_paths)
        logger.info(f"Loaded {len(self.examples)} examples from {len(file_paths)} files")
    
    def _load_and_preprocess_data(self, file_paths: List[str]) -> List[Dict[str, torch.Tensor]]:
        """Load and preprocess data from files.
        
        Args:
            file_paths: List of file paths
            
        Returns:
            List of preprocessed examples
        """
        examples = []
        
        for file_path in file_paths:
            try:
                # Check file extension
                if file_path.endswith('.json'):
                    # Load JSON file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Extract text from JSON (adjust based on your JSON structure)
                    if isinstance(data, dict):
                        # Try common fields that might contain text
                        text = data.get('text', '')
                        if not text:
                            text = data.get('content', '')
                        if not text:
                            text = data.get('body', '')
                        if not text:
                            # If no common field found, concatenate all string values
                            text = ' '.join([v for k, v in data.items() if isinstance(v, str)])
                    elif isinstance(data, list):
                        # Concatenate all string items or text fields in dictionaries
                        text_parts = []
                        for item in data:
                            if isinstance(item, str):
                                text_parts.append(item)
                            elif isinstance(item, dict):
                                text_part = item.get('text', '')
                                if not text_part:
                                    text_part = item.get('content', '')
                                if not text_part:
                                    text_part = item.get('body', '')
                                if text_part:
                                    text_parts.append(text_part)
                        text = ' '.join(text_parts)
                    else:
                        logger.warning(f"Unsupported JSON structure in {file_path}")
                        continue
                else:
                    # Load text file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                
                # Clean and normalize text
                text = self._clean_text(text)
                
                # Skip empty texts
                if not text:
                    logger.warning(f"Empty text after cleaning in {file_path}")
                    continue
                
                # Create sliding windows with stride
                token_ids = self.tokenizer.encode(text, add_special_tokens=True)
                
                # Create examples from sliding windows
                for i in range(0, len(token_ids) - 2, self.stride):  # -2 for BOS/EOS tokens
                    # Extract window
                    window = token_ids[i:i + self.max_length]
                    
                    # Skip windows that are too short
                    if len(window) < 10:  # Minimum meaningful sequence length
                        continue
                    
                    # Create input and target tensors
                    input_ids = torch.tensor(window[:-1], dtype=torch.long)
                    target_ids = torch.tensor(window[1:], dtype=torch.long)
                    
                    # Add example
                    examples.append({
                        'input_ids': input_ids,
                        'target_ids': target_ids
                    })
                    
                    # Apply data augmentation if enabled
                    if self.use_augmentation and len(window) > 20:
                        # Create augmented examples
                        augmented_examples = self._augment_sequence(window)
                        examples.extend(augmented_examples)
                
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
        
        return examples
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize Amharic text.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Replace multiple spaces with a single space
        text = ' '.join(text.split())
        
        # Remove non-Amharic characters (except punctuation and spaces)
        # This is a simplified approach - for production, use a more comprehensive cleaning
        cleaned_text = ""
        for char in text:
            # Keep Amharic Unicode range (0x1200-0x137F), punctuation, and spaces
            if (
                (0x1200 <= ord(char) <= 0x137F) or  # Amharic characters
                char.isspace() or  # Spaces
                char in '.,:;!?"()[]{}፡።፣፤፥፦፧፨'  # Common punctuation including Amharic
            ):
                cleaned_text += char
        
        return cleaned_text.strip()
    
    def _augment_sequence(self, sequence: List[int]) -> List[Dict[str, torch.Tensor]]:
        """Apply data augmentation techniques to a sequence.
        
        Args:
            sequence: Token ID sequence
            
        Returns:
            List of augmented examples
        """
        augmented_examples = []
        
        # 1. Random masking (similar to BERT)
        if len(sequence) > 10:
            masked_seq = sequence.copy()
            mask_token_id = self.tokenizer.get_unk_token_id()  # Use UNK as mask token
            
            # Mask 15% of tokens randomly
            mask_indices = torch.randperm(len(masked_seq) - 2)[:max(1, int(0.15 * (len(masked_seq) - 2)))] + 1
            for idx in mask_indices:
                masked_seq[idx] = mask_token_id
            
            # Create input and target tensors
            input_ids = torch.tensor(masked_seq[:-1], dtype=torch.long)
            target_ids = torch.tensor(sequence[1:], dtype=torch.long)  # Original sequence as target
            
            augmented_examples.append({
                'input_ids': input_ids,
                'target_ids': target_ids
            })
        
        # 2. Truncation (train on shorter sequences)
        if len(sequence) > 30:
            # Take a random subsection
            start_idx = torch.randint(0, len(sequence) - 30, (1,)).item()
            end_idx = start_idx + torch.randint(20, min(30, len(sequence) - start_idx), (1,)).item()
            
            truncated_seq = sequence[start_idx:end_idx]
            
            # Create input and target tensors
            input_ids = torch.tensor(truncated_seq[:-1], dtype=torch.long)
            target_ids = torch.tensor(truncated_seq[1:], dtype=torch.long)
            
            augmented_examples.append({
                'input_ids': input_ids,
                'target_ids': target_ids
            })
        
        return augmented_examples
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset item."""
        return self.examples[idx]


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function for DataLoader.
    
    Args:
        batch: Batch of examples
        
    Returns:
        Tuple of input and target tensors
    """
    # Get maximum sequence length in batch
    max_len = max(example['input_ids'].size(0) for example in batch)
    
    # Prepare tensors
    input_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
    target_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
    
    # Fill tensors with padded sequences
    for i, example in enumerate(batch):
        seq_len = example['input_ids'].size(0)
        input_ids[i, :seq_len] = example['input_ids']
        target_ids[i, :seq_len] = example['target_ids']
    
    return input_ids, target_ids


class ImprovedTrainer:
    """Improved trainer with advanced training techniques."""
    
    def __init__(self, 
                 model: HNetTransformer,
                 tokenizer: HybridAmharicTokenizer,
                 train_dataset: AmharicDataset,
                 val_dataset: Optional[AmharicDataset] = None,
                 batch_size: int = 16,
                 learning_rate: float = 5e-5,
                 weight_decay: float = 0.01,
                 warmup_steps: int = 1000,
                 max_grad_norm: float = 1.0,
                 num_epochs: int = 10,
                 gradient_accumulation_steps: int = 1,
                 use_mixed_precision: bool = True,
                 use_cosine_scheduler: bool = True,
                 checkpoint_dir: str = './checkpoints',
                 log_interval: int = 100,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize trainer.
        
        Args:
            model: Model to train
            tokenizer: Tokenizer for encoding/decoding
            train_dataset: Training dataset
            val_dataset: Validation dataset
            batch_size: Batch size
            learning_rate: Learning rate
            weight_decay: Weight decay
            warmup_steps: Number of warmup steps
            max_grad_norm: Maximum gradient norm
            num_epochs: Number of training epochs
            gradient_accumulation_steps: Number of steps to accumulate gradients
            use_mixed_precision: Whether to use mixed precision training
            use_cosine_scheduler: Whether to use cosine annealing scheduler
            checkpoint_dir: Directory to save checkpoints
            log_interval: Interval for logging
            device: Device to use
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()
        self.use_cosine_scheduler = use_cosine_scheduler
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.device = device
        
        # Create data loaders
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        if val_dataset:
            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=4,
                pin_memory=True if torch.cuda.is_available() else False
            )
        else:
            self.val_dataloader = None
        
        # Move model to device
        self.model.to(device)
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Create scheduler
        if use_cosine_scheduler:
            # Cosine annealing with warm restarts
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=len(self.train_dataloader) * 2,  # Restart every 2 epochs
                T_mult=2,  # Double the restart interval each time
                eta_min=learning_rate / 100  # Minimum learning rate
            )
        else:
            # Linear warmup and decay
            self.scheduler = self._get_linear_scheduler(warmup_steps)
        
        # Create loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=model.pad_idx)
        
        # Create gradient scaler for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if self.use_mixed_precision else None
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_stats = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'epochs': []
        }
    
    def _get_linear_scheduler(self, warmup_steps: int):
        """Create a learning rate scheduler with linear warmup and decay.
        
        Args:
            warmup_steps: Number of warmup steps
            
        Returns:
            Learning rate scheduler
        """
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(1.0 - current_step / (len(self.train_dataloader) * self.num_epochs)))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train(self):
        """Train the model."""
        logger.info(f"Starting training on {self.device}")
        logger.info(f"Training set size: {len(self.train_dataset)}")
        if self.val_dataset:
            logger.info(f"Validation set size: {len(self.val_dataset)}")
        
        # Training loop
        for epoch in range(self.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # Train for one epoch
            train_loss = self._train_epoch(epoch)
            
            # Validate
            val_loss = None
            if self.val_dataloader:
                val_loss = self._validate()
                logger.info(f"Validation loss: {val_loss:.4f}")
                
                # Save checkpoint if validation loss improved
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(epoch, val_loss, is_best=True)
                    logger.info(f"New best validation loss: {val_loss:.4f}")
            
            # Save regular checkpoint
            self._save_checkpoint(epoch, val_loss if val_loss is not None else train_loss)
            
            # Update training stats
            self.training_stats['train_losses'].append(train_loss)
            if val_loss is not None:
                self.training_stats['val_losses'].append(val_loss)
            self.training_stats['epochs'].append(epoch + 1)
            
            # Save training stats
            self._save_training_stats()
        
        logger.info("Training completed")
    
    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        total_samples = 0
        start_time = time.time()
        
        # Reset gradients
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(self.train_dataloader):
            # Get batch
            input_ids, target_ids = batch
            batch_size = input_ids.size(0)
            
            # Move tensors to device
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            # Forward pass with mixed precision if enabled
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    # Forward pass
                    outputs = self.model(input_ids)
                    
                    # Calculate loss
                    loss = self._compute_loss(outputs, target_ids)
                    loss = loss / self.gradient_accumulation_steps  # Normalize loss for gradient accumulation
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Update weights if gradient accumulation steps reached
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Unscale gradients for clipping
                    self.scaler.unscale_(self.optimizer)
                    
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                    # Update weights
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    
                    # Update learning rate
                    self.scheduler.step()
                    
                    # Reset gradients
                    self.optimizer.zero_grad()
                    
                    # Update global step
                    self.global_step += 1
            else:
                # Forward pass
                outputs = self.model(input_ids)
                
                # Calculate loss
                loss = self._compute_loss(outputs, target_ids)
                loss = loss / self.gradient_accumulation_steps  # Normalize loss for gradient accumulation
                
                # Backward pass
                loss.backward()
                
                # Update weights if gradient accumulation steps reached
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                    # Update weights
                    self.optimizer.step()
                    
                    # Update learning rate
                    self.scheduler.step()
                    
                    # Reset gradients
                    self.optimizer.zero_grad()
                    
                    # Update global step
                    self.global_step += 1
            
            # Accumulate loss
            total_loss += loss.item() * self.gradient_accumulation_steps * batch_size
            total_samples += batch_size
            
            # Log progress
            if (step + 1) % self.log_interval == 0 or (step + 1) == len(self.train_dataloader):
                # Calculate elapsed time and estimated time remaining
                elapsed_time = time.time() - start_time
                steps_per_sec = (step + 1) / elapsed_time
                remaining_steps = len(self.train_dataloader) - (step + 1)
                remaining_time = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                
                # Calculate current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                self.training_stats['learning_rates'].append(current_lr)
                
                # Log progress
                logger.info(
                    f"Epoch {epoch + 1}, Step {step + 1}/{len(self.train_dataloader)}, "
                    f"Loss: {total_loss / total_samples:.4f}, "
                    f"LR: {current_lr:.2e}, "
                    f"Elapsed: {self._format_time(elapsed_time)}, "
                    f"Remaining: {self._format_time(remaining_time)}"
                )
        
        # Calculate average loss
        avg_loss = total_loss / total_samples
        
        # Log epoch summary
        logger.info(
            f"Epoch {epoch + 1} completed in {self._format_time(time.time() - start_time)}, "
            f"Average loss: {avg_loss:.4f}"
        )
        
        return avg_loss
    
    def _validate(self) -> float:
        """Validate the model.
        
        Returns:
            Average validation loss
        """
        if not self.val_dataloader:
            return None
        
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                # Get batch
                input_ids, target_ids = batch
                batch_size = input_ids.size(0)
                
                # Move tensors to device
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids)
                
                # Calculate loss
                loss = self._compute_loss(outputs, target_ids)
                
                # Accumulate loss
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        # Calculate average loss
        avg_loss = total_loss / total_samples
        
        return avg_loss
    
    def _compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss.
        
        Args:
            outputs: Model outputs [batch_size, seq_len, vocab_size]
            targets: Target IDs [batch_size, seq_len]
            
        Returns:
            Loss value
        """
        # Reshape outputs and targets for loss calculation
        outputs = outputs.contiguous().view(-1, self.model.vocab_size)
        targets = targets.contiguous().view(-1)
        
        # Calculate loss
        loss = self.criterion(outputs, targets)
        
        return loss
    
    def _save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            loss: Current loss value
            is_best: Whether this is the best model so far
        """
        # Create checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': {
                'vocab_size': self.model.vocab_size,
                'd_model': self.model.d_model,
                'n_layers': len(self.model.encoder_layers),
                'n_heads': self.model.encoder_layers[0].self_attn.n_heads,
                'd_ff': self.model.encoder_layers[0].feed_forward.fc1.out_features,
                'dropout': self.model.encoder_layers[0].dropout.p,
                'pad_idx': self.model.pad_idx,
                'use_decoder': self.model.use_decoder
            }
        }
        
        # Save checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_model_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_model_path)
            logger.info(f"Best model saved to {best_model_path}")
    
    def _save_training_stats(self):
        """Save training statistics."""
        stats_path = os.path.join(self.checkpoint_dir, "training_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=4)
    
    def _format_time(self, seconds: float) -> str:
        """Format time in seconds to a readable string.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"


def prepare_datasets(tokenizer: HybridAmharicTokenizer, 
                    data_dir: str,
                    max_length: int = 512,
                    stride: int = 256,
                    val_split: float = 0.1,
                    use_augmentation: bool = False) -> Tuple[AmharicDataset, AmharicDataset]:
    """Prepare training and validation datasets.
    
    Args:
        tokenizer: Tokenizer for encoding text
        data_dir: Directory containing data files
        max_length: Maximum sequence length
        stride: Stride for sliding window
        val_split: Fraction of data to use for validation
        use_augmentation: Whether to use data augmentation
        
    Returns:
        Tuple of training and validation datasets
    """
    # Find all text and JSON files in the data directory
    file_paths = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(('.txt', '.json')):
                file_paths.append(os.path.join(root, file))
    
    # Shuffle files
    import random
    random.shuffle(file_paths)
    
    # Split into training and validation sets
    val_size = int(len(file_paths) * val_split)
    train_files = file_paths[val_size:]
    val_files = file_paths[:val_size]
    
    logger.info(f"Found {len(file_paths)} files: {len(train_files)} for training, {len(val_files)} for validation")
    
    # Create datasets
    train_dataset = AmharicDataset(
        file_paths=train_files,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride,
        use_augmentation=use_augmentation
    )
    
    val_dataset = AmharicDataset(
        file_paths=val_files,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride,
        use_augmentation=False  # No augmentation for validation
    )
    
    return train_dataset, val_dataset


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train improved Amharic language model")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing data files")
    parser.add_argument("--tokenizer_path", type=str, help="Path to pretrained tokenizer")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--stride", type=int, default=256, help="Stride for sliding window")
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of data to use for validation")
    
    # Model arguments
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feed-forward dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--use_decoder", action="store_true", help="Use decoder architecture")
    parser.add_argument("--model_path", type=str, help="Path to pretrained model")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients")
    parser.add_argument("--use_mixed_precision", action="store_true", help="Use mixed precision training")
    parser.add_argument("--use_cosine_scheduler", action="store_true", help="Use cosine annealing scheduler")
    parser.add_argument("--use_augmentation", action="store_true", help="Use data augmentation")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--log_interval", type=int, default=100, help="Interval for logging")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    
    args = parser.parse_args()
    
    # Load or create tokenizer
    if args.tokenizer_path:
        logger.info(f"Loading tokenizer from {args.tokenizer_path}")
        tokenizer = HybridAmharicTokenizer.from_pretrained(args.tokenizer_path)
    else:
        logger.info("Creating new tokenizer")
        tokenizer = HybridAmharicTokenizer(use_bpe=False)  # Character-level tokenizer
    
    # Prepare datasets
    train_dataset, val_dataset = prepare_datasets(
        tokenizer=tokenizer,
        data_dir=args.data_dir,
        max_length=args.max_length,
        stride=args.stride,
        val_split=args.val_split,
        use_augmentation=args.use_augmentation
    )
    
    # Create or load model
    if args.model_path:
        logger.info(f"Loading model from {args.model_path}")
        model = HNetTransformer.from_pretrained(args.model_path, device=args.device)
    else:
        logger.info("Creating new model")
        model = HNetTransformer(
            vocab_size=tokenizer.get_vocab_size(),
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            dropout=args.dropout,
            pad_idx=tokenizer.get_pad_token_id(),
            use_decoder=args.use_decoder
        )
    
    # Create trainer
    trainer = ImprovedTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_mixed_precision=args.use_mixed_precision,
        use_cosine_scheduler=args.use_cosine_scheduler,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=args.log_interval,
        device=args.device
    )
    
    # Train model
    trainer.train()


if __name__ == "__main__":
    main()