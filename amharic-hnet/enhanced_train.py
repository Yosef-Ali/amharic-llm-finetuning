#!/usr/bin/env python3
"""
Enhanced H-Net Training with 1000-Article Corpus
Integrates the newly collected articles with existing H-Net architecture
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
import re
from collections import Counter
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# Import environment configuration
from env_config import env_config, get_env, get_int_env, get_float_env, get_bool_env

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedAmharicTokenizer:
    def __init__(self, max_vocab_size=10000):
        self.max_vocab_size = max_vocab_size
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        
    def build_vocab(self, texts):
        """Build vocabulary from text corpus"""
        logger.info("Building vocabulary from corpus...")
        
        # Count character frequencies
        char_counts = Counter()
        for text in texts:
            char_counts.update(text)
        
        # Special tokens
        special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
        
        # Add special tokens first
        for i, token in enumerate(special_tokens):
            self.char_to_idx[token] = i
            self.idx_to_char[i] = token
        
        # Add most frequent characters
        most_common = char_counts.most_common(self.max_vocab_size - len(special_tokens))
        for char, count in most_common:
            if char not in self.char_to_idx:
                idx = len(self.char_to_idx)
                self.char_to_idx[char] = idx
                self.idx_to_char[idx] = char
        
        self.vocab_size = len(self.char_to_idx)
        logger.info(f"Vocabulary built with {self.vocab_size} characters")
        
        # Log some statistics
        logger.info(f"Most common characters: {most_common[:20]}")
        
    def encode(self, text, max_length=None):
        """Encode text to indices"""
        indices = [self.char_to_idx.get(char, self.char_to_idx['<UNK>']) for char in text]
        
        if max_length:
            if len(indices) > max_length:
                indices = indices[:max_length]
            else:
                indices.extend([self.char_to_idx['<PAD>']] * (max_length - len(indices)))
        
        return indices
    
    def decode(self, indices):
        """Decode indices to text"""
        chars = [self.idx_to_char.get(idx, '<UNK>') for idx in indices]
        text = ''.join(chars).replace('<PAD>', '').replace('<SOS>', '').replace('<EOS>', '')
        return text
    
    def save(self, filepath):
        """Save tokenizer"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'char_to_idx': self.char_to_idx,
                'idx_to_char': self.idx_to_char,
                'vocab_size': self.vocab_size,
                'max_vocab_size': self.max_vocab_size
            }, f)
        logger.info(f"Tokenizer saved to {filepath}")
    
    def load(self, filepath):
        """Load tokenizer"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.char_to_idx = data['char_to_idx']
            self.idx_to_char = data['idx_to_char']
            self.vocab_size = data['vocab_size']
            self.max_vocab_size = data['max_vocab_size']
        logger.info(f"Tokenizer loaded from {filepath}")

class AmharicHNetDataset(Dataset):
    def __init__(self, texts, tokenizer, sequence_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.data = self._prepare_data()
    
    def _prepare_data(self):
        """Prepare data for H-Net training"""
        logger.info("Preparing dataset...")
        data = []
        
        for text in tqdm(self.texts, desc="Processing texts"):
            # Clean and normalize text
            clean_text = self._clean_text(text)
            if len(clean_text) < 10:  # Skip very short texts
                continue
                
            # Encode text
            encoded = self.tokenizer.encode(clean_text)
            
            # Create sliding windows
            for i in range(0, len(encoded) - self.sequence_length, self.sequence_length // 2):
                sequence = encoded[i:i + self.sequence_length]
                if len(sequence) == self.sequence_length:
                    data.append(sequence)
        
        logger.info(f"Created {len(data)} training sequences")
        return data
    
    def _clean_text(self, text):
        """Clean and normalize Amharic text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove non-Amharic characters except basic punctuation
        text = re.sub(r'[^\u1200-\u137F\s\.\,\!\?\:\;\(\)\-\"\'፡\።\፣\፤\፥\፦\፧\፨]', '', text)
        # Normalize punctuation
        text = text.replace('  ', ' ').strip()
        return text
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        # For language modeling: input = sequence[:-1], target = sequence[1:]
        input_seq = torch.tensor(sequence[:-1], dtype=torch.long)
        target_seq = torch.tensor(sequence[1:], dtype=torch.long)
        return input_seq, target_seq

class EnhancedHNet(nn.Module):
    """Enhanced H-Net architecture for Amharic language modeling"""
    
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=3, dropout=0.2):
        super(EnhancedHNet, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layers for sequential processing
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # bidirectional
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # H-Net specific layers
        self.h_transform = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, input_ids, hidden=None):
        batch_size, seq_len = input_ids.size()
        
        # Embedding
        embedded = self.embedding(input_ids)  # (batch, seq, embed_dim)
        embedded = self.dropout(embedded)
        
        # LSTM processing
        lstm_out, hidden = self.lstm(embedded, hidden)  # (batch, seq, hidden_dim*2)
        
        # Self-attention
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Residual connection
        lstm_out = lstm_out + attended
        
        # H-Net transformation
        h_out = self.h_transform(lstm_out)  # (batch, seq, hidden_dim)
        
        # Output projection
        logits = self.output_projection(h_out)  # (batch, seq, vocab_size)
        
        return logits, hidden
    
    def generate(self, tokenizer, prompt="", max_length=100, temperature=0.8, device='cpu'):
        """Generate text using the trained model"""
        self.eval()
        
        if prompt:
            input_ids = tokenizer.encode(prompt)
        else:
            input_ids = [tokenizer.char_to_idx.get('<SOS>', 0)]
        
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
        generated = input_ids.clone()
        
        hidden = None
        
        with torch.no_grad():
            for _ in range(max_length):
                logits, hidden = self.forward(input_ids, hidden)
                
                # Apply temperature
                logits = logits[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(probs, 1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                input_ids = next_token
                
                # Stop at end token
                if next_token.item() == tokenizer.char_to_idx.get('<EOS>', -1):
                    break
        
        # Decode generated sequence
        generated_text = tokenizer.decode(generated[0].tolist())
        return generated_text

class EnhancedTrainer:
    def __init__(self, model, tokenizer, device='cpu', config=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config or {}
        self.model.to(device)
        
        # Training components
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.char_to_idx['<PAD>'])
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=self.config.get('learning_rate', 1e-3), 
            weight_decay=self.config.get('weight_decay', 1e-4)
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, 
            patience=self.config.get('patience', 3)
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (input_ids, target_ids) in enumerate(tqdm(dataloader, desc="Training")):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            logits, _ = self.model(input_ids)
            
            # Calculate loss
            loss = self.criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=self.config.get('max_grad_norm', 1.0)
            )
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Log progress
            if batch_idx % 100 == 0:
                logger.info(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, dataloader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for input_ids, target_ids in tqdm(dataloader, desc="Validating"):
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                
                logits, _ = self.model(input_ids)
                loss = self.criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
                total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def train(self, train_loader, val_loader, epochs=10, save_path="models/enhanced_hnet"):
        """Train the model"""
        logger.info(f"Starting training for {epochs} epochs...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 5
        
        # Create save directory
        os.makedirs(save_path, exist_ok=True)
        
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save model
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'epoch': epoch,
                    'val_loss': val_loss
                }, os.path.join(save_path, 'best_model.pt'))
                
                logger.info(f"New best model saved with val_loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
                
            # Generate sample text
            if epoch % 2 == 0:
                sample_text = self.model.generate(
                    self.tokenizer, 
                    prompt="ኢትዮጵያ", 
                    max_length=50, 
                    device=self.device
                )
                logger.info(f"Sample generation: {sample_text}")
        
        # Save final model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'tokenizer': self.tokenizer,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, os.path.join(save_path, 'final_model.pt'))
        
        # Plot training curves
        self.plot_training_curves(save_path)
        
        logger.info("Training completed!")
    
    def plot_training_curves(self, save_path):
        """Plot training and validation curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Enhanced H-Net Training Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_path, 'training_curves.png'))
        plt.close()
        logger.info(f"Training curves saved to {save_path}/training_curves.png")

def load_corpus_data():
    """Load the 1000-article corpus and existing data"""
    logger.info("Loading corpus data...")
    
    texts = []
    
    # Load new 1000-article corpus
    new_corpus_path = "../processed_articles/amharic_corpus.txt"
    if os.path.exists(new_corpus_path):
        with open(new_corpus_path, 'r', encoding='utf-8') as f:
            new_corpus = f.read()
            # Split by article breaks
            new_articles = new_corpus.split('--- ARTICLE BREAK ---')
            texts.extend([article.strip() for article in new_articles if article.strip()])
        logger.info(f"Loaded {len(new_articles)} articles from new corpus")
    
    # Load existing corpus
    existing_corpus_path = "data/amharic_corpus.txt"
    if os.path.exists(existing_corpus_path):
        with open(existing_corpus_path, 'r', encoding='utf-8') as f:
            existing_corpus = f.read()
            if existing_corpus.strip():
                texts.append(existing_corpus)
        logger.info("Loaded existing corpus")
    
    # Load processed files
    processed_dir = "data/processed"
    if os.path.exists(processed_dir):
        for filename in os.listdir(processed_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(processed_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        texts.append(content)
        logger.info(f"Loaded {len(os.listdir(processed_dir))} processed files")
    
    logger.info(f"Total texts loaded: {len(texts)}")
    return texts

def main():
    """Main training function"""
    logger.info("Starting Enhanced H-Net Training with 1000-Article Corpus")
    
    # Configuration with environment variable support
    config = {
        'max_vocab_size': get_int_env('MAX_VOCAB_SIZE', 5000),
        'sequence_length': get_int_env('SEQUENCE_LENGTH', 128),
        'embedding_dim': get_int_env('EMBEDDING_DIM', 256),
        'hidden_dim': get_int_env('HIDDEN_DIM', 512),
        'num_layers': get_int_env('NUM_LAYERS', 3),
        'dropout': get_float_env('DROPOUT', 0.2),
        'batch_size': get_int_env('BATCH_SIZE', 32),
        'learning_rate': get_float_env('LEARNING_RATE', 1e-3),
        'weight_decay': get_float_env('WEIGHT_DECAY', 1e-4),
        'num_epochs': get_int_env('NUM_EPOCHS', 3),
        'patience': get_int_env('PATIENCE', 5),
        'max_grad_norm': get_float_env('MAX_GRAD_NORM', 1.0),
        'save_path': get_env('SAVE_PATH', 'models/enhanced_hnet'),
        'train_split': get_float_env('TRAIN_SPLIT', 0.8)
    }
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Adjust batch size based on device
    if device.type == 'cpu':
        config['batch_size'] = min(config['batch_size'], 16)
    
    # Load data
    texts = load_corpus_data()
    if not texts:
        logger.error("No training data found!")
        return
    
    logger.info(f"Loaded {len(texts)} texts for training")
    
    # Initialize tokenizer
    tokenizer = EnhancedAmharicTokenizer(max_vocab_size=config['max_vocab_size'])
    tokenizer.build_vocab(texts)
    
    # Save tokenizer
    os.makedirs("models", exist_ok=True)
    tokenizer.save("models/enhanced_tokenizer.pkl")
    
    # Create datasets
    dataset = AmharicHNetDataset(texts, tokenizer, sequence_length=config['sequence_length'])
    
    # Split dataset
    train_size = int(config['train_split'] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Initialize model
    model = EnhancedHNet(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Initialize trainer
    trainer = EnhancedTrainer(model, tokenizer, device, config)
    
    # Train model
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['num_epochs'],
        save_path=config['save_path']
    )
    
    # Final evaluation
    logger.info("\nTraining completed! Testing generation...")
    test_prompts = ["ኢትዮጵያ", "አዲስ አበባ", "ባህል", "ትምህርት"]
    
    for prompt in test_prompts:
        generated = model.generate(tokenizer, prompt=prompt, max_length=100, device=device)
        logger.info(f"Prompt: '{prompt}' -> Generated: '{generated}'")

if __name__ == "__main__":
    main()