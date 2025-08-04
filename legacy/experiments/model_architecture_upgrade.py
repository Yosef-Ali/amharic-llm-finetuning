#!/usr/bin/env python3
"""
Model Architecture Upgrade for Amharic LLM
Implements Phase 1 roadmap: Model Architecture Upgrade
Upgrades from current small model to larger, more capable architecture
"""

import torch
import torch.nn as nn
from transformers import (
    GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
import json
import os
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import Dataset as HFDataset
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AmharicTokenizerBuilder:
    """Build enhanced tokenizer for Amharic with better coverage"""
    
    def __init__(self):
        self.amharic_chars = self._get_amharic_character_set()
        
    def _get_amharic_character_set(self):
        """Get comprehensive Amharic character set"""
        # Core Amharic characters (Ge'ez script)
        base_chars = []
        
        # Amharic consonants with vowel modifications
        for base in range(0x1200, 0x1380):  # Ethiopic block
            try:
                char = chr(base)
                base_chars.append(char)
            except ValueError:
                continue
        
        # Additional Amharic characters
        additional_chars = [
            '፩', '፪', '፫', '፬', '፭', '፮', '፯', '፰', '፱', '፲',  # Numbers
            '፼', '፽', '፾', '፿',  # More numbers
            '።', '፣', '፤', '፥', '፦', '፧', '፨',  # Punctuation
            ' ', '\n', '\t',  # Whitespace
        ]
        
        return base_chars + additional_chars
    
    def build_enhanced_tokenizer(self, base_tokenizer_name='gpt2', vocab_size=100000):
        """Build enhanced tokenizer with better Amharic coverage"""
        logger.info(f"🔤 Building enhanced Amharic tokenizer (target vocab: {vocab_size})...")
        
        # Load base tokenizer
        base_tokenizer = GPT2Tokenizer.from_pretrained(base_tokenizer_name)
        
        # Add Amharic characters to vocabulary
        new_tokens = []
        for char in self.amharic_chars:
            if char not in base_tokenizer.get_vocab():
                new_tokens.append(char)
        
        # Add common Amharic words and patterns
        common_amharic_tokens = [
            'ሰላም', 'እንዴት', 'ነህ', 'ነሽ', 'ናት', 'ነው', 'ናቸው',
            'አመሰግናለሁ', 'እባክህ', 'እባክሽ', 'ይቅርታ', 'ጤና',
            'ኢትዮጵያ', 'አማርኛ', 'አዲስ', 'አበባ', 'ሀገር',
            'ቤተሰብ', 'ወንድም', 'እህት', 'እናት', 'አባት',
            'ትምህርት', 'ቤት', 'ሰዓት', 'ቀን', 'ሳምንት',
            'ወር', 'አመት', 'ጊዜ', 'ቦታ', 'ሰው',
        ]
        
        new_tokens.extend(common_amharic_tokens)
        
        # Add tokens to tokenizer
        if new_tokens:
            base_tokenizer.add_tokens(new_tokens)
            logger.info(f"✅ Added {len(new_tokens)} new Amharic tokens")
        
        # Set special tokens
        base_tokenizer.pad_token = base_tokenizer.eos_token
        
        logger.info(f"📊 Final vocabulary size: {len(base_tokenizer)}")
        return base_tokenizer

class UpgradedAmharicModel:
    """Upgraded model architecture for Amharic LLM"""
    
    def __init__(self, config=None):
        self.config = config or self.get_upgraded_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🔧 Using device: {self.device}")
        
        # Create output directories
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def get_upgraded_config(self):
        """Get upgraded model configuration following roadmap"""
        return {
            # Model architecture (upgraded from roadmap)
            'hidden_dim': 1024,      # Upgraded from 768
            'num_layers': 16,        # Upgraded from 12
            'num_heads': 16,         # Upgraded from 12
            'ffn_dim': 4096,         # Upgraded from 3072
            'vocab_size': 75000,     # Upgraded from 50K
            'max_position_embeddings': 2048,  # Longer sequences
            
            # Training configuration
            'output_dir': 'models/amharic-upgraded',
            'batch_size': 1,         # Small for CPU
            'gradient_accumulation_steps': 16,  # Compensate with accumulation
            'num_epochs': 2,
            'learning_rate': 3e-4,
            'warmup_steps': 100,
            'logging_steps': 25,
            'save_steps': 500,
            'max_length': 512,
        }
    
    def create_upgraded_model(self, tokenizer):
        """Create upgraded model architecture"""
        logger.info("🏗️ Creating upgraded model architecture...")
        
        # Create upgraded configuration
        model_config = GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=self.config['max_position_embeddings'],
            n_embd=self.config['hidden_dim'],
            n_layer=self.config['num_layers'],
            n_head=self.config['num_heads'],
            n_inner=self.config['ffn_dim'],
            activation_function="gelu_new",
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            summary_type="cls_index",
            summary_use_proj=True,
            summary_activation=None,
            summary_proj_to_labels=True,
            summary_first_dropout=0.1,
            use_cache=True,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        
        # Create model
        model = GPT2LMHeadModel(model_config)
        model.to(self.device)
        
        # Print model statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"🔢 Total parameters: {total_params:,}")
        logger.info(f"🔢 Trainable parameters: {trainable_params:,}")
        logger.info(f"📏 Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
        
        return model, model_config
    
    def load_training_data(self):
        """Load training data for the upgraded model"""
        logger.info("📚 Loading training data for upgraded model...")
        
        all_texts = []
        data_files = [
            'data/processed/cpu_training_data.jsonl',
            'data/processed/train_data.jsonl',
            'data/processed/robust_train.jsonl',
        ]
        
        # Load JSONL files
        for file_path in data_files:
            if os.path.exists(file_path):
                logger.info(f"Loading {file_path}...")
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            if 'text' in data:
                                all_texts.append(data['text'])
                            elif 'content' in data:
                                all_texts.append(data['content'])
                        except json.JSONDecodeError:
                            continue
        
        # Load JSON files from collected directory
        collected_dir = Path('data/collected')
        if collected_dir.exists():
            for json_file in collected_dir.glob('*.json'):
                logger.info(f"Loading {json_file}...")
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict):
                                    if 'text' in item:
                                        all_texts.append(item['text'])
                                    elif 'content' in item:
                                        all_texts.append(item['content'])
                                elif isinstance(item, str):
                                    all_texts.append(item)
                        elif isinstance(data, dict):
                            if 'text' in data:
                                all_texts.append(data['text'])
                            elif 'content' in data:
                                all_texts.append(data['content'])
                except (json.JSONDecodeError, UnicodeDecodeError):
                    logger.warning(f"Could not load {json_file}")
                    continue
        
        # Filter and clean texts
        cleaned_texts = []
        for text in all_texts:
            if isinstance(text, str) and len(text.strip()) > 10:
                cleaned_texts.append(text.strip())
        
        logger.info(f"✅ Loaded {len(cleaned_texts)} training samples")
        return cleaned_texts
    
    def create_dataset(self, texts, tokenizer):
        """Create dataset for training"""
        logger.info("📊 Creating training dataset...")
        
        # Tokenize texts
        tokenized_texts = []
        for text in tqdm(texts[:3000], desc="Tokenizing"):  # Limit for memory
            tokens = tokenizer(
                text,
                truncation=True,
                padding=False,
                max_length=self.config['max_length'],
                return_tensors=None
            )
            tokenized_texts.append(tokens['input_ids'])
        
        # Create HuggingFace dataset
        dataset = HFDataset.from_dict({'input_ids': tokenized_texts})
        return dataset
    
    def train_upgraded_model(self):
        """Train the upgraded model"""
        logger.info("🚀 Starting upgraded model training...")
        
        # Build enhanced tokenizer
        tokenizer_builder = AmharicTokenizerBuilder()
        tokenizer = tokenizer_builder.build_enhanced_tokenizer()
        
        # Create upgraded model
        model, model_config = self.create_upgraded_model(tokenizer)
        
        # Load training data
        texts = self.load_training_data()
        if len(texts) == 0:
            logger.error("❌ No training data found!")
            return
        
        # Create dataset
        train_dataset = self.create_dataset(texts, tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config['output_dir'],
            overwrite_output_dir=True,
            num_train_epochs=self.config['num_epochs'],
            per_device_train_batch_size=self.config['batch_size'],
            gradient_accumulation_steps=self.config['gradient_accumulation_steps'],
            learning_rate=self.config['learning_rate'],
            warmup_steps=self.config['warmup_steps'],
            logging_steps=self.config['logging_steps'],
            save_steps=self.config['save_steps'],
            save_strategy="steps",
            evaluation_strategy="no",
            load_best_model_at_end=False,
            report_to=None,
            dataloader_pin_memory=False,
            fp16=False,  # Disable for CPU
            remove_unused_columns=False,
            gradient_checkpointing=True,  # Save memory
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        
        # Start training
        logger.info("🎯 Starting training...")
        trainer.train()
        
        # Save the model
        logger.info("💾 Saving upgraded model...")
        model.save_pretrained(self.config['output_dir'])
        tokenizer.save_pretrained(self.config['output_dir'])
        
        # Save model configuration
        config_path = Path(self.config['output_dir']) / 'upgrade_config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Upgraded model training completed! Model saved to {self.config['output_dir']}")
        
        # Generate sample text
        self.generate_sample_text(model, tokenizer)
        
        return model, tokenizer
    
    def generate_sample_text(self, model, tokenizer):
        """Generate sample text to test the upgraded model"""
        logger.info("🎭 Testing upgraded model with sample generation...")
        
        model.eval()
        
        # Test prompts in Amharic
        test_prompts = [
            "ሰላም",
            "እንዴት ነህ",
            "ኢትዮጵያ ውብ ሀገር ናት",
            "አማርኛ ቋንቋ",
            "አዲስ አበባ",
        ]
        
        for prompt in test_prompts:
            try:
                inputs = tokenizer.encode(prompt, return_tensors='pt').to(self.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_length=inputs.shape[1] + 40,
                        num_return_sequences=1,
                        temperature=0.8,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.1,
                    )
                
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                logger.info(f"🎯 '{prompt}' → '{generated_text}'")
                
            except Exception as e:
                logger.warning(f"Generation failed for '{prompt}': {e}")
    
    def compare_architectures(self):
        """Compare old vs new architecture"""
        logger.info("📊 Architecture Comparison:")
        
        # Current (small) architecture
        current = {
            'hidden_dim': 768,
            'num_layers': 12,
            'num_heads': 12,
            'ffn_dim': 3072,
            'vocab_size': 50000,
        }
        
        # Upgraded architecture
        upgraded = {
            'hidden_dim': self.config['hidden_dim'],
            'num_layers': self.config['num_layers'],
            'num_heads': self.config['num_heads'],
            'ffn_dim': self.config['ffn_dim'],
            'vocab_size': self.config['vocab_size'],
        }
        
        logger.info("\n📈 Architecture Upgrade Summary:")
        for key in current.keys():
            old_val = current[key]
            new_val = upgraded[key]
            improvement = ((new_val - old_val) / old_val) * 100
            logger.info(f"  {key}: {old_val:,} → {new_val:,} (+{improvement:.1f}%)")

def main():
    """Main function"""
    logger.info("🇪🇹 Starting Model Architecture Upgrade for Amharic LLM")
    
    # Create upgraded model trainer
    upgrade_trainer = UpgradedAmharicModel()
    
    # Show architecture comparison
    upgrade_trainer.compare_architectures()
    
    # Train upgraded model
    model, tokenizer = upgrade_trainer.train_upgraded_model()
    
    logger.info("✅ Model architecture upgrade completed!")
    logger.info("📁 Check models/amharic-upgraded/ for the upgraded model")
    logger.info("🎯 The model now has enhanced architecture following the roadmap specifications")

if __name__ == "__main__":
    main()