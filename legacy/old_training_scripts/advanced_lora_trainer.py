#!/usr/bin/env python3
"""
Advanced LoRA Training for Amharic LLM
Implements Phase 2 of the roadmap: Advanced Training with LoRA
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import json
import os
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import Dataset as HFDataset

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AmharicDataset(Dataset):
    """Custom dataset for Amharic text data"""
    
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }

class AdvancedLoRATrainer:
    """Advanced trainer with LoRA implementation"""
    
    def __init__(self, config=None):
        self.config = config or self.get_default_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🔧 Using device: {self.device}")
        
        # Create output directories
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.plots_dir = Path('training_plots')
        self.plots_dir.mkdir(exist_ok=True)
        
        # Training history
        self.training_history = {
            'losses': [],
            'learning_rates': [],
            'epochs': [],
            'perplexities': []
        }
        
    def get_default_config(self):
        """Get default training configuration"""
        return {
            'model_name': 'gpt2',
            'output_dir': 'models/amharic-lora',
            'max_length': 512,
            'batch_size': 2,
            'gradient_accumulation_steps': 4,
            'num_epochs': 3,
            'learning_rate': 5e-4,
            'warmup_steps': 100,
            'logging_steps': 50,
            'save_steps': 500,
            'eval_steps': 500,
            'lora_r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.1,
            'target_modules': ['c_attn', 'c_proj'],
        }
    
    def load_training_data(self):
        """Load and combine all available training data"""
        logger.info("📚 Loading training data...")
        
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
    
    def setup_model_and_tokenizer(self):
        """Setup model with LoRA configuration"""
        logger.info("🤖 Setting up model and tokenizer...")
        
        # Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.config['model_name'])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        model_config = GPT2Config.from_pretrained(self.config['model_name'])
        self.base_model = GPT2LMHeadModel.from_pretrained(
            self.config['model_name'],
            config=model_config
        )
        
        # Setup LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config['lora_r'],
            lora_alpha=self.config['lora_alpha'],
            lora_dropout=self.config['lora_dropout'],
            target_modules=self.config['target_modules'],
            bias="none",
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.base_model, lora_config)
        self.model.to(self.device)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"🔢 Trainable parameters: {trainable_params:,}")
        logger.info(f"🔢 Total parameters: {total_params:,}")
        logger.info(f"🔢 Trainable %: {100 * trainable_params / total_params:.2f}%")
        
        return self.model, self.tokenizer
    
    def create_dataset(self, texts):
        """Create HuggingFace dataset from texts"""
        logger.info("📊 Creating dataset...")
        
        # Tokenize all texts
        tokenized_texts = []
        for text in tqdm(texts[:5000], desc="Tokenizing"):  # Limit for memory
            tokens = self.tokenizer(
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
    
    def train(self):
        """Main training function with LoRA"""
        logger.info("🚀 Starting LoRA training...")
        
        # Setup
        self.setup_model_and_tokenizer()
        texts = self.load_training_data()
        
        if len(texts) == 0:
            logger.error("❌ No training data found!")
            return
        
        # Create dataset
        train_dataset = self.create_dataset(texts)
        
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
            eval_steps=self.config['eval_steps'],
            save_strategy="steps",
            eval_strategy="no",
            load_best_model_at_end=False,
            report_to=None,
            dataloader_pin_memory=False,
            fp16=torch.cuda.is_available(),
            remove_unused_columns=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Start training
        logger.info("🎯 Starting training...")
        trainer.train()
        
        # Save the LoRA model
        logger.info("💾 Saving LoRA model...")
        self.model.save_pretrained(self.config['output_dir'])
        self.tokenizer.save_pretrained(self.config['output_dir'])
        
        logger.info(f"✅ Training completed! Model saved to {self.config['output_dir']}")
        
        # Generate sample text
        self.generate_sample_text()
    
    def generate_sample_text(self):
        """Generate sample text to test the model"""
        logger.info("🎭 Generating sample text...")
        
        self.model.eval()
        
        # Test prompts in Amharic
        test_prompts = [
            "ሰላም",
            "እንዴት ነህ",
            "ኢትዮጵያ",
            "አዲስ አበባ",
        ]
        
        for prompt in test_prompts:
            try:
                inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_length=inputs.shape[1] + 50,
                        num_return_sequences=1,
                        temperature=0.8,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                logger.info(f"Prompt: '{prompt}' → Generated: '{generated_text}'")
                
            except Exception as e:
                logger.warning(f"Generation failed for '{prompt}': {e}")

def main():
    """Main function"""
    logger.info("🇪🇹 Starting Advanced LoRA Training for Amharic LLM")
    
    # Custom configuration
    config = {
        'model_name': 'gpt2',
        'output_dir': 'models/amharic-lora-advanced',
        'max_length': 256,  # Smaller for CPU training
        'batch_size': 1,    # Small batch for CPU
        'gradient_accumulation_steps': 8,  # Compensate with accumulation
        'num_epochs': 2,
        'learning_rate': 5e-4,
        'warmup_steps': 50,
        'logging_steps': 25,
        'save_steps': 250,
        'eval_steps': 250,
        'lora_r': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.1,
        'target_modules': ['c_attn', 'c_proj'],
    }
    
    # Create trainer and start training
    trainer = AdvancedLoRATrainer(config)
    trainer.train()
    
    logger.info("✅ Advanced LoRA training completed!")
    logger.info("📁 Check models/amharic-lora-advanced/ for the trained model")

if __name__ == "__main__":
    main()