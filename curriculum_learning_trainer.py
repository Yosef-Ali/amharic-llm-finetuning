#!/usr/bin/env python3
"""
Curriculum Learning Implementation for Amharic LLM
Implements Phase 2 of the roadmap: Curriculum Learning Setup
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
import json
import os
import re
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import Dataset as HFDataset
import numpy as np
from collections import Counter

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextComplexityAnalyzer:
    """Analyze text complexity for curriculum learning"""
    
    def __init__(self):
        # Amharic complexity indicators
        self.simple_patterns = [
            r'^[ሀ-ሿ\s]{1,20}$',  # Very short simple text
            r'^(ሰላም|እንዴት|ጤና|አመሰግናለሁ)\s*[ሀ-ሿ\s]*$',  # Common greetings
        ]
        
        self.complex_patterns = [
            r'[፩-፼]',  # Amharic numerals
            r'[ሀ-ሿ]{15,}',  # Very long words
            r'[።፣፤፥፦፧፨]',  # Complex punctuation
        ]
    
    def calculate_complexity(self, text):
        """Calculate text complexity score (0-100)"""
        if not isinstance(text, str) or len(text.strip()) == 0:
            return 0
        
        text = text.strip()
        complexity_score = 0
        
        # Length factor (0-30 points)
        length_score = min(30, len(text) / 10)
        complexity_score += length_score
        
        # Word count factor (0-20 points)
        words = text.split()
        word_count_score = min(20, len(words) / 2)
        complexity_score += word_count_score
        
        # Character diversity (0-20 points)
        unique_chars = len(set(text))
        char_diversity_score = min(20, unique_chars / 3)
        complexity_score += char_diversity_score
        
        # Complex patterns (0-20 points)
        complex_pattern_score = 0
        for pattern in self.complex_patterns:
            if re.search(pattern, text):
                complex_pattern_score += 5
        complexity_score += min(20, complex_pattern_score)
        
        # Simple patterns (reduce score)
        for pattern in self.simple_patterns:
            if re.search(pattern, text):
                complexity_score -= 10
        
        # Sentence structure (0-10 points)
        sentence_count = len(re.split(r'[።!?]', text))
        if sentence_count > 1:
            complexity_score += min(10, sentence_count * 2)
        
        return max(0, min(100, complexity_score))
    
    def categorize_complexity(self, score):
        """Categorize complexity score into levels"""
        if score < 20:
            return 'beginner'
        elif score < 40:
            return 'elementary'
        elif score < 60:
            return 'intermediate'
        elif score < 80:
            return 'advanced'
        else:
            return 'expert'

class CurriculumDataset(Dataset):
    """Dataset with curriculum learning support"""
    
    def __init__(self, texts, complexity_scores, tokenizer, max_length=512, difficulty_level='beginner'):
        self.texts = texts
        self.complexity_scores = complexity_scores
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.difficulty_level = difficulty_level
        
        # Filter texts by difficulty level
        self.filtered_indices = self._filter_by_difficulty()
        
    def _filter_by_difficulty(self):
        """Filter texts based on difficulty level"""
        analyzer = TextComplexityAnalyzer()
        indices = []
        
        for i, score in enumerate(self.complexity_scores):
            category = analyzer.categorize_complexity(score)
            
            if self.difficulty_level == 'beginner' and category in ['beginner', 'elementary']:
                indices.append(i)
            elif self.difficulty_level == 'intermediate' and category in ['elementary', 'intermediate']:
                indices.append(i)
            elif self.difficulty_level == 'advanced' and category in ['intermediate', 'advanced']:
                indices.append(i)
            elif self.difficulty_level == 'expert' and category in ['advanced', 'expert']:
                indices.append(i)
            elif self.difficulty_level == 'all':
                indices.append(i)
        
        return indices
    
    def __len__(self):
        return len(self.filtered_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.filtered_indices[idx]
        text = str(self.texts[actual_idx])
        
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

class CurriculumLearningTrainer:
    """Curriculum Learning Trainer for Amharic LLM"""
    
    def __init__(self, config=None):
        self.config = config or self.get_default_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🔧 Using device: {self.device}")
        
        # Create output directories
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.complexity_analyzer = TextComplexityAnalyzer()
        
        # Curriculum stages
        self.curriculum_stages = ['beginner', 'intermediate', 'advanced', 'expert']
        self.current_stage = 0
        
    def get_default_config(self):
        """Get default training configuration"""
        return {
            'model_name': 'gpt2',
            'output_dir': 'models/amharic-curriculum',
            'max_length': 256,
            'batch_size': 2,
            'gradient_accumulation_steps': 4,
            'epochs_per_stage': 1,
            'learning_rate': 5e-4,
            'warmup_steps': 50,
            'logging_steps': 25,
            'save_steps': 250,
        }
    
    def load_and_analyze_data(self):
        """Load training data and analyze complexity"""
        logger.info("📚 Loading and analyzing training data...")
        
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
            if isinstance(text, str) and len(text.strip()) > 5:
                cleaned_texts.append(text.strip())
        
        # Analyze complexity
        logger.info("🧠 Analyzing text complexity...")
        complexity_scores = []
        for text in tqdm(cleaned_texts, desc="Analyzing complexity"):
            score = self.complexity_analyzer.calculate_complexity(text)
            complexity_scores.append(score)
        
        # Print complexity distribution
        self.print_complexity_distribution(complexity_scores)
        
        logger.info(f"✅ Loaded {len(cleaned_texts)} training samples with complexity analysis")
        return cleaned_texts, complexity_scores
    
    def print_complexity_distribution(self, complexity_scores):
        """Print distribution of complexity scores"""
        categories = [self.complexity_analyzer.categorize_complexity(score) for score in complexity_scores]
        category_counts = Counter(categories)
        
        logger.info("📊 Complexity Distribution:")
        for category, count in category_counts.items():
            percentage = (count / len(complexity_scores)) * 100
            logger.info(f"  {category.capitalize()}: {count} samples ({percentage:.1f}%)")
    
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer"""
        logger.info("🤖 Setting up model and tokenizer...")
        
        # Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.config['model_name'])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        model_config = GPT2Config.from_pretrained(self.config['model_name'])
        # Make model smaller for CPU training
        model_config.n_layer = 6
        model_config.n_head = 6
        model_config.n_embd = 384
        
        self.model = GPT2LMHeadModel(model_config)
        self.model.to(self.device)
        
        logger.info(f"🔢 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        return self.model, self.tokenizer
    
    def train_curriculum_stage(self, texts, complexity_scores, stage_name):
        """Train on a specific curriculum stage"""
        logger.info(f"🎯 Training curriculum stage: {stage_name}")
        
        # Create dataset for this stage
        dataset = CurriculumDataset(
            texts, complexity_scores, self.tokenizer,
            max_length=self.config['max_length'],
            difficulty_level=stage_name
        )
        
        if len(dataset) == 0:
            logger.warning(f"⚠️ No data for stage {stage_name}, skipping...")
            return
        
        logger.info(f"📊 Stage {stage_name}: {len(dataset)} samples")
        
        # Convert to HuggingFace dataset
        hf_dataset = HFDataset.from_dict({
            'input_ids': [dataset[i]['input_ids'].tolist() for i in range(min(len(dataset), 1000))]
        })
        
        # Training arguments for this stage
        stage_output_dir = self.output_dir / f"stage_{stage_name}"
        stage_output_dir.mkdir(exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=str(stage_output_dir),
            overwrite_output_dir=True,
            num_train_epochs=self.config['epochs_per_stage'],
            per_device_train_batch_size=self.config['batch_size'],
            gradient_accumulation_steps=self.config['gradient_accumulation_steps'],
            learning_rate=self.config['learning_rate'],
            warmup_steps=self.config['warmup_steps'],
            logging_steps=self.config['logging_steps'],
            save_steps=self.config['save_steps'],
            save_strategy="steps",
            eval_strategy="no",
            load_best_model_at_end=False,
            report_to=None,
            dataloader_pin_memory=False,
            fp16=False,  # Disable for CPU
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
            train_dataset=hf_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Train this stage
        logger.info(f"🚀 Starting training for stage: {stage_name}")
        trainer.train()
        
        # Save stage checkpoint
        stage_checkpoint_dir = stage_output_dir / "final"
        self.model.save_pretrained(stage_checkpoint_dir)
        self.tokenizer.save_pretrained(stage_checkpoint_dir)
        
        logger.info(f"✅ Completed stage: {stage_name}")
    
    def train(self):
        """Main curriculum training function"""
        logger.info("🚀 Starting Curriculum Learning Training...")
        
        # Setup
        self.setup_model_and_tokenizer()
        texts, complexity_scores = self.load_and_analyze_data()
        
        if len(texts) == 0:
            logger.error("❌ No training data found!")
            return
        
        # Train through curriculum stages
        for stage in self.curriculum_stages:
            self.train_curriculum_stage(texts, complexity_scores, stage)
        
        # Save final model
        final_dir = self.output_dir / "final_curriculum_model"
        self.model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)
        
        logger.info(f"✅ Curriculum training completed! Final model saved to {final_dir}")
        
        # Generate sample text
        self.generate_sample_text()
    
    def generate_sample_text(self):
        """Generate sample text to test the model"""
        logger.info("🎭 Generating sample text...")
        
        self.model.eval()
        
        # Test prompts in Amharic with different complexity levels
        test_prompts = [
            "ሰላም",  # Simple
            "እንዴት ነህ",  # Elementary
            "ኢትዮጵያ ውብ ሀገር ናት",  # Intermediate
            "የአማርኛ ቋንቋ ታሪክ በጣም ረጅም ነው",  # Advanced
        ]
        
        for prompt in test_prompts:
            try:
                inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_length=inputs.shape[1] + 30,
                        num_return_sequences=1,
                        temperature=0.8,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                complexity = self.complexity_analyzer.calculate_complexity(prompt)
                category = self.complexity_analyzer.categorize_complexity(complexity)
                
                logger.info(f"[{category.upper()}] '{prompt}' → '{generated_text}'")
                
            except Exception as e:
                logger.warning(f"Generation failed for '{prompt}': {e}")

def main():
    """Main function"""
    logger.info("🇪🇹 Starting Curriculum Learning Training for Amharic LLM")
    
    # Custom configuration
    config = {
        'model_name': 'gpt2',
        'output_dir': 'models/amharic-curriculum',
        'max_length': 128,  # Smaller for CPU training
        'batch_size': 1,    # Small batch for CPU
        'gradient_accumulation_steps': 8,
        'epochs_per_stage': 1,
        'learning_rate': 5e-4,
        'warmup_steps': 25,
        'logging_steps': 10,
        'save_steps': 100,
    }
    
    # Create trainer and start training
    trainer = CurriculumLearningTrainer(config)
    trainer.train()
    
    logger.info("✅ Curriculum learning training completed!")
    logger.info("📁 Check models/amharic-curriculum/ for the trained models")

if __name__ == "__main__":
    main()