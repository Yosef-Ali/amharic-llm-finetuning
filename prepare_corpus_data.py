#!/usr/bin/env python3
"""
Prepare the 30k+ Amharic corpus for training with the clean implementation.

This script converts your processed corpus into JSONL format suitable for
src/amharichnet training pipeline.
"""

import json
import os
from pathlib import Path
import random
from typing import List, Dict, Any


def load_sentences_file(file_path: str) -> List[str]:
    """Load sentences from the sentences.txt file."""
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                sentences.append(line)
    return sentences


def load_corpus_file(file_path: str) -> List[str]:
    """Load paragraphs from the corpus.txt file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split on the article break marker
    articles = content.split('--- ARTICLE BREAK ---')
    
    # Clean and filter articles
    cleaned_articles = []
    for article in articles:
        article = article.strip()
        if article and len(article) > 50:  # Filter very short articles
            cleaned_articles.append(article)
    
    return cleaned_articles


def load_json_corpus(file_path: str) -> List[Dict[str, Any]]:
    """Load the structured JSON corpus."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('articles', [])


def create_jsonl_dataset(texts: List[str], output_path: str, text_type: str = "sentence"):
    """Create JSONL dataset from texts."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, text in enumerate(texts):
            record = {
                "id": f"{text_type}_{i:06d}",
                "text": text,
                "type": text_type,
                "language": "amharic"
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"✅ Created {output_path} with {len(texts)} {text_type}s")


def split_data(texts: List[str], train_ratio: float = 0.8, val_ratio: float = 0.1):
    """Split data into train/val/test sets."""
    random.shuffle(texts)
    
    n = len(texts)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_texts = texts[:train_end]
    val_texts = texts[train_end:val_end]
    test_texts = texts[val_end:]
    
    return train_texts, val_texts, test_texts


def main():
    """Main function to prepare corpus data."""
    print("🚀 Preparing your 30k+ Amharic corpus for training...")
    
    # Set random seed for reproducible splits
    random.seed(1337)
    
    # Define paths
    base_dir = Path("data/processed/processed_articles")
    sentences_file = base_dir / "amharic_sentences.txt"
    corpus_file = base_dir / "amharic_corpus.txt"
    json_file = base_dir / "amharic_corpus.json"
    
    output_dir = Path("data/training")
    output_dir.mkdir(exist_ok=True)
    
    # Check which files exist
    print("📁 Checking available data files...")
    
    if sentences_file.exists():
        print(f"✅ Found sentences file: {sentences_file}")
        sentences = load_sentences_file(str(sentences_file))
        print(f"   📊 {len(sentences)} sentences loaded")
        
        # Split sentences for training
        train_sentences, val_sentences, test_sentences = split_data(sentences)
        
        # Create JSONL files for sentences
        create_jsonl_dataset(train_sentences, str(output_dir / "sentences_train.jsonl"), "sentence")
        create_jsonl_dataset(val_sentences, str(output_dir / "sentences_val.jsonl"), "sentence") 
        create_jsonl_dataset(test_sentences, str(output_dir / "sentences_test.jsonl"), "sentence")
        
    else:
        print(f"❌ Sentences file not found: {sentences_file}")
    
    if corpus_file.exists():
        print(f"✅ Found corpus file: {corpus_file}")
        articles = load_corpus_file(str(corpus_file))
        print(f"   📊 {len(articles)} articles loaded")
        
        # Split articles for training
        train_articles, val_articles, test_articles = split_data(articles)
        
        # Create JSONL files for articles
        create_jsonl_dataset(train_articles, str(output_dir / "articles_train.jsonl"), "article")
        create_jsonl_dataset(val_articles, str(output_dir / "articles_val.jsonl"), "article")
        create_jsonl_dataset(test_articles, str(output_dir / "articles_test.jsonl"), "article")
        
    else:
        print(f"❌ Corpus file not found: {corpus_file}")
    
    if json_file.exists():
        print(f"✅ Found JSON corpus: {json_file}")
        json_articles = load_json_corpus(str(json_file))
        print(f"   📊 {len(json_articles)} structured articles loaded")
        
        # Extract sentences from JSON structure
        all_json_sentences = []
        for article in json_articles:
            if 'sentences' in article:
                all_json_sentences.extend(article['sentences'])
        
        if all_json_sentences:
            train_json, val_json, test_json = split_data(all_json_sentences)
            create_jsonl_dataset(train_json, str(output_dir / "json_sentences_train.jsonl"), "json_sentence")
            create_jsonl_dataset(val_json, str(output_dir / "json_sentences_val.jsonl"), "json_sentence")
            create_jsonl_dataset(test_json, str(output_dir / "json_sentences_test.jsonl"), "json_sentence")
    
    else:
        print(f"❌ JSON corpus not found: {json_file}")
    
    # Create a combined dataset (recommended for training)
    print("\n🔄 Creating combined training dataset...")
    
    all_training_texts = []
    
    # Combine sentences if available
    if sentences_file.exists():
        sentences = load_sentences_file(str(sentences_file))
        all_training_texts.extend(sentences)
    
    if all_training_texts:
        print(f"📊 Total combined texts: {len(all_training_texts)}")
        
        # Create final train/val/test split
        final_train, final_val, final_test = split_data(all_training_texts, 0.8, 0.1)
        
        # Create the main training files
        create_jsonl_dataset(final_train, str(output_dir / "train.jsonl"), "combined")
        create_jsonl_dataset(final_val, str(output_dir / "val.jsonl"), "combined")
        create_jsonl_dataset(final_test, str(output_dir / "test.jsonl"), "combined")
        
        print(f"\n✅ Training data prepared!")
        print(f"   📁 Training samples: {len(final_train)}")
        print(f"   📁 Validation samples: {len(final_val)}")
        print(f"   📁 Test samples: {len(final_test)}")
        print(f"   📂 Output directory: {output_dir}")
        
        # Create updated config
        config_content = f"""# Updated config for your prepared corpus data
data:
  train_path: "data/training/train.jsonl"
  val_path: "data/training/val.jsonl"
  test_path: "data/training/test.jsonl"
  batch_size: 16
  num_workers: 4
  max_length: 256

model:
  name: "AmharicHNet"
  hidden_dim: 512
  num_layers: 8
  checkpoint: null

train:
  seed: 1337
  epochs: 15
  lr: 0.0003
  output_dir: "outputs/corpus_training"

# Dataset statistics
dataset_stats:
  total_samples: {len(all_training_texts)}
  train_samples: {len(final_train)}
  val_samples: {len(final_val)}
  test_samples: {len(final_test)}
"""
        
        with open("configs/prepared_corpus.yaml", 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        print(f"   ⚙️ Created config: configs/prepared_corpus.yaml")
        
        print(f"\n🚀 Ready to train! Run:")
        print(f"   PYTHONPATH=src python -m amharichnet.cli train --config configs/prepared_corpus.yaml")
    
    else:
        print("❌ No training data could be prepared. Check your data files.")


if __name__ == "__main__":
    main()