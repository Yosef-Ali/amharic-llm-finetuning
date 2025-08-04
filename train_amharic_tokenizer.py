#!/usr/bin/env python3
"""
Train the Amharic tokenizer on your 30k+ corpus and integrate with training pipeline.
"""

import json
import sys
from pathlib import Path
sys.path.append('src')

from amharichnet.data.amharic_tokenizer import create_amharic_tokenizer


def load_corpus_texts():
    """Load all your corpus texts for tokenizer training."""
    texts = []
    
    # Load from sentences file
    sentences_file = Path("data/processed/processed_articles/amharic_sentences.txt")
    if sentences_file.exists():
        print(f"📖 Loading sentences from {sentences_file}")
        with open(sentences_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
        print(f"   ✅ Loaded {len(texts)} sentences")
    
    # Load from JSONL training data
    train_file = Path("data/training/train.jsonl")
    if train_file.exists():
        print(f"📖 Loading training data from {train_file}")
        jsonl_texts = []
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'text' in data:
                        jsonl_texts.append(data['text'])
                except:
                    continue
        print(f"   ✅ Loaded {len(jsonl_texts)} training texts")
        texts.extend(jsonl_texts)
    
    # Load from corpus file
    corpus_file = Path("data/processed/processed_articles/amharic_corpus.txt")
    if corpus_file.exists():
        print(f"📖 Loading corpus from {corpus_file}")
        with open(corpus_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Split on article breaks
            articles = content.split('--- ARTICLE BREAK ---')
            corpus_texts = [article.strip() for article in articles if article.strip()]
        print(f"   ✅ Loaded {len(corpus_texts)} articles")
        texts.extend(corpus_texts)
    
    print(f"🎯 Total texts for tokenizer training: {len(texts)}")
    return texts


def main():
    print("🚀 Training Amharic Tokenizer on Your 30k+ Corpus")
    print("=" * 50)
    
    # Load your corpus
    corpus_texts = load_corpus_texts()
    
    if not corpus_texts:
        print("❌ No corpus texts found! Check your data paths.")
        return
    
    # Create and train tokenizer
    print(f"\n🔤 Training tokenizer with vocab_size=8000...")
    tokenizer = create_amharic_tokenizer(
        corpus_texts=corpus_texts, 
        vocab_size=8000
    )
    
    # Save the trained tokenizer
    output_dir = Path("models/tokenizer")
    output_dir.mkdir(exist_ok=True)
    
    vocab_path = output_dir / "amharic_vocab.json"
    tokenizer.save_vocab(str(vocab_path))
    print(f"💾 Saved tokenizer vocabulary to: {vocab_path}")
    
    # Test the trained tokenizer
    print(f"\n🧪 Testing trained tokenizer:")
    test_texts = [
        "ሰላም ዓለም! እንዴት ነህ?",
        "ኢትዮጵያ የአፍሪካ አገር ነች።",
        "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ነች።"
    ]
    
    for i, test_text in enumerate(test_texts, 1):
        encoded = tokenizer.encode(test_text, max_len=32)
        decoded = tokenizer.decode(encoded)
        
        print(f"\n   Test {i}:")
        print(f"   Original: {test_text}")
        print(f"   Encoded:  {encoded[:15]}...")
        print(f"   Decoded:  {decoded}")
    
    # Show vocabulary statistics
    print(f"\n📊 Tokenizer Statistics:")
    print(f"   - Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"   - Trained on {len(corpus_texts)} texts")
    print(f"   - Special tokens: {len(tokenizer.special_tokens)}")
    
    # Create updated training config
    print(f"\n⚙️ Creating updated training config...")
    
    config_content = f"""# Updated config with trained Amharic tokenizer
data:
  train_path: "data/training/train.jsonl"
  val_path: "data/training/val.jsonl"
  tokenizer: "amharic"               # Use new Amharic tokenizer
  tokenizer_path: "models/tokenizer/amharic_vocab.json"
  batch_size: 8
  num_workers: 2
  max_length: 128

model:
  name: "AmharicHNet"
  vocab_size: {tokenizer.get_vocab_size()}    # Actual vocab size from tokenizer
  hidden_dim: 256
  num_layers: 4
  num_heads: 8
  dropout: 0.1
  max_seq_len: 128
  checkpoint: null

train:
  seed: 1337
  epochs: 100
  lr: 0.0001                        # Higher LR with better tokenization
  weight_decay: 0.01
  precision: "fp16"
  device: "auto"
  output_dir: "outputs/amharic_tokenized_training"

# Tokenizer info
tokenizer_stats:
  vocab_size: {tokenizer.get_vocab_size()}
  trained_on_texts: {len(corpus_texts)}
  special_tokens: {len(tokenizer.special_tokens)}
"""
    
    config_path = Path("configs/amharic_tokenized.yaml")
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"✅ Created training config: {config_path}")
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Update training pipeline to use new tokenizer")
    print(f"   2. Run training with: PYTHONPATH=src python -m amharichnet.cli train --config configs/amharic_tokenized.yaml")
    print(f"   3. Compare results with previous character-level training")
    
    print(f"\n🎉 Amharic tokenizer training complete!")
    print(f"   Expected improvement: Loss should drop from ~9.0 to ~3-5 range")


if __name__ == "__main__":
    main()