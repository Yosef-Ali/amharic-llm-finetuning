#!/usr/bin/env python3
"""
Comprehensive Performance Analysis for Enhanced H-Net
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from collections import Counter
import json

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_train import EnhancedHNet, EnhancedAmharicTokenizer, AmharicHNetDataset

class PerformanceAnalyzer:
    def __init__(self, model_path, tokenizer_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = EnhancedAmharicTokenizer()
        self.tokenizer.load(tokenizer_path)
        
        # Load model
        self.model = EnhancedHNet(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=256,
            hidden_dim=512,
            num_layers=3,
            dropout=0.2
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.checkpoint = checkpoint
        
    def analyze_training_metrics(self):
        """Analyze training performance metrics"""
        print("📊 TRAINING PERFORMANCE ANALYSIS")
        print("=" * 50)
        
        # Training info
        print(f"🔢 Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"📚 Vocabulary Size: {self.tokenizer.vocab_size}")
        print(f"💾 Device: {self.device}")
        print(f"📈 Training Epoch: {self.checkpoint.get('epoch', 'N/A')}")
        print(f"📉 Final Validation Loss: {self.checkpoint.get('val_loss', 'N/A'):.6f}")
        
        # Model architecture
        print(f"\n🏗️ MODEL ARCHITECTURE:")
        print(f"   • Embedding Dimension: 256")
        print(f"   • Hidden Dimension: 512") 
        print(f"   • LSTM Layers: 3 (bidirectional)")
        print(f"   • Attention Heads: 8")
        print(f"   • Dropout: 0.2")
        
        # Training curves
        if 'train_losses' in self.checkpoint and 'val_losses' in self.checkpoint:
            train_losses = self.checkpoint['train_losses']
            val_losses = self.checkpoint['val_losses']
            
            print(f"\n📈 TRAINING PROGRESS:")
            print(f"   • Initial Train Loss: {train_losses[0]:.6f}")
            print(f"   • Final Train Loss: {train_losses[-1]:.6f}")
            print(f"   • Loss Reduction: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.2f}%")
            print(f"   • Best Validation Loss: {min(val_losses):.6f}")
            
    def test_generation_quality(self):
        """Test text generation quality with various prompts"""
        print(f"\n🎯 TEXT GENERATION QUALITY TEST")
        print("=" * 50)
        
        test_cases = [
            {
                "category": "Geography", 
                "prompts": ["ኢትዮጵያ", "አዲስ አበባ", "ላሊበላ", "ጎንደር", "አክሱም"]
            },
            {
                "category": "Culture",
                "prompts": ["ባህል", "ቡና", "እንጀራ", "ሙዚቃ", "ዳንስ"]
            },
            {
                "category": "Education", 
                "prompts": ["ትምህርት", "ዩኒቨርሲቲ", "ሳይንስ", "ተማሪ", "መምህር"]
            },
            {
                "category": "Technology",
                "prompts": ["ኮምፒዩተር", "ኢንተርኔት", "ሞባይል", "ቴክኖሎጂ", "ፕሮግራም"]
            }
        ]
        
        generation_results = {}
        
        for test_case in test_cases:
            category = test_case["category"]
            prompts = test_case["prompts"]
            
            print(f"\n📝 {category} Generation:")
            print("-" * 30)
            
            category_results = []
            
            for prompt in prompts:
                try:
                    start_time = time.time()
                    generated = self.model.generate(
                        tokenizer=self.tokenizer,
                        prompt=prompt,
                        max_length=50,
                        temperature=0.7,
                        device=self.device
                    )
                    generation_time = time.time() - start_time
                    
                    # Calculate metrics
                    amharic_ratio = self._calculate_amharic_ratio(generated)
                    coherence_score = self._calculate_coherence_score(generated)
                    
                    result = {
                        "prompt": prompt,
                        "generated": generated,
                        "generation_time": generation_time,
                        "amharic_ratio": amharic_ratio,
                        "coherence_score": coherence_score,
                        "length": len(generated)
                    }
                    
                    category_results.append(result)
                    
                    print(f"   • Prompt: '{prompt}'")
                    print(f"     Generated: '{generated[:80]}{'...' if len(generated) > 80 else ''}'")
                    print(f"     Time: {generation_time:.3f}s | Amharic: {amharic_ratio:.1%} | Length: {len(generated)}")
                    
                except Exception as e:
                    print(f"   ❌ Failed for '{prompt}': {e}")
                    
            generation_results[category] = category_results
        
        return generation_results
    
    def test_inference_speed(self):
        """Test model inference speed"""
        print(f"\n⚡ INFERENCE SPEED TEST")
        print("=" * 50)
        
        # Test different sequence lengths
        test_lengths = [10, 25, 50, 100]
        speed_results = {}
        
        for length in test_lengths:
            times = []
            for _ in range(10):  # Average over 10 runs
                start_time = time.time()
                try:
                    generated = self.model.generate(
                        tokenizer=self.tokenizer,
                        prompt="ኢትዮጵያ",
                        max_length=length,
                        temperature=0.7,
                        device=self.device
                    )
                    times.append(time.time() - start_time)
                except Exception as e:
                    print(f"Error at length {length}: {e}")
                    break
            
            if times:
                avg_time = np.mean(times)
                tokens_per_sec = length / avg_time if avg_time > 0 else 0
                speed_results[length] = {
                    "avg_time": avg_time,
                    "tokens_per_sec": tokens_per_sec,
                    "std_time": np.std(times)
                }
                
                print(f"📏 Length {length:3d}: {avg_time:.3f}s ± {np.std(times):.3f}s ({tokens_per_sec:.1f} tokens/sec)")
        
        return speed_results
    
    def analyze_vocabulary_coverage(self):
        """Analyze vocabulary and character coverage"""
        print(f"\n📚 VOCABULARY ANALYSIS")
        print("=" * 50)
        
        # Load corpus for analysis
        corpus_path = "processed_articles/amharic_corpus.txt"
        if os.path.exists(corpus_path):
            with open(corpus_path, 'r', encoding='utf-8') as f:
                corpus_text = f.read()
            
            # Character frequency analysis
            char_freq = Counter(corpus_text)
            total_chars = len(corpus_text)
            
            print(f"📄 Corpus Statistics:")
            print(f"   • Total Characters: {total_chars:,}")
            print(f"   • Unique Characters: {len(char_freq)}")
            print(f"   • Vocabulary Coverage: {self.tokenizer.vocab_size} / {len(char_freq)} ({self.tokenizer.vocab_size/len(char_freq)*100:.1f}%)")
            
            # Most common characters
            print(f"\n🔤 Top 20 Characters:")
            for i, (char, freq) in enumerate(char_freq.most_common(20)):
                if char in self.tokenizer.char_to_idx:
                    status = "✅"
                else:
                    status = "❌"
                print(f"   {i+1:2d}. '{char}' ({freq:,}, {freq/total_chars*100:.2f}%) {status}")
            
            # Amharic script analysis
            amharic_chars = sum(1 for char in corpus_text if '\u1200' <= char <= '\u137F')
            amharic_ratio = amharic_chars / total_chars
            print(f"\n🇪🇹 Amharic Script Analysis:")
            print(f"   • Amharic Characters: {amharic_chars:,} ({amharic_ratio:.1%})")
            print(f"   • Other Characters: {total_chars - amharic_chars:,} ({(1-amharic_ratio):.1%})")
    
    def test_model_confidence(self):
        """Test model confidence and uncertainty"""
        print(f"\n🎯 MODEL CONFIDENCE ANALYSIS")
        print("=" * 50)
        
        test_prompts = ["ኢትዮጵያ", "አዲስ አበባ", "ባህል", "ትምህርት"]
        
        for prompt in test_prompts:
            with torch.no_grad():
                # Encode prompt
                input_ids = self.tokenizer.encode(prompt)
                input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
                
                # Get model predictions
                logits, _ = self.model(input_tensor)
                probs = F.softmax(logits, dim=-1)
                
                # Calculate confidence metrics
                max_probs = torch.max(probs, dim=-1)[0]
                avg_confidence = torch.mean(max_probs).item()
                min_confidence = torch.min(max_probs).item()
                max_confidence = torch.max(max_probs).item()
                
                # Entropy (uncertainty measure)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                avg_entropy = torch.mean(entropy).item()
                
                print(f"📝 Prompt: '{prompt}'")
                print(f"   • Avg Confidence: {avg_confidence:.3f}")
                print(f"   • Min/Max Confidence: {min_confidence:.3f} / {max_confidence:.3f}")
                print(f"   • Avg Entropy: {avg_entropy:.3f}")
    
    def _calculate_amharic_ratio(self, text):
        """Calculate ratio of Amharic characters in text"""
        if not text:
            return 0.0
        amharic_chars = sum(1 for char in text if '\u1200' <= char <= '\u137F')
        return amharic_chars / len(text)
    
    def _calculate_coherence_score(self, text):
        """Simple coherence score based on repeated patterns"""
        if not text or len(text) < 5:
            return 0.0
        
        # Penalize excessive repetition
        unique_chars = len(set(text))
        repetition_penalty = unique_chars / len(text)
        
        # Reward proper word boundaries (spaces)
        space_ratio = text.count(' ') / len(text)
        space_score = min(space_ratio * 10, 1.0)  # Cap at 1.0
        
        # Reward sentence endings
        sentence_endings = text.count('።') + text.count('.') + text.count('!')
        sentence_score = min(sentence_endings / 5, 1.0)  # Cap at 1.0
        
        return (repetition_penalty + space_score + sentence_score) / 3
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        print("🔍 ENHANCED H-NET PERFORMANCE REPORT")
        print("=" * 60)
        
        # Basic metrics
        self.analyze_training_metrics()
        
        # Generation quality
        generation_results = self.test_generation_quality()
        
        # Speed analysis
        speed_results = self.test_inference_speed()
        
        # Vocabulary analysis
        self.analyze_vocabulary_coverage()
        
        # Confidence analysis
        self.test_model_confidence()
        
        # Summary
        print(f"\n📋 PERFORMANCE SUMMARY")
        print("=" * 50)
        
        # Calculate overall metrics
        all_results = []
        for category_results in generation_results.values():
            all_results.extend(category_results)
        
        if all_results:
            avg_amharic_ratio = np.mean([r['amharic_ratio'] for r in all_results])
            avg_coherence = np.mean([r['coherence_score'] for r in all_results])
            avg_gen_time = np.mean([r['generation_time'] for r in all_results])
            avg_length = np.mean([r['length'] for r in all_results])
            
            print(f"✅ Text Generation:")
            print(f"   • Average Amharic Ratio: {avg_amharic_ratio:.1%}")
            print(f"   • Average Coherence Score: {avg_coherence:.3f}")
            print(f"   • Average Generation Time: {avg_gen_time:.3f}s")
            print(f"   • Average Output Length: {avg_length:.1f} characters")
        
        if speed_results:
            max_speed = max(r['tokens_per_sec'] for r in speed_results.values())
            print(f"⚡ Inference Speed:")
            print(f"   • Max Speed: {max_speed:.1f} tokens/second")
            print(f"   • Device: {self.device}")
        
        print(f"\n🎯 Model Status: {'✅ READY FOR PRODUCTION' if avg_amharic_ratio > 0.8 else '⚠️ NEEDS IMPROVEMENT'}")

def main():
    """Main performance analysis"""
    model_path = "models/enhanced_hnet/best_model.pt"
    tokenizer_path = "models/enhanced_tokenizer.pkl"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return
        
    if not os.path.exists(tokenizer_path):
        print(f"❌ Tokenizer not found at {tokenizer_path}")
        return
    
    analyzer = PerformanceAnalyzer(model_path, tokenizer_path)
    analyzer.generate_performance_report()

if __name__ == "__main__":
    main()