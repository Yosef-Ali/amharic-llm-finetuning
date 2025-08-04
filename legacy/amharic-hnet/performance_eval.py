#!/usr/bin/env python3
"""
Performance Evaluation Script for Amharic H-Net Model
Step 4: Comprehensive performance metrics and analysis
"""

import torch
import pickle
import time
import os
import json
import numpy as np
from collections import Counter
from enhanced_train import EnhancedHNet, EnhancedAmharicTokenizer

class AmharicModelEvaluator:
    def __init__(self, model_path="models/enhanced_hnet_epoch_1.pt", tokenizer_path="models/enhanced_tokenizer.pkl"):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.evaluation_results = {}
        
    def load_model_components(self):
        """Load model and tokenizer"""
        print("Loading model components...")
        
        # Load tokenizer
        self.tokenizer = EnhancedAmharicTokenizer()
        if os.path.exists(self.tokenizer_path):
            self.tokenizer.load(self.tokenizer_path)
            print(f"✓ Tokenizer loaded: {self.tokenizer.vocab_size} vocab size")
        else:
            print("✗ Tokenizer not found")
            return False
            
        # Load model
        self.model = EnhancedHNet(vocab_size=self.tokenizer.vocab_size)
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            print("✓ Model loaded successfully")
        else:
            print("✗ Model checkpoint not found, using untrained model")
            
        self.model.to(self.device)
        self.model.eval()
        return True
        
    def evaluate_model_architecture(self):
        """Evaluate model architecture and parameters"""
        print("\n" + "="*60)
        print("MODEL ARCHITECTURE EVALUATION")
        print("="*60)
        
        if self.model is None:
            print("Model not loaded")
            return
            
        # Parameter count
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Model size in MB
        model_size_mb = total_params * 4 / (1024 * 1024)
        
        architecture_metrics = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": model_size_mb,
            "vocab_size": self.tokenizer.vocab_size,
            "device": str(self.device)
        }
        
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Model Size: {model_size_mb:.2f} MB")
        print(f"Vocabulary Size: {self.tokenizer.vocab_size}")
        print(f"Device: {self.device}")
        
        self.evaluation_results["architecture"] = architecture_metrics
        
    def evaluate_generation_speed(self):
        """Evaluate text generation speed"""
        print("\n" + "="*60)
        print("GENERATION SPEED EVALUATION")
        print("="*60)
        
        if self.model is None or self.tokenizer is None:
            print("Model or tokenizer not loaded")
            return
            
        test_prompts = ["ሰላም", "ኢትዮጵያ", "አዲስ አበባ"]
        generation_times = []
        
        for prompt in test_prompts:
            start_time = time.time()
            try:
                generated = self.model.generate(
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                    max_length=50,
                    temperature=0.8,
                    device=self.device
                )
                end_time = time.time()
                generation_time = end_time - start_time
                generation_times.append(generation_time)
                
                print(f"Prompt: '{prompt}' | Time: {generation_time:.3f}s | Length: {len(generated)}")
            except Exception as e:
                print(f"Error generating for '{prompt}': {e}")
                
        if generation_times:
            avg_time = np.mean(generation_times)
            print(f"\nAverage Generation Time: {avg_time:.3f}s")
            print(f"Tokens per Second: {50/avg_time:.1f}")
            
            self.evaluation_results["generation_speed"] = {
                "average_time_seconds": avg_time,
                "tokens_per_second": 50/avg_time,
                "individual_times": generation_times
            }
            
    def evaluate_text_quality(self):
        """Evaluate generated text quality"""
        print("\n" + "="*60)
        print("TEXT QUALITY EVALUATION")
        print("="*60)
        
        if self.model is None or self.tokenizer is None:
            print("Model or tokenizer not loaded")
            return
            
        test_cases = [
            {"prompt": "ሰላም", "expected_pattern": "amharic_greeting"},
            {"prompt": "ኢትዮጵያ", "expected_pattern": "country_name"},
            {"prompt": "ትምህርት", "expected_pattern": "education_topic"},
        ]
        
        quality_scores = []
        
        for case in test_cases:
            prompt = case["prompt"]
            print(f"\nTesting: '{prompt}'")
            
            try:
                generated = self.model.generate(
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                    max_length=100,
                    temperature=0.7,
                    device=self.device
                )
                
                # Basic quality metrics
                amharic_char_ratio = self._calculate_amharic_ratio(generated)
                repetition_score = self._calculate_repetition_score(generated)
                length_score = min(len(generated) / 50, 1.0)  # Normalize to 50 chars
                
                quality_score = (amharic_char_ratio + (1 - repetition_score) + length_score) / 3
                quality_scores.append(quality_score)
                
                print(f"Generated: {generated[:100]}...")
                print(f"Amharic Ratio: {amharic_char_ratio:.2f}")
                print(f"Repetition Score: {repetition_score:.2f}")
                print(f"Quality Score: {quality_score:.2f}")
                
            except Exception as e:
                print(f"Error: {e}")
                quality_scores.append(0.0)
                
        if quality_scores:
            avg_quality = np.mean(quality_scores)
            print(f"\nOverall Quality Score: {avg_quality:.2f}/1.00")
            
            self.evaluation_results["text_quality"] = {
                "average_quality_score": avg_quality,
                "individual_scores": quality_scores
            }
            
    def _calculate_amharic_ratio(self, text):
        """Calculate ratio of Amharic characters in text"""
        amharic_chars = 0
        total_chars = len(text)
        
        for char in text:
            # Amharic Unicode range: U+1200-U+137F
            if '\u1200' <= char <= '\u137F':
                amharic_chars += 1
                
        return amharic_chars / total_chars if total_chars > 0 else 0
        
    def _calculate_repetition_score(self, text):
        """Calculate repetition score (higher = more repetitive)"""
        if len(text) < 10:
            return 0
            
        # Count character frequencies
        char_counts = Counter(text)
        total_chars = len(text)
        
        # Calculate entropy-based repetition score
        entropy = 0
        for count in char_counts.values():
            prob = count / total_chars
            if prob > 0:
                entropy -= prob * np.log2(prob)
                
        # Normalize entropy (higher entropy = less repetitive)
        max_entropy = np.log2(min(len(char_counts), total_chars))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        return 1 - normalized_entropy  # Return repetition score
        
    def evaluate_memory_usage(self):
        """Evaluate memory usage"""
        print("\n" + "="*60)
        print("MEMORY USAGE EVALUATION")
        print("="*60)
        
        if self.model is None:
            print("Model not loaded")
            return
            
        # Model memory
        model_memory = sum(p.numel() * p.element_size() for p in self.model.parameters())
        model_memory_mb = model_memory / (1024 * 1024)
        
        print(f"Model Memory Usage: {model_memory_mb:.2f} MB")
        
        # Test generation memory
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        try:
            test_generation = self.model.generate(
                tokenizer=self.tokenizer,
                prompt="ሰላም",
                max_length=100,
                temperature=0.8,
                device=self.device
            )
            
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
                print(f"Peak GPU Memory: {peak_memory:.2f} MB")
                
                self.evaluation_results["memory_usage"] = {
                    "model_memory_mb": model_memory_mb,
                    "peak_gpu_memory_mb": peak_memory
                }
            else:
                self.evaluation_results["memory_usage"] = {
                    "model_memory_mb": model_memory_mb,
                    "peak_gpu_memory_mb": 0
                }
                
        except Exception as e:
            print(f"Error during memory evaluation: {e}")
            
    def save_evaluation_report(self, output_path="evaluation_report.json"):
        """Save evaluation results to file"""
        print(f"\nSaving evaluation report to {output_path}...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_results, f, indent=2, ensure_ascii=False)
            
        print("✓ Evaluation report saved")
        
    def run_full_evaluation(self):
        """Run complete evaluation suite"""
        print("Starting Comprehensive Model Evaluation...")
        print("="*80)
        
        if not self.load_model_components():
            print("Failed to load model components")
            return
            
        # Run all evaluations
        self.evaluate_model_architecture()
        self.evaluate_generation_speed()
        self.evaluate_text_quality()
        self.evaluate_memory_usage()
        
        # Save results
        self.save_evaluation_report()
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        return self.evaluation_results

def main():
    """Main evaluation function"""
    evaluator = AmharicModelEvaluator()
    results = evaluator.run_full_evaluation()
    
    # Print summary
    if results:
        print("\n📊 EVALUATION SUMMARY:")
        if "architecture" in results:
            arch = results["architecture"]
            print(f"   • Model Size: {arch['model_size_mb']:.1f} MB")
            print(f"   • Parameters: {arch['total_parameters']:,}")
            
        if "generation_speed" in results:
            speed = results["generation_speed"]
            print(f"   • Generation Speed: {speed['tokens_per_second']:.1f} tokens/sec")
            
        if "text_quality" in results:
            quality = results["text_quality"]
            print(f"   • Text Quality: {quality['average_quality_score']:.2f}/1.00")
            
        if "memory_usage" in results:
            memory = results["memory_usage"]
            print(f"   • Memory Usage: {memory['model_memory_mb']:.1f} MB")

if __name__ == "__main__":
    main()