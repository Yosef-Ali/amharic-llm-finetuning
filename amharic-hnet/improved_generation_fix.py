#!/usr/bin/env python3
"""
Improved Generation with Repetition Fixes
Implements the recommendations from deep analysis to solve repetition issues
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
from typing import List, Optional, Tuple
import time

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_train import EnhancedHNet, EnhancedAmharicTokenizer

class ImprovedAmharicGenerator:
    def __init__(self, model_path: str = "models/enhanced_hnet/best_model.pt", 
                 tokenizer_path: str = "models/enhanced_tokenizer.pkl"):
        """Initialize the improved generator"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model(model_path, tokenizer_path)
        
    def load_model(self, model_path: str, tokenizer_path: str):
        """Load model and tokenizer"""
        try:
            # Load tokenizer
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
            
            print("✅ Improved generator loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def generate_with_repetition_penalty(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.5,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        word_repetition_penalty: float = 1.5,
        sequence_repetition_penalty: float = 2.0,
        min_length: int = 20
    ) -> str:
        """Generate text with advanced repetition penalties"""
        
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        if not input_ids:
            input_ids = [0]  # Start token
        
        input_tensor = torch.tensor([input_ids], device=self.device)
        generated_ids = input_ids.copy()
        
        # Track generated tokens for repetition penalty
        token_counts = {}
        word_history = []
        sequence_history = []
        hidden = None  # Initialize hidden state
        
        with torch.no_grad():
            for step in range(max_length):
                # Get model predictions
                logits, hidden = self.model(input_tensor, hidden)
                logits = logits[:, -1, :]  # Get last token logits
                
                # Apply temperature
                logits = logits / temperature
                
                # Apply repetition penalties
                logits = self._apply_repetition_penalties(
                    logits, generated_ids, token_counts, word_history, sequence_history,
                    repetition_penalty, word_repetition_penalty, sequence_repetition_penalty
                )
                
                # Apply nucleus sampling (top-p)
                next_token_id = self._nucleus_sampling(logits, top_p)
                
                # Add to generated sequence
                generated_ids.append(next_token_id)
                
                # Update input tensor
                input_tensor = torch.cat([
                    input_tensor, 
                    torch.tensor([[next_token_id]], device=self.device)
                ], dim=1)
                
                # Update tracking
                token_counts[next_token_id] = token_counts.get(next_token_id, 0) + 1
                
                # Update word and sequence history
                self._update_history(next_token_id, word_history, sequence_history)
                
                # Check for early stopping
                if step >= min_length and self._should_stop(generated_ids):
                    break
        
        # Decode and return
        generated_text = self.tokenizer.decode(generated_ids)
        return generated_text
    
    def _apply_repetition_penalties(
        self, 
        logits: torch.Tensor, 
        generated_ids: List[int],
        token_counts: dict,
        word_history: List[str],
        sequence_history: List[str],
        repetition_penalty: float,
        word_repetition_penalty: float,
        sequence_repetition_penalty: float
    ) -> torch.Tensor:
        """Apply various repetition penalties"""
        
        # 1. Token-level repetition penalty
        for token_id, count in token_counts.items():
            if count > 0:
                penalty = repetition_penalty ** count
                logits[0, token_id] = logits[0, token_id] / penalty
        
        # 2. Recent token penalty (stronger for very recent tokens)
        recent_window = min(10, len(generated_ids))
        for i, token_id in enumerate(generated_ids[-recent_window:]):
            distance_penalty = 1.0 + (recent_window - i) * 0.1
            penalty = repetition_penalty * distance_penalty
            logits[0, token_id] = logits[0, token_id] / penalty
        
        # 3. Sequence repetition penalty
        if len(generated_ids) >= 3:
            # Check for repeating sequences
            for seq_len in range(2, min(6, len(generated_ids) // 2)):
                if len(generated_ids) >= seq_len * 2:
                    recent_seq = generated_ids[-seq_len:]
                    prev_seq = generated_ids[-seq_len*2:-seq_len]
                    
                    if recent_seq == prev_seq:
                        # Penalize tokens that would continue the repetition
                        for token_id in recent_seq:
                            logits[0, token_id] = logits[0, token_id] / sequence_repetition_penalty
        
        return logits
    
    def _nucleus_sampling(self, logits: torch.Tensor, top_p: float) -> int:
        """Apply nucleus (top-p) sampling"""
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        
        # Calculate cumulative probabilities
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Set logits to -inf for tokens to remove
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[0, indices_to_remove] = float('-inf')
        
        # Sample from the filtered distribution
        probs = F.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1).item()
        
        return next_token_id
    
    def _update_history(self, token_id: int, word_history: List[str], sequence_history: List[str]):
        """Update word and sequence history for tracking"""
        # Convert token to character
        char = self.tokenizer.decode([token_id])
        
        # Update sequence history (keep last 20 characters)
        sequence_history.append(char)
        if len(sequence_history) > 20:
            sequence_history.pop(0)
        
        # Update word history (simple space-based splitting)
        if char == ' ' or char in '።፣፤፥፦፧፨':
            if sequence_history:
                word = ''.join(sequence_history[:-1]).strip()
                if word:
                    word_history.append(word)
                    if len(word_history) > 10:
                        word_history.pop(0)
                sequence_history.clear()
    
    def _should_stop(self, generated_ids: List[int]) -> bool:
        """Determine if generation should stop"""
        # Stop if we hit a natural ending
        if len(generated_ids) > 0:
            last_char = self.tokenizer.decode([generated_ids[-1]])
            if last_char in '።!':
                return True
        
        return False
    
    def generate_multiple_samples(
        self, 
        prompt: str, 
        num_samples: int = 5,
        **kwargs
    ) -> List[Tuple[str, dict]]:
        """Generate multiple samples with quality scoring"""
        samples = []
        
        for i in range(num_samples):
            # Vary parameters slightly for diversity
            temp = kwargs.get('temperature', 1.5) + (i - num_samples//2) * 0.1
            temp = max(0.5, min(2.0, temp))  # Clamp temperature
            
            generated = self.generate_with_repetition_penalty(
                prompt=prompt,
                temperature=temp,
                **{k: v for k, v in kwargs.items() if k != 'temperature'}
            )
            
            # Score the generation
            score = self._score_generation(generated, prompt)
            
            samples.append((generated, {
                'temperature': temp,
                'score': score,
                'length': len(generated)
            }))
        
        # Sort by score (higher is better)
        samples.sort(key=lambda x: x[1]['score'], reverse=True)
        return samples
    
    def _score_generation(self, text: str, prompt: str) -> float:
        """Score generation quality"""
        if len(text) < 10:
            return 0.0
        
        score = 1.0
        
        # 1. Amharic character ratio (higher is better)
        amharic_chars = sum(1 for c in text if '\u1200' <= c <= '\u137F')
        amharic_ratio = amharic_chars / len(text)
        score *= amharic_ratio
        
        # 2. Repetition penalty (lower repetition is better)
        words = text.split()
        if len(words) > 1:
            unique_ratio = len(set(words)) / len(words)
            score *= unique_ratio
        
        # 3. Length bonus (reasonable length is better)
        length_score = min(1.0, len(text) / 50)  # Optimal around 50 chars
        score *= length_score
        
        # 4. Diversity bonus
        unique_chars = len(set(text))
        diversity_score = min(1.0, unique_chars / 20)  # Good diversity around 20 unique chars
        score *= diversity_score
        
        return score

def demo_improved_generation():
    """Demonstrate the improved generation"""
    print("🚀 Improved Amharic Generation Demo")
    print("Implementing repetition penalties and nucleus sampling")
    print("="*60)
    
    try:
        generator = ImprovedAmharicGenerator()
        
        test_prompts = [
            "ሰላም",
            "ትምህርት", 
            "ቤተሰብ",
            "ኢትዮጵያ",
            "ባህል"
        ]
        
        for prompt in test_prompts:
            print(f"\n📝 Prompt: '{prompt}'")
            print("-" * 40)
            
            # Generate multiple samples
            samples = generator.generate_multiple_samples(
                prompt=prompt,
                num_samples=3,
                max_length=80,
                temperature=1.5,
                top_p=0.9,
                repetition_penalty=1.3,
                word_repetition_penalty=1.8,
                sequence_repetition_penalty=2.5
            )
            
            for i, (text, info) in enumerate(samples):
                print(f"\n{i+1}. Generated (Score: {info['score']:.3f}, T: {info['temperature']:.1f}):")
                print(f"   '{text}'")
                print(f"   Length: {info['length']} chars")
        
        print("\n" + "="*60)
        print("✅ Improved generation demo complete!")
        print("Notice the reduced repetition and better quality.")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("Make sure the model files exist and training is complete.")

if __name__ == "__main__":
    demo_improved_generation()