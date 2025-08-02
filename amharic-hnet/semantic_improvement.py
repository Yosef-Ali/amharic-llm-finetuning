#!/usr/bin/env python3
"""
Semantic Improvement for Amharic H-Net Model
Addresses the issue of generating Amharic characters without meaningful content
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
from typing import List, Optional, Tuple, Dict
import re

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_train import EnhancedHNet, EnhancedAmharicTokenizer

class SemanticAmharicGenerator:
    def __init__(self, model_path: str = "models/enhanced_hnet/best_model.pt", 
                 tokenizer_path: str = "models/enhanced_tokenizer.pkl"):
        """Initialize the semantic generator with word-aware improvements"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model(model_path, tokenizer_path)
        
        # Common Amharic word patterns and structures
        self.amharic_word_patterns = [
            # Common prefixes
            'በ', 'ለ', 'ከ', 'ወደ', 'እስከ', 'ስለ', 'ያለ',
            # Common suffixes  
            'ች', 'ዎች', 'ቶች', 'ዎች', 'ነት', 'ነው', 'ናል',
            # Common word roots
            'ሰላም', 'ቤት', 'ሰው', 'ልጅ', 'እናት', 'አባት', 'ወንድም', 'እህት',
            'ትምህርት', 'ሥራ', 'ገንዘብ', 'ጊዜ', 'ቦታ', 'ሀገር', 'ከተማ',
            'ባህል', 'ቋንቋ', 'ታሪክ', 'ሕዝብ', 'መንግስት', 'ፖለቲካ'
        ]
        
        # Amharic sentence connectors
        self.sentence_connectors = ['እና', 'ነገር ግን', 'ስለዚህ', 'ከዚህ በፊት', 'በተጨማሪ']
        
        # Amharic punctuation
        self.amharic_punctuation = ['።', '፣', '፤', '፥', '፦', '፧', '፨']
        
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
            
            print("✅ Semantic generator loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def generate_semantic_text(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.2,
        top_p: float = 0.85,
        word_boundary_bonus: float = 1.5,
        semantic_coherence_penalty: float = 0.8
    ) -> str:
        """Generate semantically coherent Amharic text"""
        
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        if not input_ids:
            input_ids = [0]  # Start token
        
        input_tensor = torch.tensor([input_ids], device=self.device)
        generated_ids = input_ids.copy()
        
        # Track for semantic coherence
        current_word = ""
        generated_words = []
        last_was_space = False
        hidden = None
        
        with torch.no_grad():
            for step in range(max_length):
                # Get model predictions
                logits, hidden = self.model(input_tensor, hidden)
                logits = logits[:, -1, :]  # Get last token logits
                
                # Apply temperature
                logits = logits / temperature
                
                # Apply semantic improvements
                logits = self._apply_semantic_bonuses(
                    logits, current_word, generated_words, last_was_space,
                    word_boundary_bonus, semantic_coherence_penalty
                )
                
                # Apply nucleus sampling
                next_token_id = self._nucleus_sampling(logits, top_p)
                
                # Decode the character
                next_char = self.tokenizer.decode([next_token_id])
                
                # Update word tracking
                if next_char == ' ' or next_char in self.amharic_punctuation:
                    if current_word.strip():
                        generated_words.append(current_word.strip())
                    current_word = ""
                    last_was_space = True
                else:
                    current_word += next_char
                    last_was_space = False
                
                # Add to generated sequence
                generated_ids.append(next_token_id)
                
                # Update input tensor
                input_tensor = torch.cat([
                    input_tensor, 
                    torch.tensor([[next_token_id]], device=self.device)
                ], dim=1)
                
                # Check for natural stopping
                if self._should_stop_semantic(generated_words, next_char):
                    break
        
        # Decode and post-process
        generated_text = self.tokenizer.decode(generated_ids)
        return self._post_process_text(generated_text)
    
    def _apply_semantic_bonuses(
        self, 
        logits: torch.Tensor, 
        current_word: str,
        generated_words: List[str],
        last_was_space: bool,
        word_boundary_bonus: float,
        semantic_coherence_penalty: float
    ) -> torch.Tensor:
        """Apply semantic bonuses to encourage meaningful text"""
        
        # 1. Word boundary bonus - encourage spaces after reasonable word lengths
        if len(current_word) >= 3 and not last_was_space:
            space_token_id = self.tokenizer.char_to_idx.get(' ', -1)
            if space_token_id != -1:
                logits[0, space_token_id] *= word_boundary_bonus
        
        # 2. Punctuation bonus at appropriate intervals
        if len(generated_words) >= 3 and not last_was_space:
            for punct in ['።', '፣']:
                punct_id = self.tokenizer.char_to_idx.get(punct, -1)
                if punct_id != -1:
                    logits[0, punct_id] *= 1.3
        
        # 3. Encourage common Amharic character combinations
        if current_word:
            last_char = current_word[-1]
            # Boost probability of characters that commonly follow current character
            common_combinations = self._get_common_combinations(last_char)
            for char, boost in common_combinations.items():
                char_id = self.tokenizer.char_to_idx.get(char, -1)
                if char_id != -1:
                    logits[0, char_id] *= boost
        
        # 4. Penalize excessive repetition within current word
        if len(current_word) >= 2:
            last_chars = current_word[-2:]
            for char in last_chars:
                char_id = self.tokenizer.char_to_idx.get(char, -1)
                if char_id != -1:
                    logits[0, char_id] *= semantic_coherence_penalty
        
        return logits
    
    def _get_common_combinations(self, char: str) -> Dict[str, float]:
        """Get common character combinations in Amharic"""
        combinations = {
            # Common combinations based on Amharic phonology
            'ል': {'ም': 1.2, 'ን': 1.2, 'ት': 1.1, 'ክ': 1.1},
            'ም': {'ህ': 1.3, 'ር': 1.2, 'ን': 1.1, 'ስ': 1.1},
            'ር': {'ት': 1.3, 'ስ': 1.2, 'ን': 1.1, 'ክ': 1.1},
            'ት': {'ም': 1.2, 'ር': 1.2, 'ን': 1.1, 'ክ': 1.1},
            'ን': {'ት': 1.2, 'ግ': 1.1, 'ድ': 1.1, 'ስ': 1.1},
            'ስ': {'ት': 1.2, 'ር': 1.1, 'ላ': 1.1, 'ም': 1.1},
            'ክ': {'ር': 1.2, 'ት': 1.1, 'ን': 1.1, 'ል': 1.1},
            'ግ': {'ር': 1.2, 'ን': 1.1, 'ዛ': 1.1, 'ብ': 1.1},
            'ብ': {'ር': 1.2, 'ት': 1.1, 'ን': 1.1, 'ል': 1.1},
            'ድ': {'ር': 1.2, 'ን': 1.1, 'ግ': 1.1, 'ል': 1.1}
        }
        
        return combinations.get(char, {})
    
    def _nucleus_sampling(self, logits: torch.Tensor, top_p: float) -> int:
        """Apply nucleus (top-p) sampling"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[0, indices_to_remove] = float('-inf')
        
        probs = F.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1).item()
        
        return next_token_id
    
    def _should_stop_semantic(self, generated_words: List[str], last_char: str) -> bool:
        """Determine if generation should stop based on semantic completeness"""
        # Stop if we have a reasonable number of words and end with punctuation
        if len(generated_words) >= 5 and last_char in ['።', '!', '?']:
            return True
        
        # Stop if we have many words (avoid overly long generation)
        if len(generated_words) >= 15:
            return True
            
        return False
    
    def _post_process_text(self, text: str) -> str:
        """Post-process generated text for better readability"""
        # Remove excessive spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Ensure proper spacing around punctuation
        for punct in self.amharic_punctuation:
            text = re.sub(f'\s*{re.escape(punct)}\s*', punct + ' ', text)
        
        # Remove trailing spaces
        text = text.strip()
        
        # Ensure text ends with proper punctuation
        if text and text[-1] not in self.amharic_punctuation:
            text += '።'
        
        return text
    
    def analyze_semantic_quality(self, text: str) -> Dict:
        """Analyze the semantic quality of generated text"""
        words = text.split()
        
        # Word-level analysis
        avg_word_length = np.mean([len(word.strip('።፣፤፥፦፧፨')) for word in words]) if words else 0
        
        # Check for known Amharic patterns
        known_patterns = sum(1 for word in words if any(pattern in word for pattern in self.amharic_word_patterns))
        pattern_ratio = known_patterns / len(words) if words else 0
        
        # Punctuation analysis
        punct_count = sum(1 for char in text if char in self.amharic_punctuation)
        punct_ratio = punct_count / len(text) if text else 0
        
        # Word boundary analysis
        space_count = text.count(' ')
        word_boundary_ratio = space_count / len(text) if text else 0
        
        return {
            'avg_word_length': avg_word_length,
            'pattern_ratio': pattern_ratio,
            'punct_ratio': punct_ratio,
            'word_boundary_ratio': word_boundary_ratio,
            'total_words': len(words),
            'semantic_score': (pattern_ratio + word_boundary_ratio + min(punct_ratio * 10, 1)) / 3
        }

def demo_semantic_improvement():
    """Demonstrate semantic improvements"""
    print("🧠 Semantic Amharic Generation Demo")
    print("Focusing on meaningful word structures and coherence")
    print("=" * 60)
    
    try:
        generator = SemanticAmharicGenerator()
        
        test_prompts = [
            "ባህል",
            "ሰላም", 
            "ትምህርት",
            "ኢትዮጵያ",
            "ቤተሰብ"
        ]
        
        for prompt in test_prompts:
            print(f"\n📝 Prompt: '{prompt}'")
            print("-" * 40)
            
            # Generate with semantic improvements
            generated = generator.generate_semantic_text(
                prompt=prompt,
                max_length=80,
                temperature=1.2,
                top_p=0.85,
                word_boundary_bonus=1.5,
                semantic_coherence_penalty=0.8
            )
            
            print(f"Generated: '{generated}'")
            
            # Analyze quality
            analysis = generator.analyze_semantic_quality(generated)
            print(f"Quality Analysis:")
            print(f"  - Average word length: {analysis['avg_word_length']:.1f}")
            print(f"  - Pattern recognition: {analysis['pattern_ratio']:.2f}")
            print(f"  - Word boundaries: {analysis['word_boundary_ratio']:.2f}")
            print(f"  - Semantic score: {analysis['semantic_score']:.3f}")
            print(f"  - Total words: {analysis['total_words']}")
        
        print("\n" + "="*60)
        print("✅ Semantic improvement demo complete!")
        print("Notice the improved word structure and coherence.")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_semantic_improvement()