"""
Advanced Amharic Tokenizer with subword support.
This replaces the basic character tokenizer with proper Amharic language processing.
"""

from __future__ import annotations
from typing import List, Dict, Set, Optional, Tuple
import re
import json
from pathlib import Path


class AmharicTextNormalizer:
    """Normalize Amharic text for better tokenization."""
    
    def __init__(self):
        # Amharic Unicode ranges
        self.amharic_range = (0x1200, 0x137F)  # Ethiopic block
        self.amharic_supplement = (0x1380, 0x139F)  # Ethiopic Supplement
        
        # Common Amharic punctuation normalization
        self.punct_map = {
            '፡': '.',  # Ethiopic full stop
            '፤': ';',  # Ethiopic semicolon  
            '፥': ':',  # Ethiopic colon
            '፦': ':',  # Ethiopic preface colon
            '፧': '?',  # Ethiopic question mark
            '፨': '!',  # Ethiopic exclamation mark
        }
    
    def is_amharic_char(self, char: str) -> bool:
        """Check if character is Amharic."""
        code = ord(char)
        return (self.amharic_range[0] <= code <= self.amharic_range[1] or
                self.amharic_supplement[0] <= code <= self.amharic_supplement[1])
    
    def normalize_punctuation(self, text: str) -> str:
        """Normalize Amharic punctuation to standard forms."""
        for amh_punct, std_punct in self.punct_map.items():
            text = text.replace(amh_punct, std_punct)
        return text
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize Amharic text."""
        # Basic cleaning
        text = text.strip()
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize punctuation
        text = self.normalize_punctuation(text)
        
        # Remove non-Amharic, non-ASCII characters (keep some punctuation)
        cleaned_chars = []
        for char in text:
            if (self.is_amharic_char(char) or 
                char.isascii() or 
                char in ' .,!?;:-()[]{}"\'\n\t'):
                cleaned_chars.append(char)
        
        return ''.join(cleaned_chars)


class AmharicSubwordTokenizer:
    """Subword tokenizer optimized for Amharic language."""
    
    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.normalizer = AmharicTextNormalizer()
        
        # Special tokens
        self.special_tokens = {
            "<pad>": 0,
            "<unk>": 1, 
            "<bos>": 2,  # Beginning of sequence
            "<eos>": 3,  # End of sequence
            "<mask>": 4,
        }
        
        # Initialize vocabulary
        self.vocab: Dict[str, int] = self.special_tokens.copy()
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.special_tokens.items()}
        self._next_id = len(self.special_tokens)
        
        # Amharic syllable patterns (common combinations)
        self.amharic_syllables = self._build_amharic_syllables()
        
        # Subword patterns
        self.subwords: Dict[str, int] = {}
        self.word_frequencies: Dict[str, int] = {}
        
    def _build_amharic_syllables(self) -> List[str]:
        """Build common Amharic syllable patterns."""
        # Basic Amharic syllables (this is a simplified version)
        # In a full implementation, you'd have comprehensive syllable patterns
        base_chars = ['ሀ', 'ለ', 'ሐ', 'መ', 'ሠ', 'ረ', 'ሰ', 'ቀ', 'በ', 'ተ', 'ቸ', 
                     'ሐ', 'ነ', 'ኘ', 'አ', 'ከ', 'ወ', 'ዐ', 'ዘ', 'ዠ', 'የ', 'ደ', 
                     'ጀ', 'ገ', 'ጠ', 'ጨ', 'ጰ', 'ጸ', 'ፀ', 'ፈ', 'ፐ']
        
        # Common syllable endings/modifications
        syllable_patterns = []
        for base in base_chars[:10]:  # Use subset for demo
            # Add base character and common variations
            syllable_patterns.extend([base, base + 'ን', base + 'ም', base + 'ት'])
        
        return syllable_patterns
    
    def _add_to_vocab(self, token: str) -> int:
        """Add token to vocabulary."""
        if token not in self.vocab:
            self.vocab[token] = self._next_id
            self.id_to_token[self._next_id] = token
            self._next_id += 1
        return self.vocab[token]
    
    def train_on_text(self, texts: List[str]) -> None:
        """Train tokenizer on corpus texts."""
        print(f"🔤 Training Amharic tokenizer on {len(texts)} texts...")
        
        # Step 1: Collect word frequencies
        word_freq = {}
        for text in texts:
            cleaned = self.normalizer.clean_text(text)
            words = cleaned.split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Step 2: Add common Amharic syllables
        for syllable in self.amharic_syllables:
            self._add_to_vocab(syllable)
        
        # Step 3: Add frequent words (up to vocab limit)
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        added_words = 0
        
        for word, freq in sorted_words:
            if len(self.vocab) >= self.vocab_size:
                break
                
            # Add word if it's common enough and not too long
            if freq >= 3 and len(word) <= 15 and word not in self.vocab:
                self._add_to_vocab(word)
                added_words += 1
        
        # Step 4: Add character fallbacks for remaining Amharic characters
        for text in texts[:100]:  # Sample to avoid overprocessing
            for char in text:
                if (self.normalizer.is_amharic_char(char) and 
                    char not in self.vocab and
                    len(self.vocab) < self.vocab_size):
                    self._add_to_vocab(char)
        
        print(f"✅ Tokenizer trained:")
        print(f"   - Total vocabulary: {len(self.vocab)}")
        print(f"   - Words added: {added_words}")
        print(f"   - Amharic syllables: {len(self.amharic_syllables)}")
    
    def _tokenize_word(self, word: str) -> List[str]:
        """Tokenize a single word using subword approach."""
        if word in self.vocab:
            return [word]
        
        # Try to break into known subwords/syllables
        tokens = []
        i = 0
        while i < len(word):
            # Try longest match first
            found = False
            for length in range(min(8, len(word) - i), 0, -1):
                subword = word[i:i+length]
                if subword in self.vocab:
                    tokens.append(subword)
                    i += length
                    found = True
                    break
            
            if not found:
                # Fallback to character level
                char = word[i]
                if char in self.vocab:
                    tokens.append(char)
                else:
                    tokens.append("<unk>")
                i += 1
        
        return tokens
    
    def encode(self, text: str, max_len: int = 128, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        # Clean and normalize
        cleaned = self.normalizer.clean_text(text)
        
        # Tokenize
        words = cleaned.split()
        all_tokens = []
        
        if add_special_tokens:
            all_tokens.append("<bos>")
        
        for word in words:
            word_tokens = self._tokenize_word(word)
            all_tokens.extend(word_tokens)
        
        if add_special_tokens:
            all_tokens.append("<eos>")
        
        # Convert to IDs
        token_ids = []
        for token in all_tokens:
            token_ids.append(self.vocab.get(token, self.vocab["<unk>"]))
        
        # Truncate or pad
        token_ids = token_ids[:max_len]
        while len(token_ids) < max_len:
            token_ids.append(self.vocab["<pad>"])
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        tokens = []
        for id in token_ids:
            if id == self.vocab["<pad>"]:
                break
            token = self.id_to_token.get(id, "<unk>")
            if token not in ["<bos>", "<eos>", "<pad>"]:
                tokens.append(token)
        
        # Join tokens (basic reconstruction)
        text = " ".join(tokens)
        return text
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)
    
    def save_vocab(self, path: str) -> None:
        """Save vocabulary to file."""
        vocab_data = {
            "vocab": self.vocab,
            "vocab_size": len(self.vocab),
            "special_tokens": self.special_tokens
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    
    def load_vocab(self, path: str) -> None:
        """Load vocabulary from file."""
        with open(path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.vocab = vocab_data["vocab"]
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self._next_id = len(self.vocab)


# Factory function for compatibility
def create_amharic_tokenizer(corpus_texts: Optional[List[str]] = None, 
                           vocab_size: int = 8000) -> AmharicSubwordTokenizer:
    """Create and optionally train an Amharic tokenizer."""
    tokenizer = AmharicSubwordTokenizer(vocab_size=vocab_size)
    
    if corpus_texts:
        tokenizer.train_on_text(corpus_texts)
    
    return tokenizer


# Example usage and testing
if __name__ == "__main__":
    # Test the tokenizer
    tokenizer = AmharicSubwordTokenizer(vocab_size=1000)
    
    # Sample Amharic texts for training
    sample_texts = [
        "ሰላም ዓለም! እንዴት ነህ?",
        "ኢትዮጵያ የአፍሪካ አገር ነች።",
        "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ነች።",
        "አማርኛ የኢትዮጵያ ቋንቋ ነው።"
    ]
    
    tokenizer.train_on_text(sample_texts)
    
    # Test encoding/decoding
    test_text = "ሰላም ዓለም! እንዴት ነህ?"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original: {test_text}")
    print(f"Encoded: {encoded[:10]}...")
    print(f"Decoded: {decoded}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")