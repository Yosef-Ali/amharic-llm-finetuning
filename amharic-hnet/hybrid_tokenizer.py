import os
import re
import json
import torch
import pickle
import logging
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Union, Any

try:
    import tokenizers
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False
    logging.warning("HuggingFace tokenizers library not found. Install with: pip install tokenizers")


class HybridAmharicTokenizer:
    """Hybrid tokenizer for Amharic that combines subword tokenization with character-level fallback.
    
    This tokenizer first attempts to use BPE (Byte-Pair Encoding) for efficient subword tokenization.
    If a token is unknown or for specific use cases, it falls back to character-level tokenization.
    This approach provides better handling of Amharic's morphological complexity.
    """
    
    def __init__(self, vocab_size: int = 32000, min_frequency: int = 2, 
                 special_tokens: List[str] = None, use_bpe: bool = True):
        """Initialize the tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size for BPE tokenizer
            min_frequency: Minimum frequency for a token to be included in vocabulary
            special_tokens: List of special tokens to add to vocabulary
            use_bpe: Whether to use BPE tokenization (if False, uses character-level only)
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.use_bpe = use_bpe and TOKENIZERS_AVAILABLE
        
        # Default special tokens
        self.special_tokens = [
            "<PAD>",  # Padding token
            "<UNK>",  # Unknown token
            "<BOS>",  # Beginning of sequence token
            "<EOS>",  # End of sequence token
            "<MASK>"  # Mask token for masked language modeling
        ]
        
        # Update with user-provided special tokens if any
        if special_tokens:
            self.special_tokens.extend([token for token in special_tokens if token not in self.special_tokens])
            
        # Initialize tokenizer components
        self.tokenizer = None
        self.char_to_id = {}
        self.id_to_char = {}
        self.word_to_id = {}
        self.id_to_word = {}
        
        # Initialize vocabulary
        self._initialize_vocab()
    
    def _initialize_vocab(self):
        """Initialize vocabulary with special tokens"""
        # Add special tokens to vocabulary
        for i, token in enumerate(self.special_tokens):
            self.word_to_id[token] = i
            self.id_to_word[i] = token
        
        # Initialize character vocabulary with Amharic Unicode range
        # Ethiopic Unicode block: U+1200 to U+137F
        char_idx = len(self.special_tokens)
        for code_point in range(0x1200, 0x1380):
            char = chr(code_point)
            self.char_to_id[char] = char_idx
            self.id_to_char[char_idx] = char
            char_idx += 1
        
        # Add common punctuation and numbers
        for char in ".,;:!?()[]{}፡።፣፤፥፦፧0123456789":
            if char not in self.char_to_id:
                self.char_to_id[char] = char_idx
                self.id_to_char[char_idx] = char
                char_idx += 1
    
    def train_bpe_tokenizer(self, corpus_files: List[str], output_dir: str = None):
        """Train a BPE tokenizer on the given corpus files.
        
        Args:
            corpus_files: List of file paths containing Amharic text for training
            output_dir: Directory to save the trained tokenizer
            
        Returns:
            True if training was successful, False otherwise
        """
        if not self.use_bpe:
            logging.warning("BPE tokenization is disabled or tokenizers library not available")
            return False
        
        if not corpus_files:
            logging.error("No corpus files provided for training")
            return False
        
        # Check if files exist
        for file_path in corpus_files:
            if not os.path.exists(file_path):
                logging.error(f"Corpus file not found: {file_path}")
                return False
        
        try:
            # Initialize a BPE tokenizer
            self.tokenizer = Tokenizer(models.BPE())
            
            # Configure pre-tokenization (split on whitespace and punctuation)
            self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
                pre_tokenizers.WhitespaceSplit(),
                pre_tokenizers.Punctuation()
            ])
            
            # Configure decoder
            self.tokenizer.decoder = decoders.ByteLevel()
            
            # Configure trainer
            trainer = trainers.BpeTrainer(
                vocab_size=self.vocab_size,
                min_frequency=self.min_frequency,
                special_tokens=self.special_tokens,
                show_progress=True
            )
            
            # Train tokenizer
            self.tokenizer.train(files=corpus_files, trainer=trainer)
            
            # Add post-processor for special tokens
            self.tokenizer.post_processor = processors.TemplateProcessing(
                single=f"<BOS> $A <EOS>",
                special_tokens=[
                    ("<BOS>", self.tokenizer.token_to_id("<BOS>")),
                    ("<EOS>", self.tokenizer.token_to_id("<EOS>"))
                ]
            )
            
            # Save tokenizer if output directory is provided
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                self.tokenizer.save(os.path.join(output_dir, "bpe_tokenizer.json"))
                
                # Save vocabulary mapping
                vocab = self.tokenizer.get_vocab()
                with open(os.path.join(output_dir, "vocab.json"), "w", encoding="utf-8") as f:
                    json.dump(vocab, f, ensure_ascii=False, indent=2)
            
            # Update word_to_id and id_to_word with BPE vocabulary
            vocab = self.tokenizer.get_vocab()
            for token, idx in vocab.items():
                if token not in self.word_to_id:
                    self.word_to_id[token] = idx
                    self.id_to_word[idx] = token
            
            logging.info(f"BPE tokenizer trained successfully with vocabulary size: {len(vocab)}")
            return True
            
        except Exception as e:
            logging.error(f"Error training BPE tokenizer: {e}")
            return False
    
    def load_tokenizer(self, tokenizer_path: str):
        """Load a pre-trained tokenizer.
        
        Args:
            tokenizer_path: Path to the tokenizer file or directory
            
        Returns:
            True if loading was successful, False otherwise
        """
        if not self.use_bpe:
            logging.warning("BPE tokenization is disabled or tokenizers library not available")
            return False
        
        try:
            # Check if path exists
            if not os.path.exists(tokenizer_path):
                logging.error(f"Tokenizer path not found: {tokenizer_path}")
                return False
            
            # If directory, look for tokenizer.json
            if os.path.isdir(tokenizer_path):
                tokenizer_file = os.path.join(tokenizer_path, "bpe_tokenizer.json")
                if not os.path.exists(tokenizer_file):
                    logging.error(f"Tokenizer file not found in directory: {tokenizer_path}")
                    return False
                tokenizer_path = tokenizer_file
            
            # Load tokenizer
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
            
            # Load vocabulary
            vocab = self.tokenizer.get_vocab()
            for token, idx in vocab.items():
                self.word_to_id[token] = idx
                self.id_to_word[idx] = token
            
            logging.info(f"Tokenizer loaded successfully with vocabulary size: {len(vocab)}")
            return True
            
        except Exception as e:
            logging.error(f"Error loading tokenizer: {e}")
            return False
    
    def save(self, output_dir: str):
        """Save the tokenizer and vocabulary.
        
        Args:
            output_dir: Directory to save the tokenizer
            
        Returns:
            True if saving was successful, False otherwise
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save BPE tokenizer if available
            if self.use_bpe and self.tokenizer:
                self.tokenizer.save(os.path.join(output_dir, "bpe_tokenizer.json"))
            
            # Save vocabulary mappings
            with open(os.path.join(output_dir, "word_to_id.json"), "w", encoding="utf-8") as f:
                json.dump(self.word_to_id, f, ensure_ascii=False, indent=2)
            
            with open(os.path.join(output_dir, "id_to_word.json"), "w", encoding="utf-8") as f:
                json.dump({str(k): v for k, v in self.id_to_word.items()}, f, ensure_ascii=False, indent=2)
            
            with open(os.path.join(output_dir, "char_to_id.json"), "w", encoding="utf-8") as f:
                json.dump(self.char_to_id, f, ensure_ascii=False, indent=2)
            
            with open(os.path.join(output_dir, "id_to_char.json"), "w", encoding="utf-8") as f:
                json.dump({str(k): v for k, v in self.id_to_char.items()}, f, ensure_ascii=False, indent=2)
            
            # Save configuration
            config = {
                "vocab_size": self.vocab_size,
                "min_frequency": self.min_frequency,
                "special_tokens": self.special_tokens,
                "use_bpe": self.use_bpe
            }
            
            with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            logging.info(f"Tokenizer saved to {output_dir}")
            return True
            
        except Exception as e:
            logging.error(f"Error saving tokenizer: {e}")
            return False
    
    @classmethod
    def from_pretrained(cls, tokenizer_path: str):
        """Load a pre-trained tokenizer from a directory.
        
        Args:
            tokenizer_path: Path to the tokenizer directory
            
        Returns:
            HybridAmharicTokenizer instance
        """
        if not os.path.exists(tokenizer_path):
            raise ValueError(f"Tokenizer path not found: {tokenizer_path}")
        
        # Load configuration
        config_path = os.path.join(tokenizer_path, "config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"Config file not found: {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # Create tokenizer instance
        tokenizer = cls(
            vocab_size=config.get("vocab_size", 32000),
            min_frequency=config.get("min_frequency", 2),
            special_tokens=config.get("special_tokens", None),
            use_bpe=config.get("use_bpe", True)
        )
        
        # Load vocabulary mappings
        word_to_id_path = os.path.join(tokenizer_path, "word_to_id.json")
        if os.path.exists(word_to_id_path):
            with open(word_to_id_path, "r", encoding="utf-8") as f:
                tokenizer.word_to_id = json.load(f)
        
        id_to_word_path = os.path.join(tokenizer_path, "id_to_word.json")
        if os.path.exists(id_to_word_path):
            with open(id_to_word_path, "r", encoding="utf-8") as f:
                id_to_word_str = json.load(f)
                tokenizer.id_to_word = {int(k): v for k, v in id_to_word_str.items()}
        
        char_to_id_path = os.path.join(tokenizer_path, "char_to_id.json")
        if os.path.exists(char_to_id_path):
            with open(char_to_id_path, "r", encoding="utf-8") as f:
                tokenizer.char_to_id = json.load(f)
        
        id_to_char_path = os.path.join(tokenizer_path, "id_to_char.json")
        if os.path.exists(id_to_char_path):
            with open(id_to_char_path, "r", encoding="utf-8") as f:
                id_to_char_str = json.load(f)
                tokenizer.id_to_char = {int(k): v for k, v in id_to_char_str.items()}
        
        # Load BPE tokenizer if available
        if tokenizer.use_bpe and TOKENIZERS_AVAILABLE:
            bpe_tokenizer_path = os.path.join(tokenizer_path, "bpe_tokenizer.json")
            if os.path.exists(bpe_tokenizer_path):
                tokenizer.tokenizer = Tokenizer.from_file(bpe_tokenizer_path)
        
        return tokenizer
    
    def _tokenize_with_bpe(self, text: str) -> List[str]:
        """Tokenize text using BPE tokenizer.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        if not self.tokenizer:
            raise ValueError("BPE tokenizer not initialized. Call train_bpe_tokenizer() or load_tokenizer() first.")
        
        # Use the HuggingFace tokenizers library for BPE tokenization
        encoding = self.tokenizer.encode(text)
        return encoding.tokens
    
    def _tokenize_with_char(self, text: str) -> List[str]:
        """Tokenize text at character level.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of character tokens
        """
        return list(text)
    
    def tokenize(self, text: str, use_bpe: bool = None) -> List[str]:
        """Tokenize text into tokens.
        
        Args:
            text: Text to tokenize
            use_bpe: Whether to use BPE tokenization (overrides instance setting)
            
        Returns:
            List of tokens
        """
        if text is None or text == "":
            return []
        
        # Normalize text
        text = self._normalize_text(text)
        
        # Determine tokenization method
        should_use_bpe = self.use_bpe if use_bpe is None else use_bpe
        
        if should_use_bpe and self.tokenizer:
            return self._tokenize_with_bpe(text)
        else:
            return self._tokenize_with_char(text)
    
    def _normalize_text(self, text: str) -> str:
        """Normalize Amharic text.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Trim whitespace
        text = text.strip()
        
        return text
    
    def encode(self, text: str, add_special_tokens: bool = True, 
               fallback_to_char: bool = True) -> List[int]:
        """Encode text to token IDs.
        
        Args:
            text: Text to encode
            add_special_tokens: Whether to add special tokens (BOS/EOS)
            fallback_to_char: Whether to fall back to character-level encoding for unknown tokens
            
        Returns:
            List of token IDs
        """
        if not text:
            return []
        
        # Normalize text
        text = self._normalize_text(text)
        
        # Try BPE tokenization first if available
        if self.use_bpe and self.tokenizer:
            try:
                # Use the HuggingFace tokenizers library for BPE encoding
                encoding = self.tokenizer.encode(text)
                token_ids = encoding.ids
                
                # If not using special tokens from the tokenizer's post-processor
                if not add_special_tokens:
                    # Check if the first and last tokens are special tokens
                    if len(token_ids) >= 2:
                        bos_id = self.word_to_id.get("<BOS>")
                        eos_id = self.word_to_id.get("<EOS>")
                        
                        if token_ids[0] == bos_id and token_ids[-1] == eos_id:
                            token_ids = token_ids[1:-1]
                
                return token_ids
            except Exception as e:
                logging.warning(f"BPE encoding failed: {e}. Falling back to character-level encoding.")
                if not fallback_to_char:
                    raise
        
        # Fall back to character-level encoding
        token_ids = []
        
        # Add BOS token if requested
        if add_special_tokens and "<BOS>" in self.word_to_id:
            token_ids.append(self.word_to_id["<BOS>"])
        
        # Encode each character
        for char in text:
            if char in self.char_to_id:
                token_ids.append(self.char_to_id[char])
            else:
                # Use UNK token for unknown characters
                token_ids.append(self.word_to_id["<UNK>"])
        
        # Add EOS token if requested
        if add_special_tokens and "<EOS>" in self.word_to_id:
            token_ids.append(self.word_to_id["<EOS>"])
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs to decode
            skip_special_tokens: Whether to skip special tokens in the output
            
        Returns:
            Decoded text
        """
        if not token_ids:
            return ""
        
        # Try BPE decoding first if available
        if self.use_bpe and self.tokenizer:
            try:
                # Use the HuggingFace tokenizers library for BPE decoding
                decoded = self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
                return decoded
            except Exception as e:
                logging.warning(f"BPE decoding failed: {e}. Falling back to manual decoding.")
        
        # Fall back to manual decoding
        special_token_ids = set()
        if skip_special_tokens:
            for token in self.special_tokens:
                if token in self.word_to_id:
                    special_token_ids.add(self.word_to_id[token])
        
        # Decode each token
        decoded_tokens = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in special_token_ids:
                continue
                
            if token_id in self.id_to_word:
                # Word from vocabulary
                decoded_tokens.append(self.id_to_word[token_id])
            elif token_id in self.id_to_char:
                # Character from character vocabulary
                decoded_tokens.append(self.id_to_char[token_id])
            else:
                # Unknown token ID
                decoded_tokens.append("<UNK>")
        
        # Join tokens (for character-level, this reconstructs the text)
        return "".join(decoded_tokens)
    
    def encode_batch(self, texts: List[str], add_special_tokens: bool = True, 
                     fallback_to_char: bool = True) -> List[List[int]]:
        """Encode a batch of texts to token IDs.
        
        Args:
            texts: List of texts to encode
            add_special_tokens: Whether to add special tokens (BOS/EOS)
            fallback_to_char: Whether to fall back to character-level encoding for unknown tokens
            
        Returns:
            List of lists of token IDs
        """
        return [self.encode(text, add_special_tokens, fallback_to_char) for text in texts]
    
    def decode_batch(self, batch_token_ids: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        """Decode a batch of token IDs back to texts.
        
        Args:
            batch_token_ids: List of lists of token IDs to decode
            skip_special_tokens: Whether to skip special tokens in the output
            
        Returns:
            List of decoded texts
        """
        return [self.decode(token_ids, skip_special_tokens) for token_ids in batch_token_ids]
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size.
        
        Returns:
            Size of the vocabulary
        """
        if self.use_bpe and self.tokenizer:
            return len(self.tokenizer.get_vocab())
        return len(self.word_to_id)
    
    def get_pad_token_id(self) -> int:
        """Get the ID of the padding token.
        
        Returns:
            ID of the padding token
        """
        return self.word_to_id.get("<PAD>", 0)
    
    def get_unk_token_id(self) -> int:
        """Get the ID of the unknown token.
        
        Returns:
            ID of the unknown token
        """
        return self.word_to_id.get("<UNK>", 1)
    
    def get_bos_token_id(self) -> int:
        """Get the ID of the beginning of sequence token.
        
        Returns:
            ID of the BOS token
        """
        return self.word_to_id.get("<BOS>", 2)
    
    def get_eos_token_id(self) -> int:
        """Get the ID of the end of sequence token.
        
        Returns:
            ID of the EOS token
        """
        return self.word_to_id.get("<EOS>", 3)


def train_bpe_tokenizer(corpus_files: List[str], output_dir: str, vocab_size: int = 32000, 
                        min_frequency: int = 2, special_tokens: List[str] = None):
    """Utility function to train a BPE tokenizer on Amharic corpus files.
    
    Args:
        corpus_files: List of file paths containing Amharic text for training
        output_dir: Directory to save the trained tokenizer
        vocab_size: Maximum vocabulary size
        min_frequency: Minimum frequency for a token to be included in vocabulary
        special_tokens: List of special tokens to add to vocabulary
        
    Returns:
        Trained HybridAmharicTokenizer instance
    """
    tokenizer = HybridAmharicTokenizer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        use_bpe=True
    )
    
    success = tokenizer.train_bpe_tokenizer(corpus_files, output_dir)
    if not success:
        logging.warning("Failed to train BPE tokenizer. Using character-level tokenizer instead.")
    
    return tokenizer


def main():
    """Main function to demonstrate tokenizer usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train or use a hybrid Amharic tokenizer')
    parser.add_argument('--train', action='store_true', help='Train a new tokenizer')
    parser.add_argument('--corpus', type=str, nargs='+', help='Corpus files for training')
    parser.add_argument('--output-dir', type=str, default='./tokenizer', help='Output directory for tokenizer')
    parser.add_argument('--vocab-size', type=int, default=32000, help='Vocabulary size for BPE tokenizer')
    parser.add_argument('--min-frequency', type=int, default=2, help='Minimum frequency for tokens')
    parser.add_argument('--load', type=str, help='Load a pre-trained tokenizer')
    parser.add_argument('--text', type=str, help='Text to tokenize/encode/decode')
    parser.add_argument('--char-only', action='store_true', help='Use character-level tokenization only')
    
    args = parser.parse_args()
    
    if args.train:
        if not args.corpus:
            parser.error("--train requires --corpus")
        
        print(f"Training tokenizer on {len(args.corpus)} corpus files...")
        tokenizer = HybridAmharicTokenizer(
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            use_bpe=not args.char_only
        )
        
        success = tokenizer.train_bpe_tokenizer(args.corpus, args.output_dir)
        if success:
            print(f"Tokenizer trained successfully and saved to {args.output_dir}")
        else:
            print("Failed to train tokenizer")
    
    elif args.load:
        print(f"Loading tokenizer from {args.load}...")
        tokenizer = HybridAmharicTokenizer.from_pretrained(args.load)
        print(f"Tokenizer loaded with vocabulary size: {tokenizer.get_vocab_size()}")
    
    else:
        tokenizer = HybridAmharicTokenizer(use_bpe=not args.char_only)
        print("Created new tokenizer with default settings")
    
    if args.text:
        print(f"\nInput text: {args.text}")
        
        # Tokenize
        tokens = tokenizer.tokenize(args.text)
        print(f"Tokens: {tokens}")
        
        # Encode
        token_ids = tokenizer.encode(args.text)
        print(f"Token IDs: {token_ids}")
        
        # Decode
        decoded = tokenizer.decode(token_ids)
        print(f"Decoded: {decoded}")


if __name__ == "__main__":
    main()