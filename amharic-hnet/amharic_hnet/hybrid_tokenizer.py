# Amharic H-Net: Hybrid Tokenizer Implementation

import os
import json
import regex as re
from typing import Dict, List, Optional, Tuple, Union, Set

import torch
from transformers import PreTrainedTokenizer
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors, trainers
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing


class AmharicTextPreprocessor:
    """Preprocessor for Amharic text to improve tokenization quality."""
    
    def __init__(
        self,
        remove_non_amharic: bool = True,
        normalize_spaces: bool = True,
        normalize_punctuation: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_numbers: bool = False,
        min_chars: int = 2,
        max_chars: int = 1000,
    ):
        """Initialize the preprocessor with the specified options.
        
        Args:
            remove_non_amharic: Whether to remove non-Amharic characters
            normalize_spaces: Whether to normalize spaces
            normalize_punctuation: Whether to normalize punctuation
            remove_urls: Whether to remove URLs
            remove_emails: Whether to remove email addresses
            remove_numbers: Whether to remove numbers
            min_chars: Minimum number of characters for a valid text
            max_chars: Maximum number of characters for a valid text
        """
        self.remove_non_amharic = remove_non_amharic
        self.normalize_spaces = normalize_spaces
        self.normalize_punctuation = normalize_punctuation
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_numbers = remove_numbers
        self.min_chars = min_chars
        self.max_chars = max_chars
        
        # Amharic Unicode range: 0x1200-0x137F (Ethiopic)
        self.amharic_pattern = re.compile(r'[^\u1200-\u137F\s\p{P}]')
        
        # URL pattern
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        
        # Email pattern
        self.email_pattern = re.compile(r'\S+@\S+\.[a-zA-Z]{2,}')
        
        # Number pattern
        self.number_pattern = re.compile(r'\d+')
        
        # Multiple spaces pattern
        self.spaces_pattern = re.compile(r'\s+')
        
        # Punctuation normalization mapping
        self.punct_map = {
            '«': '"',
            '»': '"',
            ''': "'",
            ''': "'",
            '"': '"',
            '„': '"',
            '‟': '"',
            '‹': '<',
            '›': '>',
            '–': '-',
            '—': '-',
            '―': '-',
            '…': '...'
        }
        
    def clean_text(self, text: str) -> str:
        """Clean the text according to the specified options.
        
        Args:
            text: The text to clean
            
        Returns:
            The cleaned text
        """
        if not text or len(text) < self.min_chars:
            return ""
            
        if len(text) > self.max_chars:
            text = text[:self.max_chars]
            
        # Remove URLs
        if self.remove_urls:
            text = self.url_pattern.sub(' ', text)
            
        # Remove emails
        if self.remove_emails:
            text = self.email_pattern.sub(' ', text)
            
        # Remove numbers
        if self.remove_numbers:
            text = self.number_pattern.sub(' ', text)
            
        # Remove non-Amharic characters
        if self.remove_non_amharic:
            text = self.amharic_pattern.sub(' ', text)
            
        # Normalize punctuation
        if self.normalize_punctuation:
            for orig, repl in self.punct_map.items():
                text = text.replace(orig, repl)
                
        # Normalize spaces
        if self.normalize_spaces:
            text = self.spaces_pattern.sub(' ', text)
            text = text.strip()
            
        return text
        
    def batch_clean(self, texts: List[str]) -> List[str]:
        """Clean a batch of texts.
        
        Args:
            texts: List of texts to clean
            
        Returns:
            List of cleaned texts
        """
        return [self.clean_text(text) for text in texts]


class HybridAmharicTokenizer(PreTrainedTokenizer):
    """Hybrid tokenizer for Amharic text that combines character-level and subword tokenization.
    
    This tokenizer is designed to handle the morphological complexity of Amharic by using
    a combination of character-level tokenization for rare words and subword tokenization
    for common words and patterns.
    """
    
    vocab_files_names = {"vocab_file": "vocab.json", "merges_file": "merges.txt"}
    model_input_names = ["input_ids", "attention_mask"]
    
    def __init__(
        self,
        vocab_file=None,
        merges_file=None,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        **kwargs
    ):
        """Initialize the tokenizer.
        
        Args:
            vocab_file: Path to the vocabulary file
            merges_file: Path to the merges file for BPE
            unk_token: Unknown token
            sep_token: Separator token
            pad_token: Padding token
            cls_token: Classification token
            mask_token: Mask token for masked language modeling
        """
        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )
        
        # Initialize the text preprocessor
        self.preprocessor = AmharicTextPreprocessor()
        
        # Initialize the tokenizer backend
        if vocab_file is not None and os.path.isfile(vocab_file):
            self._tokenizer = self._create_tokenizer_from_files(vocab_file, merges_file)
        else:
            # Initialize with a minimal vocabulary for special tokens
            self._tokenizer = Tokenizer(WordPiece(unk_token=unk_token))
            self._tokenizer.pre_tokenizer = Whitespace()
            self._tokenizer.decoder = decoders.WordPiece()
            
            # Add special tokens
            special_tokens = [unk_token, sep_token, pad_token, cls_token, mask_token]
            self._tokenizer.add_special_tokens(special_tokens)
        
        # Set up token mappings
        self._update_special_tokens()
        
    def _create_tokenizer_from_files(self, vocab_file, merges_file):
        """Create a tokenizer from vocabulary and merges files.
        
        Args:
            vocab_file: Path to the vocabulary file
            merges_file: Path to the merges file
            
        Returns:
            A Tokenizer instance
        """
        tokenizer = Tokenizer(WordPiece(vocab_file, unk_token=self.unk_token))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.decoder = decoders.WordPiece()
        
        # Add post-processor for special tokens
        tokenizer.post_processor = TemplateProcessing(
            single=f"{self.cls_token} $A {self.sep_token}",
            pair=f"{self.cls_token} $A {self.sep_token} $B {self.sep_token}",
            special_tokens=[
                (self.cls_token, self.convert_tokens_to_ids(self.cls_token)),
                (self.sep_token, self.convert_tokens_to_ids(self.sep_token)),
            ],
        )
        
        return tokenizer
        
    def _update_special_tokens(self):
        """Update the special token mappings."""
        self.unk_token_id = self.convert_tokens_to_ids(self.unk_token)
        self.sep_token_id = self.convert_tokens_to_ids(self.sep_token)
        self.pad_token_id = self.convert_tokens_to_ids(self.pad_token)
        self.cls_token_id = self.convert_tokens_to_ids(self.cls_token)
        self.mask_token_id = self.convert_tokens_to_ids(self.mask_token)
        
    def _tokenize(self, text):
        """Tokenize a string.
        
        Args:
            text: Text to tokenize
            
        Returns:
            A list of tokens
        """
        # Preprocess the text
        text = self.preprocessor.clean_text(text)
        
        # Tokenize using the backend tokenizer
        tokens = self._tokenizer.encode(text).tokens
        
        return tokens
        
    def _convert_token_to_id(self, token):
        """Convert a token to an ID using the vocabulary.
        
        Args:
            token: Token to convert
            
        Returns:
            The ID of the token
        """
        return self._tokenizer.token_to_id(token) or self.unk_token_id
        
    def _convert_id_to_token(self, index):
        """Convert an ID to a token using the vocabulary.
        
        Args:
            index: ID to convert
            
        Returns:
            The token corresponding to the ID
        """
        return self._tokenizer.id_to_token(index) or self.unk_token
        
    def convert_tokens_to_string(self, tokens):
        """Convert a sequence of tokens to a single string.
        
        Args:
            tokens: List of tokens to convert
            
        Returns:
            The string representation of the tokens
        """
        return self._tokenizer.decode(self.convert_tokens_to_ids(tokens))
        
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """Build model inputs from a sequence by adding special tokens.
        
        Args:
            token_ids_0: List of IDs for the first sequence
            token_ids_1: Optional list of IDs for the second sequence
            
        Returns:
            List of IDs with special tokens added
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        else:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id] + token_ids_1 + [self.sep_token_id]
            
    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """Get a mask indicating special tokens.
        
        Args:
            token_ids_0: List of IDs for the first sequence
            token_ids_1: Optional list of IDs for the second sequence
            already_has_special_tokens: Whether the input already has special tokens
            
        Returns:
            A mask with 1 for special tokens and 0 for other tokens
        """
        if already_has_special_tokens:
            return [1 if token_id in [self.cls_token_id, self.sep_token_id, self.pad_token_id] else 0 for token_id in token_ids_0]
        
        if token_ids_1 is None:
            return [1] + [0] * len(token_ids_0) + [1]
        else:
            return [1] + [0] * len(token_ids_0) + [1] + [0] * len(token_ids_1) + [1]
            
    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        """Create token type IDs from sequences.
        
        Args:
            token_ids_0: List of IDs for the first sequence
            token_ids_1: Optional list of IDs for the second sequence
            
        Returns:
            List of token type IDs
        """
        if token_ids_1 is None:
            return [0] * (len(token_ids_0) + 2)
        else:
            return [0] * (len(token_ids_0) + 1) + [1] * (len(token_ids_1) + 1)
            
    def save_vocabulary(self, save_directory, filename_prefix=None):
        """Save the tokenizer vocabulary to a file.
        
        Args:
            save_directory: Directory to save the vocabulary
            filename_prefix: Optional prefix for the vocabulary files
            
        Returns:
            Tuple of saved vocabulary files
        """
        prefix = f"{filename_prefix}-" if filename_prefix else ""
        vocab_file = os.path.join(save_directory, f"{prefix}vocab.json")
        merges_file = os.path.join(save_directory, f"{prefix}merges.txt")
        
        # Save the vocabulary
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self._tokenizer.get_vocab(), f, ensure_ascii=False)
            
        # Save an empty merges file for compatibility
        with open(merges_file, "w", encoding="utf-8") as f:
            f.write("")
            
        return (vocab_file, merges_file)
        
    @classmethod
    def from_pretrained(cls, pretrained_model_path, **kwargs):
        """Load a tokenizer from a pretrained model.
        
        Args:
            pretrained_model_path: Path to the pretrained model directory
            **kwargs: Additional arguments to pass to the tokenizer
            
        Returns:
            A HybridAmharicTokenizer instance
        """
        vocab_file = os.path.join(pretrained_model_path, "vocab.json")
        merges_file = os.path.join(pretrained_model_path, "merges.txt")
        
        # Check if the vocabulary file exists
        if not os.path.isfile(vocab_file):
            raise ValueError(f"Vocabulary file not found at {vocab_file}")
            
        # Create the tokenizer
        tokenizer = cls(vocab_file=vocab_file, merges_file=merges_file, **kwargs)
        
        return tokenizer
        
    def train(self, files, vocab_size=30000, min_frequency=2, special_tokens=None):
        """Train the tokenizer on a corpus.
        
        Args:
            files: List of files to train on
            vocab_size: Size of the vocabulary
            min_frequency: Minimum frequency for a token to be included
            special_tokens: List of special tokens to add to the vocabulary
            
        Returns:
            The trained tokenizer
        """
        # Initialize a new tokenizer for training
        tokenizer = Tokenizer(WordPiece(unk_token=self.unk_token))
        tokenizer.pre_tokenizer = Whitespace()
        
        # Add special tokens if provided
        if special_tokens is None:
            special_tokens = [self.unk_token, self.sep_token, self.pad_token, self.cls_token, self.mask_token]
        
        # Create a trainer
        trainer = trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens
        )
        
        # Train the tokenizer
        tokenizer.train(files, trainer)
        
        # Update the internal tokenizer
        self._tokenizer = tokenizer
        self._update_special_tokens()
        
        return self