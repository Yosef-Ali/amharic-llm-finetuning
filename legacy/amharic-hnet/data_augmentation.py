#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data augmentation script for Amharic text.

This script provides functions for augmenting Amharic text data to improve
model training by generating additional samples through various techniques.
"""

import os
import re
import json
import random
import logging
import argparse
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class AmharicDataAugmenter:
    """Data augmenter for Amharic text data."""
    
    def __init__(self, 
                 random_seed: int = 42,
                 char_swap_prob: float = 0.1,
                 word_dropout_prob: float = 0.1,
                 word_swap_prob: float = 0.1,
                 synonym_replace_prob: float = 0.1,
                 max_augmentations_per_text: int = 3):
        """Initialize data augmenter.
        
        Args:
            random_seed: Random seed for reproducibility
            char_swap_prob: Probability of swapping characters
            word_dropout_prob: Probability of dropping words
            word_swap_prob: Probability of swapping words
            synonym_replace_prob: Probability of replacing words with synonyms
            max_augmentations_per_text: Maximum number of augmentations per text
        """
        self.random_seed = random_seed
        self.char_swap_prob = char_swap_prob
        self.word_dropout_prob = word_dropout_prob
        self.word_swap_prob = word_swap_prob
        self.synonym_replace_prob = synonym_replace_prob
        self.max_augmentations_per_text = max_augmentations_per_text
        
        # Set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Load Amharic synonyms if available
        self.synonyms = self._load_synonyms()
    
    def _load_synonyms(self, synonym_file: Optional[str] = None) -> Dict[str, List[str]]:
        """Load Amharic synonyms.
        
        Args:
            synonym_file: Path to synonym file
            
        Returns:
            Dictionary of synonyms
        """
        synonyms = {}
        
        # Try to load synonyms from file
        if synonym_file and os.path.exists(synonym_file):
            try:
                with open(synonym_file, 'r', encoding='utf-8') as f:
                    synonyms = json.load(f)
                logger.info(f"Loaded {len(synonyms)} synonyms from {synonym_file}")
            except Exception as e:
                logger.error(f"Error loading synonyms from {synonym_file}: {e}")
        
        # Add some common Amharic synonyms if file not available
        if not synonyms:
            # These are just examples and should be expanded with actual Amharic synonyms
            synonyms = {
                "ጥሩ": ["ደስ የሚል", "ጥሩ ነው", "ጥሩ ነገር"],
                "ትልቅ": ["ግዙፍ", "ከፍተኛ", "ሰፊ"],
                "ትንሽ": ["አነስተኛ", "ጥቂት", "ትንሽ ነገር"],
                "ቆንጆ": ["ውብ", "ቆንጆ ነው", "ውብ ነው"],
                "ጥቁር": ["ጨለማ", "ጥቁር ነው"],
                "ነጭ": ["ነጣ", "ነጭ ነው"],
                "ቀይ": ["ቀይ ነው", "ቀይ ቀለም"],
                "ሰማያዊ": ["ሰማያዊ ነው", "ሰማያዊ ቀለም"],
                "አረንጓዴ": ["አረንጓዴ ነው", "አረንጓዴ ቀለም"],
                "ቢጫ": ["ቢጫ ነው", "ቢጫ ቀለም"],
                "ሰው": ["ሰውዬ", "ሰውየው", "ሰዎች"],
                "ሴት": ["ሴቷ", "ሴቲቱ", "ሴቶች"],
                "ወንድ": ["ወንዱ", "ወንዶች"],
                "ልጅ": ["ህፃን", "ልጆች", "ልጁ"],
                "አገር": ["ሀገር", "አገሬ", "ሀገሬ"],
                "ከተማ": ["ከተማው", "ከተሞች"],
                "ቤት": ["መኖሪያ", "ቤቱ", "ቤቶች"],
                "መኪና": ["መኪናው", "መኪኖች", "አውቶሞቢል"],
                "ትምህርት": ["ትምህርት ቤት", "መማር"],
                "መጽሐፍ": ["መጽሐፉ", "መጽሐፎች", "መጻህፍት"],
                "ሥራ": ["ሥራው", "ሥራዎች", "ተግባር"],
                "ጊዜ": ["ሰዓት", "ጊዜው", "ጊዜያት"],
                "ቀን": ["ቀኑ", "ቀናት", "ዕለት"],
                "ሌሊት": ["ሌሊቱ", "ሌሊቶች", "ምሽት"],
                "ዓመት": ["ዓመቱ", "ዓመታት"],
                "ወር": ["ወሩ", "ወራት"],
                "ሳምንት": ["ሳምንቱ", "ሳምንታት"],
                "ሰዓት": ["ሰዓቱ", "ሰዓታት"],
                "ደቂቃ": ["ደቂቃው", "ደቂቃዎች"],
                "ሰከንድ": ["ሰከንዱ", "ሰከንዶች"],
            }
            logger.info(f"Using {len(synonyms)} default Amharic synonyms")
        
        return synonyms
    
    def character_swap(self, text: str) -> str:
        """Swap characters in text.
        
        Args:
            text: Text to augment
            
        Returns:
            Augmented text
        """
        if not text or len(text) < 2:
            return text
        
        chars = list(text)
        
        # Find positions to swap
        for i in range(len(chars) - 1):
            # Skip spaces and punctuation
            if chars[i].isspace() or chars[i] in ".,!?;:()[]{}":
                continue
            
            # Swap with probability
            if random.random() < self.char_swap_prob:
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
        
        return ''.join(chars)
    
    def word_dropout(self, text: str) -> str:
        """Drop words from text.
        
        Args:
            text: Text to augment
            
        Returns:
            Augmented text
        """
        if not text:
            return text
        
        words = text.split()
        if len(words) < 2:
            return text
        
        # Drop words with probability
        result = []
        for word in words:
            if random.random() >= self.word_dropout_prob:
                result.append(word)
        
        # Ensure at least one word remains
        if not result:
            result = [random.choice(words)]
        
        return ' '.join(result)
    
    def word_swap(self, text: str) -> str:
        """Swap words in text.
        
        Args:
            text: Text to augment
            
        Returns:
            Augmented text
        """
        if not text:
            return text
        
        words = text.split()
        if len(words) < 2:
            return text
        
        # Swap words with probability
        for i in range(len(words) - 1):
            if random.random() < self.word_swap_prob:
                words[i], words[i + 1] = words[i + 1], words[i]
        
        return ' '.join(words)
    
    def synonym_replace(self, text: str) -> str:
        """Replace words with synonyms.
        
        Args:
            text: Text to augment
            
        Returns:
            Augmented text
        """
        if not text or not self.synonyms:
            return text
        
        words = text.split()
        
        # Replace words with synonyms with probability
        for i in range(len(words)):
            word = words[i]
            
            # Check if word has synonyms
            if word in self.synonyms and random.random() < self.synonym_replace_prob:
                # Replace with random synonym
                words[i] = random.choice(self.synonyms[word])
        
        return ' '.join(words)
    
    def augment_text(self, text: str) -> List[str]:
        """Augment text using multiple techniques.
        
        Args:
            text: Text to augment
            
        Returns:
            List of augmented texts
        """
        if not text:
            return []
        
        augmented_texts = []
        
        # Apply augmentation techniques
        augmented_texts.append(self.character_swap(text))
        augmented_texts.append(self.word_dropout(text))
        augmented_texts.append(self.word_swap(text))
        augmented_texts.append(self.synonym_replace(text))
        
        # Remove duplicates and original text
        augmented_texts = [t for t in augmented_texts if t != text]
        augmented_texts = list(set(augmented_texts))
        
        # Limit number of augmentations
        if len(augmented_texts) > self.max_augmentations_per_text:
            augmented_texts = random.sample(augmented_texts, self.max_augmentations_per_text)
        
        return augmented_texts
    
    def augment_file(self, file_path: Union[str, Path], output_file: Optional[Union[str, Path]] = None) -> List[str]:
        """Augment text file.
        
        Args:
            file_path: Path to text file
            output_file: Path to output file
            
        Returns:
            List of augmented texts
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.warning(f"File {file_path} does not exist")
            return []
        
        # Read file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return []
        
        # Augment lines
        original_texts = [line.strip() for line in lines if line.strip()]
        augmented_texts = []
        
        for text in tqdm(original_texts, desc="Augmenting texts"):
            augmented = self.augment_text(text)
            augmented_texts.extend(augmented)
        
        logger.info(f"Generated {len(augmented_texts)} augmented texts from {len(original_texts)} original texts")
        
        # Save augmented texts
        if output_file:
            output_file = Path(output_file)
            with open(output_file, 'w', encoding='utf-8') as f:
                for text in augmented_texts:
                    f.write(f"{text}\n")
            logger.info(f"Saved augmented texts to {output_file}")
        
        return augmented_texts
    
    def augment_json_file(self, file_path: Union[str, Path], text_field: str = 'text', output_file: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
        """Augment JSON file.
        
        Args:
            file_path: Path to JSON file
            text_field: Field containing text
            output_file: Path to output file
            
        Returns:
            List of augmented items
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.warning(f"File {file_path} does not exist")
            return []
        
        # Read file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return []
        
        # Augment data
        augmented_items = []
        
        # Handle list of objects
        if isinstance(data, list):
            original_items = [item for item in data if isinstance(item, dict) and text_field in item]
            
            for item in tqdm(original_items, desc="Augmenting items"):
                text = item[text_field]
                augmented_texts = self.augment_text(text)
                
                for augmented_text in augmented_texts:
                    # Create new item with augmented text
                    new_item = item.copy()
                    new_item[text_field] = augmented_text
                    new_item['augmented'] = True
                    augmented_items.append(new_item)
        
        # Handle single object
        elif isinstance(data, dict) and text_field in data:
            text = data[text_field]
            augmented_texts = self.augment_text(text)
            
            for augmented_text in augmented_texts:
                # Create new item with augmented text
                new_item = data.copy()
                new_item[text_field] = augmented_text
                new_item['augmented'] = True
                augmented_items.append(new_item)
        
        logger.info(f"Generated {len(augmented_items)} augmented items")
        
        # Save augmented items
        if output_file:
            output_file = Path(output_file)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(augmented_items, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved augmented items to {output_file}")
        
        return augmented_items
    
    def augment_directory(self, 
                         dir_path: Union[str, Path], 
                         output_dir: Optional[Union[str, Path]] = None,
                         file_pattern: str = "*.json",
                         text_field: str = 'text') -> None:
        """Augment directory of files.
        
        Args:
            dir_path: Path to directory
            output_dir: Path to output directory
            file_pattern: Pattern to match files
            text_field: Field containing text (for JSON files)
        """
        dir_path = Path(dir_path)
        if not dir_path.exists() or not dir_path.is_dir():
            logger.error(f"Directory {dir_path} does not exist or is not a directory")
            return
        
        # Create output directory
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get files
        files = list(dir_path.glob(file_pattern))
        logger.info(f"Found {len(files)} files in {dir_path}")
        
        # Process files
        for file in tqdm(files, desc="Augmenting files"):
            # Determine output file
            if output_dir:
                output_file = output_dir / f"augmented_{file.name}"
            else:
                output_file = None
            
            # Augment file
            if file_pattern.endswith('.json'):
                self.augment_json_file(file, text_field=text_field, output_file=output_file)
            else:
                self.augment_file(file, output_file=output_file)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Augment Amharic text data")
    
    # Input arguments
    parser.add_argument("--input_dir", type=str, help="Input directory")
    parser.add_argument("--input_file", type=str, help="Input file")
    parser.add_argument("--file_pattern", type=str, default="*.json", help="File pattern to match")
    parser.add_argument("--text_field", type=str, default="text", help="Field containing text (for JSON files)")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--output_file", type=str, help="Output file")
    
    # Augmentation arguments
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--char_swap_prob", type=float, default=0.1, help="Probability of swapping characters")
    parser.add_argument("--word_dropout_prob", type=float, default=0.1, help="Probability of dropping words")
    parser.add_argument("--word_swap_prob", type=float, default=0.1, help="Probability of swapping words")
    parser.add_argument("--synonym_replace_prob", type=float, default=0.1, help="Probability of replacing words with synonyms")
    parser.add_argument("--max_augmentations_per_text", type=int, default=3, help="Maximum number of augmentations per text")
    parser.add_argument("--synonym_file", type=str, help="Path to synonym file")
    
    args = parser.parse_args()
    
    # Create augmenter
    augmenter = AmharicDataAugmenter(
        random_seed=args.random_seed,
        char_swap_prob=args.char_swap_prob,
        word_dropout_prob=args.word_dropout_prob,
        word_swap_prob=args.word_swap_prob,
        synonym_replace_prob=args.synonym_replace_prob,
        max_augmentations_per_text=args.max_augmentations_per_text
    )
    
    # Load synonyms
    if args.synonym_file:
        augmenter._load_synonyms(args.synonym_file)
    
    # Process input
    if args.input_dir:
        augmenter.augment_directory(
            dir_path=args.input_dir,
            output_dir=args.output_dir,
            file_pattern=args.file_pattern,
            text_field=args.text_field
        )
    elif args.input_file:
        if args.input_file.endswith('.json'):
            augmenter.augment_json_file(
                file_path=args.input_file,
                text_field=args.text_field,
                output_file=args.output_file
            )
        else:
            augmenter.augment_file(
                file_path=args.input_file,
                output_file=args.output_file
            )
    else:
        logger.error("Either --input_dir or --input_file must be specified")


if __name__ == "__main__":
    main()