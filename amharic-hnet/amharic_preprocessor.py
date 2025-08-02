#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data preprocessing script for Amharic text.

This script provides functions for cleaning, normalizing, and preparing
Amharic text data for training language models.
"""

import os
import re
import json
import logging
import argparse
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import pandas as pd
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class AmharicTextPreprocessor:
    """Preprocessor for Amharic text data."""
    
    def __init__(self, 
                 remove_non_amharic: bool = True,
                 normalize_spaces: bool = True,
                 normalize_punctuation: bool = True,
                 remove_urls: bool = True,
                 remove_emails: bool = True,
                 remove_numbers: bool = False,
                 min_length: int = 10,
                 max_length: Optional[int] = None):
        """Initialize preprocessor.
        
        Args:
            remove_non_amharic: Whether to remove non-Amharic characters
            normalize_spaces: Whether to normalize spaces
            normalize_punctuation: Whether to normalize punctuation
            remove_urls: Whether to remove URLs
            remove_emails: Whether to remove email addresses
            remove_numbers: Whether to remove numbers
            min_length: Minimum text length to keep
            max_length: Maximum text length to keep
        """
        self.remove_non_amharic = remove_non_amharic
        self.normalize_spaces = normalize_spaces
        self.normalize_punctuation = normalize_punctuation
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_numbers = remove_numbers
        self.min_length = min_length
        self.max_length = max_length
        
        # Compile regular expressions
        self._compile_regexes()
    
    def _compile_regexes(self):
        """Compile regular expressions for text cleaning."""
        # Amharic Unicode range: U+1200 to U+137F
        self.amharic_regex = re.compile(r'[^\u1200-\u137F\s.,!?;:"\'\'\(\)\[\]\{\}\-–—]')
        
        # URLs
        self.url_regex = re.compile(r'https?://\S+|www\.\S+')
        
        # Email addresses
        self.email_regex = re.compile(r'\S+@\S+\.[a-zA-Z]{2,}')
        
        # Numbers
        self.number_regex = re.compile(r'\d+')
        
        # Multiple spaces
        self.space_regex = re.compile(r'\s+')
        
        # Punctuation normalization
        self.punct_map = {
            '፡፡': '።',  # Normalize double colon to Ethiopian full stop
            '፥': '፣',   # Normalize colon to Ethiopian comma
            '…': '...',  # Normalize ellipsis
            ''': '\'',   # Normalize quotes
            ''': '\'',   # Normalize quotes
            '"': '\"',  # Normalize quotes
            '-': '-',    # Normalize hyphens
            '–': '-',    # Normalize en dash
            '—': '-',    # Normalize em dash
        }
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing unwanted characters and normalizing.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Remove URLs
        if self.remove_urls:
            text = self.url_regex.sub(' ', text)
        
        # Remove email addresses
        if self.remove_emails:
            text = self.email_regex.sub(' ', text)
        
        # Remove numbers
        if self.remove_numbers:
            text = self.number_regex.sub(' ', text)
        
        # Remove non-Amharic characters
        if self.remove_non_amharic:
            text = self.amharic_regex.sub(' ', text)
        
        # Normalize punctuation
        if self.normalize_punctuation:
            for old, new in self.punct_map.items():
                text = text.replace(old, new)
        
        # Normalize spaces
        if self.normalize_spaces:
            text = self.space_regex.sub(' ', text)
            text = text.strip()
        
        return text
    
    def filter_text(self, text: str) -> Optional[str]:
        """Filter text based on length.
        
        Args:
            text: Text to filter
            
        Returns:
            Filtered text or None if it doesn't meet criteria
        """
        if not text:
            return None
        
        # Check minimum length
        if self.min_length and len(text) < self.min_length:
            return None
        
        # Check maximum length
        if self.max_length and len(text) > self.max_length:
            return None
        
        return text
    
    def process_text(self, text: str) -> Optional[str]:
        """Process text by cleaning and filtering.
        
        Args:
            text: Text to process
            
        Returns:
            Processed text or None if it doesn't meet criteria
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Filter text
        return self.filter_text(cleaned_text)
    
    def process_file(self, file_path: Union[str, Path]) -> List[str]:
        """Process text file.
        
        Args:
            file_path: Path to text file
            
        Returns:
            List of processed texts
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
        
        # Process lines
        processed_lines = []
        for line in lines:
            processed_line = self.process_text(line)
            if processed_line:
                processed_lines.append(processed_line)
        
        return processed_lines
    
    def process_json_file(self, file_path: Union[str, Path], text_field: str = 'text') -> List[str]:
        """Process JSON file.
        
        Args:
            file_path: Path to JSON file
            text_field: Field containing text
            
        Returns:
            List of processed texts
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
        
        # Process data
        processed_texts = []
        
        # Handle list of objects
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and text_field in item:
                    processed_text = self.process_text(item[text_field])
                    if processed_text:
                        processed_texts.append(processed_text)
        
        # Handle single object
        elif isinstance(data, dict) and text_field in data:
            processed_text = self.process_text(data[text_field])
            if processed_text:
                processed_texts.append(processed_text)
        
        return processed_texts
    
    def process_directory(self, 
                         dir_path: Union[str, Path], 
                         output_dir: Optional[Union[str, Path]] = None,
                         file_pattern: str = "*.json",
                         text_field: str = 'text',
                         output_format: str = 'txt',
                         num_workers: int = 1) -> None:
        """Process directory of files.
        
        Args:
            dir_path: Path to directory
            output_dir: Path to output directory
            file_pattern: Pattern to match files
            text_field: Field containing text (for JSON files)
            output_format: Output format ('txt' or 'json')
            num_workers: Number of workers for parallel processing
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
        if num_workers > 1:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                if file_pattern.endswith('.json'):
                    process_func = partial(self.process_json_file, text_field=text_field)
                else:
                    process_func = self.process_file
                
                results = list(tqdm(executor.map(process_func, files), total=len(files)))
        else:
            # Sequential processing
            results = []
            for file in tqdm(files):
                if file_pattern.endswith('.json'):
                    result = self.process_json_file(file, text_field=text_field)
                else:
                    result = self.process_file(file)
                results.append(result)
        
        # Flatten results
        all_texts = [text for result in results for text in result]
        logger.info(f"Processed {len(all_texts)} texts")
        
        # Save results
        if output_dir:
            if output_format == 'txt':
                output_file = output_dir / 'processed_texts.txt'
                with open(output_file, 'w', encoding='utf-8') as f:
                    for text in all_texts:
                        f.write(f"{text}\n")
                logger.info(f"Saved processed texts to {output_file}")
            elif output_format == 'json':
                output_file = output_dir / 'processed_texts.json'
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(all_texts, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved processed texts to {output_file}")
            else:
                logger.error(f"Unknown output format: {output_format}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Preprocess Amharic text data")
    
    # Input arguments
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory")
    parser.add_argument("--file_pattern", type=str, default="*.json", help="File pattern to match")
    parser.add_argument("--text_field", type=str, default="text", help="Field containing text (for JSON files)")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--output_format", type=str, default="txt", choices=["txt", "json"], help="Output format")
    
    # Preprocessing arguments
    parser.add_argument("--remove_non_amharic", action="store_true", help="Remove non-Amharic characters")
    parser.add_argument("--normalize_spaces", action="store_true", help="Normalize spaces")
    parser.add_argument("--normalize_punctuation", action="store_true", help="Normalize punctuation")
    parser.add_argument("--remove_urls", action="store_true", help="Remove URLs")
    parser.add_argument("--remove_emails", action="store_true", help="Remove email addresses")
    parser.add_argument("--remove_numbers", action="store_true", help="Remove numbers")
    parser.add_argument("--min_length", type=int, default=10, help="Minimum text length to keep")
    parser.add_argument("--max_length", type=int, help="Maximum text length to keep")
    
    # Other arguments
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for parallel processing")
    
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = AmharicTextPreprocessor(
        remove_non_amharic=args.remove_non_amharic,
        normalize_spaces=args.normalize_spaces,
        normalize_punctuation=args.normalize_punctuation,
        remove_urls=args.remove_urls,
        remove_emails=args.remove_emails,
        remove_numbers=args.remove_numbers,
        min_length=args.min_length,
        max_length=args.max_length
    )
    
    # Process directory
    preprocessor.process_directory(
        dir_path=args.input_dir,
        output_dir=args.output_dir,
        file_pattern=args.file_pattern,
        text_field=args.text_field,
        output_format=args.output_format,
        num_workers=args.num_workers
    )


if __name__ == "__main__":
    main()