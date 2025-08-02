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
            '