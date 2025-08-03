#!/usr/bin/env python3
"""
Amharic Data Collector - Local Implementation
Collects and processes Amharic text data for LLM training

Features:
- Web scraping for Amharic content
- Text cleaning and preprocessing
- Quality filtering
- Export to training formats
"""

import os
import re
import json
import time
import requests
from typing import List, Dict, Optional
from pathlib import Path
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

class AmharicDataCollector:
    """Enhanced Amharic text data collector with quality filtering"""
    
    def __init__(self, output_dir: str = "data/collected"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Amharic character range for validation
        self.amharic_pattern = re.compile(r'[\u1200-\u137F]+')
        
        # Quality thresholds
        self.min_text_length = 50
        self.min_amharic_ratio = 0.7
        
        # Data storage
        self.collected_texts = []
        
    def is_amharic_text(self, text: str) -> bool:
        """Check if text contains sufficient Amharic content"""
        if not text or len(text) < self.min_text_length:
            return False
            
        # Count Amharic characters
        amharic_matches = self.amharic_pattern.findall(text)
        amharic_chars = sum(len(match) for match in amharic_matches)
        
        # Count total non-whitespace characters
        total_chars = len(re.sub(r'\s+', '', text))
        
        if total_chars == 0:
            return False
            
        amharic_ratio = amharic_chars / total_chars
        return amharic_ratio >= self.min_amharic_ratio
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize Amharic text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove HTML entities
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        return text.strip()
    
    def scrape_website(self, url: str, max_pages: int = 10) -> List[str]:
        """Scrape Amharic content from a website"""
        texts = []
        visited_urls = set()
        urls_to_visit = [url]
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        
        for _ in range(max_pages):
            if not urls_to_visit:
                break
                
            current_url = urls_to_visit.pop(0)
            if current_url in visited_urls:
                continue
                
            try:
                print(f"Scraping: {current_url}")
                response = requests.get(current_url, headers=headers, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract text content
                for tag in soup.find_all(['p', 'div', 'article', 'section']):
                    text = tag.get_text(strip=True)
                    if self.is_amharic_text(text):
                        cleaned_text = self.clean_text(text)
                        if cleaned_text:
                            texts.append(cleaned_text)
                
                # Find more URLs (same domain only)
                base_domain = urlparse(url).netloc
                for link in soup.find_all('a', href=True):
                    link_url = urljoin(current_url, link['href'])
                    if urlparse(link_url).netloc == base_domain and link_url not in visited_urls:
                        urls_to_visit.append(link_url)
                
                visited_urls.add(current_url)
                time.sleep(1)  # Be respectful
                
            except Exception as e:
                print(f"Error scraping {current_url}: {e}")
                continue
        
        return texts
    
    def collect_from_sources(self, sources: List[str]) -> None:
        """Collect data from multiple sources"""
        print(f"Starting data collection from {len(sources)} sources...")
        
        for i, source in enumerate(sources, 1):
            print(f"\n[{i}/{len(sources)}] Processing: {source}")
            
            try:
                texts = self.scrape_website(source, max_pages=5)
                self.collected_texts.extend(texts)
                print(f"Collected {len(texts)} texts from {source}")
                
            except Exception as e:
                print(f"Failed to process {source}: {e}")
                continue
        
        print(f"\nTotal collected texts: {len(self.collected_texts)}")
    
    def save_data(self, filename: str = None) -> str:
        """Save collected data to files"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"amharic_corpus_{timestamp}"
        
        # Save as JSON
        json_path = self.output_dir / f"{filename}.json"
        data = {
            'metadata': {
                'total_texts': len(self.collected_texts),
                'collection_date': datetime.now().isoformat(),
                'min_text_length': self.min_text_length,
                'min_amharic_ratio': self.min_amharic_ratio
            },
            'texts': self.collected_texts
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # Save as plain text
        txt_path = self.output_dir / f"{filename}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            for text in self.collected_texts:
                f.write(text + '\n\n')
        
        # Save statistics
        stats_path = self.output_dir / f"{filename}_stats.json"
        stats = self.get_statistics()
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"\nData saved to:")
        print(f"  JSON: {json_path}")
        print(f"  Text: {txt_path}")
        print(f"  Stats: {stats_path}")
        
        return str(json_path)
    
    def get_statistics(self) -> Dict:
        """Generate statistics about collected data"""
        if not self.collected_texts:
            return {}
        
        text_lengths = [len(text) for text in self.collected_texts]
        word_counts = [len(text.split()) for text in self.collected_texts]
        
        return {
            'total_texts': len(self.collected_texts),
            'total_characters': sum(text_lengths),
            'total_words': sum(word_counts),
            'avg_text_length': sum(text_lengths) / len(text_lengths),
            'avg_word_count': sum(word_counts) / len(word_counts),
            'min_text_length': min(text_lengths),
            'max_text_length': max(text_lengths)
        }

def main():
    """Main function for data collection"""
    # Known Amharic content sources
    amharic_sources = [
        "https://www.ethiopianreporter.com",
        "https://www.fanabc.com",
        "https://www.ebc.et",
        "https://addisfortune.news",
        "https://www.ethiopianherald.com.et"
    ]
    
    # Initialize collector
    collector = AmharicDataCollector()
    
    print("Amharic Data Collector - Starting Collection...")
    print(f"Target sources: {len(amharic_sources)}")
    
    # Collect data
    collector.collect_from_sources(amharic_sources)
    
    # Save results
    if collector.collected_texts:
        output_file = collector.save_data()
        
        # Print statistics
        stats = collector.get_statistics()
        print("\n=== Collection Statistics ===")
        print(f"Total texts: {stats['total_texts']}")
        print(f"Total characters: {stats['total_characters']:,}")
        print(f"Total words: {stats['total_words']:,}")
        print(f"Average text length: {stats['avg_text_length']:.1f} characters")
        print(f"Average word count: {stats['avg_word_count']:.1f} words")
        
        print(f"\n✅ Data collection completed successfully!")
        print(f"📁 Output file: {output_file}")
    else:
        print("❌ No data collected. Please check the sources and try again.")

if __name__ == "__main__":
    main()