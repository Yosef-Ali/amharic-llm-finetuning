#!/usr/bin/env python3
"""
Enhanced Amharic Data Collector - Phase 1.1 Implementation
Follows the Grand Implementation Plan for scalable data collection

Features:
- Rate limiting and retry logic
- Multi-source data integration
- Progress tracking and resumption
- Data quality validation
- Automated preprocessing
"""

import requests
import time
import json
import os
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import logging
from tqdm import tqdm
import random
from urllib.parse import urljoin

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DataSource:
    """Configuration for a data source"""
    name: str
    base_url: str
    api_endpoint: str
    rate_limit: float  # seconds between requests
    max_retries: int = 3
    timeout: int = 30

class EnhancedAmharicCollector:
    """Enhanced data collector with rate limiting and multi-source support"""
    
    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "raw").mkdir(exist_ok=True)
        (self.output_dir / "processed").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        
        # Data sources configuration
        self.sources = {
            'wikipedia': DataSource(
                name="Amharic Wikipedia",
                base_url="https://am.wikipedia.org",
                api_endpoint="/w/api.php",
                rate_limit=2.0,  # 2 seconds between requests
                max_retries=5
            ),
            'ena': DataSource(
                name="Ethiopian News Agency",
                base_url="https://www.ena.et",
                api_endpoint="/api/news",
                rate_limit=3.0,  # 3 seconds between requests
                max_retries=3
            )
        }
        
        # Progress tracking
        self.progress_file = self.output_dir / "metadata" / "collection_progress.json"
        self.stats = {
            'total_articles': 0,
            'successful_collections': 0,
            'failed_collections': 0,
            'total_words': 0,
            'sources_used': [],
            'last_updated': None
        }
        
        # Load existing progress
        self.load_progress()
    
    def load_progress(self):
        """Load existing collection progress"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    self.stats.update(json.load(f))
                logger.info(f"Loaded progress: {self.stats['successful_collections']} articles collected")
            except Exception as e:
                logger.warning(f"Could not load progress: {e}")
    
    def save_progress(self):
        """Save collection progress"""
        self.stats['last_updated'] = time.time()
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Could not save progress: {e}")
    
    def make_request_with_retry(self, url: str, params: Dict, source: DataSource) -> Optional[Dict]:
        """Make HTTP request with retry logic and rate limiting"""
        
        for attempt in range(source.max_retries):
            try:
                # Rate limiting
                time.sleep(source.rate_limit + random.uniform(0, 1))
                
                response = requests.get(
                    url, 
                    params=params, 
                    timeout=source.timeout,
                    headers={
                        'User-Agent': 'AmharicLLM-DataCollector/1.0 (Educational Research)'
                    }
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limited
                    wait_time = 2 ** attempt * 5  # Exponential backoff
                    logger.warning(f"Rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.warning(f"HTTP {response.status_code} for {url}")
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < source.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        return None
    
    def collect_wikipedia_articles(self, target_count: int = 500) -> List[Dict]:
        """Collect articles from Amharic Wikipedia with improved rate limiting"""
        source = self.sources['wikipedia']
        articles = []
        
        logger.info(f"Starting Wikipedia collection (target: {target_count} articles)")
        
        # Get list of article titles
        titles = self.get_wikipedia_titles(target_count * 2)  # Get more titles than needed
        
        if not titles:
            logger.error("Could not fetch article titles")
            return articles
        
        # Collect articles with progress bar
        pbar = tqdm(titles[:target_count], desc="Collecting Wikipedia articles")
        
        for title in pbar:
            try:
                article_data = self.get_wikipedia_article(title, source)
                if article_data and self.validate_article(article_data):
                    articles.append(article_data)
                    self.stats['successful_collections'] += 1
                    
                    # Save article immediately
                    self.save_article(article_data, 'wikipedia')
                    
                    # Update progress
                    pbar.set_postfix({
                        'collected': len(articles),
                        'words': sum(len(a.get('content', '').split()) for a in articles)
                    })
                    
                    # Save progress every 10 articles
                    if len(articles) % 10 == 0:
                        self.save_progress()
                else:
                    self.stats['failed_collections'] += 1
                    
            except Exception as e:
                logger.error(f"Error collecting article '{title}': {e}")
                self.stats['failed_collections'] += 1
        
        logger.info(f"Wikipedia collection complete: {len(articles)} articles")
        return articles
    
    def get_wikipedia_titles(self, limit: int = 1000) -> List[str]:
        """Get list of article titles from Wikipedia"""
        source = self.sources['wikipedia']
        url = urljoin(source.base_url, source.api_endpoint)
        
        titles = []
        apcontinue = None
        
        while len(titles) < limit:
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'allpages',
                'aplimit': min(50, limit - len(titles)),
                'apnamespace': 0
            }
            
            if apcontinue:
                params['apcontinue'] = apcontinue
            
            data = self.make_request_with_retry(url, params, source)
            
            if not data or 'query' not in data:
                logger.error("Failed to fetch article titles")
                break
            
            pages = data['query'].get('allpages', [])
            titles.extend([page['title'] for page in pages])
            
            # Check for continuation
            if 'continue' in data and 'apcontinue' in data['continue']:
                apcontinue = data['continue']['apcontinue']
            else:
                break
        
        logger.info(f"Fetched {len(titles)} article titles")
        return titles
    
    def get_wikipedia_article(self, title: str, source: DataSource) -> Optional[Dict]:
        """Get full article content from Wikipedia"""
        url = urljoin(source.base_url, source.api_endpoint)
        
        params = {
            'action': 'query',
            'format': 'json',
            'titles': title,
            'prop': 'extracts',
            'exintro': False,
            'explaintext': True,
            'exsectionformat': 'plain'
        }
        
        data = self.make_request_with_retry(url, params, source)
        
        if not data or 'query' not in data:
            return None
        
        pages = data['query'].get('pages', {})
        
        for page_id, page_data in pages.items():
            if page_id != '-1' and 'extract' in page_data:
                return {
                    'title': page_data.get('title', title),
                    'content': page_data['extract'],
                    'source': 'wikipedia',
                    'url': f"https://am.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    'collected_at': time.time()
                }
        
        return None
    
    def validate_article(self, article: Dict) -> bool:
        """Validate article quality"""
        content = article.get('content', '')
        
        # Basic validation criteria
        if len(content) < 100:  # Too short
            return False
        
        if len(content.split()) < 20:  # Too few words
            return False
        
        # Check for Amharic content (basic check)
        amharic_chars = sum(1 for char in content if '\u1200' <= char <= '\u137F')
        if amharic_chars < len(content) * 0.3:  # At least 30% Amharic characters
            return False
        
        return True
    
    def save_article(self, article: Dict, source_name: str):
        """Save individual article to file"""
        filename = f"{source_name}_{article['title'][:50].replace('/', '_')}.json"
        filepath = self.output_dir / "raw" / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(article, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Could not save article: {e}")
    
    def run_collection(self, target_articles: int = 1000):
        """Run the complete data collection process"""
        logger.info(f"Starting enhanced data collection (target: {target_articles} articles)")
        
        # Collect from Wikipedia
        wikipedia_articles = self.collect_wikipedia_articles(target_articles)
        
        # Update statistics
        self.stats['total_articles'] = len(wikipedia_articles)
        self.stats['total_words'] = sum(len(article.get('content', '').split()) for article in wikipedia_articles)
        self.stats['sources_used'] = ['wikipedia']
        
        # Save final progress
        self.save_progress()
        
        # Generate summary
        self.generate_summary()
        
        logger.info(f"Collection complete! {self.stats['total_articles']} articles, {self.stats['total_words']} words")
    
    def generate_summary(self):
        """Generate collection summary"""
        summary = {
            'collection_stats': self.stats,
            'data_quality': {
                'avg_article_length': self.stats['total_words'] / max(self.stats['total_articles'], 1),
                'success_rate': self.stats['successful_collections'] / max(self.stats['successful_collections'] + self.stats['failed_collections'], 1)
            },
            'next_steps': [
                'Run data preprocessing pipeline',
                'Validate data quality',
                'Upload to Kaggle dataset',
                'Begin model training'
            ]
        }
        
        summary_file = self.output_dir / "metadata" / "collection_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Summary saved to {summary_file}")

def main():
    """Main execution function"""
    collector = EnhancedAmharicCollector()
    
    # Start with a smaller target for testing
    target = 100  # Start with 100 articles to test the system
    
    logger.info("=== Enhanced Amharic Data Collection - Phase 1.1 ===")
    logger.info(f"Target: {target} articles")
    logger.info("Following Grand Implementation Plan...")
    
    collector.run_collection(target)
    
    print("\n🎉 Data collection complete!")
    print(f"📊 Collected: {collector.stats['total_articles']} articles")
    print(f"📝 Total words: {collector.stats['total_words']}")
    print(f"✅ Success rate: {collector.stats['successful_collections'] / max(collector.stats['successful_collections'] + collector.stats['failed_collections'], 1):.1%}")
    print("\n📁 Data saved to:")
    print(f"   Raw articles: {collector.output_dir / 'raw'}")
    print(f"   Metadata: {collector.output_dir / 'metadata'}")
    print("\n🚀 Next: Run data preprocessing pipeline")

if __name__ == "__main__":
    main()