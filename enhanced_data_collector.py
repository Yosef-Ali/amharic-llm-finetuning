#!/usr/bin/env python3
"""
Improved Enhanced Amharic Data Collector
Fixes collection failures and implements robust data gathering from multiple sources
"""

import requests
from bs4 import BeautifulSoup
import json
import os
from pathlib import Path
import time
import random
from typing import List, Dict, Optional
import logging
from datetime import datetime
import re
from urllib.parse import urljoin, urlparse
import feedparser
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RobustAmharicCollector:
    """Improved data collector with better error handling and multiple sources"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.collected_dir = self.data_dir / "collected"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories
        self.collected_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup robust session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Headers to appear more like a browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'am,en-US;q=0.7,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # Data sources with RSS feeds and direct URLs
        self.sources = {
            "bbc_amharic": {
                "name": "BBC Amharic",
                "rss": "https://feeds.bbci.co.uk/amharic/rss.xml",
                "base_url": "https://www.bbc.com/amharic",
                "rate_limit": 2.0
            },
            "voa_amharic": {
                "name": "VOA Amharic",
                "rss": "https://amharic.voanews.com/api/zrqite$miq",
                "base_url": "https://amharic.voanews.com",
                "rate_limit": 3.0
            },
            "ena": {
                "name": "Ethiopian News Agency",
                "base_url": "https://www.ena.et",
                "rate_limit": 4.0
            },
            "wikipedia": {
                "name": "Amharic Wikipedia",
                "api_url": "https://am.wikipedia.org/w/api.php",
                "rate_limit": 1.0
            }
        }
        
        # Quality metrics
        self.min_text_length = 100
        self.max_text_length = 5000
        self.amharic_char_ratio_threshold = 0.6
        
        # Collection stats
        self.stats = {
            "total_collected": 0,
            "successful_collections": 0,
            "failed_collections": 0,
            "sources_used": [],
            "collection_time": None,
            "quality_scores": []
        }
    
    def is_amharic_text(self, text: str) -> bool:
        """Check if text contains sufficient Amharic characters"""
        if not text:
            return False
        
        # Amharic Unicode range: U+1200-U+137F
        amharic_chars = sum(1 for char in text if '\u1200' <= char <= '\u137F')
        total_chars = len(text.replace(' ', '').replace('\n', ''))
        
        if total_chars == 0:
            return False
        
        ratio = amharic_chars / total_chars
        return ratio >= self.amharic_char_ratio_threshold
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep Amharic punctuation
        text = re.sub(r'[^\u1200-\u137F\s.,;:!?()\[\]"\'-]', '', text)
        return text.strip()
    
    def assess_quality(self, text: str) -> float:
        """Assess text quality (0-100 score)"""
        if not text:
            return 0.0
        
        score = 0.0
        
        # Length check (30 points)
        if self.min_text_length <= len(text) <= self.max_text_length:
            score += 30
        elif len(text) > self.min_text_length:
            score += 20
        
        # Amharic content ratio (40 points)
        if self.is_amharic_text(text):
            amharic_chars = sum(1 for char in text if '\u1200' <= char <= '\u137F')
            total_chars = len(text.replace(' ', ''))
            ratio = amharic_chars / max(total_chars, 1)
            score += 40 * ratio
        
        # Sentence structure (20 points)
        sentences = text.split('።')
        if len(sentences) >= 2:
            score += 20
        elif len(sentences) == 1 and len(text) > 50:
            score += 10
        
        # Diversity check (10 points)
        unique_words = len(set(text.split()))
        if unique_words > 10:
            score += 10
        elif unique_words > 5:
            score += 5
        
        return min(score, 100.0)
    
    def collect_from_rss(self, source_name: str, max_articles: int = 50) -> List[Dict]:
        """Collect articles from RSS feeds"""
        source = self.sources.get(source_name)
        if not source or 'rss' not in source:
            return []
        
        articles = []
        try:
            logger.info(f"📡 Collecting from {source['name']} RSS feed...")
            
            # Parse RSS feed
            feed = feedparser.parse(source['rss'])
            
            for entry in feed.entries[:max_articles]:
                try:
                    # Get article content
                    if hasattr(entry, 'link'):
                        article_text = self.extract_article_content(entry.link)
                        if article_text:
                            cleaned_text = self.clean_text(article_text)
                            if self.is_amharic_text(cleaned_text):
                                quality_score = self.assess_quality(cleaned_text)
                                
                                articles.append({
                                    'title': getattr(entry, 'title', 'No title'),
                                    'content': cleaned_text,
                                    'source': source_name,
                                    'url': entry.link,
                                    'quality_score': quality_score,
                                    'timestamp': datetime.now().isoformat(),
                                    'word_count': len(cleaned_text.split()),
                                    'char_count': len(cleaned_text)
                                })
                                
                                self.stats['successful_collections'] += 1
                                self.stats['quality_scores'].append(quality_score)
                                logger.info(f"✅ Collected article: {entry.title[:50]}... (Quality: {quality_score:.1f})")
                            else:
                                self.stats['failed_collections'] += 1
                                logger.warning(f"❌ Low Amharic content: {entry.title[:50]}...")
                    
                    # Rate limiting
                    time.sleep(source.get('rate_limit', 2.0))
                    
                except Exception as e:
                    self.stats['failed_collections'] += 1
                    logger.error(f"Error processing entry: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error collecting from {source_name}: {e}")
        
        return articles
    
    def extract_article_content(self, url: str) -> str:
        """Extract main content from article URL"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()
            
            # Try different content selectors
            content_selectors = [
                'article',
                '.article-content',
                '.content',
                '.post-content',
                '.entry-content',
                'main',
                '.main-content'
            ]
            
            content = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    content = ' '.join([elem.get_text(strip=True) for elem in elements])
                    break
            
            # Fallback to body content
            if not content:
                body = soup.find('body')
                if body:
                    content = body.get_text(strip=True)
            
            return content
        
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return ""
    
    def collect_wikipedia_articles(self, max_articles: int = 100) -> List[Dict]:
        """Collect articles from Amharic Wikipedia with improved API usage"""
        articles = []
        source = self.sources['wikipedia']
        
        try:
            logger.info(f"📚 Collecting from {source['name']}...")
            
            # Get random articles
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'random',
                'rnnamespace': 0,
                'rnlimit': max_articles
            }
            
            response = self.session.get(source['api_url'], params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if 'query' in data and 'random' in data['query']:
                for page in data['query']['random']:
                    try:
                        # Get page content
                        content_params = {
                            'action': 'query',
                            'format': 'json',
                            'pageids': page['id'],
                            'prop': 'extracts',
                            'exintro': False,
                            'explaintext': True,
                            'exsectionformat': 'plain'
                        }
                        
                        content_response = self.session.get(source['api_url'], params=content_params, timeout=15)
                        content_response.raise_for_status()
                        content_data = content_response.json()
                        
                        if 'query' in content_data and 'pages' in content_data['query']:
                            page_data = list(content_data['query']['pages'].values())[0]
                            if 'extract' in page_data:
                                content = page_data['extract']
                                cleaned_content = self.clean_text(content)
                                
                                if self.is_amharic_text(cleaned_content) and len(cleaned_content) >= self.min_text_length:
                                    quality_score = self.assess_quality(cleaned_content)
                                    
                                    articles.append({
                                        'title': page['title'],
                                        'content': cleaned_content,
                                        'source': 'wikipedia',
                                        'page_id': page['id'],
                                        'quality_score': quality_score,
                                        'timestamp': datetime.now().isoformat(),
                                        'word_count': len(cleaned_content.split()),
                                        'char_count': len(cleaned_content)
                                    })
                                    
                                    self.stats['successful_collections'] += 1
                                    self.stats['quality_scores'].append(quality_score)
                                    logger.info(f"✅ Wikipedia article: {page['title'][:50]}... (Quality: {quality_score:.1f})")
                                else:
                                    self.stats['failed_collections'] += 1
                                    logger.warning(f"❌ Low quality Wikipedia content: {page['title'][:50]}...")
                        
                        # Rate limiting
                        time.sleep(source.get('rate_limit', 1.0))
                        
                    except Exception as e:
                        self.stats['failed_collections'] += 1
                        logger.error(f"Error processing Wikipedia page {page['id']}: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Error collecting from Wikipedia: {e}")
        
        return articles
    
    def generate_conversational_data(self, base_articles: List[Dict], target_samples: int = 1000) -> List[Dict]:
        """Generate conversational and instruction data from collected articles"""
        conversations = []
        
        # Conversation templates
        conversation_templates = [
            {
                "pattern": "ስለ {topic} ንገረኝ",
                "response_start": "{topic} ስለ"
            },
            {
                "pattern": "{topic} ማለት ምንድን ነው?",
                "response_start": "{topic} ማለት"
            },
            {
                "pattern": "የ{topic} ጠቀሜታ ምንድን ነው?",
                "response_start": "የ{topic} ጠቀሜታ"
            }
        ]
        
        # Instruction templates
        instruction_templates = [
            {
                "instruction": "የሚከተለውን ጽሑፍ ማጠቃለል",
                "type": "summarization"
            },
            {
                "instruction": "የሚከተለውን ጽሑፍ ወደ ቀላል አማርኛ መቀየር",
                "type": "simplification"
            },
            {
                "instruction": "ስለ ይህ ርዕስ ጥያቄዎችን መመለስ",
                "type": "qa"
            }
        ]
        
        for article in base_articles:
            if len(conversations) >= target_samples:
                break
            
            content = article['content']
            title = article.get('title', 'ርዕስ')
            
            # Generate conversations
            for template in conversation_templates:
                if len(conversations) >= target_samples:
                    break
                
                # Extract key topics from title
                topic = title.split()[0] if title.split() else "ርዕስ"
                
                question = template["pattern"].format(topic=topic)
                
                # Create response from article content
                sentences = content.split('።')
                response = '។'.join(sentences[:3]) + '።' if len(sentences) >= 3 else content[:200] + '...'
                
                conversations.append({
                    "input": question,
                    "output": response,
                    "type": "conversation",
                    "source_article": article.get('title', ''),
                    "timestamp": datetime.now().isoformat()
                })
            
            # Generate instruction data
            for template in instruction_templates:
                if len(conversations) >= target_samples:
                    break
                
                # Create instruction-following examples
                if template["type"] == "summarization":
                    sentences = content.split('።')
                    summary = '።'.join(sentences[:2]) + '።' if len(sentences) >= 2 else content[:100]
                    
                    conversations.append({
                        "instruction": template["instruction"],
                        "input": content[:500],
                        "output": summary,
                        "type": "instruction",
                        "timestamp": datetime.now().isoformat()
                    })
        
        logger.info(f"💬 Generated {len(conversations)} conversational examples")
        return conversations
    
    def collect_all_data(self, target_samples: int = 5000) -> Dict:
        """Collect data from all sources with improved error handling"""
        logger.info(f"🚀 Starting robust data collection (target: {target_samples} samples)")
        start_time = datetime.now()
        
        all_data = {
            "articles": [],
            "conversations": [],
            "metadata": {
                "collection_start": start_time.isoformat(),
                "target_samples": target_samples,
                "sources_attempted": []
            }
        }
        
        # Collect from RSS sources
        for source_name in ["bbc_amharic", "voa_amharic"]:
            try:
                logger.info(f"📡 Attempting {source_name}...")
                articles = self.collect_from_rss(source_name, max_articles=50)
                all_data["articles"].extend(articles)
                all_data["metadata"]["sources_attempted"].append(source_name)
                self.stats["sources_used"].append(source_name)
                logger.info(f"✅ {source_name}: {len(articles)} articles collected")
            except Exception as e:
                logger.error(f"❌ {source_name} failed: {e}")
        
        # Collect from Wikipedia
        try:
            logger.info(f"📚 Attempting Wikipedia...")
            wiki_articles = self.collect_wikipedia_articles(max_articles=100)
            all_data["articles"].extend(wiki_articles)
            all_data["metadata"]["sources_attempted"].append("wikipedia")
            if wiki_articles:
                self.stats["sources_used"].append("wikipedia")
            logger.info(f"✅ Wikipedia: {len(wiki_articles)} articles collected")
        except Exception as e:
            logger.error(f"❌ Wikipedia failed: {e}")
        
        # Generate conversational data
        if all_data["articles"]:
            try:
                conversations = self.generate_conversational_data(
                    all_data["articles"], 
                    target_samples=min(2000, target_samples // 2)
                )
                all_data["conversations"] = conversations
                self.stats["total_collected"] = len(all_data["articles"]) + len(conversations)
            except Exception as e:
                logger.error(f"❌ Conversation generation failed: {e}")
        
        # Save collected data
        self.save_collected_data(all_data)
        
        # Create training files
        self.create_training_files(all_data)
        
        # Update final stats
        self.stats["collection_time"] = (datetime.now() - start_time).total_seconds()
        self.save_collection_stats()
        
        # Print summary
        success_rate = (self.stats["successful_collections"] / 
                       max(self.stats["successful_collections"] + self.stats["failed_collections"], 1)) * 100
        avg_quality = sum(self.stats["quality_scores"]) / max(len(self.stats["quality_scores"]), 1)
        
        logger.info(f"\n📊 Collection Summary:")
        logger.info(f"✅ Articles collected: {len(all_data['articles'])}")
        logger.info(f"💬 Conversations generated: {len(all_data['conversations'])}")
        logger.info(f"📈 Success rate: {success_rate:.1f}%")
        logger.info(f"⭐ Average quality: {avg_quality:.1f}/100")
        logger.info(f"🎯 Total samples: {self.stats['total_collected']}")
        logger.info(f"⏱️ Collection time: {self.stats['collection_time']:.2f} seconds")
        
        return all_data
    
    def save_collected_data(self, data: Dict):
        """Save collected data to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw collected data
        output_file = self.collected_dir / f"robust_amharic_data_{timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Saved collected data to {output_file}")
    
    def create_training_files(self, data: Dict):
        """Create training files in different formats"""
        logger.info("📝 Creating training files...")
        
        # Create text file for language modeling
        text_file = self.processed_dir / "robust_train.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            # Add articles
            for article in data["articles"]:
                f.write(f"ርዕስ: {article.get('title', '')}\n")
                f.write(f"{article['content']}\n\n")
            
            # Add conversations
            for conv in data["conversations"]:
                if conv["type"] == "conversation":
                    f.write(f"ጥያቄ: {conv['input']}\n")
                    f.write(f"መልስ: {conv['output']}\n\n")
                elif conv["type"] == "instruction":
                    f.write(f"መመሪያ: {conv['instruction']}\n")
                    if conv.get("input"):
                        f.write(f"ግብዓት: {conv['input']}\n")
                    f.write(f"ውጤት: {conv['output']}\n\n")
        
        # Create JSONL file for instruction tuning
        jsonl_file = self.processed_dir / "robust_train.jsonl"
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            # Add articles as text completion tasks
            for article in data["articles"]:
                json.dump({
                    "text": f"ርዕስ: {article.get('title', '')}\n{article['content']}",
                    "source": article['source'],
                    "quality_score": article.get('quality_score', 0)
                }, f, ensure_ascii=False)
                f.write("\n")
            
            # Add conversations
            for conv in data["conversations"]:
                json.dump(conv, f, ensure_ascii=False)
                f.write("\n")
        
        logger.info(f"📝 Created training files: {text_file}, {jsonl_file}")
    
    def save_collection_stats(self):
        """Save collection statistics"""
        stats_file = self.data_dir / "metadata" / "robust_collection_stats.json"
        stats_file.parent.mkdir(exist_ok=True)
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📊 Saved collection stats to {stats_file}")

def main():
    """Main execution function"""
    print("🇪🇹 Robust Amharic Data Collector")
    print("====================================")
    
    collector = RobustAmharicCollector()
    
    # Collect data with higher target
    target_samples = 5000
    data = collector.collect_all_data(target_samples)
    
    print("\n🎯 Next Steps:")
    print("1. Run: python smart_train.py")
    print("2. Run: python smart_amharic_app.py")
    print("3. Check data quality in data/processed/")

if __name__ == "__main__":
    main()