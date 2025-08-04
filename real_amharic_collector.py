#!/usr/bin/env python3
"""
REAL Amharic Data Collector - Actually Scrapes Web Content
This script truly collects Amharic text from various sources
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re
from pathlib import Path
from datetime import datetime
import hashlib

class RealAmharicCollector:
    def __init__(self):
        self.data_dir = Path("data/real_collected")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        self.collected_hashes = set()  # To avoid duplicates
        
    def is_amharic_text(self, text):
        """Check if text contains significant Amharic content"""
        if not text:
            return False
        amharic_chars = len(re.findall(r'[\u1200-\u137F]', text))
        total_chars = len(text)
        return (amharic_chars / total_chars) > 0.5 if total_chars > 0 else False
    
    def clean_text(self, text):
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove HTML entities
        text = re.sub(r'&[a-zA-Z]+;', '', text)
        # Keep only relevant characters
        text = re.sub(r'[^\u1200-\u137F\s\.\!\?\፡\፣\፤\፥\፦\፧\።\u0020-\u007E]', '', text)
        return text.strip()
    
    def collect_from_bbc_amharic(self, num_articles=50):
        """Collect articles from BBC Amharic"""
        print("📰 Collecting from BBC Amharic...")
        articles = []
        base_url = "https://www.bbc.com"
        
        try:
            # Get main page
            response = self.session.get(f"{base_url}/amharic")
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find article links
            links = []
            for a in soup.find_all('a', href=True):
                href = a['href']
                if '/amharic/articles/' in href or '/amharic/news-' in href:
                    if href.startswith('/'):
                        href = base_url + href
                    links.append(href)
            
            # Collect articles
            for i, link in enumerate(links[:num_articles]):
                try:
                    print(f"  Collecting article {i+1}/{min(len(links), num_articles)}: {link}")
                    article_response = self.session.get(link)
                    article_soup = BeautifulSoup(article_response.content, 'html.parser')
                    
                    # Extract text from paragraphs
                    paragraphs = article_soup.find_all(['p', 'h1', 'h2'])
                    text_parts = []
                    
                    for p in paragraphs:
                        text = p.get_text().strip()
                        if self.is_amharic_text(text) and len(text) > 20:
                            text_parts.append(text)
                    
                    if text_parts:
                        full_text = ' '.join(text_parts)
                        clean_text = self.clean_text(full_text)
                        
                        # Check for duplicates
                        text_hash = hashlib.md5(clean_text.encode()).hexdigest()
                        if text_hash not in self.collected_hashes:
                            self.collected_hashes.add(text_hash)
                            
                            articles.append({
                                'id': f'bbc_{i:04d}',
                                'source': 'BBC Amharic',
                                'url': link,
                                'text': clean_text,
                                'word_count': len(clean_text.split()),
                                'timestamp': datetime.now().isoformat()
                            })
                    
                    time.sleep(1)  # Be respectful
                    
                except Exception as e:
                    print(f"  Error collecting article: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error accessing BBC Amharic: {e}")
            
        print(f"✅ Collected {len(articles)} articles from BBC Amharic")
        return articles
    
    def collect_from_voa_amharic(self, num_articles=50):
        """Collect articles from VOA Amharic"""
        print("📻 Collecting from VOA Amharic...")
        articles = []
        base_url = "https://amharic.voanews.com"
        
        try:
            response = self.session.get(f"{base_url}/z/3303")
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find article links
            links = []
            for article in soup.find_all('div', class_='media-block'):
                link_elem = article.find('a')
                if link_elem and link_elem.get('href'):
                    href = link_elem['href']
                    if href.startswith('/'):
                        href = base_url + href
                    links.append(href)
            
            # Collect articles
            for i, link in enumerate(links[:num_articles]):
                try:
                    print(f"  Collecting article {i+1}/{min(len(links), num_articles)}")
                    article_response = self.session.get(link)
                    article_soup = BeautifulSoup(article_response.content, 'html.parser')
                    
                    # Find article content
                    content_div = article_soup.find('div', class_='wsw')
                    if not content_div:
                        content_div = article_soup.find('div', class_='content')
                    
                    if content_div:
                        paragraphs = content_div.find_all('p')
                        text_parts = []
                        
                        for p in paragraphs:
                            text = p.get_text().strip()
                            if self.is_amharic_text(text) and len(text) > 20:
                                text_parts.append(text)
                        
                        if text_parts:
                            full_text = ' '.join(text_parts)
                            clean_text = self.clean_text(full_text)
                            
                            # Check for duplicates
                            text_hash = hashlib.md5(clean_text.encode()).hexdigest()
                            if text_hash not in self.collected_hashes:
                                self.collected_hashes.add(text_hash)
                                
                                articles.append({
                                    'id': f'voa_{i:04d}',
                                    'source': 'VOA Amharic',
                                    'url': link,
                                    'text': clean_text,
                                    'word_count': len(clean_text.split()),
                                    'timestamp': datetime.now().isoformat()
                                })
                    
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"  Error collecting article: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error accessing VOA Amharic: {e}")
            
        print(f"✅ Collected {len(articles)} articles from VOA Amharic")
        return articles
    
    def collect_wikipedia_amharic(self, num_articles=100):
        """Collect from Amharic Wikipedia using API"""
        print("📚 Collecting from Amharic Wikipedia...")
        articles = []
        base_url = "https://am.wikipedia.org/w/api.php"
        
        try:
            # Get random articles
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'random',
                'rnlimit': 50,
                'rnnamespace': 0
            }
            
            collected = 0
            while collected < num_articles:
                response = self.session.get(base_url, params=params)
                data = response.json()
                
                for page in data['query']['random']:
                    try:
                        # Get page content
                        content_params = {
                            'action': 'query',
                            'format': 'json',
                            'pageids': page['id'],
                            'prop': 'extracts',
                            'exintro': True,
                            'explaintext': True
                        }
                        
                        content_response = self.session.get(base_url, params=content_params)
                        content_data = content_response.json()
                        
                        page_content = content_data['query']['pages'][str(page['id'])]
                        if 'extract' in page_content:
                            text = page_content['extract']
                            
                            if self.is_amharic_text(text) and len(text) > 100:
                                clean_text = self.clean_text(text)
                                
                                # Check for duplicates
                                text_hash = hashlib.md5(clean_text.encode()).hexdigest()
                                if text_hash not in self.collected_hashes:
                                    self.collected_hashes.add(text_hash)
                                    
                                    articles.append({
                                        'id': f'wiki_{collected:04d}',
                                        'source': 'Wikipedia Amharic',
                                        'title': page['title'],
                                        'text': clean_text,
                                        'word_count': len(clean_text.split()),
                                        'timestamp': datetime.now().isoformat()
                                    })
                                    
                                    collected += 1
                                    print(f"  Collected {collected}/{num_articles}: {page['title']}")
                                    
                                    if collected >= num_articles:
                                        break
                        
                        time.sleep(0.5)
                        
                    except Exception as e:
                        print(f"  Error processing page: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error accessing Wikipedia: {e}")
            
        print(f"✅ Collected {len(articles)} articles from Wikipedia")
        return articles
    
    def save_collected_data(self, data, source_name):
        """Save collected data to JSON file"""
        if not data:
            print(f"⚠️  No data to save for {source_name}")
            return
            
        filename = self.data_dir / f"{source_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        # Calculate statistics
        total_words = sum(item['word_count'] for item in data)
        avg_words = total_words // len(data) if data else 0
        
        print(f"📊 Statistics for {source_name}:")
        print(f"   - Articles: {len(data)}")
        print(f"   - Total words: {total_words:,}")
        print(f"   - Average words/article: {avg_words}")
        print(f"   - Saved to: {filename}")
        
        return filename
    
    def create_training_corpus(self):
        """Combine all collected data into training corpus"""
        print("\n🔄 Creating training corpus...")
        
        all_texts = []
        all_metadata = []
        
        # Load all collected JSON files
        for json_file in self.data_dir.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    all_texts.append(item['text'])
                    all_metadata.append({
                        'source': item['source'],
                        'word_count': item['word_count']
                    })
        
        # Create training files
        train_dir = Path("data/processed")
        train_dir.mkdir(parents=True, exist_ok=True)
        
        # Plain text corpus
        corpus_file = train_dir / "amharic_corpus.txt"
        with open(corpus_file, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(all_texts))
            
        # Metadata
        metadata_file = train_dir / "corpus_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump({
                'total_documents': len(all_texts),
                'total_words': sum(m['word_count'] for m in all_metadata),
                'sources': list(set(m['source'] for m in all_metadata)),
                'created': datetime.now().isoformat()
            }, f, indent=2)
            
        print(f"✅ Created training corpus with {len(all_texts)} documents")
        print(f"📁 Saved to: {corpus_file}")
        
        return corpus_file
    
    def run_full_collection(self, articles_per_source=30):
        """Run complete data collection pipeline"""
        print("\n🚀 Starting REAL Amharic Data Collection")
        print("="*50)
        
        # Collect from each source
        all_data = []
        
        # BBC Amharic
        bbc_data = self.collect_from_bbc_amharic(articles_per_source)
        if bbc_data:
            self.save_collected_data(bbc_data, "bbc_amharic")
            all_data.extend(bbc_data)
        
        # VOA Amharic
        voa_data = self.collect_from_voa_amharic(articles_per_source)
        if voa_data:
            self.save_collected_data(voa_data, "voa_amharic")
            all_data.extend(voa_data)
        
        # Wikipedia Amharic
        wiki_data = self.collect_wikipedia_amharic(articles_per_source * 2)
        if wiki_data:
            self.save_collected_data(wiki_data, "wikipedia_amharic")
            all_data.extend(wiki_data)
        
        # Create combined corpus
        if all_data:
            self.create_training_corpus()
            
        print("\n✅ Data collection complete!")
        print(f"📊 Total articles collected: {len(all_data)}")
        print(f"💾 Total words: {sum(item['word_count'] for item in all_data):,}")
        
        return all_data

if __name__ == "__main__":
    print("🇪🇹 Real Amharic Data Collector")
    print("This will actually scrape real Amharic content from the web")
    print("="*60)
    
    collector = RealAmharicCollector()
    
    # You can run full collection or test individual sources
    print("\nOptions:")
    print("1) Full collection (BBC + VOA + Wikipedia)")
    print("2) Test BBC Amharic only")
    print("3) Test VOA Amharic only") 
    print("4) Test Wikipedia Amharic only")
    
    choice = input("\nEnter choice (1-4): ")
    
    if choice == "1":
        collector.run_full_collection(articles_per_source=30)
    elif choice == "2":
        data = collector.collect_from_bbc_amharic(10)
        collector.save_collected_data(data, "bbc_test")
    elif choice == "3":
        data = collector.collect_from_voa_amharic(10)
        collector.save_collected_data(data, "voa_test")
    elif choice == "4":
        data = collector.collect_wikipedia_amharic(20)
        collector.save_collected_data(data, "wikipedia_test")
    else:
        print("Invalid choice!")
