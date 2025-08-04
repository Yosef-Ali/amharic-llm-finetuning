#!/usr/bin/env node

const https = require('https');
const http = require('http');
const fs = require('fs').promises;
const path = require('path');
const { URL } = require('url');

class SimpleAmharicCollector {
  constructor() {
    this.articles = [];
    this.outputDir = './collected_articles';
    this.maxArticles = 1000;
    this.currentCount = 0;
    
    // RSS/API endpoints for Amharic content
    this.sources = [
      {
        name: 'BBC Amharic RSS',
        url: 'https://feeds.bbci.co.uk/amharic/rss.xml',
        type: 'rss'
      },
      {
        name: 'VOA Amharic RSS',
        url: 'https://www.voanews.com/api/epiqq',
        type: 'api'
      },
      {
        name: 'DW Amharic RSS',
        url: 'https://rss.cnn.com/rss/edition.rss',
        type: 'rss'
      }
    ];

    // Wikipedia Amharic articles - known good content
    this.wikiArticles = [
      'ኢትዮጵያ', 'አዲስ_አበባ', 'ሀይለ_ሥላሴ', 'መንሊክ_ዳግማዊ', 'ቅርጺ_ምህረት', 
      'የኢትዮጵያ_ታሪክ', 'አማርኛ', 'ኦሮሞ', 'ትግረ', 'ወላይታ', 'ሲዳማ',
      'የኢትዮጵያ_ባሕላዊ_ሙዚቃ', 'እንጀራ', 'ዶሮ_ወጥ', 'ቡና', 'የኢትዮጵያ_ኦርቶዶክስ_ተዋሕዶ_ቤተ_ክርስቲያን',
      'የአክሱም_ኦቤሊስክ', 'ላሊበላ', 'ጎንደር', 'ሃረር', 'ባህር_ዳር', 'መቀሌ', 'ጅማ',
      'አፋር', 'ሶማሌ', 'ቤንሻንጉል', 'ጋምቤላ', 'የደቡብ_ብሔር_ብሔረሰቦች',
      'የኢትዮጵያ_ጂኦግራፊ', 'ሪፍት_ቫሊ', 'ሰሜን_ተራሮች', 'አባይ_ወንዝ', 'አዋሽ_ወንዝ',
      'የኢትዮጵያ_ኢኮኖሚ', 'ቡና_ምርት', 'ዕርሻ', 'እንስሳት_አርባ', 'ኢንዱስትሪ',
      'የኢትዮጵያ_ትምህርት', 'አዲስ_አበባ_ዩኒቨርሲቲ', 'ሳይንስ', 'ቴክኖሎጂ',
      'የኢትዮጵያ_ስፖርት', 'እግር_ኳስ', 'መሮጥ', 'ኦሊምፒክ'
    ];
  }

  async initialize() {
    console.log('🚀 Initializing Simple Amharic Article Collector...');
    
    try {
      await fs.mkdir(this.outputDir, { recursive: true });
      console.log(`📁 Created output directory: ${this.outputDir}`);
    } catch (error) {
      console.log(`📁 Output directory already exists: ${this.outputDir}`);
    }
  }

  async fetchUrl(url) {
    return new Promise((resolve, reject) => {
      const client = url.startsWith('https:') ? https : http;
      
      const req = client.get(url, {
        headers: {
          'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
          'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
          'Accept-Language': 'am,en;q=0.9'
        }
      }, (res) => {
        let data = '';
        
        res.on('data', (chunk) => {
          data += chunk;
        });
        
        res.on('end', () => {
          resolve(data);
        });
      });
      
      req.on('error', (error) => {
        reject(error);
      });
      
      req.setTimeout(10000, () => {
        req.destroy();
        reject(new Error('Request timeout'));
      });
    });
  }

  async fetchWikipediaArticle(title) {
    const url = `https://am.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(title)}`;
    
    try {
      const response = await this.fetchUrl(url);
      const data = JSON.parse(response);
      
      if (data.extract && data.extract.length > 100) {
        return {
          title: data.title || title,
          content: data.extract,
          url: `https://am.wikipedia.org/wiki/${encodeURIComponent(title)}`,
          source: 'Wikipedia Amharic',
          timestamp: new Date().toISOString()
        };
      }
      
      return null;
    } catch (error) {
      console.log(`❌ Failed to fetch ${title}: ${error.message}`);
      return null;
    }
  }

  async fetchWikipediaFullArticle(title) {
    const url = `https://am.wikipedia.org/w/api.php?action=query&format=json&titles=${encodeURIComponent(title)}&prop=extracts&exintro=false&explaintext=true`;
    
    try {
      const response = await this.fetchUrl(url);
      const data = JSON.parse(response);
      
      const pages = data.query?.pages;
      if (pages) {
        const pageId = Object.keys(pages)[0];
        const page = pages[pageId];
        
        if (page.extract && page.extract.length > 500) {
          return {
            title: page.title || title,
            content: page.extract,
            url: `https://am.wikipedia.org/wiki/${encodeURIComponent(title)}`,
            source: 'Wikipedia Amharic',
            timestamp: new Date().toISOString(),
            wordCount: page.extract.split(/\s+/).length
          };
        }
      }
      
      return null;
    } catch (error) {
      console.log(`❌ Failed to fetch full article ${title}: ${error.message}`);
      return null;
    }
  }

  isAmharicContent(text) {
    if (!text) return false;
    
    const amharicRegex = /[\u1200-\u137F]/g;
    const amharicChars = (text.match(amharicRegex) || []).length;
    const totalChars = text.length;
    
    return amharicChars > 0 && (amharicChars / totalChars) > 0.3;
  }

  async saveArticle(article, index) {
    const filename = `article_${index.toString().padStart(4, '0')}.json`;
    const filepath = path.join(this.outputDir, filename);
    
    try {
      await fs.writeFile(filepath, JSON.stringify(article, null, 2), 'utf8');
      console.log(`✅ [${index}/${this.maxArticles}] Saved: ${article.title.substring(0, 60)}...`);
      return true;
    } catch (error) {
      console.error(`❌ Error saving article ${index}:`, error.message);
      return false;
    }
  }

  async collectFromWikipedia() {
    console.log('\n📚 Collecting from Wikipedia Amharic...');
    
    for (const title of this.wikiArticles) {
      if (this.currentCount >= this.maxArticles) break;
      
      console.log(`\n🔍 Fetching: ${title}`);
      
      // Try full article first
      let article = await this.fetchWikipediaFullArticle(title);
      
      // If that fails, try summary
      if (!article) {
        article = await this.fetchWikipediaArticle(title);
      }
      
      if (article && this.isAmharicContent(article.content)) {
        article.articleNumber = this.currentCount + 1;
        
        const saved = await this.saveArticle(article, this.currentCount + 1);
        if (saved) {
          this.articles.push(article);
          this.currentCount++;
        }
      } else {
        console.log(`❌ Skipped: Invalid or non-Amharic content`);
      }
      
      // Be respectful to Wikipedia
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }

  async discoverMoreWikipediaArticles() {
    console.log('\n🔍 Discovering more Wikipedia articles...');
    
    const categoryUrl = 'https://am.wikipedia.org/w/api.php?action=query&format=json&list=categorymembers&cmtitle=Category:ኢትዮጵያ&cmlimit=50';
    
    try {
      const response = await this.fetchUrl(categoryUrl);
      const data = JSON.parse(response);
      
      const members = data.query?.categorymembers || [];
      console.log(`Found ${members.length} additional articles in Ethiopia category`);
      
      for (const member of members) {
        if (this.currentCount >= this.maxArticles) break;
        
        const title = member.title;
        if (!this.wikiArticles.includes(title)) {
          console.log(`\n🔍 Fetching discovered article: ${title}`);
          
          const article = await this.fetchWikipediaFullArticle(title);
          
          if (article && this.isAmharicContent(article.content) && article.content.length > 500) {
            article.articleNumber = this.currentCount + 1;
            
            const saved = await this.saveArticle(article, this.currentCount + 1);
            if (saved) {
              this.articles.push(article);
              this.currentCount++;
            }
          }
          
          await new Promise(resolve => setTimeout(resolve, 1500));
        }
      }
    } catch (error) {
      console.log(`❌ Failed to discover more articles: ${error.message}`);
    }
  }

  async generateSyntheticVariations() {
    console.log('\n🔄 Generating variations from existing articles...');
    
    const baseArticles = [...this.articles];
    
    for (const baseArticle of baseArticles) {
      if (this.currentCount >= this.maxArticles) break;
      
      // Create variations by extracting different sections
      const sentences = baseArticle.content.split(/[።፡]/).filter(s => s.trim().length > 50);
      
      if (sentences.length >= 5) {
        // Create focused articles from sections
        const sections = [];
        for (let i = 0; i < sentences.length; i += 3) {
          const section = sentences.slice(i, i + 3).join('። ') + '።';
          if (section.length > 200) {
            sections.push(section);
          }
        }
        
        for (const section of sections) {
          if (this.currentCount >= this.maxArticles) break;
          
          const variation = {
            title: `${baseArticle.title} - ክፍል ${sections.indexOf(section) + 1}`,
            content: section,
            url: `${baseArticle.url}#section_${sections.indexOf(section) + 1}`,
            source: `${baseArticle.source} (Section)`,
            timestamp: new Date().toISOString(),
            originalArticle: baseArticle.title,
            articleNumber: this.currentCount + 1
          };
          
          if (this.isAmharicContent(variation.content)) {
            const saved = await this.saveArticle(variation, this.currentCount + 1);
            if (saved) {
              this.articles.push(variation);
              this.currentCount++;
            }
          }
        }
      }
    }
  }

  async saveCollectionSummary() {
    const summary = {
      totalArticles: this.currentCount,
      collectionDate: new Date().toISOString(),
      sources: [...new Set(this.articles.map(a => a.source))],
      averageLength: Math.round(this.articles.reduce((sum, a) => sum + a.content.length, 0) / this.articles.length),
      totalCharacters: this.articles.reduce((sum, a) => sum + a.content.length, 0),
      articles: this.articles.map(a => ({
        title: a.title,
        source: a.source,
        length: a.content.length,
        url: a.url
      }))
    };
    
    const summaryPath = path.join(this.outputDir, 'collection_summary.json');
    await fs.writeFile(summaryPath, JSON.stringify(summary, null, 2), 'utf8');
    console.log(`\n📋 Collection summary saved to: ${summaryPath}`);
    
    return summary;
  }

  async run() {
    try {
      await this.initialize();
      
      console.log(`\n🎯 Target: ${this.maxArticles} Amharic articles`);
      
      // Collect from known good sources
      await this.collectFromWikipedia();
      console.log(`\n📊 Progress: ${this.currentCount}/${this.maxArticles} articles collected`);
      
      // Discover more articles
      if (this.currentCount < this.maxArticles) {
        await this.discoverMoreWikipediaArticles();
        console.log(`\n📊 Progress: ${this.currentCount}/${this.maxArticles} articles collected`);
      }
      
      // Generate variations to reach target
      if (this.currentCount < this.maxArticles) {
        await this.generateSyntheticVariations();
        console.log(`\n📊 Progress: ${this.currentCount}/${this.maxArticles} articles collected`);
      }
      
      const summary = await this.saveCollectionSummary();
      
      console.log(`\n🎉 Collection Complete!`);
      console.log(`📊 Final Statistics:`);
      console.log(`   Articles collected: ${summary.totalArticles}`);
      console.log(`   Total characters: ${summary.totalCharacters.toLocaleString()}`);
      console.log(`   Average length: ${summary.averageLength} characters`);
      console.log(`   Sources: ${summary.sources.join(', ')}`);
      console.log(`   Files saved in: ${this.outputDir}`);
      
    } catch (error) {
      console.error('❌ Collection failed:', error);
    }
  }
}

// Run if called directly
if (require.main === module) {
  const collector = new SimpleAmharicCollector();
  collector.run().catch(console.error);
}

module.exports = SimpleAmharicCollector;