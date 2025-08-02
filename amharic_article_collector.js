#!/usr/bin/env node

const { chromium } = require('playwright');
const fs = require('fs').promises;
const path = require('path');

class AmharicArticleCollector {
  constructor() {
    this.browser = null;
    this.page = null;
    this.articles = [];
    this.outputDir = './collected_articles';
    this.maxArticles = 1000;
    this.currentCount = 0;
    
    // Major Amharic news sources
    this.sources = [
      {
        name: 'BBC Amharic',
        baseUrl: 'https://www.bbc.com/amharic',
        articleSelector: 'article',
        titleSelector: 'h3 a, h2 a, .gs-c-promo-heading__title',
        contentSelector: '.story-body, .gs-c-article-body',
        linkSelector: 'a'
      },
      {
        name: 'VOA Amharic',
        baseUrl: 'https://amharic.voanews.com',
        articleSelector: '.media-block',
        titleSelector: 'h4 a, h3 a',
        contentSelector: '.entry-content, .article-content',
        linkSelector: 'a'
      },
      {
        name: 'DW Amharic',
        baseUrl: 'https://www.dw.com/am',
        articleSelector: '.news',
        titleSelector: 'h2 a, h3 a',
        contentSelector: '.longText, .article-content',
        linkSelector: 'a'
      },
      {
        name: 'Fana Broadcasting',
        baseUrl: 'https://www.fanabc.com',
        articleSelector: '.post',
        titleSelector: 'h2 a, h3 a',
        contentSelector: '.entry-content, .post-content',
        linkSelector: 'a'
      },
      {
        name: 'EBC News',
        baseUrl: 'https://www.ebc.et',
        articleSelector: '.news-item',
        titleSelector: 'h2 a, h3 a',
        contentSelector: '.content, .article-body',
        linkSelector: 'a'
      }
    ];
  }

  async initialize() {
    console.log('Initializing Amharic Article Collector...');
    
    // Create output directory
    try {
      await fs.mkdir(this.outputDir, { recursive: true });
      console.log(`Created output directory: ${this.outputDir}`);
    } catch (error) {
      console.log(`Output directory already exists: ${this.outputDir}`);
    }

    // Launch browser
    this.browser = await chromium.launch({ 
      headless: false,
      args: ['--no-sandbox', '--disable-setuid-sandbox']
    });
    this.page = await this.browser.newPage();
    
    // Set user agent and other headers
    await this.page.setUserAgent('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36');
    
    console.log('Browser initialized successfully');
  }

  async extractArticleLinks(source) {
    console.log(`\nExtracting links from: ${source.name}`);
    
    try {
      await this.page.goto(source.baseUrl, { waitUntil: 'networkidle' });
      await this.page.waitForTimeout(2000);

      // Get all article links
      const links = await this.page.evaluate((selector) => {
        const elements = document.querySelectorAll('a');
        const links = [];
        
        elements.forEach(el => {
          const href = el.href;
          const text = el.textContent?.trim();
          
          if (href && text && text.length > 10) {
            // Filter for likely article URLs
            if (href.includes('/news/') || 
                href.includes('/article/') || 
                href.includes('/story/') ||
                href.includes(window.location.hostname)) {
              links.push({
                url: href,
                title: text,
                source: window.location.hostname
              });
            }
          }
        });
        
        return [...new Set(links.map(l => l.url))].slice(0, 50); // Limit per source
      }, source.linkSelector);

      console.log(`Found ${links.length} potential article links from ${source.name}`);
      return links;
      
    } catch (error) {
      console.error(`Error extracting links from ${source.name}:`, error.message);
      return [];
    }
  }

  async extractArticleContent(url, source) {
    try {
      await this.page.goto(url, { waitUntil: 'networkidle', timeout: 30000 });
      await this.page.waitForTimeout(1000);

      const article = await this.page.evaluate((selectors) => {
        // Try different selectors for title
        let title = '';
        const titleSelectors = ['h1', 'h2', '.title', '.headline', selectors.titleSelector];
        for (const sel of titleSelectors) {
          const el = document.querySelector(sel);
          if (el && el.textContent.trim()) {
            title = el.textContent.trim();
            break;
          }
        }

        // Try different selectors for content
        let content = '';
        const contentSelectors = ['.article-body', '.story-body', '.entry-content', '.post-content', selectors.contentSelector];
        for (const sel of contentSelectors) {
          const el = document.querySelector(sel);
          if (el && el.textContent.trim().length > 100) {
            content = el.textContent.trim();
            break;
          }
        }

        // If no content found with selectors, try to get main text
        if (!content) {
          const paragraphs = Array.from(document.querySelectorAll('p'))
            .map(p => p.textContent.trim())
            .filter(text => text.length > 50);
          content = paragraphs.join('\n\n');
        }

        return {
          title: title || 'No title found',
          content: content || 'No content found',
          url: window.location.href,
          timestamp: new Date().toISOString()
        };
      }, source);

      // Validate that this is likely Amharic content
      if (this.isAmharicContent(article.content) && article.content.length > 200) {
        return article;
      }
      
      return null;
      
    } catch (error) {
      console.error(`Error extracting content from ${url}:`, error.message);
      return null;
    }
  }

  isAmharicContent(text) {
    // Check for Amharic script characters (Ethiopian)
    const amharicRegex = /[\u1200-\u137F]/;
    const amharicChars = (text.match(/[\u1200-\u137F]/g) || []).length;
    const totalChars = text.length;
    
    // At least 10% should be Amharic characters
    return amharicChars > 0 && (amharicChars / totalChars) > 0.1;
  }

  async saveArticle(article, index) {
    const filename = `article_${index.toString().padStart(4, '0')}.json`;
    const filepath = path.join(this.outputDir, filename);
    
    try {
      await fs.writeFile(filepath, JSON.stringify(article, null, 2), 'utf8');
      console.log(`✓ Saved article ${index}: ${article.title.substring(0, 50)}...`);
      return true;
    } catch (error) {
      console.error(`Error saving article ${index}:`, error.message);
      return false;
    }
  }

  async collectArticles() {
    console.log(`\n🚀 Starting collection of ${this.maxArticles} Amharic articles...\n`);
    
    for (const source of this.sources) {
      if (this.currentCount >= this.maxArticles) break;
      
      console.log(`\n📰 Processing source: ${source.name}`);
      
      // Get article links from source
      const links = await this.extractArticleLinks(source);
      
      // Process each link
      for (const link of links) {
        if (this.currentCount >= this.maxArticles) break;
        
        console.log(`\n[${this.currentCount + 1}/${this.maxArticles}] Processing: ${link.substring(0, 80)}...`);
        
        const article = await this.extractArticleContent(link, source);
        
        if (article && article.content.length > 200) {
          article.source = source.name;
          article.sourceUrl = source.baseUrl;
          article.articleNumber = this.currentCount + 1;
          
          const saved = await this.saveArticle(article, this.currentCount + 1);
          if (saved) {
            this.articles.push(article);
            this.currentCount++;
            
            // Progress indicator
            if (this.currentCount % 10 === 0) {
              console.log(`\n🎯 Progress: ${this.currentCount}/${this.maxArticles} articles collected`);
            }
          }
        } else {
          console.log('❌ Skipped: Not valid Amharic content or too short');
        }
        
        // Small delay to be respectful
        await this.page.waitForTimeout(1000);
      }
    }
    
    console.log(`\n✅ Collection complete! Gathered ${this.currentCount} Amharic articles.`);
  }

  async saveCollectionSummary() {
    const summary = {
      totalArticles: this.currentCount,
      collectionDate: new Date().toISOString(),
      sources: this.sources.map(s => s.name),
      articles: this.articles.map(a => ({
        title: a.title,
        source: a.source,
        url: a.url,
        length: a.content.length
      }))
    };
    
    const summaryPath = path.join(this.outputDir, 'collection_summary.json');
    await fs.writeFile(summaryPath, JSON.stringify(summary, null, 2), 'utf8');
    console.log(`📋 Collection summary saved to: ${summaryPath}`);
  }

  async cleanup() {
    if (this.browser) {
      await this.browser.close();
      console.log('🔒 Browser closed');
    }
  }

  async run() {
    try {
      await this.initialize();
      await this.collectArticles();
      await this.saveCollectionSummary();
      
      console.log(`\n🎉 Successfully collected ${this.currentCount} Amharic articles!`);
      console.log(`📁 Articles saved in: ${this.outputDir}`);
      
    } catch (error) {
      console.error('❌ Collection failed:', error);
    } finally {
      await this.cleanup();
    }
  }
}

// Run if called directly
if (require.main === module) {
  const collector = new AmharicArticleCollector();
  collector.run().catch(console.error);
}

module.exports = AmharicArticleCollector;