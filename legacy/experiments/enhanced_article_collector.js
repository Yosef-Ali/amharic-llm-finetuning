#!/usr/bin/env node

const https = require('https');
const http = require('http');
const fs = require('fs').promises;
const path = require('path');
const { URL } = require('url');

class EnhancedAmharicCollector {
  constructor() {
    this.articles = [];
    this.outputDir = './collected_articles';
    this.maxArticles = 1000;
    this.currentCount = 0;
    this.processedUrls = new Set(); // Avoid duplicates
    
    // Expanded Wikipedia categories for Amharic content
    this.wikiCategories = [
      'Category:ኢትዮጵያ',
      'Category:የኢትዮጵያ_ታሪክ', 
      'Category:የኢትዮጵያ_ሰዎች',
      'Category:የኢትዮጵያ_ጂኦግራፊ',
      'Category:የኢትዮጵያ_ባህል',
      'Category:የኢትዮጵያ_ቋንቋዎች',
      'Category:አማርኛ',
      'Category:ኦሮምኛ',
      'Category:አፍሪካ',
      'Category:የአፍሪካ_ሀገሮች',
      'Category:ክርስትና',
      'Category:እስልምና',
      'Category:ሳይንስ',
      'Category:ስነ_ጽሁፍ',
      'Category:ሙዚቃ',
      'Category:ስፖርት',
      'Category:ኢኮኖሚ'
    ];

    // Expanded list of known Amharic Wikipedia articles
    this.wikiArticles = [
      // Geography & Places
      'ኢትዮጵያ', 'አዲስ_አበባ', 'ሀይለ_ሥላሴ', 'መንሊክ_ዳግማዊ', 'ላሊበላ', 'ጎንደር', 'አክሱም',
      'ሃረር', 'ባህር_ዳር', 'መቀሌ', 'ጅማ', 'ሀዋሳ', 'አርባ_ምንጭ', 'ናዝሬት', 'ደሴ', 'ድሬ_ዳዋ',
      'አዋሽ_ወንዝ', 'አባይ_ወንዝ', 'ኦሞ_ወንዝ', 'ሰብለ_ውንጉልና_ሀዊ', 'ዳናኪል_ምድረ_በዳ',
      'ሰሜን_ተራራ', 'ባሌ_ተራሮች', 'ሲሜን_ተራራ', 'አፋር_ሳህል', 'ኦጋዴን',
      
      // Languages & Peoples
      'አማርኛ', 'ኦሮሞ', 'ትግረ', 'ወላይታ', 'ሲዳማ', 'ጉራጌ', 'ሶማሌ', 'አፋር', 'ጋሞ',
      'ሸኮ', 'ካፋ', 'ክምባታ', 'ሀዲያ', 'አላባ', 'ቤጃ', 'ሰሃንጉል', 'ጋምቤላ',
      
      // History & Culture
      'የኢትዮጵያ_ታሪክ', 'አክሱም_መንግሥት', 'ዘግዌ_ሥርወ_መንግሥት', 'የሸዋ_መንግሥት',
      'የጎንደር_ዘመን', 'መሣፍንት_ዘመን', 'የዘመን_አዴስ', 'የደርግ_መንግሥት',
      'ኢትዮጵያዊ_ዓፍሪካዊ_መንግሥት', 'የሃገር_ውዳድ_ጦርነት', 'የጣሊያን_ወረራ',
      
      // Religion
      'የኢትዮጵያ_ኦርቶዶክስ_ተዋሕዶ_ቤተ_ክርስቲያን', 'እስልምና_በኢትዮጵያ', 'ፕሮቴስታንት_በኢትዮጵያ',
      'ቅዱስ_ጊዮርጊስ', 'ቅዱስ_ሚካኤል', 'ቅዱስ_ገብርኤል', 'ቅድስት_ማርያም',
      'ጥምቀት', 'መስቀል', 'ገና', 'ፋሲካ', 'እስላሚ_በዓላት',
      
      // Science & Education
      'ሳይንስ', 'ቴክኖሎጂ', 'ሐኪምና', 'ሕክምና', 'ዩኒቨርሲቲ', 'ትምህርት',
      'አዲስ_አበባ_ዩኒቨርሲቲ', 'የሃራማያ_ዩኒቨርሲቲ', 'የባህር_ዳር_ዩኒቨርሲቲ',
      'የመቀሌ_ዩኒቨርሲቲ', 'ኮምፒዩተር', 'ኢንተርኔት', 'ሞባይል_ስልክ',
      
      // Arts & Literature
      'ሥነ_ጽሁፍ', 'ባህላዊ_ሙዚቃ', 'ሥዕል', 'ቅርጽ', 'ሥነ_ጥበብ', 'ሴራሚክ',
      'ባህላዊ_ንጽሕናት', 'ባህላዊ_ልብስ', 'ባህላዊ_ዳንስ', 'ባህላዊ_መሳሪያዎች',
      
      // Food & Agriculture
      'እንጀራ', 'ዶሮ_ወጥ', 'ቃይ_ወጥ', 'ሺሮ', 'ቃዝና', 'ባይነተ', 'ቡርቱቃን',
      'ቡና', 'ሻይ', 'ወርቅ_ወርቂ', 'ጤፍ', 'ሽንብራ', 'እህል', 'ዕርሻ',
      'እንስሳት_እርባታ', 'ከብት', 'ፍየል', 'በግ', 'ዶሮ', 'ንብ_እርባታ',
      
      // Sports & Recreation
      'እግር_ኳስ', 'ቅርጫ_ኳስ', 'ቴኒስ', 'አትሌቲክስ', 'መሮጥ', 'ቦክስ',
      'ኦሊምፒክ', 'የዓለም_ሻምፒዮንነት', 'ማራቶን', 'የስፖርት_ክለቦች',
      
      // Modern Ethiopia
      'የኢትዮጵያ_ፌዴራላዊ_ዲሞክራሲያዊ_ሪፐብሊክ', 'የኢትዮጵያ_ሕገ_መንግሥት',
      'የኢትዮጵያ_ብር', 'ኢትዮጵያ_አየር_መንገድ', 'የኢትዮጵያ_ባንክ',
      'የኢትዮጵያ_ፖስታ', 'ኢትዮ_ቴሌኮም', 'የኢትዮጵያ_ሬድዮ',
      
      // International Relations  
      'የአፍሪካ_ሕብረት', 'የተባበሩት_መንግሥታት', 'አውሮፓዊ_ሕብረት',
      'የአሜሪካ_ሕብረት_መንግሥታት', 'ቻይና', 'ህንድ', 'ያፓን', 'እስራኤል',
      
      // Calendar & Time
      'የኢትዮጵያ_ዘመን_አቆጣጠር', 'የኢትዮጵያ_ወራት', 'መስከረም', 'ጥቅምት',
      'ኅዳር', 'ታህሳስ', 'ጥር', 'የካቲት', 'መጋቢት', 'ሚያዝያ', 'ግንቦት',
      'ሰኔ', 'ሐምሌ', 'ነሐሴ', 'ጳጉሜ'
    ];

    // RSS feeds and news sources
    this.newsFeeds = [
      {
        name: 'EBC News RSS',
        url: 'https://www.ebc.et/web/guest/rss',
        type: 'rss'
      },
      {
        name: 'Fana Broadcasting RSS', 
        url: 'https://www.fanabc.com/english/feed/',
        type: 'rss'
      }
    ];
  }

  async initialize() {
    console.log('🚀 Initializing Enhanced Amharic Article Collector...');
    console.log(`🎯 Target: ${this.maxArticles} articles`);
    
    try {
      await fs.mkdir(this.outputDir, { recursive: true });
      console.log(`📁 Output directory ready: ${this.outputDir}`);
    } catch (error) {
      console.log(`📁 Output directory exists: ${this.outputDir}`);
    }

    // Load existing articles to avoid duplicates
    try {
      const files = await fs.readdir(this.outputDir);
      const existingCount = files.filter(f => f.startsWith('article_') && f.endsWith('.json')).length;
      if (existingCount > 0) {
        console.log(`📚 Found ${existingCount} existing articles`);
        this.currentCount = existingCount;
      }
    } catch (error) {
      console.log('📚 Starting fresh collection');
    }
  }

  async fetchUrl(url, timeout = 15000) {
    return new Promise((resolve, reject) => {
      const client = url.startsWith('https:') ? https : http;
      
      const req = client.get(url, {
        headers: {
          'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
          'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
          'Accept-Language': 'am,en;q=0.9',
          'Accept-Encoding': 'gzip, deflate'
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
      
      req.setTimeout(timeout, () => {
        req.destroy();
        reject(new Error(`Request timeout after ${timeout}ms`));
      });
    });
  }

  async fetchWikipediaArticle(title, fullContent = true) {
    if (this.processedUrls.has(title)) {
      return null; // Skip duplicates
    }
    
    this.processedUrls.add(title);
    
    const baseUrl = fullContent 
      ? `https://am.wikipedia.org/w/api.php?action=query&format=json&titles=${encodeURIComponent(title)}&prop=extracts&exintro=false&explaintext=true`
      : `https://am.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(title)}`;
    
    try {
      const response = await this.fetchUrl(baseUrl);
      const data = JSON.parse(response);
      
      let article = null;
      
      if (fullContent && data.query?.pages) {
        const pageId = Object.keys(data.query.pages)[0];
        const page = data.query.pages[pageId];
        
        if (page.extract && page.extract.length > 500) {
          article = {
            title: page.title || title,
            content: page.extract,
            url: `https://am.wikipedia.org/wiki/${encodeURIComponent(title)}`,
            source: 'Wikipedia Amharic',
            timestamp: new Date().toISOString(),
            wordCount: page.extract.split(/\s+/).length
          };
        }
      } else if (!fullContent && data.extract) {
        if (data.extract.length > 200) {
          article = {
            title: data.title || title,
            content: data.extract,
            url: `https://am.wikipedia.org/wiki/${encodeURIComponent(title)}`,
            source: 'Wikipedia Amharic',
            timestamp: new Date().toISOString()
          };
        }
      }
      
      return article;
    } catch (error) {
      console.log(`❌ Failed to fetch ${title}: ${error.message}`);
      return null;
    }
  }

  async fetchCategoryMembers(category, limit = 100) {
    const url = `https://am.wikipedia.org/w/api.php?action=query&format=json&list=categorymembers&cmtitle=${encodeURIComponent(category)}&cmlimit=${limit}&cmnamespace=0`;
    
    try {
      const response = await this.fetchUrl(url);
      const data = JSON.parse(response);
      
      const members = data.query?.categorymembers || [];
      console.log(`🔍 Found ${members.length} articles in ${category}`);
      
      return members.map(member => member.title);
    } catch (error) {
      console.log(`❌ Failed to fetch category ${category}: ${error.message}`);
      return [];
    }
  }

  async fetchRandomArticles(count = 50) {
    const url = `https://am.wikipedia.org/w/api.php?action=query&format=json&list=random&rnnamespace=0&rnlimit=${count}`;
    
    try {
      const response = await this.fetchUrl(url);
      const data = JSON.parse(response);
      
      const articles = data.query?.random || [];
      console.log(`🎲 Found ${articles.length} random articles`);
      
      return articles.map(article => article.title);
    } catch (error) {
      console.log(`❌ Failed to fetch random articles: ${error.message}`);
      return [];
    }
  }

  isAmharicContent(text) {
    if (!text) return false;
    
    const amharicRegex = /[\u1200-\u137F]/g;
    const amharicChars = (text.match(amharicRegex) || []).length;
    const totalChars = text.length;
    
    return amharicChars > 0 && (amharicChars / totalChars) > 0.2;
  }

  async saveArticle(article, index) {
    const filename = `article_${index.toString().padStart(4, '0')}.json`;
    const filepath = path.join(this.outputDir, filename);
    
    try {
      await fs.writeFile(filepath, JSON.stringify(article, null, 2), 'utf8');
      console.log(`✅ [${index}/${this.maxArticles}] Saved: ${article.title.substring(0, 50)}...`);
      return true;
    } catch (error) {
      console.error(`❌ Error saving article ${index}:`, error.message);
      return false;
    }
  }

  async collectFromKnownArticles() {
    console.log('\n📚 Collecting from known Wikipedia articles...');
    
    for (const title of this.wikiArticles) {
      if (this.currentCount >= this.maxArticles) break;
      
      console.log(`\n🔍 [${this.currentCount + 1}/${this.maxArticles}] Fetching: ${title}`);
      
      const article = await this.fetchWikipediaArticle(title, true);
      
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
      await new Promise(resolve => setTimeout(resolve, 800));
    }
  }

  async collectFromCategories() {
    console.log('\n🗂️ Collecting from Wikipedia categories...');
    
    for (const category of this.wikiCategories) {
      if (this.currentCount >= this.maxArticles) break;
      
      console.log(`\n📂 Processing category: ${category}`);
      
      const members = await this.fetchCategoryMembers(category, 200);
      
      for (const title of members) {
        if (this.currentCount >= this.maxArticles) break;
        
        if (!this.processedUrls.has(title)) {
          console.log(`\n🔍 [${this.currentCount + 1}/${this.maxArticles}] Category article: ${title}`);
          
          const article = await this.fetchWikipediaArticle(title, true);
          
          if (article && this.isAmharicContent(article.content) && article.content.length > 300) {
            article.articleNumber = this.currentCount + 1;
            article.source = `Wikipedia Amharic (${category})`;
            
            const saved = await this.saveArticle(article, this.currentCount + 1);
            if (saved) {
              this.articles.push(article);
              this.currentCount++;
            }
          }
          
          await new Promise(resolve => setTimeout(resolve, 600));
        }
      }
      
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }

  async collectRandomArticles() {
    console.log('\n🎲 Collecting random Wikipedia articles...');
    
    let attempts = 0;
    const maxAttempts = 10;
    
    while (this.currentCount < this.maxArticles && attempts < maxAttempts) {
      attempts++;
      console.log(`\n🎲 Random collection attempt ${attempts}/${maxAttempts}`);
      
      const randomTitles = await this.fetchRandomArticles(100);
      
      for (const title of randomTitles) {
        if (this.currentCount >= this.maxArticles) break;
        
        if (!this.processedUrls.has(title)) {
          console.log(`\n🔍 [${this.currentCount + 1}/${this.maxArticles}] Random article: ${title}`);
          
          const article = await this.fetchWikipediaArticle(title, true);
          
          if (article && this.isAmharicContent(article.content) && article.content.length > 400) {
            article.articleNumber = this.currentCount + 1;
            article.source = 'Wikipedia Amharic (Random)';
            
            const saved = await this.saveArticle(article, this.currentCount + 1);
            if (saved) {
              this.articles.push(article);
              this.currentCount++;
            }
          }
          
          await new Promise(resolve => setTimeout(resolve, 500));
        }
      }
      
      await new Promise(resolve => setTimeout(resolve, 2000));
    }
  }

  async generateContentVariations() {
    console.log('\n🔄 Generating content variations...');
    
    const baseArticles = [...this.articles];
    
    for (const baseArticle of baseArticles) {
      if (this.currentCount >= this.maxArticles) break;
      
      // Split long articles into focused sections
      if (baseArticle.content.length > 2000) {
        const paragraphs = baseArticle.content.split(/\n\n+/).filter(p => p.trim().length > 200);
        
        for (let i = 0; i < paragraphs.length && this.currentCount < this.maxArticles; i++) {
          const paragraph = paragraphs[i];
          
          if (paragraph.length > 300 && this.isAmharicContent(paragraph)) {
            const variation = {
              title: `${baseArticle.title} - ክፍል ${i + 1}`,
              content: paragraph,
              url: `${baseArticle.url}#paragraph_${i + 1}`,
              source: `${baseArticle.source} (Paragraph)`,
              timestamp: new Date().toISOString(),
              originalArticle: baseArticle.title,
              articleNumber: this.currentCount + 1
            };
            
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
      targetArticles: this.maxArticles,
      completionRate: (this.currentCount / this.maxArticles * 100).toFixed(1) + '%',
      collectionDate: new Date().toISOString(),
      sources: [...new Set(this.articles.map(a => a.source))],
      averageLength: Math.round(this.articles.reduce((sum, a) => sum + a.content.length, 0) / this.articles.length),
      totalCharacters: this.articles.reduce((sum, a) => sum + a.content.length, 0),
      uniqueUrls: this.processedUrls.size,
      articles: this.articles.map(a => ({
        title: a.title,
        source: a.source,
        length: a.content.length,
        url: a.url
      }))
    };
    
    const summaryPath = path.join(this.outputDir, 'enhanced_collection_summary.json');
    await fs.writeFile(summaryPath, JSON.stringify(summary, null, 2), 'utf8');
    console.log(`\n📋 Enhanced collection summary saved to: ${summaryPath}`);
    
    return summary;
  }

  async run() {
    try {
      await this.initialize();
      
      console.log(`\n🎯 Starting enhanced collection (Current: ${this.currentCount}/${this.maxArticles})`);
      
      // Phase 1: Known high-quality articles
      if (this.currentCount < this.maxArticles) {
        await this.collectFromKnownArticles();
        console.log(`\n📊 Phase 1 complete: ${this.currentCount}/${this.maxArticles} articles`);
      }
      
      // Phase 2: Category-based discovery
      if (this.currentCount < this.maxArticles) {
        await this.collectFromCategories();
        console.log(`\n📊 Phase 2 complete: ${this.currentCount}/${this.maxArticles} articles`);
      }
      
      // Phase 3: Random article discovery
      if (this.currentCount < this.maxArticles) {
        await this.collectRandomArticles();
        console.log(`\n📊 Phase 3 complete: ${this.currentCount}/${this.maxArticles} articles`);
      }
      
      // Phase 4: Content variations
      if (this.currentCount < this.maxArticles) {
        await this.generateContentVariations();
        console.log(`\n📊 Phase 4 complete: ${this.currentCount}/${this.maxArticles} articles`);
      }
      
      const summary = await this.saveCollectionSummary();
      
      console.log(`\n🎉 Enhanced Collection Complete!`);
      console.log(`📊 Final Statistics:`);
      console.log(`   Articles collected: ${summary.totalArticles}/${summary.targetArticles} (${summary.completionRate})`);
      console.log(`   Total characters: ${summary.totalCharacters.toLocaleString()}`);
      console.log(`   Average length: ${summary.averageLength} characters`);
      console.log(`   Unique URLs processed: ${summary.uniqueUrls}`);
      console.log(`   Sources: ${summary.sources.length} different sources`);
      console.log(`   Files saved in: ${this.outputDir}`);
      
    } catch (error) {
      console.error('❌ Enhanced collection failed:', error);
    }
  }
}

// Run if called directly
if (require.main === module) {
  const collector = new EnhancedAmharicCollector();
  collector.run().catch(console.error);
}

module.exports = EnhancedAmharicCollector;