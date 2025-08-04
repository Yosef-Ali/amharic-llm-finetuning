#!/usr/bin/env node

const fs = require('fs').promises;
const path = require('path');

class ArticleProcessor {
  constructor(inputDir = './collected_articles', outputDir = './processed_articles') {
    this.inputDir = inputDir;
    this.outputDir = outputDir;
    this.processedArticles = [];
  }

  async initialize() {
    try {
      await fs.mkdir(this.outputDir, { recursive: true });
      console.log(`Created output directory: ${this.outputDir}`);
    } catch (error) {
      console.log(`Output directory already exists: ${this.outputDir}`);
    }
  }

  // Clean and normalize Amharic text
  cleanAmharicText(text) {
    if (!text) return '';
    
    return text
      // Remove extra whitespace
      .replace(/\s+/g, ' ')
      // Remove non-Amharic characters except basic punctuation
      .replace(/[^\u1200-\u137F\s\.\,\!\?\:\;\(\)\-\"\'\፡\።\፣\፤\፥\፦\፧\፨]/g, '')
      // Normalize Amharic punctuation
      .replace(/\s*፡\s*/g, '፡ ')
      .replace(/\s*\።\s*/g, '። ')
      .replace(/\s*\፣\s*/g, '፣ ')
      // Remove multiple consecutive punctuation
      .replace(/([፡።፣፤])\1+/g, '$1')
      // Fix spacing around punctuation
      .replace(/\s+([፡።፣፤])/g, '$1')
      .replace(/([፡።፣፤])\s+/g, '$1 ')
      // Trim
      .trim();
  }

  // Validate Amharic content quality
  validateArticle(article) {
    const issues = [];
    
    // Check title
    if (!article.title || article.title.length < 10) {
      issues.push('Title too short or missing');
    }
    
    // Check content length
    if (!article.content || article.content.length < 200) {
      issues.push('Content too short');
    }
    
    // Check Amharic character percentage
    const amharicChars = (article.content.match(/[\u1200-\u137F]/g) || []).length;
    const totalChars = article.content.length;
    const amharicPercentage = amharicChars / totalChars;
    
    if (amharicPercentage < 0.1) {
      issues.push('Not enough Amharic characters');
    }
    
    // Check for meaningful content (not just repeated characters)
    const uniqueChars = new Set(article.content).size;
    if (uniqueChars < 20) {
      issues.push('Content appears to be repetitive');
    }
    
    return {
      isValid: issues.length === 0,
      issues: issues,
      amharicPercentage: amharicPercentage,
      contentLength: article.content.length,
      uniqueCharacters: uniqueChars
    };
  }

  // Extract sentences from article
  extractSentences(text) {
    // Split on Amharic sentence endings
    const sentences = text
      .split(/[።፡]/)
      .map(s => s.trim())
      .filter(s => s.length > 10)
      .map(s => this.cleanAmharicText(s));
    
    return sentences;
  }

  // Process a single article
  async processArticle(articlePath) {
    try {
      const content = await fs.readFile(articlePath, 'utf8');
      const article = JSON.parse(content);
      
      // Clean the article
      const cleanedArticle = {
        ...article,
        title: this.cleanAmharicText(article.title),
        content: this.cleanAmharicText(article.content),
        originalLength: article.content?.length || 0,
        processedAt: new Date().toISOString()
      };
      
      // Validate the article
      const validation = this.validateArticle(cleanedArticle);
      cleanedArticle.validation = validation;
      
      if (validation.isValid) {
        // Extract sentences
        cleanedArticle.sentences = this.extractSentences(cleanedArticle.content);
        cleanedArticle.sentenceCount = cleanedArticle.sentences.length;
        
        // Calculate some basic stats
        cleanedArticle.stats = {
          characterCount: cleanedArticle.content.length,
          wordCount: cleanedArticle.content.split(/\s+/).length,
          amharicCharacterCount: (cleanedArticle.content.match(/[\u1200-\u137F]/g) || []).length,
          amharicPercentage: validation.amharicPercentage
        };
        
        return cleanedArticle;
      } else {
        console.log(`❌ Skipping invalid article: ${path.basename(articlePath)} - ${validation.issues.join(', ')}`);
        return null;
      }
      
    } catch (error) {
      console.error(`Error processing ${articlePath}:`, error.message);
      return null;
    }
  }

  // Process all articles
  async processAllArticles() {
    console.log(`\n📝 Processing articles from: ${this.inputDir}`);
    
    try {
      const files = await fs.readdir(this.inputDir);
      const articleFiles = files.filter(f => f.endsWith('.json') && f.startsWith('article_'));
      
      console.log(`Found ${articleFiles.length} article files to process`);
      
      let validCount = 0;
      let invalidCount = 0;
      
      for (const file of articleFiles) {
        const filePath = path.join(this.inputDir, file);
        const processed = await this.processArticle(filePath);
        
        if (processed) {
          // Save processed article
          const outputPath = path.join(this.outputDir, `processed_${file}`);
          await fs.writeFile(outputPath, JSON.stringify(processed, null, 2), 'utf8');
          
          this.processedArticles.push(processed);
          validCount++;
          
          if (validCount % 10 === 0) {
            console.log(`✅ Processed ${validCount} valid articles...`);
          }
        } else {
          invalidCount++;
        }
      }
      
      console.log(`\n📊 Processing Summary:`);
      console.log(`   Valid articles: ${validCount}`);
      console.log(`   Invalid articles: ${invalidCount}`);
      console.log(`   Success rate: ${((validCount / (validCount + invalidCount)) * 100).toFixed(1)}%`);
      
      return { validCount, invalidCount };
      
    } catch (error) {
      console.error('Error processing articles:', error);
      throw error;
    }
  }

  // Create a consolidated corpus file
  async createCorpus() {
    console.log('\n📚 Creating consolidated corpus...');
    
    const corpus = {
      metadata: {
        totalArticles: this.processedArticles.length,
        createdAt: new Date().toISOString(),
        totalSentences: this.processedArticles.reduce((sum, a) => sum + a.sentenceCount, 0),
        totalCharacters: this.processedArticles.reduce((sum, a) => sum + a.stats.characterCount, 0),
        totalWords: this.processedArticles.reduce((sum, a) => sum + a.stats.wordCount, 0)
      },
      articles: this.processedArticles.map(a => ({
        title: a.title,
        content: a.content,
        sentences: a.sentences,
        source: a.source,
        url: a.url,
        stats: a.stats
      }))
    };
    
    // Save full corpus
    const corpusPath = path.join(this.outputDir, 'amharic_corpus.json');
    await fs.writeFile(corpusPath, JSON.stringify(corpus, null, 2), 'utf8');
    
    // Save text-only version for training
    const textCorpus = this.processedArticles
      .map(a => a.content)
      .join('\n\n--- ARTICLE BREAK ---\n\n');
    
    const textPath = path.join(this.outputDir, 'amharic_corpus.txt');
    await fs.writeFile(textPath, textCorpus, 'utf8');
    
    // Save sentences only
    const allSentences = this.processedArticles
      .flatMap(a => a.sentences)
      .join('\n');
    
    const sentencesPath = path.join(this.outputDir, 'amharic_sentences.txt');
    await fs.writeFile(sentencesPath, allSentences, 'utf8');
    
    console.log(`📄 Corpus files created:`);
    console.log(`   Full corpus: ${corpusPath}`);
    console.log(`   Text corpus: ${textPath}`);
    console.log(`   Sentences: ${sentencesPath}`);
    
    return corpus.metadata;
  }

  async run() {
    try {
      console.log('🔧 Initializing Article Processor...');
      await this.initialize();
      
      const results = await this.processAllArticles();
      
      if (results.validCount > 0) {
        const corpusMetadata = await this.createCorpus();
        
        console.log(`\n🎉 Processing Complete!`);
        console.log(`📊 Final Statistics:`);
        console.log(`   Total articles processed: ${corpusMetadata.totalArticles}`);
        console.log(`   Total sentences: ${corpusMetadata.totalSentences}`);
        console.log(`   Total characters: ${corpusMetadata.totalCharacters.toLocaleString()}`);
        console.log(`   Total words: ${corpusMetadata.totalWords.toLocaleString()}`);
        console.log(`   Files saved in: ${this.outputDir}`);
      } else {
        console.log('❌ No valid articles found to process');
      }
      
    } catch (error) {
      console.error('❌ Processing failed:', error);
    }
  }
}

// Run if called directly
if (require.main === module) {
  const processor = new ArticleProcessor();
  processor.run().catch(console.error);
}

module.exports = ArticleProcessor;