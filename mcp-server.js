#!/usr/bin/env node

const { chromium } = require('playwright');
const http = require('http');
const url = require('url');

class PlaywrightMCPServer {
  constructor() {
    this.browser = null;
    this.page = null;
  }

  async initialize() {
    this.browser = await chromium.launch({ headless: false });
    this.page = await this.browser.newPage();
    console.log('Playwright MCP Server initialized');
  }

  async handleRequest(req, res) {
    const parsedUrl = url.parse(req.url, true);
    const { pathname, query } = parsedUrl;

    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Content-Type', 'application/json');

    try {
      switch (pathname) {
        case '/navigate':
          await this.page.goto(query.url);
          res.end(JSON.stringify({ success: true, url: query.url }));
          break;
        
        case '/screenshot':
          const screenshot = await this.page.screenshot({ encoding: 'base64' });
          res.end(JSON.stringify({ success: true, screenshot }));
          break;
        
        case '/click':
          await this.page.click(query.selector);
          res.end(JSON.stringify({ success: true, selector: query.selector }));
          break;
        
        case '/type':
          await this.page.fill(query.selector, query.text);
          res.end(JSON.stringify({ success: true, selector: query.selector, text: query.text }));
          break;
        
        case '/content':
          const content = await this.page.content();
          res.end(JSON.stringify({ success: true, content }));
          break;
        
        default:
          res.statusCode = 404;
          res.end(JSON.stringify({ error: 'Unknown endpoint' }));
      }
    } catch (error) {
      res.statusCode = 500;
      res.end(JSON.stringify({ error: error.message }));
    }
  }

  async start(port = 3334) {
    await this.initialize();
    
    const server = http.createServer((req, res) => {
      this.handleRequest(req, res);
    });

    server.listen(port, () => {
      console.log(`Playwright MCP Server running on port ${port}`);
    });

    process.on('SIGINT', async () => {
      if (this.browser) {
        await this.browser.close();
      }
      process.exit(0);
    });
  }
}

if (require.main === module) {
  const server = new PlaywrightMCPServer();
  server.start();
}

module.exports = PlaywrightMCPServer;