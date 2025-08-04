#!/usr/bin/env python3
"""
Production Server for Amharic Language AI
Enterprise-grade deployment with monitoring and logging
"""

import os
import sys
import json
import logging
import signal
import time
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from flask import Flask
    from werkzeug.serving import WSGIRequestHandler
    import gunicorn.app.base
    GUNICORN_AVAILABLE = True
except ImportError:
    GUNICORN_AVAILABLE = False

from amharichnet.api.hybrid_api import create_hybrid_api_server
from amharichnet.hybrid.amharic_language_ai import LanguageAIConfig

class ProductionServer:
    """Production server for Amharic Language AI."""
    
    def __init__(self, config_path: str = "production_config.json"):
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.api_server = None
        self.running = False
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load production configuration."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Configuration file not found: {config_path}")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default production configuration."""
        return {
            "api": {"host": "0.0.0.0", "port": 8000, "workers": 4},
            "processing": {"max_workers": 8, "quality_threshold": 0.7},
            "storage": {"logs_path": "logs", "results_path": "outputs/production"}
        }
    
    def setup_logging(self):
        """Setup production logging."""
        logs_path = Path(self.config.get("storage", {}).get("logs_path", "logs"))
        logs_path.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(logs_path / "amharic_ai.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger("AmharicAI")
        self.logger.info("🚀 Production logging initialized")
    
    def create_api_server(self):
        """Create API server with production configuration."""
        ai_config = LanguageAIConfig(
            max_generation_length=500,
            quality_threshold=self.config.get("processing", {}).get("quality_threshold", 0.7),
            batch_size=self.config.get("processing", {}).get("max_batch_size", 50),
            enable_caching=True
        )
        
        self.api_server = create_hybrid_api_server(config=ai_config)
        self.logger.info("✅ API server created with production configuration")
    
    def setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers."""
        def signal_handler(signum, frame):
            self.logger.info(f"📢 Received signal {signum}, initiating graceful shutdown...")
            self.shutdown()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def start(self):
        """Start the production server."""
        try:
            self.logger.info("🚀 Starting Amharic Language AI Production Server")
            self.logger.info(f"   Version: {self.config.get('system', {}).get('version', '1.0.0')}")
            self.logger.info(f"   Environment: {self.config.get('system', {}).get('environment', 'production')}")
            
            self.create_api_server()
            self.setup_signal_handlers()
            
            api_config = self.config.get("api", {})
            host = api_config.get("host", "0.0.0.0")
            port = api_config.get("port", 8000)
            
            self.running = True
            self.logger.info(f"🌐 Server starting on http://{host}:{port}")
            
            if GUNICORN_AVAILABLE and not os.getenv("FLASK_DEBUG"):
                self.start_with_gunicorn()
            else:
                self.start_with_flask()
                
        except Exception as e:
            self.logger.error(f"❌ Failed to start server: {e}")
            raise
    
    def start_with_gunicorn(self):
        """Start with Gunicorn for production."""
        class GunicornApp(gunicorn.app.base.BaseApplication):
            def __init__(self, app, options=None):
                self.options = options or {}
                self.application = app
                super().__init__()
            
            def load_config(self):
                for key, value in self.options.items():
                    self.cfg.set(key.lower(), value)
            
            def load(self):
                return self.application
        
        options = {
            'bind': f"{self.config['api']['host']}:{self.config['api']['port']}",
            'workers': self.config['api'].get('workers', 4),
            'worker_class': 'sync',
            'worker_connections': 1000,
            'timeout': self.config['api'].get('timeout', 120),
            'keepalive': 5,
            'max_requests': self.config['api'].get('max_requests', 10000),
            'preload_app': True,
            'accesslog': str(Path(self.config['storage']['logs_path']) / 'access.log'),
            'errorlog': str(Path(self.config['storage']['logs_path']) / 'error.log'),
            'loglevel': 'info'
        }
        
        self.logger.info("🔥 Starting with Gunicorn (Production Mode)")
        GunicornApp(self.api_server.app, options).run()
    
    def start_with_flask(self):
        """Start with Flask development server."""
        self.logger.info("⚠️  Starting with Flask (Development Mode)")
        api_config = self.config.get("api", {})
        
        self.api_server.run(
            host=api_config.get("host", "0.0.0.0"),
            port=api_config.get("port", 8000),
            debug=False,
            threaded=True
        )
    
    def shutdown(self):
        """Graceful shutdown."""
        if self.running:
            self.logger.info("🛑 Shutting down Amharic Language AI Server...")
            self.running = False
            # Additional cleanup would go here
            self.logger.info("✅ Server shutdown complete")

def main():
    """Main production server entry point."""
    print("🚀 AMHARIC LANGUAGE AI - PRODUCTION SERVER")
    print("=" * 60)
    
    server = ProductionServer()
    server.start()

if __name__ == "__main__":
    main()
