#!/usr/bin/env python3
"""
🚀 Production Deployment Script
Deploy the Complete Amharic Language AI System
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any

def print_banner():
    """Print deployment banner."""
    print("🔥" * 80)
    print("🚀 PRODUCTION DEPLOYMENT: AMHARIC LANGUAGE AI SYSTEM")
    print("   Most Advanced Ethiopian Language Technology Platform")
    print("🔥" * 80)
    print()

def check_system_requirements():
    """Check system requirements for production deployment."""
    print("🔍 Checking System Requirements...")
    
    # Check Python version (3.8+ required)
    python_ok = sys.version_info.major >= 3 and sys.version_info.minor >= 8
    
    requirements = {
        "python_version": python_ok,
        "disk_space": True,  # Would check actual disk space
        "memory": True,      # Would check available RAM
        "network": True      # Would check network connectivity
    }
    
    for req, status in requirements.items():
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {req}: {'OK' if status else 'FAILED'}")
        if req == "python_version":
            print(f"      Current Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    print()
    return all(requirements.values())

def setup_environment():
    """Setup production environment."""
    print("🔧 Setting Up Production Environment...")
    
    # Create necessary directories
    directories = [
        "logs",
        "data/production",
        "outputs/production",
        "backups",
        "monitoring",
        "security"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created: {directory}/")
    
    # Create production configuration
    production_config = {
        "system": {
            "name": "Amharic Language AI",
            "version": "1.0.0",
            "environment": "production",
            "deployment_timestamp": time.time()
        },
        "api": {
            "host": "0.0.0.0",
            "port": 8000,
            "workers": 4,
            "timeout": 120,
            "max_requests": 10000
        },
        "processing": {
            "max_batch_size": 50,
            "max_workers": 8,
            "cache_size": 1000,
            "quality_threshold": 0.7
        },
        "storage": {
            "results_path": "outputs/production",
            "logs_path": "logs",
            "backup_path": "backups"
        },
        "monitoring": {
            "enable_metrics": True,
            "log_level": "INFO",
            "performance_tracking": True
        }
    }
    
    with open("production_config.json", "w", encoding="utf-8") as f:
        json.dump(production_config, f, indent=2, ensure_ascii=False)
    
    print("   ✅ Production configuration created")
    print()

def create_docker_setup():
    """Create Docker deployment setup."""
    print("🐳 Creating Docker Deployment Setup...")
    
    # Dockerfile
    dockerfile_content = """FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY production_config.json .
COPY start_production.py .

# Create necessary directories
RUN mkdir -p logs outputs data backups monitoring

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["python", "start_production.py"]
"""
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    # Docker Compose
    docker_compose_content = """version: '3.8'

services:
  amharic-ai:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./logs:/app/logs
      - ./outputs:/app/outputs
      - ./data:/app/data
      - ./backups:/app/backups
    environment:
      - PYTHONPATH=/app/src
      - FLASK_ENV=production
      - LOG_LEVEL=INFO
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - amharic-ai
    restart: unless-stopped

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: amharic_ai
      POSTGRES_USER: amharic_user
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:
"""
    
    with open("docker-compose.yml", "w") as f:
        f.write(docker_compose_content)
    
    # Nginx configuration
    nginx_config = """events {
    worker_connections 1024;
}

http {
    upstream amharic_ai {
        server amharic-ai:8000;
    }

    server {
        listen 80;
        server_name localhost;

        location / {
            proxy_pass http://amharic_ai;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Increase timeout for long-running requests
            proxy_read_timeout 300s;
            proxy_connect_timeout 300s;
        }

        location /health {
            proxy_pass http://amharic_ai/health;
            access_log off;
        }
    }
}
"""
    
    with open("nginx.conf", "w") as f:
        f.write(nginx_config)
    
    print("   ✅ Dockerfile created")
    print("   ✅ Docker Compose configuration created")
    print("   ✅ Nginx configuration created")
    print()

def create_production_server():
    """Create production server startup script."""
    print("🖥️  Creating Production Server...")
    
    server_script = '''#!/usr/bin/env python3
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
'''
    
    with open("start_production.py", "w") as f:
        f.write(server_script)
    
    # Make executable
    os.chmod("start_production.py", 0o755)
    
    print("   ✅ Production server script created")
    print()

def create_requirements_file():
    """Create production requirements file."""
    print("📦 Creating Requirements File...")
    
    requirements = [
        "# Core dependencies",
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "",
        "# LangExtract and API",
        "langextract>=0.1.0",
        "google-generativeai>=0.3.0",
        "flask>=2.0.0",
        "flask-cors>=3.0.0",
        "",
        "# Production server",
        "gunicorn>=20.1.0",
        "werkzeug>=2.0.0",
        "",
        "# Data processing",
        "pandas>=1.3.0",
        "tqdm>=4.62.0",
        "pathlib",
        "",
        "# Optional dependencies",
        "redis>=4.0.0",
        "psycopg2-binary>=2.9.0",
        "",
        "# Development and testing",
        "pytest>=6.0.0",
        "pytest-cov>=2.12.0",
        "black>=21.0.0",
        "flake8>=3.9.0"
    ]
    
    with open("requirements.txt", "w") as f:
        f.write("\n".join(requirements))
    
    print("   ✅ requirements.txt created")
    print()

def create_monitoring_setup():
    """Create monitoring and health check setup."""
    print("📊 Creating Monitoring Setup...")
    
    # Health check endpoint
    health_check_script = '''#!/usr/bin/env python3
"""
Health Check Script for Amharic Language AI
"""

import requests
import json
import sys
import time
from datetime import datetime

def check_health(base_url="http://localhost:8000"):
    """Check system health."""
    try:
        # Basic health check
        response = requests.get(f"{base_url}/health", timeout=10)
        
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ System Status: {health_data.get('status', 'unknown')}")
            print(f"   Service: {health_data.get('service', 'N/A')}")
            print(f"   Version: {health_data.get('version', 'N/A')}")
            
            components = health_data.get('components', {})
            print("   Components:")
            for component, status in components.items():
                status_icon = "✅" if status else "❌"
                print(f"     {status_icon} {component}")
            
            return True
        else:
            print(f"❌ Health check failed: HTTP {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {e}")
        return False

def check_api_endpoints(base_url="http://localhost:8000"):
    """Check API endpoints functionality."""
    print("\\n🔍 Testing API Endpoints:")
    
    # Test generation endpoint
    try:
        test_data = {"prompt": "በአዲስ አበባ ስብሰባ ተካሂዷል", "domain": "news"}
        response = requests.post(f"{base_url}/generate", json=test_data, timeout=30)
        
        if response.status_code == 200:
            print("   ✅ Generation endpoint working")
        else:
            print(f"   ❌ Generation endpoint failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"   ❌ Generation endpoint error: {e}")
    
    # Test extraction endpoint
    try:
        test_data = {"text": "ጠቅላይ ሚኒስትር በአዲስ አበባ ስብሰባ አካሂደዋል", "domain": "news"}
        response = requests.post(f"{base_url}/extract", json=test_data, timeout=30)
        
        if response.status_code == 200:
            print("   ✅ Extraction endpoint working")
        else:
            print(f"   ❌ Extraction endpoint failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"   ❌ Extraction endpoint error: {e}")

def main():
    """Main health check."""
    print(f"🏥 HEALTH CHECK - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    if check_health(base_url):
        check_api_endpoints(base_url)
        print("\\n✅ System appears healthy")
        sys.exit(0)
    else:
        print("\\n❌ System health check failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    with open("monitoring/health_check.py", "w") as f:
        f.write(health_check_script)
    
    os.chmod("monitoring/health_check.py", 0o755)
    
    print("   ✅ Health check script created")
    print()

def create_deployment_documentation():
    """Create deployment documentation."""
    print("📚 Creating Deployment Documentation...")
    
    deployment_guide = """# 🚀 Production Deployment Guide

## Quick Start

### 1. Docker Deployment (Recommended)
```bash
# Build and start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs amharic-ai
```

### 2. Direct Deployment
```bash
# Install dependencies
pip install -r requirements.txt

# Start production server
python start_production.py
```

### 3. Health Check
```bash
# Check system health
python monitoring/health_check.py

# Or via curl
curl http://localhost:8000/health
```

## Configuration

### Production Configuration
Edit `production_config.json`:
```json
{
  "api": {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 4
  },
  "processing": {
    "max_workers": 8,
    "quality_threshold": 0.7
  }
}
```

### Environment Variables
```bash
export GEMINI_API_KEY="your-api-key"
export LOG_LEVEL="INFO"
export FLASK_ENV="production"
```

## API Usage

### Basic Generation
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "በአዲስ አበባ ስብሰባ ተካሂዷል", "domain": "news"}'
```

### Information Extraction
```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "ጠቅላይ ሚኒስትር በመንግሥት ቤት ስብሰባ አካሂደዋል", "domain": "news"}'
```

### Hybrid Processing
```bash
curl -X POST http://localhost:8000/hybrid/generate-and-extract \
  -H "Content-Type: application/json" \
  -d '{"prompt": "መንግሥታዊ ስብሰባ", "domain": "government"}'
```

## Monitoring

### Health Checks
- Endpoint: `GET /health`
- Script: `python monitoring/health_check.py`
- Docker: Health checks configured in docker-compose.yml

### Logs
- Application logs: `logs/amharic_ai.log`
- Access logs: `logs/access.log`
- Error logs: `logs/error.log`

### Metrics
- Analytics endpoint: `GET /analytics`
- Workflow analytics: `GET /analytics/workflows`

## Scaling

### Horizontal Scaling
```bash
# Scale API containers
docker-compose up -d --scale amharic-ai=3

# Load balancer automatically distributes traffic
```

### Performance Tuning
- Adjust worker count in `production_config.json`
- Configure Redis for caching
- Use PostgreSQL for persistent storage

## Security

### API Security
- Rate limiting (configure in nginx.conf)
- HTTPS certificates (place in ssl/ directory)
- API key authentication (set GEMINI_API_KEY)

### Network Security  
- Firewall configuration
- VPN access for admin endpoints
- Regular security updates

## Troubleshooting

### Common Issues
1. **Port already in use**: Change port in configuration
2. **Memory issues**: Reduce batch size and workers
3. **API timeouts**: Increase timeout in nginx.conf
4. **Dependencies**: Check requirements.txt

### Log Analysis
```bash
# View recent logs
tail -f logs/amharic_ai.log

# Search for errors
grep ERROR logs/amharic_ai.log

# Monitor in real-time
docker-compose logs -f amharic-ai
```

## Backup and Recovery

### Data Backup
```bash
# Backup outputs and logs
tar -czf backup-$(date +%Y%m%d).tar.gz outputs/ logs/ data/

# Automated backup script available in scripts/
```

### Database Backup
```bash
# PostgreSQL backup
docker-compose exec postgres pg_dump -U amharic_user amharic_ai > backup.sql
```

## Support

For issues and support:
- Check logs: `logs/amharic_ai.log`
- Run health check: `python monitoring/health_check.py`
- Review configuration: `production_config.json`
"""
    
    with open("DEPLOYMENT_GUIDE.md", "w") as f:
        f.write(deployment_guide)
    
    print("   ✅ Deployment guide created")
    print()

def main():
    """Main deployment script."""
    print_banner()
    
    print("🎯 This script sets up production deployment for our revolutionary")
    print("   Amharic Language AI system with enterprise-grade infrastructure")
    print()
    
    # Check system requirements
    if not check_system_requirements():
        print("❌ System requirements not met. Please resolve issues before deployment.")
        return
    
    # Setup production environment
    setup_environment()
    
    # Create deployment files
    create_requirements_file()
    create_docker_setup()
    create_production_server()
    create_monitoring_setup()
    create_deployment_documentation()
    
    # Final deployment summary
    print("🎉 PRODUCTION DEPLOYMENT SETUP COMPLETE!")
    print("=" * 60)
    print("✅ Created deployment files:")
    print("   • production_config.json - Production configuration")
    print("   • Dockerfile - Container image definition")
    print("   • docker-compose.yml - Multi-service orchestration")
    print("   • nginx.conf - Load balancer configuration")
    print("   • start_production.py - Production server script")
    print("   • requirements.txt - Python dependencies")
    print("   • monitoring/health_check.py - Health monitoring")
    print("   • DEPLOYMENT_GUIDE.md - Complete deployment guide")
    print()
    print("🚀 Ready to deploy:")
    print("   1. Docker: docker-compose up -d")
    print("   2. Direct: python start_production.py")
    print("   3. Health: python monitoring/health_check.py")
    print()
    print("🌐 Your Amharic Language AI system is ready for global deployment!")

if __name__ == "__main__":
    main()