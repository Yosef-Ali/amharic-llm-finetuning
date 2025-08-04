# 🚀 Production Deployment Guide

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
curl -X POST http://localhost:8000/generate   -H "Content-Type: application/json"   -d '{"prompt": "በአዲስ አበባ ስብሰባ ተካሂዷል", "domain": "news"}'
```

### Information Extraction
```bash
curl -X POST http://localhost:8000/extract   -H "Content-Type: application/json"   -d '{"text": "ጠቅላይ ሚኒስትር በመንግሥት ቤት ስብሰባ አካሂደዋል", "domain": "news"}'
```

### Hybrid Processing
```bash
curl -X POST http://localhost:8000/hybrid/generate-and-extract   -H "Content-Type: application/json"   -d '{"prompt": "መንግሥታዊ ስብሰባ", "domain": "government"}'
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
