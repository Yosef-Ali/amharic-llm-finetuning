FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y     build-essential     curl     && rm -rf /var/lib/apt/lists/*

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
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3     CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["python", "start_production.py"]
