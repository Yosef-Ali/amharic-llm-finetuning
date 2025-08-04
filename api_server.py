#!/usr/bin/env python3
"""REST API server for Amharic H-Net text generation."""

import sys
from pathlib import Path
import json
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
import uvicorn

from generate import AmharicGenerator
from amharichnet.evaluation import AmharicTextEvaluator


# API Models
class GenerationRequest(BaseModel):
    prompt: str = Field(default="", description="Starting text prompt")
    category: str = Field(default="general", description="Text category (general, news, educational, cultural, conversation)")
    length: int = Field(default=50, ge=10, le=200, description="Generated text length")
    temperature: float = Field(default=1.0, ge=0.1, le=2.0, description="Generation temperature")
    top_k: int = Field(default=50, ge=1, le=100, description="Top-k sampling")
    top_p: float = Field(default=0.9, ge=0.1, le=1.0, description="Top-p (nucleus) sampling")
    max_time: int = Field(default=30, ge=5, le=120, description="Maximum generation time in seconds")


class GenerationResponse(BaseModel):
    text: str
    prompt: str
    category: str
    length: int
    quality_score: float
    generation_time: float
    model_info: Dict[str, str]


class BatchGenerationRequest(BaseModel):
    requests: List[GenerationRequest] = Field(..., max_items=10, description="Batch of generation requests")


class EvaluationRequest(BaseModel):
    text: str = Field(..., description="Text to evaluate")


class EvaluationResponse(BaseModel):
    text: str
    scores: Dict[str, float]
    evaluation_time: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime: str
    requests_served: int


# Rate limiting
class RateLimiter:
    def __init__(self):
        self.requests = defaultdict(list)
        self.daily_limits = {"free": 100, "premium": 1000}
        self.hourly_limits = {"free": 20, "premium": 200}
    
    def is_allowed(self, client_ip: str, tier: str = "free") -> bool:
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)
        
        # Clean old requests
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip] 
            if req_time > day_ago
        ]
        
        recent_requests = [
            req_time for req_time in self.requests[client_ip]
            if req_time > hour_ago
        ]
        
        daily_count = len(self.requests[client_ip])
        hourly_count = len(recent_requests)
        
        return (
            daily_count < self.daily_limits[tier] and
            hourly_count < self.hourly_limits[tier]
        )
    
    def record_request(self, client_ip: str):
        self.requests[client_ip].append(datetime.now())


# Initialize components
app = FastAPI(
    title="Amharic H-Net API",
    description="REST API for Amharic text generation using H-Net architecture",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components
generator = None
evaluator = None
rate_limiter = RateLimiter()
start_time = time.time()
request_count = 0


async def get_rate_limit_info(request: Request):
    """Check rate limiting."""
    client_ip = request.client.host
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )
    return client_ip


@app.on_event("startup")
async def startup_event():
    """Initialize the API server."""
    global generator, evaluator
    
    print("🚀 Starting Amharic H-Net API Server...")
    
    try:
        # Initialize generator
        print("   Loading Amharic text generator...")
        generator = AmharicGenerator()
        print("   ✅ Generator loaded successfully")
        
        # Initialize evaluator
        print("   Loading text evaluator...")
        evaluator = AmharicTextEvaluator()
        print("   ✅ Evaluator loaded successfully")
        
        print("🎉 API Server ready!")
        
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        raise


@app.get("/", response_class=HTMLResponse)
async def root():
    """API homepage with basic information."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Amharic H-Net API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }
            h1 { color: #2c3e50; }
            .endpoint { background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { font-weight: bold; color: #27ae60; }
            code { background: #34495e; color: white; padding: 2px 5px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🇪🇹 Amharic H-Net API</h1>
            <p>REST API for Amharic text generation using Hierarchical Network architecture.</p>
            
            <h2>Available Endpoints:</h2>
            
            <div class="endpoint">
                <div class="method">POST</div> <code>/generate</code>
                <p>Generate Amharic text from a prompt</p>
            </div>
            
            <div class="endpoint">
                <div class="method">POST</div> <code>/generate/batch</code>
                <p>Generate multiple texts in batch</p>
            </div>
            
            <div class="endpoint">
                <div class="method">POST</div> <code>/evaluate</code>
                <p>Evaluate Amharic text quality</p>
            </div>
            
            <div class="endpoint">
                <div class="method">GET</div> <code>/health</code>
                <p>API health status</p>
            </div>
            
            <div class="endpoint">
                <div class="method">GET</div> <code>/docs</code>
                <p>Interactive API documentation</p>
            </div>
            
            <h2>Quick Example:</h2>
            <pre><code>curl -X POST "http://localhost:8000/generate" \\
  -H "Content-Type: application/json" \\
  -d '{"prompt": "ኢትዮጵያ", "length": 50}'</code></pre>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global request_count, start_time
    
    uptime_seconds = time.time() - start_time
    uptime_str = f"{uptime_seconds/3600:.1f} hours"
    
    return HealthResponse(
        status="healthy",
        model_loaded=generator is not None,
        uptime=uptime_str,
        requests_served=request_count
    )


@app.post("/generate", response_model=GenerationResponse)
async def generate_text(
    request_data: GenerationRequest,
    client_ip: str = Depends(get_rate_limit_info)
):
    """Generate Amharic text."""
    global request_count
    
    if not generator:
        raise HTTPException(status_code=503, detail="Generator not available")
    
    try:
        start_time_gen = time.time()
        
        # Generate text (H-Net generator only supports category, prompt, length)
        generated_text = generator.generate_text(
            category=request_data.category,
            prompt=request_data.prompt,
            length=request_data.length
        )
        
        # Evaluate quality
        quality_scores = evaluator.evaluate_text(generated_text)
        quality_score = quality_scores.get('overall_quality', 0.0)
        
        generation_time = time.time() - start_time_gen
        
        # Record request
        rate_limiter.record_request(client_ip)
        request_count += 1
        
        return GenerationResponse(
            text=generated_text,
            prompt=request_data.prompt,
            category=request_data.category,
            length=len(generated_text.split()),
            quality_score=quality_score,
            generation_time=generation_time,
            model_info={
                "architecture": "H-Net",
                "language": "Amharic",
                "parameters": "6.85M"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/generate/batch")
async def generate_batch(
    batch_request: BatchGenerationRequest,
    client_ip: str = Depends(get_rate_limit_info)
):
    """Generate multiple texts in batch."""
    global request_count
    
    if not generator:
        raise HTTPException(status_code=503, detail="Generator not available")
    
    # Check batch size limits
    if len(batch_request.requests) > 10:
        raise HTTPException(status_code=400, detail="Batch size limited to 10 requests")
    
    try:
        results = []
        start_time_batch = time.time()
        
        for req in batch_request.requests:
            start_time_gen = time.time()
            
            # Generate text (H-Net generator only supports category, prompt, length)
            generated_text = generator.generate_text(
                category=req.category,
                prompt=req.prompt,
                length=req.length
            )
            
            # Evaluate quality
            quality_scores = evaluator.evaluate_text(generated_text)
            quality_score = quality_scores.get('overall_quality', 0.0)
            
            generation_time = time.time() - start_time_gen
            
            results.append(GenerationResponse(
                text=generated_text,
                prompt=req.prompt,
                category=req.category,
                length=len(generated_text.split()),
                quality_score=quality_score,
                generation_time=generation_time,
                model_info={
                    "architecture": "H-Net",
                    "language": "Amharic",
                    "parameters": "6.85M"
                }
            ))
        
        batch_time = time.time() - start_time_batch
        
        # Record request
        rate_limiter.record_request(client_ip)
        request_count += len(batch_request.requests)
        
        return {
            "results": results,
            "batch_size": len(results),
            "total_time": batch_time,
            "avg_time_per_request": batch_time / len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch generation failed: {str(e)}")


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_text_endpoint(
    request_data: EvaluationRequest,
    client_ip: str = Depends(get_rate_limit_info)
):
    """Evaluate Amharic text quality."""
    global request_count
    
    if not evaluator:
        raise HTTPException(status_code=503, detail="Evaluator not available")
    
    try:
        start_time_eval = time.time()
        
        # Evaluate text
        scores = evaluator.evaluate_text(request_data.text)
        
        evaluation_time = time.time() - start_time_eval
        
        # Record request
        rate_limiter.record_request(client_ip)
        request_count += 1
        
        return EvaluationResponse(
            text=request_data.text,
            scores=scores,
            evaluation_time=evaluation_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.get("/stats")
async def get_stats():
    """Get API usage statistics."""
    global request_count, start_time
    
    uptime_seconds = time.time() - start_time
    
    return {
        "uptime_hours": uptime_seconds / 3600,
        "total_requests": request_count,
        "requests_per_hour": request_count / (uptime_seconds / 3600) if uptime_seconds > 0 else 0,
        "model_status": "loaded" if generator else "not_loaded",
        "evaluator_status": "loaded" if evaluator else "not_loaded"
    }


if __name__ == "__main__":
    print("🚀 Starting Amharic H-Net API Server...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🏠 Homepage: http://localhost:8000/")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info"
    )