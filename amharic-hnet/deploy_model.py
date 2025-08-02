#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Deployment script for Amharic H-Net model.

This script provides functions for deploying the model as a web service,
including a REST API and a simple web interface.
"""

import os
import json
import time
import logging
import argparse
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path

import torch
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Global variables
model = None
tokenizer = None
generator = None
use_onnx = False
onnx_session = None
model_type = "hnet"


def load_model(model_path: Union[str, Path], 
              model_type: str = "hnet",
              use_onnx: bool = False,
              device: str = "cpu") -> None:
    """Load model and tokenizer.
    
    Args:
        model_path: Path to model directory
        model_type: Type of model (hnet or improved)
        use_onnx: Whether to use ONNX model
        device: Device to use for inference
    """
    global model, tokenizer, generator, onnx_session
    
    model_path = Path(model_path)
    
    try:
        if use_onnx:
            # Load ONNX model
            import onnxruntime as ort
            
            # Find ONNX model file
            onnx_files = list(model_path.glob("*.onnx"))
            if not onnx_files:
                logger.error(f"No ONNX model found in {model_path}")
                return
            
            onnx_path = onnx_files[0]
            logger.info(f"Loading ONNX model from {onnx_path}")
            
            # Create ONNX Runtime session
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            onnx_session = ort.InferenceSession(str(onnx_path), session_options)
            
            # Load tokenizer
            from transformers import AutoTokenizer
            logger.info(f"Loading tokenizer from {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            logger.info(f"ONNX model and tokenizer loaded successfully")
        else:
            # Load model and tokenizer based on model type
            if model_type == "hnet":
                # Load original H-Net model
                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                logger.info(f"Loading original H-Net model from {model_path}")
                model = AutoModelForCausalLM.from_pretrained(model_path)
                model.to(device)
                model.eval()
                
                logger.info(f"Loading tokenizer from {model_path}")
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                
                # Create generator
                from amharic_hnet.generator import AmharicGenerator
                generator = AmharicGenerator(model, tokenizer)
            else:
                # Load improved H-Net model
                from amharic_hnet.model import HNetTransformer
                from transformers import AutoTokenizer
                
                logger.info(f"Loading improved H-Net model from {model_path}")
                model = HNetTransformer.from_pretrained(model_path)
                model.to(device)
                model.eval()
                
                logger.info(f"Loading tokenizer from {model_path}")
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                
                # Create generator
                from amharic_hnet.generator import ImprovedAmharicGenerator
                generator = ImprovedAmharicGenerator(model, tokenizer)
            
            logger.info(f"Model and tokenizer loaded successfully")
    except ImportError as e:
        logger.error(f"Failed to import required packages: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load model and tokenizer: {e}")
        raise


def generate_text(prompt: str, 
                 max_length: int = 100,
                 temperature: float = 0.7,
                 top_k: int = 50,
                 top_p: float = 0.9,
                 repetition_penalty: float = 1.2,
                 num_return_sequences: int = 1) -> List[str]:
    """Generate text from prompt.
    
    Args:
        prompt: Input prompt
        max_length: Maximum length of generated text
        temperature: Temperature for sampling
        top_k: Top-k sampling parameter
        top_p: Top-p sampling parameter
        repetition_penalty: Repetition penalty parameter
        num_return_sequences: Number of sequences to return
        
    Returns:
        List of generated texts
    """
    global model, tokenizer, generator, onnx_session, use_onnx
    
    if use_onnx:
        # Generate text using ONNX model
        try:
            # Tokenize prompt
            inputs = tokenizer(prompt, return_tensors="np")
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            
            # Run inference
            ort_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            
            # Generate text iteratively
            for _ in range(max_length):
                outputs = onnx_session.run(None, ort_inputs)
                next_token_logits = outputs[0][0, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply top-k sampling
                top_k_logits, top_k_indices = torch.topk(torch.tensor(next_token_logits), top_k)
                
                # Apply top-p sampling
                probs = torch.softmax(top_k_logits, dim=-1).numpy()
                cumulative_probs = np.cumsum(probs)
                top_p_mask = cumulative_probs < top_p
                top_p_mask[np.argmax(top_p_mask)] = True
                filtered_indices = top_k_indices[top_p_mask].numpy()
                filtered_probs = probs[top_p_mask]
                filtered_probs = filtered_probs / np.sum(filtered_probs)
                
                # Sample next token
                next_token = np.random.choice(filtered_indices, p=filtered_probs)
                
                # Append next token to input_ids
                input_ids = np.concatenate([input_ids, [[next_token]]], axis=1)
                attention_mask = np.concatenate([attention_mask, [[1]]], axis=1)
                
                # Update ort_inputs
                ort_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
                
                # Check if end of sequence
                if next_token == tokenizer.eos_token_id:
                    break
            
            # Decode generated text
            generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            
            return [generated_text]
        except Exception as e:
            logger.error(f"Failed to generate text using ONNX model: {e}")
            return [f"Error: {e}"]
    else:
        # Generate text using PyTorch model
        try:
            if generator is None:
                logger.error("Generator not initialized")
                return ["Error: Generator not initialized"]
            
            # Generate text
            generated_texts = generator.generate(
                prompt,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_return_sequences=num_return_sequences,
            )
            
            return generated_texts
        except Exception as e:
            logger.error(f"Failed to generate text: {e}")
            return [f"Error: {e}"]


@app.route("/")
def index():
    """Render index page."""
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    """Generate text from prompt."""
    try:
        # Get request data
        data = request.json
        prompt = data.get("prompt", "")
        max_length = int(data.get("max_length", 100))
        temperature = float(data.get("temperature", 0.7))
        top_k = int(data.get("top_k", 50))
        top_p = float(data.get("top_p", 0.9))
        repetition_penalty = float(data.get("repetition_penalty", 1.2))
        num_return_sequences = int(data.get("num_return_sequences", 1))
        
        # Validate parameters
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        
        if max_length < 1 or max_length > 1000:
            return jsonify({"error": "max_length must be between 1 and 1000"}), 400
        
        if temperature < 0.1 or temperature > 2.0:
            return jsonify({"error": "temperature must be between 0.1 and 2.0"}), 400
        
        if top_k < 1 or top_k > 100:
            return jsonify({"error": "top_k must be between 1 and 100"}), 400
        
        if top_p < 0.1 or top_p > 1.0:
            return jsonify({"error": "top_p must be between 0.1 and 1.0"}), 400
        
        if repetition_penalty < 1.0 or repetition_penalty > 2.0:
            return jsonify({"error": "repetition_penalty must be between 1.0 and 2.0"}), 400
        
        if num_return_sequences < 1 or num_return_sequences > 5:
            return jsonify({"error": "num_return_sequences must be between 1 and 5"}), 400
        
        # Generate text
        start_time = time.time()
        generated_texts = generate_text(
            prompt,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
        )
        generation_time = time.time() - start_time
        
        # Return response
        return jsonify({
            "generated_texts": generated_texts,
            "generation_time": generation_time,
        })
    except Exception as e:
        logger.error(f"Error in generate endpoint: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


def create_templates(output_dir: Union[str, Path]) -> None:
    """Create HTML templates for web interface.
    
    Args:
        output_dir: Directory to save templates
    """
    output_dir = Path(output_dir)
    templates_dir = output_dir / "templates"
    static_dir = output_dir / "static"
    
    # Create directories
    templates_dir.mkdir(parents=True, exist_ok=True)
    static_dir.mkdir(parents=True, exist_ok=True)
    
    # Create index.html
    index_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Amharic H-Net Text Generator</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <div class="container">
        <h1>Amharic H-Net Text Generator</h1>
        <div class="form-container">
            <div class="form-group">
                <label for="prompt">Prompt:</label>
                <textarea id="prompt" rows="5" placeholder="Enter your prompt in Amharic..."></textarea>
            </div>
            <div class="form-row">
                <div class="form-group">
                    <label for="max-length">Max Length:</label>
                    <input type="number" id="max-length" value="100" min="1" max="1000">
                </div>
                <div class="form-group">
                    <label for="temperature">Temperature:</label>
                    <input type="number" id="temperature" value="0.7" min="0.1" max="2.0" step="0.1">
                </div>
            </div>
            <div class="form-row">
                <div class="form-group">
                    <label for="top-k">Top K:</label>
                    <input type="number" id="top-k" value="50" min="1" max="100">
                </div>
                <div class="form-group">
                    <label for="top-p">Top P:</label>
                    <input type="number" id="top-p" value="0.9" min="0.1" max="1.0" step="0.1">
                </div>
            </div>
            <div class="form-row">
                <div class="form-group">
                    <label for="repetition-penalty">Repetition Penalty:</label>
                    <input type="number" id="repetition-penalty" value="1.2" min="1.0" max="2.0" step="0.1">
                </div>
                <div class="form-group">
                    <label for="num-sequences">Number of Sequences:</label>
                    <input type="number" id="num-sequences" value="1" min="1" max="5">
                </div>
            </div>
            <div class="form-group">
                <button id="generate-btn">Generate</button>
            </div>
        </div>
        <div class="results-container">
            <h2>Generated Text</h2>
            <div id="loading" class="loading hidden">Generating...</div>
            <div id="results"></div>
            <div id="generation-time" class="generation-time"></div>
        </div>
    </div>
    <script src="{{ url_for('static', filename='script.js') }}"></script>
</body>
</html>
"""
    
    # Create style.css
    style_css = """
* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    color: #333;
    background-color: #f5f5f5;
}

.container {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
}

h1 {
    text-align: center;
    margin-bottom: 20px;
    color: #2c3e50;
}

h2 {
    margin-bottom: 15px;
    color: #2c3e50;
}

.form-container {
    background-color: #fff;
    border-radius: 5px;
    padding: 20px;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    margin-bottom: 20px;
}

.form-group {
    margin-bottom: 15px;
}

.form-row {
    display: flex;
    gap: 20px;
}

.form-row .form-group {
    flex: 1;
}

label {
    display: block;
    margin-bottom: 5px;
    font-weight: 600;
}

input, textarea {
    width: 100%;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 16px;
}

textarea {
    resize: vertical;
}

button {
    background-color: #3498db;
    color: #fff;
    border: none;
    padding: 10px 20px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 16px;
    transition: background-color 0.3s;
}

button:hover {
    background-color: #2980b9;
}

.results-container {
    background-color: #fff;
    border-radius: 5px;
    padding: 20px;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
}

.result {
    margin-bottom: 20px;
    padding: 15px;
    background-color: #f9f9f9;
    border-radius: 4px;
    border-left: 4px solid #3498db;
}

.result:last-child {
    margin-bottom: 0;
}

.result h3 {
    margin-bottom: 10px;
    color: #2c3e50;
}

.result p {
    white-space: pre-wrap;
    font-size: 16px;
    line-height: 1.6;
}

.loading {
    text-align: center;
    padding: 20px;
    font-size: 18px;
    color: #666;
}

.hidden {
    display: none;
}

.generation-time {
    text-align: right;
    margin-top: 10px;
    font-size: 14px;
    color: #666;
}

@media (max-width: 600px) {
    .form-row {
        flex-direction: column;
        gap: 0;
    }
}
"""
    
    # Create script.js
    script_js = """
document.addEventListener('DOMContentLoaded', function() {
    const generateBtn = document.getElementById('generate-btn');
    const promptInput = document.getElementById('prompt');
    const maxLengthInput = document.getElementById('max-length');
    const temperatureInput = document.getElementById('temperature');
    const topKInput = document.getElementById('top-k');
    const topPInput = document.getElementById('top-p');
    const repetitionPenaltyInput = document.getElementById('repetition-penalty');
    const numSequencesInput = document.getElementById('num-sequences');
    const resultsContainer = document.getElementById('results');
    const loadingIndicator = document.getElementById('loading');
    const generationTimeElement = document.getElementById('generation-time');

    generateBtn.addEventListener('click', generateText);

    async function generateText() {
        // Get input values
        const prompt = promptInput.value.trim();
        const maxLength = parseInt(maxLengthInput.value);
        const temperature = parseFloat(temperatureInput.value);
        const topK = parseInt(topKInput.value);
        const topP = parseFloat(topPInput.value);
        const repetitionPenalty = parseFloat(repetitionPenaltyInput.value);
        const numSequences = parseInt(numSequencesInput.value);

        // Validate input
        if (!prompt) {
            alert('Please enter a prompt.');
            return;
        }

        // Show loading indicator
        loadingIndicator.classList.remove('hidden');
        resultsContainer.innerHTML = '';
        generationTimeElement.textContent = '';

        try {
            // Send request to server
            const response = await fetch('/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    prompt,
                    max_length: maxLength,
                    temperature,
                    top_k: topK,
                    top_p: topP,
                    repetition_penalty: repetitionPenalty,
                    num_return_sequences: numSequences
                })
            });

            const data = await response.json();

            // Hide loading indicator
            loadingIndicator.classList.add('hidden');

            // Check for error
            if (data.error) {
                resultsContainer.innerHTML = `<div class="result"><p>Error: ${data.error}</p></div>`;
                return;
            }

            // Display results
            const generatedTexts = data.generated_texts;
            let resultsHTML = '';

            generatedTexts.forEach((text, index) => {
                resultsHTML += `
                    <div class="result">
                        <h3>Sequence ${index + 1}</h3>
                        <p>${text}</p>
                    </div>
                `;
            });

            resultsContainer.innerHTML = resultsHTML;
            generationTimeElement.textContent = `Generation time: ${data.generation_time.toFixed(2)} seconds`;
        } catch (error) {
            // Hide loading indicator
            loadingIndicator.classList.add('hidden');

            // Display error
            resultsContainer.innerHTML = `<div class="result"><p>Error: ${error.message}</p></div>`;
        }
    }
});
"""
    
    # Write files
    with open(templates_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(index_html)
    
    with open(static_dir / "style.css", "w", encoding="utf-8") as f:
        f.write(style_css)
    
    with open(static_dir / "script.js", "w", encoding="utf-8") as f:
        f.write(script_js)
    
    logger.info(f"Templates created in {templates_dir} and {static_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Deploy Amharic H-Net model as a web service")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to model directory")
    parser.add_argument("--model_type", type=str, default="improved", choices=["hnet", "improved"], help="Type of model")
    parser.add_argument("--use_onnx", action="store_true", help="Use ONNX model")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use for inference")
    
    # Server arguments
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run server on")
    parser.add_argument("--port", type=int, default=5000, help="Port to run server on")
    parser.add_argument("--debug", action="store_true", help="Run server in debug mode")
    
    # Template arguments
    parser.add_argument("--create_templates", action="store_true", help="Create HTML templates")
    parser.add_argument("--templates_dir", type=str, default=".", help="Directory to save templates")
    
    args = parser.parse_args()
    
    # Create templates
    if args.create_templates:
        create_templates(args.templates_dir)
        return
    
    # Set global variables
    global model_type, use_onnx
    model_type = args.model_type
    use_onnx = args.use_onnx
    
    # Load model
    load_model(
        model_path=args.model_path,
        model_type=args.model_type,
        use_onnx=args.use_onnx,
        device=args.device
    )
    
    # Run server
    logger.info(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()