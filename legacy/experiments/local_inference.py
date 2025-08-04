#!/usr/bin/env python3
"""
Local Amharic Model Inference - Works Offline
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import gradio as gr
from pathlib import Path

class LocalAmharicInference:
    def __init__(self):
        self.model_dir = Path("models/amharic-gpt2-local")
        self.device = torch.device('cpu')
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the local model"""
        if self.model is None:
            print("Loading model...")
            self.model = GPT2LMHeadModel.from_pretrained(self.model_dir)
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_dir)
            self.model.to(self.device)
            self.model.eval()
            print("✅ Model loaded")
    
    def generate_text(self, prompt, max_length=100, temperature=0.8):
        """Generate Amharic text"""
        self.load_model()
        
        # Encode prompt
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated
    
    def create_gradio_interface(self):
        """Create web interface"""
        def generate_wrapper(prompt, max_length, temperature):
            return self.generate_text(prompt, int(max_length), temperature)
        
        interface = gr.Interface(
            fn=generate_wrapper,
            inputs=[
                gr.Textbox(label="የአማርኛ ጽሑፍ ያስገቡ (Enter Amharic text)", 
                          placeholder="ሰላም...", lines=2),
                gr.Slider(50, 200, value=100, step=10, label="Maximum Length"),
                gr.Slider(0.1, 1.5, value=0.8, step=0.1, label="Temperature"),
            ],
            outputs=gr.Textbox(label="የመነጨ ጽሑፍ (Generated text)", lines=5),
            title="🇪🇹 Amharic Text Generator (Local)",
            description="Generate Amharic text using locally trained model",
            examples=[
                ["ሰላም ወንድሜ", 100, 0.8],
                ["ኢትዮጵያ", 150, 0.9],
                ["ዛሬ ጥሩ", 100, 0.7],
            ]
        )
        
        return interface
    
    def run_server(self):
        """Run Gradio server"""
        print("Starting Gradio server...")
        interface = self.create_gradio_interface()
        interface.launch(server_name="127.0.0.1", server_port=7860, share=False)

if __name__ == "__main__":
    inference = LocalAmharicInference()
    
    # Check if model exists
    if not (inference.model_dir / "config.json").exists():
        print("❌ No trained model found. Run local_trainer.py first!")
    else:
        inference.run_server()
