#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

class SimpleAmharicTest:
    def __init__(self):
        self.model_path = "models/amharic-gpt2-local"
        self.device = torch.device("cpu")
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        if self.model is None:
            print(f"Loading model from {self.model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully!")
    
    def generate_response(self, prompt):
        self.load_model()
        
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=200, truncation=True)
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 50,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if len(response) > len(prompt):
                response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            return f"Error: {e}"

if __name__ == "__main__":
    app = SimpleAmharicTest()
    
    print("🇪🇹 Simple Amharic H-Net Test")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
            
        response = app.generate_response(user_input)
        print(f"Bot: {response}\n")