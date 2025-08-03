#!/usr/bin/env python3
"""
Amharic LLM Hugging Face Spaces App
Free deployment template for Amharic language model

Features:
- Gradio interface for easy interaction
- Amharic text generation
- Model loading from Hugging Face Hub
- Responsive web interface
"""

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json
from datetime import datetime

class AmharicLLMInterface:
    """Amharic LLM interface for Hugging Face Spaces"""
    
    def __init__(self, model_name="./amharic-hnet"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the Amharic model and tokenizer"""
        try:
            print(f"Loading model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            # Set padding token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            # Fallback to a smaller model for demonstration
            self.model_name = "microsoft/DialoGPT-small"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print("⚠️ Using fallback model for demonstration")
    
    def generate_text(self, prompt, max_length=150, temperature=0.8, top_p=0.9, do_sample=True):
        """Generate Amharic text from prompt"""
        if not prompt.strip():
            return "እባክዎ ጽሑፍ ያስገቡ (Please enter some text)"
        
        try:
            # Encode the prompt
            inputs = self.tokenizer.encode(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            # Move to device if CUDA available
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            
            # Generate text
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=len(inputs[0]) + max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1,
                    early_stopping=True
                )
            
            # Decode the generated text
            generated_text = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            # Remove the original prompt from the output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text if generated_text else "ምንም ጽሑፍ አልተፈጠረም (No text generated)"
            
        except Exception as e:
            return f"ስህተት ተከስቷል (Error occurred): {str(e)}"
    
    def get_model_info(self):
        """Get model information"""
        try:
            num_params = sum(p.numel() for p in self.model.parameters())
            return {
                "model_name": self.model_name,
                "parameters": f"{num_params:,}",
                "vocab_size": len(self.tokenizer),
                "device": "CUDA" if torch.cuda.is_available() else "CPU"
            }
        except:
            return {"error": "Could not retrieve model info"}

# Initialize the model interface
print("Initializing Amharic LLM...")
llm_interface = AmharicLLMInterface()

# Define the Gradio interface
def generate_amharic_text(prompt, max_length, temperature, top_p, do_sample):
    """Wrapper function for Gradio interface"""
    return llm_interface.generate_text(
        prompt=prompt,
        max_length=int(max_length),
        temperature=float(temperature),
        top_p=float(top_p),
        do_sample=bool(do_sample)
    )

def get_examples():
    """Get example prompts"""
    return [
        ["ሰላም", 100, 0.8, 0.9, True],
        ["ኢትዮጵያ", 120, 0.7, 0.9, True],
        ["ትምህርት", 100, 0.8, 0.9, True],
        ["ቴክኖሎጂ", 100, 0.9, 0.9, True],
        ["ባህል", 100, 0.8, 0.9, True]
    ]

# Create Gradio interface
with gr.Blocks(title="Amharic LLM - አማርኛ ቋንቋ ሞዴል", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🇪🇹 Amharic Language Model - አማርኛ ቋንቋ ሞዴል
        
        This is an AI model trained to generate text in Amharic (አማርኛ). 
        Enter a prompt in Amharic and the model will continue the text.
        
        **አማርኛ ጽሑፍ ለማመንጨት የሚያገለግል AI ሞዴል ነው። አማርኛ ጽሑፍ ያስገቡ እና ሞዴሉ ይቀጥላል።**
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            prompt_input = gr.Textbox(
                label="Prompt (አማርኛ ጽሑፍ ያስገቡ)",
                placeholder="ሰላም፣ እንዴት ነዎት?",
                lines=3,
                max_lines=5
            )
            
            with gr.Row():
                max_length = gr.Slider(
                    minimum=10,
                    maximum=300,
                    value=100,
                    step=10,
                    label="Max Length (ከፍተኛ ርዝመት)"
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.8,
                    step=0.1,
                    label="Temperature (ፈጠራ ደረጃ)"
                )
            
            with gr.Row():
                top_p = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.1,
                    label="Top-p (ምርጫ ወሰን)"
                )
                do_sample = gr.Checkbox(
                    value=True,
                    label="Enable Sampling (ናሙና አወሳሰድ)"
                )
            
            generate_btn = gr.Button(
                "Generate Text (ጽሑፍ አመንጭ)", 
                variant="primary",
                size="lg"
            )
        
        with gr.Column(scale=2):
            output_text = gr.Textbox(
                label="Generated Text (የተፈጠረ ጽሑፍ)",
                lines=8,
                max_lines=15,
                interactive=False
            )
    
    # Examples section
    gr.Markdown("### Examples (ምሳሌዎች)")
    examples = gr.Examples(
        examples=get_examples(),
        inputs=[prompt_input, max_length, temperature, top_p, do_sample],
        outputs=output_text,
        fn=generate_amharic_text,
        cache_examples=False
    )
    
    # Model information
    with gr.Accordion("Model Information (የሞዴል መረጃ)", open=False):
        model_info = llm_interface.get_model_info()
        info_text = f"""
        **Model Name:** {model_info.get('model_name', 'Unknown')}
        **Parameters:** {model_info.get('parameters', 'Unknown')}
        **Vocabulary Size:** {model_info.get('vocab_size', 'Unknown')}
        **Device:** {model_info.get('device', 'Unknown')}
        """
        gr.Markdown(info_text)
    
    # Usage instructions
    with gr.Accordion("How to Use (አጠቃቀም መመሪያ)", open=False):
        gr.Markdown(
            """
            ### English Instructions:
            1. Enter an Amharic text prompt in the input box
            2. Adjust the generation parameters if needed:
               - **Max Length**: Maximum number of tokens to generate
               - **Temperature**: Controls randomness (higher = more creative)
               - **Top-p**: Controls diversity of word selection
               - **Enable Sampling**: Whether to use random sampling
            3. Click "Generate Text" to create Amharic text
            
            ### አማርኛ መመሪያ:
            1. በግቤት ሳጥን ውስጥ አማርኛ ጽሑፍ ያስገቡ
            2. አስፈላጊ ከሆነ የማመንጨት መለኪያዎችን ያስተካክሉ
            3. "ጽሑፍ አመንጭ" የሚለውን ቁልፍ ይጫኑ
            """
        )
    
    # Connect the generate button
    generate_btn.click(
        fn=generate_amharic_text,
        inputs=[prompt_input, max_length, temperature, top_p, do_sample],
        outputs=output_text
    )
    
    # Footer
    gr.Markdown(
        """
        ---
        **Note:** This model is for research and educational purposes. 
        Generated text may not always be accurate or appropriate.
        
        **ማስታወሻ:** ይህ ሞዴል ለምርምር እና ትምህርት ዓላማ ነው። የተፈጠረው ጽሑፍ ሁልጊዜ ትክክል ወይም ተገቢ ላይሆን ይችላል።
        """
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )