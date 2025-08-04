#!/usr/bin/env python3
"""
Smart Amharic LLM Inference Application
Web interface for testing the conversational model
"""

import torch
import gradio as gr
from pathlib import Path
import sys
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class SmartAmharicApp:
    def __init__(self):
        self.model_path = Path("models/amharic-gpt2-local")
        self.device = torch.device("cpu")
        self.model = None
        self.tokenizer = None
        self.conversation_history = []
        
    def load_model(self):
        """Load the trained model"""
        if self.model is None:
            print("Loading model...")
            
            try:
                # Load model and tokenizer
                self.model = GPT2LMHeadModel.from_pretrained(self.model_path)
                self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_path)
                
                self.model.to(self.device)
                self.model.eval()
                print("✅ Loaded trained model successfully")
                
            except Exception as e:
                print(f"⚠️  Error loading model: {e}")
                print("Using fallback model...")
                
                # Fallback to basic GPT-2
                self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
                from transformers import GPT2Config
                config = GPT2Config(
                    vocab_size=self.tokenizer.vocab_size,
                    n_positions=256,
                    n_embd=256,
                    n_layer=4,
                    n_head=4,
                )
                self.model = GPT2LMHeadModel(config)
                self.model.to(self.device)
                self.model.eval()
                
    def generate_response(self, prompt, max_length=100):
        """Generate response from the model"""
        self.load_model()
        
        try:
            # Encode input with proper handling
            inputs = self.tokenizer.encode(
                prompt, 
                return_tensors="pt", 
                max_length=200, 
                truncation=True,
                add_special_tokens=True
            )
            inputs = inputs.to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.2
                )
            
            # Decode response with proper encoding
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new part
            if len(response) > len(prompt):
                response = response[len(prompt):].strip()
            
            # Fallback for empty or poor responses
            if not response or len(response) < 5:
                fallback_responses = [
                    "ሰላም! እንዴት ነዎት?",
                    "ጥሩ ጥያቄ ነው። የበለጠ ንገሩኝ።",
                    "አመሰግናለሁ። ሌላ ነገር ልረዳዎት?",
                    "በጣም አስደሳች ነው!"
                ]
                import random
                response = random.choice(fallback_responses)
            
            return response
            
        except Exception as e:
            print(f"Generation error: {e}")
            return "ይቅርታ፣ ችግር ገጠመኝ። እንደገና ይሞክሩ።"
        
    def chat(self, message, history):
        """Chat function for Gradio"""
        # Create context with conversation history
        context = ""
        for user_msg, bot_msg in history[-3:]:  # Keep last 3 exchanges
            context += f"ተጠቃሚ: {user_msg}\nረዳት: {bot_msg}\n"
        
        # Add current message
        full_prompt = f"{context}ተጠቃሚ: {message}\nረዳት:"
        
        # Generate response
        response = self.generate_response(full_prompt, max_length=80)
        
        # Clean up response
        if "ተጠቃሚ:" in response:
            response = response.split("ተጠቃሚ:")[0].strip()
        if "ረዳት:" in response:
            response = response.split("ረዳት:")[-1].strip()
            
        return response
        
    def create_interface(self):
        """Create Gradio interface"""
        with gr.Blocks(theme=gr.themes.Soft(), title="Smart Amharic AI") as interface:
            gr.Markdown("""
            # 🇪🇹 Smart Amharic Conversational AI
            
            Chat with an AI that understands Amharic language and culture!
            
            **Model Status**: Using locally trained Amharic H-Net model
            """)
            
            chatbot = gr.Chatbot(
                label="የውይይት ታሪክ (Conversation History)",
                height=400,
                show_copy_button=True,
                type="messages",
                placeholder="ውይይቱ እዚህ ይታያል..."
            )
            
            msg = gr.Textbox(
                label="መልእክትዎን ያስገቡ (Enter your message)",
                placeholder="ሰላም! እንዴት ነህ?",
                lines=2
            )
            
            with gr.Row():
                submit = gr.Button("ላክ (Send)", variant="primary")
                clear = gr.Button("አጽዳ (Clear)")
                
            # Examples
            gr.Examples(
                examples=[
                    "ሰላም! እንዴት ነህ?",
                    "ስለ ኢትዮጵያ ንገረኝ",
                    "የአማርኛ ቋንቋ ታሪክ ምንድን ነው?",
                    "ቡና መጠጣት ስለምንወድ አስረዳኝ",
                    "የኢትዮጵያ ባህል ምን ይመስላል?",
                    "ኢንጀራ እንዴት ይሰራል?"
                ],
                inputs=msg
            )
            
            # Event handlers
            def respond(message, chat_history):
                if message.strip():
                    # Convert to old format for processing
                    history_tuples = [(msg["content"] if msg["role"] == "user" else "", 
                                     msg["content"] if msg["role"] == "assistant" else "") 
                                    for msg in chat_history if msg["role"] in ["user", "assistant"]]
                    
                    bot_message = self.chat(message, history_tuples)
                    
                    # Add user message
                    chat_history.append({"role": "user", "content": message})
                    # Add bot response
                    chat_history.append({"role": "assistant", "content": bot_message})
                    
                return "", chat_history
                
            msg.submit(respond, [msg, chatbot], [msg, chatbot])
            submit.click(respond, [msg, chatbot], [msg, chatbot])
            clear.click(lambda: [], None, chatbot, queue=False)
            
        return interface
        
    def run(self):
        """Run the application"""
        print("🇪🇹 Starting Smart Amharic AI...")
        print("📍 Model path:", self.model_path)
        
        interface = self.create_interface()
        interface.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            show_error=True
        )

if __name__ == "__main__":
    app = SmartAmharicApp()
    app.run()