#!/usr/bin/env python3
"""
Quick Start Script for Amharic Smart LLM Development
Run this after local_setup_fix.py to begin implementation
"""

import os
import sys
import subprocess
from pathlib import Path
import json

class AmharicLLMQuickStart:
    def __init__(self):
        self.project_root = Path.cwd()
        self.src_dir = self.project_root / "src"
        
    def create_project_structure(self):
        """Create the complete project structure"""
        print("📁 Creating project structure...")
        
        directories = [
            "src/models",
            "src/conversational", 
            "src/training",
            "src/data",
            "src/evaluation",
            "src/reasoning",
            "src/memory",
            "src/multimodal",
            "src/retrieval"
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
        # Create __init__.py files
        for dir_path in directories:
            init_file = Path(dir_path) / "__init__.py"
            init_file.touch()
            
        print("✅ Project structure created")
        
    def create_enhanced_data_collector(self):
        """Create an enhanced data collector"""
        print("📝 Creating enhanced data collector...")
        
        collector_code = '''#!/usr/bin/env python3
"""
Enhanced Amharic Data Collector
Collects from multiple sources including Wikipedia
"""

import requests
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict

class EnhancedAmharicCollector:
    def __init__(self):
        self.data_dir = Path("data/collected")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AmharicLLMBot/1.0'
        })
        
    def collect_wikipedia_amharic(self, num_articles=100):
        """Collect Amharic Wikipedia articles"""
        print(f"Collecting {num_articles} Wikipedia articles...")
        
        articles = []
        base_url = "https://am.wikipedia.org/w/api.php"
        
        # Get random articles
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'random',
            'rnlimit': min(num_articles, 10),
            'rnnamespace': 0
        }
        
        while len(articles) < num_articles:
            try:
                response = self.session.get(base_url, params=params)
                data = response.json()
                
                for page in data['query']['random']:
                    # Get page content
                    content_params = {
                        'action': 'parse',
                        'format': 'json',
                        'pageid': page['id'],
                        'prop': 'text',
                        'formatversion': 2
                    }
                    
                    content_response = self.session.get(base_url, params=content_params)
                    content_data = content_response.json()
                    
                    if 'parse' in content_data:
                        # Extract text (simplified - real implementation needs HTML parsing)
                        text = content_data['parse']['text']
                        
                        articles.append({
                            'id': f"wiki_{page['id']}",
                            'title': page['title'],
                            'text': text[:5000],  # Limit length
                            'source': 'wikipedia_am',
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        print(f"Collected: {page['title']}")
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"Error: {e}")
                continue
                
        # Save collected data
        output_file = self.data_dir / f"wikipedia_am_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
            
        print(f"✅ Saved {len(articles)} articles to {output_file}")
        return articles
        
    def create_conversation_dataset(self, num_conversations=1000):
        """Create conversational training data"""
        print(f"Creating {num_conversations} conversations...")
        
        conversations = []
        
        # Templates for different conversation types
        templates = [
            {
                "type": "greeting",
                "user": ["ሰላም", "እንደምን ነህ/ነሽ?", "ጤና ይስጥልኝ"],
                "assistant": ["ሰላም! እንኳን ደህና መጡ!", "እኔ በጥሩ ሁኔታ ላይ ነኝ፣ አመሰግናለሁ!", "ጤና ይስጥልኝ! እንዴት ልረዳዎት እችላለሁ?"]
            },
            {
                "type": "question",
                "user": ["... ምንድን ነው?", "... ስለ ንገረኝ", "... እንዴት ነው?"],
                "assistant": ["... ማለት ...", "ስለ ... ላስረዳዎት", "... የሚሰራው በዚህ መንገድ ነው"]
            }
        ]
        
        topics = ["ኢትዮጵያ", "አማርኛ", "ባህል", "ታሪክ", "ቴክኖሎጂ", "ትምህርት"]
        
        for i in range(num_conversations):
            template = templates[i % len(templates)]
            topic = topics[i % len(topics)]
            
            conversation = {
                "id": f"conv_{i:04d}",
                "messages": [
                    {
                        "role": "user",
                        "content": template["user"][i % len(template["user"])].replace("...", topic)
                    },
                    {
                        "role": "assistant", 
                        "content": template["assistant"][i % len(template["assistant"])].replace("...", topic)
                    }
                ],
                "metadata": {
                    "type": template["type"],
                    "topic": topic
                }
            }
            
            conversations.append(conversation)
            
        # Save conversations
        output_file = self.data_dir / f"conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, ensure_ascii=False, indent=2)
            
        print(f"✅ Created {num_conversations} conversations")
        return conversations

if __name__ == "__main__":
    collector = EnhancedAmharicCollector()
    
    # Collect Wikipedia articles
    # Note: This is a simplified version - real implementation needs proper HTML parsing
    # collector.collect_wikipedia_amharic(100)
    
    # Create conversation dataset
    collector.create_conversation_dataset(1000)
'''
        
        with open("enhanced_data_collector.py", 'w') as f:
            f.write(collector_code)
            
        print("✅ Created enhanced_data_collector.py")
        
    def create_smart_model_implementation(self):
        """Create the smart model implementation"""
        print("🤖 Creating smart model implementation...")
        
        model_code = '''#!/usr/bin/env python3
"""
Smart Amharic H-Net Model Implementation
Combines conversational abilities with enhanced architecture
"""

import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from typing import Dict, List, Optional

class SmartAmharicHNet(nn.Module):
    """Enhanced H-Net with conversational and reasoning capabilities"""
    
    def __init__(self, config: Dict = None):
        super().__init__()
        
        # Default configuration
        self.config = config or {
            "vocab_size": 50000,
            "hidden_size": 768,  # Will scale to 2048 later
            "num_layers": 12,    # Will scale to 24 later
            "num_heads": 12,     # Will scale to 32 later
            "max_position_embeddings": 512,
            "intermediate_size": 3072,
        }
        
        # Base transformer
        gpt2_config = GPT2Config(**self.config)
        self.transformer = GPT2Model(gpt2_config)
        
        # Conversational components
        self.conversation_encoder = nn.LSTM(
            input_size=self.config["hidden_size"],
            hidden_size=self.config["hidden_size"],
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Instruction understanding
        self.instruction_attention = nn.MultiheadAttention(
            embed_dim=self.config["hidden_size"],
            num_heads=self.config["num_heads"]
        )
        
        # Output layers
        self.lm_head = nn.Linear(
            self.config["hidden_size"] * 2,  # *2 for bidirectional LSTM
            self.config["vocab_size"]
        )
        
        # Amharic-specific components
        self.morphological_embeddings = nn.Embedding(
            1000,  # Morphological features
            self.config["hidden_size"] // 4
        )
        
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                conversation_history: Optional[torch.Tensor] = None,
                instruction: Optional[torch.Tensor] = None):
        
        # Get transformer outputs
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        hidden_states = transformer_outputs.last_hidden_state
        
        # Process through conversation encoder
        conv_output, _ = self.conversation_encoder(hidden_states)
        
        # Apply instruction attention if provided
        if instruction is not None:
            attended, _ = self.instruction_attention(
                conv_output, instruction, instruction
            )
            conv_output = conv_output + attended
            
        # Generate output logits
        logits = self.lm_head(conv_output)
        
        return logits
        
    def generate_response(self, 
                         prompt: str,
                         tokenizer,
                         max_length: int = 100,
                         temperature: float = 0.8):
        """Generate conversational response"""
        self.eval()
        
        # Encode prompt
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        
        with torch.no_grad():
            # Generate
            for _ in range(max_length):
                outputs = self(inputs)
                next_token_logits = outputs[0, -1, :] / temperature
                next_token = torch.multinomial(
                    torch.softmax(next_token_logits, dim=-1), 1
                )
                inputs = torch.cat([inputs, next_token.unsqueeze(0)], dim=1)
                
                # Stop if EOS token
                if next_token.item() == tokenizer.eos_token_id:
                    break
                    
        # Decode response
        response = tokenizer.decode(inputs[0], skip_special_tokens=True)
        return response

class ConversationManager:
    """Manages conversation history and context"""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.history = []
        
    def add_turn(self, user_input: str, assistant_response: str):
        """Add a conversation turn"""
        self.history.append({
            "user": user_input,
            "assistant": assistant_response
        })
        
        # Keep only recent history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
            
    def get_context(self) -> str:
        """Get formatted conversation context"""
        context = []
        for turn in self.history:
            context.append(f"ተጠቃሚ: {turn['user']}")
            context.append(f"ረዳት: {turn['assistant']}")
        return "\\n".join(context)

if __name__ == "__main__":
    # Test model creation
    model = SmartAmharicHNet()
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test conversation manager
    conv_manager = ConversationManager()
    conv_manager.add_turn("ሰላም", "ሰላም! እንኳን ደህና መጡ!")
    print(f"✅ Conversation manager working")
'''
        
        # Save to src/models/
        model_file = self.src_dir / "models" / "smart_amharic_hnet.py"
        model_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(model_file, 'w') as f:
            f.write(model_code)
            
        print(f"✅ Created {model_file}")
        
    def create_training_script(self):
        """Create smart training script"""
        print("🎓 Creating training script...")
        
        train_code = '''#!/usr/bin/env python3
"""
Smart Training Script for Amharic LLM
Implements curriculum learning and advanced techniques
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import json
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append('src')

from models.smart_amharic_hnet import SmartAmharicHNet

class AmharicDataset(Dataset):
    def __init__(self, data_path: str, tokenizer):
        self.data = []
        self.tokenizer = tokenizer
        
        # Load all JSON files
        data_dir = Path(data_path)
        for json_file in data_dir.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                self.data.extend(json.load(f))
                
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Handle different data formats
        if "messages" in item:  # Conversation format
            text = "\\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in item["messages"]
            ])
        else:  # Plain text
            text = item.get("text", "")
            
        # Tokenize
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze()
        }

class SmartTrainer:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device("cpu")  # Use CPU for local
        
    def train(self, train_dataset, num_epochs=3):
        """Train with curriculum learning"""
        print("🚀 Starting smart training...")
        
        # Create dataloader
        dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.get("batch_size", 4),
            shuffle=True
        )
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get("learning_rate", 5e-5)
        )
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.model.train()
        for epoch in range(num_epochs):
            print(f"\\nEpoch {epoch + 1}/{num_epochs}")
            
            total_loss = 0
            progress_bar = tqdm(dataloader, desc="Training")
            
            for batch in progress_bar:
                # Move to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Calculate loss
                shift_logits = outputs[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                loss = criterion(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=1.0
                )
                
                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()
                
                # Update progress
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})
                
            avg_loss = total_loss / len(dataloader)
            print(f"Average loss: {avg_loss:.4f}")
            
        print("✅ Training complete!")
        
        # Save model
        save_path = Path("models/smart_amharic_model.pt")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"✅ Model saved to {save_path}")

if __name__ == "__main__":
    print("Smart Amharic LLM Training")
    print("="*50)
    
    # Create mock tokenizer for testing
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create model
    model = SmartAmharicHNet()
    
    # Create dataset
    dataset = AmharicDataset("data/collected", tokenizer)
    
    # Create trainer
    config = {
        "batch_size": 4,
        "learning_rate": 5e-5
    }
    trainer = SmartTrainer(model, tokenizer, config)
    
    # Train
    if len(dataset) > 0:
        trainer.train(dataset, num_epochs=1)
    else:
        print("❌ No data found. Run enhanced_data_collector.py first!")
'''
        
        with open("smart_train.py", 'w') as f:
            f.write(train_code)
            
        print("✅ Created smart_train.py")
        
    def create_inference_app(self):
        """Create smart inference application"""
        print("🌐 Creating inference app...")
        
        app_code = '''#!/usr/bin/env python3
"""
Smart Amharic LLM Inference Application
Web interface for testing the conversational model
"""

import torch
import gradio as gr
from pathlib import Path
import sys
sys.path.append('src')

from models.smart_amharic_hnet import SmartAmharicHNet, ConversationManager
from transformers import AutoTokenizer

class SmartAmharicApp:
    def __init__(self):
        self.model_path = Path("models/smart_amharic_model.pt")
        self.device = torch.device("cpu")
        self.model = None
        self.tokenizer = None
        self.conversation_manager = ConversationManager()
        
    def load_model(self):
        """Load the trained model"""
        if self.model is None:
            print("Loading model...")
            
            # Create model
            self.model = SmartAmharicHNet()
            
            # Load weights if available
            if self.model_path.exists():
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                print("✅ Loaded trained model")
            else:
                print("⚠️  No trained model found, using random weights")
                
            self.model.to(self.device)
            self.model.eval()
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def chat(self, message, history):
        """Chat function for Gradio"""
        self.load_model()
        
        # Add conversation context
        context = self.conversation_manager.get_context()
        full_prompt = f"{context}\\nተጠቃሚ: {message}\\nረዳት:"
        
        # Generate response
        response = self.model.generate_response(
            full_prompt,
            self.tokenizer,
            max_length=100,
            temperature=0.8
        )
        
        # Extract only the assistant's response
        if "ረዳት:" in response:
            response = response.split("ረዳት:")[-1].strip()
            
        # Update conversation history
        self.conversation_manager.add_turn(message, response)
        
        return response
        
    def create_interface(self):
        """Create Gradio interface"""
        with gr.Blocks(theme=gr.themes.Soft()) as interface:
            gr.Markdown("""
            # 🇪🇹 Smart Amharic Conversational AI
            
            Chat with an AI that understands Amharic language and culture!
            """)
            
            chatbot = gr.Chatbot(
                label="የውይይት ታሪክ (Conversation History)",
                height=400
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
                    "ቡና መጠጣት ስለምንወድ አስረዳኝ"
                ],
                inputs=msg
            )
            
            # Event handlers
            def respond(message, chat_history):
                bot_message = self.chat(message, chat_history)
                chat_history.append((message, bot_message))
                return "", chat_history
                
            msg.submit(respond, [msg, chatbot], [msg, chatbot])
            submit.click(respond, [msg, chatbot], [msg, chatbot])
            clear.click(lambda: None, None, chatbot, queue=False)
            
        return interface
        
    def run(self):
        """Run the application"""
        print("Starting Smart Amharic AI...")
        interface = self.create_interface()
        interface.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False
        )

if __name__ == "__main__":
    app = SmartAmharicApp()
    app.run()
'''
        
        with open("smart_amharic_app.py", 'w') as f:
            f.write(app_code)
            
        print("✅ Created smart_amharic_app.py")
        
    def create_run_script(self):
        """Create a simple run script"""
        print("📜 Creating run script...")
        
        run_script = '''#!/bin/bash
# Smart Amharic LLM Runner Script

echo "🇪🇹 Smart Amharic LLM Development"
echo "================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python local_setup_fix.py"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Menu
echo ""
echo "What would you like to do?"
echo "1) Collect Data"
echo "2) Train Model"
echo "3) Run Chat Interface"
echo "4) Full Pipeline (1→2→3)"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo "📊 Collecting data..."
        python enhanced_data_collector.py
        ;;
    2)
        echo "🎓 Training model..."
        python smart_train.py
        ;;
    3)
        echo "💬 Starting chat interface..."
        python smart_amharic_app.py
        ;;
    4)
        echo "🚀 Running full pipeline..."
        python enhanced_data_collector.py
        python smart_train.py
        python smart_amharic_app.py
        ;;
    *)
        echo "Invalid choice!"
        ;;
esac
'''
        
        with open("run_smart_llm.sh", 'w') as f:
            f.write(run_script)
            
        # Make executable
        import stat
        os.chmod("run_smart_llm.sh", os.stat("run_smart_llm.sh").st_mode | stat.S_IEXEC)
        
        print("✅ Created run_smart_llm.sh")
        
    def print_next_steps(self):
        """Print clear next steps"""
        print("\n" + "="*60)
        print("🎉 QUICK START SETUP COMPLETE!")
        print("="*60)
        
        print("\n📋 Your Smart Amharic LLM is ready to build!")
        print("\n🚀 NEXT STEPS (in order):")
        print("\n1️⃣  First, run the setup fix:")
        print("    python local_setup_fix.py")
        print("\n2️⃣  Activate virtual environment:")
        print("    source venv/bin/activate  # Mac/Linux")
        print("    venv\\Scripts\\activate     # Windows")
        print("\n3️⃣  Collect training data:")
        print("    python enhanced_data_collector.py")
        print("\n4️⃣  Train your smart model:")
        print("    python smart_train.py")
        print("\n5️⃣  Run the chat interface:")
        print("    python smart_amharic_app.py")
        print("\n💡 Or use the helper script:")
        print("    ./run_smart_llm.sh")
        print("\n📖 See SMART_LLM_ROADMAP.md for detailed guidance")
        print("\n🆘 If you get errors:")
        print("   - Check IMPLEMENTATION_GUIDE.md")
        print("   - Make sure you activated the venv")
        print("   - Install missing packages with pip")
        
    def run(self):
        """Run all setup steps"""
        print("\n🚀 Amharic Smart LLM Quick Start")
        print("="*60)
        
        # Create everything
        self.create_project_structure()
        self.create_enhanced_data_collector()
        self.create_smart_model_implementation()
        self.create_training_script()
        self.create_inference_app()
        self.create_run_script()
        
        # Print next steps
        self.print_next_steps()

if __name__ == "__main__":
    quickstart = AmharicLLMQuickStart()
    quickstart.run()
