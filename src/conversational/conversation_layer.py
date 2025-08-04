#!/usr/bin/env python3
"""
Conversational Layer for Smart Amharic LLM
Implements multi-turn dialogue capabilities with context awareness
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
import json
from pathlib import Path

class ConversationalHNet(nn.Module):
    """Enhanced H-Net with conversational capabilities"""
    
    def __init__(self, base_model, config=None):
        super().__init__()
        
        self.base_model = base_model
        self.config = config or {
            "max_context_length": 512,
            "conversation_memory_size": 10,
            "context_encoding_dim": 256,
            "instruction_embedding_dim": 128
        }
        
        # Get model dimensions from base model
        if hasattr(base_model, 'config'):
            self.hidden_size = base_model.config.hidden_size
        else:
            self.hidden_size = 768  # Default GPT-2 size
        
        # Conversational components
        self.conversation_encoder = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.config["context_encoding_dim"],
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Context attention mechanism
        self.context_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Instruction processing layer
        self.instruction_processor = nn.Linear(
            self.hidden_size, 
            self.config["instruction_embedding_dim"]
        )
        
        # Context fusion layer
        self.context_fusion = nn.Linear(
            self.hidden_size + self.config["context_encoding_dim"] + self.config["instruction_embedding_dim"],
            self.hidden_size
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        # Conversation memory
        self.conversation_memory = []
        
    def encode_conversation_context(self, conversation_history: List[str]) -> torch.Tensor:
        """Encode conversation history into context vector"""
        if not conversation_history:
            return torch.zeros(1, self.config["context_encoding_dim"])
        
        # For now, use a simple encoding - in practice, you'd tokenize and embed
        # This is a placeholder that returns appropriate tensor shape
        batch_size = 1
        seq_len = min(len(conversation_history), self.config["conversation_memory_size"])
        
        # Create dummy context encoding (replace with actual tokenization)
        context_embeddings = torch.randn(batch_size, seq_len, self.hidden_size)
        
        # Encode through LSTM
        lstm_output, (hidden, cell) = self.conversation_encoder(context_embeddings)
        
        # Use final hidden state as context
        context_vector = hidden[-1]  # Take last layer's hidden state
        
        return context_vector
    
    def process_instruction(self, instruction_text: str) -> torch.Tensor:
        """Process instruction into embedding"""
        # Placeholder for instruction processing
        # In practice, you'd tokenize and embed the instruction
        instruction_embedding = torch.randn(1, self.hidden_size)
        
        # Process through instruction layer
        processed_instruction = self.instruction_processor(instruction_embedding)
        
        return processed_instruction
    
    def apply_context_attention(self, input_embeddings: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Apply attention mechanism with conversation context"""
        # Expand context to match input sequence length
        seq_len = input_embeddings.size(1)
        context_expanded = context.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Apply attention
        attended_output, attention_weights = self.context_attention(
            query=input_embeddings,
            key=context_expanded,
            value=context_expanded
        )
        
        return attended_output
    
    def forward(self, input_ids, attention_mask=None, conversation_history=None, instruction=None, **kwargs):
        """Forward pass with conversational enhancements"""
        
        # Get base model embeddings
        if hasattr(self.base_model, 'transformer'):
            # GPT-2 style model
            transformer_outputs = self.base_model.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
            hidden_states = transformer_outputs.last_hidden_state
        else:
            # Fallback for other model types
            hidden_states = self.base_model(input_ids, attention_mask=attention_mask, **kwargs).last_hidden_state
        
        # Encode conversation context
        if conversation_history:
            context_vector = self.encode_conversation_context(conversation_history)
        else:
            context_vector = torch.zeros(1, self.config["context_encoding_dim"])
        
        # Process instruction if provided
        if instruction:
            instruction_vector = self.process_instruction(instruction)
        else:
            instruction_vector = torch.zeros(1, self.config["instruction_embedding_dim"])
        
        # Apply context attention
        attended_states = self.apply_context_attention(hidden_states, context_vector)
        
        # Fuse all information
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Expand context and instruction vectors to match sequence length
        context_expanded = context_vector.unsqueeze(1).expand(batch_size, seq_len, -1)
        instruction_expanded = instruction_vector.unsqueeze(1).expand(batch_size, seq_len, -1)
        
        # Concatenate all features
        fused_features = torch.cat([
            attended_states,
            context_expanded,
            instruction_expanded
        ], dim=-1)
        
        # Apply fusion layer
        enhanced_hidden_states = self.context_fusion(fused_features)
        enhanced_hidden_states = self.layer_norm(enhanced_hidden_states)
        
        # Get final logits
        if hasattr(self.base_model, 'lm_head'):
            logits = self.base_model.lm_head(enhanced_hidden_states)
        else:
            # Create a simple output layer if not available
            if not hasattr(self, 'output_layer'):
                vocab_size = getattr(self.base_model.config, 'vocab_size', 50257)
                self.output_layer = nn.Linear(self.hidden_size, vocab_size)
            logits = self.output_layer(enhanced_hidden_states)
        
        return type('ModelOutput', (), {
            'logits': logits,
            'hidden_states': enhanced_hidden_states,
            'context_vector': context_vector,
            'instruction_vector': instruction_vector
        })()
    
    def update_conversation_memory(self, user_input: str, model_response: str):
        """Update conversation memory with new exchange"""
        self.conversation_memory.append({
            'user': user_input,
            'assistant': model_response,
            'timestamp': torch.tensor(len(self.conversation_memory))
        })
        
        # Keep only recent conversations
        if len(self.conversation_memory) > self.config["conversation_memory_size"]:
            self.conversation_memory = self.conversation_memory[-self.config["conversation_memory_size"]:]
    
    def get_conversation_context(self) -> List[str]:
        """Get current conversation context as list of strings"""
        context = []
        for exchange in self.conversation_memory:
            context.append(f"ተጠቃሚ: {exchange['user']}")
            context.append(f"ረዳት: {exchange['assistant']}")
        return context
    
    def reset_conversation(self):
        """Reset conversation memory"""
        self.conversation_memory = []
    
    def save_conversation_state(self, filepath: str):
        """Save conversation state to file"""
        state = {
            'conversation_memory': self.conversation_memory,
            'config': self.config
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2, default=str)
    
    def load_conversation_state(self, filepath: str):
        """Load conversation state from file"""
        if Path(filepath).exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)
                self.conversation_memory = state.get('conversation_memory', [])
                self.config.update(state.get('config', {}))

class AmharicConversationManager:
    """High-level conversation management"""
    
    def __init__(self, model: ConversationalHNet):
        self.model = model
        self.conversation_templates = {
            'greeting': {
                'amharic': ['ሰላም', 'ሰላም ነህ', 'እንዴት ነህ', 'ደህና ነህ'],
                'response': 'ሰላም! እኔ ደህና ነኝ፣ አመሰግናለሁ። እንዴት ልረዳህ?'
            },
            'question': {
                'indicators': ['ምንድን', 'እንዴት', 'መቼ', 'የት', 'ለምን', 'ማን'],
                'response_prefix': 'በተመለከተ'
            },
            'instruction': {
                'indicators': ['አስረዳኝ', 'ንገረኝ', 'ፃፍልኝ', 'ተርጉምልኝ'],
                'response_prefix': 'እሺ፣ '
            }
        }
    
    def classify_input_type(self, user_input: str) -> str:
        """Classify the type of user input"""
        user_input_lower = user_input.lower()
        
        # Check for greetings
        for greeting in self.conversation_templates['greeting']['amharic']:
            if greeting in user_input_lower:
                return 'greeting'
        
        # Check for questions
        for indicator in self.conversation_templates['question']['indicators']:
            if indicator in user_input_lower:
                return 'question'
        
        # Check for instructions
        for indicator in self.conversation_templates['instruction']['indicators']:
            if indicator in user_input_lower:
                return 'instruction'
        
        return 'general'
    
    def generate_contextual_response(self, user_input: str, max_length: int = 100) -> str:
        """Generate response with conversation context"""
        input_type = self.classify_input_type(user_input)
        conversation_history = self.model.get_conversation_context()
        
        # For now, return template-based responses
        # In practice, you'd use the model to generate responses
        
        if input_type == 'greeting':
            response = self.conversation_templates['greeting']['response']
        elif input_type == 'question':
            response = f"ጥያቄህን ተረድቻለሁ። {user_input} በተመለከተ መረጃ እሰጣለሁ።"
        elif input_type == 'instruction':
            response = f"እሺ፣ {user_input} ላደርግልህ እሞክራለሁ።"
        else:
            response = "ተረድቻለሁ። ተጨማሪ መረጃ ልስጥህ?"
        
        # Update conversation memory
        self.model.update_conversation_memory(user_input, response)
        
        return response