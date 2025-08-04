"""
Advanced Transformer-based H-Net for Amharic Language Generation
Enhanced with Multi-Head Attention, Layer Normalization, and Optimized Architecture
"""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Optional, Tuple, Dict, Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn import MultiheadAttention, LayerNorm, Dropout
except Exception:
    torch, nn, F = None, None, None


@dataclass
class TransformerHNetConfig:
    """Configuration for Transformer-based H-Net."""
    vocab_size: int = 3087  # Amharic tokenizer vocab size
    hidden_dim: int = 512   # Increased from 256 for better capacity
    num_layers: int = 8     # Increased from 6 for better depth
    num_heads: int = 8      # Multi-head attention
    dropout: float = 0.1
    max_seq_len: int = 512
    intermediate_size: int = 2048  # FFN intermediate size
    layer_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int = 0
    activation_function: str = "gelu"  # Better than ReLU for language models
    
    # Amharic-specific configurations
    use_hierarchical_attention: bool = True
    use_positional_bias: bool = True
    attention_window_size: int = 64  # Local attention window


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE) for better position encoding."""
    
    def __init__(self, dim: int, max_seq_len: int = 512, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Pre-compute frequency matrix
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate rotary embeddings for given sequence length."""
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim//2]
        
        # Create complex embedding
        emb = torch.polar(torch.ones_like(freqs), freqs)  # e^(i * freqs)
        return emb


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embedding to input tensor."""
    # x: [batch, seq_len, heads, head_dim]
    # cos, sin: [seq_len, head_dim//2]
    
    batch_size, seq_len, num_heads, head_dim = x.shape
    
    # Ensure cos/sin dimensions match
    cos = cos[:seq_len, :head_dim//2]  # [seq_len, head_dim//2]
    sin = sin[:seq_len, :head_dim//2]  # [seq_len, head_dim//2]
    
    # Expand to match x dimensions
    cos = cos.unsqueeze(0).unsqueeze(2).expand(batch_size, seq_len, num_heads, head_dim//2)
    sin = sin.unsqueeze(0).unsqueeze(2).expand(batch_size, seq_len, num_heads, head_dim//2)
    
    # Split x into even/odd dimensions
    x1 = x[..., 0::2]  # [batch, seq_len, heads, head_dim//2] - even indices
    x2 = x[..., 1::2]  # [batch, seq_len, heads, head_dim//2] - odd indices
    
    # Apply rotation: [cos * x1 - sin * x2, sin * x1 + cos * x2]
    x_rotated_1 = x1 * cos - x2 * sin
    x_rotated_2 = x1 * sin + x2 * cos
    
    # Interleave back to original order
    x_rotated = torch.zeros_like(x)
    x_rotated[..., 0::2] = x_rotated_1
    x_rotated[..., 1::2] = x_rotated_2
    
    return x_rotated


class MultiHeadSelfAttention(nn.Module):
    """Enhanced Multi-Head Self-Attention with RoPE and optional local attention."""
    
    def __init__(self, config: TransformerHNetConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        
        assert self.hidden_dim % self.num_heads == 0
        
        # Linear projections
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        
        # Rotary positional embedding
        self.rope = RotaryPositionalEmbedding(self.head_dim, config.max_seq_len)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Local attention window (optional)
        self.attention_window_size = config.attention_window_size if hasattr(config, 'attention_window_size') else None
        
    def forward(self, 
                hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # Compute Q, K, V
        query = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        value = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply RoPE - simplified implementation
        device = hidden_states.device
        t = torch.arange(seq_len, device=device, dtype=torch.float)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2, device=device).float() / self.head_dim))
        freqs = torch.outer(t, inv_freq)
        cos = torch.cos(freqs)  # [seq_len, head_dim//2]
        sin = torch.sin(freqs)  # [seq_len, head_dim//2]
        
        query = apply_rotary_pos_emb(query, cos, sin)
        key = apply_rotary_pos_emb(key, cos, sin)
        
        # Handle past key-value for generation
        if past_key_value is not None:
            past_key, past_value = past_key_value
            # Ensure dimensions match before concatenation
            if past_key.shape[2] == key.shape[2]:  # Check if num_heads match
                key = torch.cat([past_key, key], dim=1)
                value = torch.cat([past_value, value], dim=1)
            else:
                # If dimensions don't match, skip concatenation for now
                pass
        
        # Transpose for attention computation: [batch, heads, seq_len, head_dim]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Apply attention mask
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Apply local attention window (if specified)
        if hasattr(self, 'attention_window_size') and self.attention_window_size is not None and seq_len > self.attention_window_size:
            # Create local attention mask
            local_mask = torch.triu(torch.ones(seq_len, seq_len, device=attn_scores.device), diagonal=self.attention_window_size + 1)
            local_mask = local_mask + torch.tril(torch.ones(seq_len, seq_len, device=attn_scores.device), diagonal=-self.attention_window_size - 1)
            local_mask = local_mask.bool()
            attn_scores = attn_scores.masked_fill(local_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Softmax attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        attn_output = self.o_proj(attn_output)
        
        # Prepare cache for next iteration
        present_key_value = (key, value) if use_cache else None
        
        return attn_output, present_key_value


class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network with GELU activation."""
    
    def __init__(self, config: TransformerHNetConfig):
        super().__init__()
        self.dense_1 = nn.Linear(config.hidden_dim, config.intermediate_size)
        self.dense_2 = nn.Linear(config.intermediate_size, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        
        # Use GELU activation for better performance
        if config.activation_function == "gelu":
            self.activation = F.gelu
        elif config.activation_function == "relu":
            self.activation = F.relu
        else:
            self.activation = F.gelu
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense_1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        return hidden_states


class TransformerLayer(nn.Module):
    """Single Transformer layer with pre-layer normalization."""
    
    def __init__(self, config: TransformerHNetConfig):
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(config)
        self.feed_forward = PositionwiseFeedForward(config)
        
        # Layer normalization (Pre-LN architecture)
        self.ln_1 = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.ln_2 = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        # Pre-LN Self-Attention
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output, present_key_value = self.self_attention(
            hidden_states, attention_mask, past_key_value, use_cache
        )
        hidden_states = residual + self.dropout(attn_output)
        
        # Pre-LN Feed-Forward
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        ff_output = self.feed_forward(hidden_states)
        hidden_states = residual + self.dropout(ff_output)
        
        return hidden_states, present_key_value


class TransformerHNet(nn.Module):
    """Advanced Transformer-based Hierarchical Network for Amharic."""
    
    def __init__(self, config: TransformerHNetConfig):
        super().__init__()
        if torch is None or nn is None:
            self.available = False
            return
            
        self.available = True
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        
        # Tie embeddings with output layer (common practice)
        self.lm_head.weight = self.token_embedding.weight
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize model weights using best practices."""
        if isinstance(module, nn.Linear):
            # Use Xavier/Glorot initialization for linear layers
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
                labels: Optional[torch.Tensor] = None,
                use_cache: bool = False) -> Dict[str, Any]:
        
        if not self.available:
            return {"loss": torch.tensor(0.0), "logits": torch.zeros(1, 1, self.config.vocab_size)}
        
        batch_size, seq_len = input_ids.shape
        
        # Create causal attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.tril(torch.ones(seq_len, seq_len)).to(input_ids.device)
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        # Token embeddings
        hidden_states = self.token_embedding(input_ids)
        hidden_states = self.dropout(hidden_states)
        
        # Apply transformer layers
        present_key_values = []
        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            hidden_states, present_key_value = layer(
                hidden_states, attention_mask, past_key_value, use_cache
            )
            if use_cache:
                present_key_values.append(present_key_value)
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Shift labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate cross-entropy loss
            loss_fn = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": present_key_values if use_cache else None
        }
    
    def generate(self, 
                 input_ids: torch.Tensor,
                 max_length: int = 50,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 top_p: float = 0.9,
                 do_sample: bool = True,
                 pad_token_id: Optional[int] = None) -> torch.Tensor:
        """Generate text using the model."""
        
        if not self.available:
            return input_ids
        
        self.eval()
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        # Set pad token
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        
        # Initialize generation
        generated = input_ids
        past_key_values = None
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                outputs = self.forward(
                    generated, 
                    past_key_values=past_key_values,
                    use_cache=True
                )
                
                # Get next token logits
                next_token_logits = outputs["logits"][:, -1, :]  # [batch_size, vocab_size]
                past_key_values = outputs["past_key_values"]
                
                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Scatter back to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits = next_token_logits.masked_fill(indices_to_remove, float('-inf'))
                
                # Sample next token
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for end of sequence
                if next_token.item() == pad_token_id:
                    break
        
        return generated
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        if not self.available:
            return 0
        return sum(p.numel() for p in self.parameters())
    
    def step(self) -> torch.Tensor:
        """Training step for compatibility with existing training loop."""
        if not self.available:
            return torch.tensor(0.0)
        
        # Generate dummy input for testing
        batch_size, seq_len = 4, 32
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        labels = input_ids.clone()
        
        output = self.forward(input_ids, labels=labels)
        return output["loss"] if output["loss"] is not None else torch.tensor(0.0)


def create_transformer_model(cfg) -> TransformerHNet:
    """Create enhanced Transformer H-Net model from config."""
    model_cfg = TransformerHNetConfig(
        vocab_size=getattr(cfg.model, 'vocab_size', 3087),
        hidden_dim=getattr(cfg.model, 'hidden_dim', 512),
        num_layers=getattr(cfg.model, 'num_layers', 8),
        num_heads=getattr(cfg.model, 'num_heads', 8),
        dropout=getattr(cfg.model, 'dropout', 0.1),
        max_seq_len=getattr(cfg.model, 'max_seq_len', 512),
        intermediate_size=getattr(cfg.model, 'intermediate_size', 2048),
        use_cache=getattr(cfg.model, 'use_cache', True),
        activation_function=getattr(cfg.model, 'activation_function', 'gelu')
    )
    return TransformerHNet(model_cfg)


# Alias for backward compatibility
AdvancedAmharicHNet = TransformerHNet