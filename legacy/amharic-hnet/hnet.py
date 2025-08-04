import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """Injects positional information into the input sequence."""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class HNetAmharic(nn.Module):
    """A standard Transformer model for Amharic."""
    
    def __init__(self, vocab_size, d_model=256, n_layers=6, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following standard practices."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, input_ids, labels=None):
        """
        input_ids: (batch_size, seq_len)
        labels: (batch_size, seq_len) - for training
        """
        # Embed tokens and add positional encoding
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Transformer processing
        x = self.transformer_encoder(x)
        
        # Output projection
        logits = self.output_proj(x)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
        
        return {
            "logits": logits,
            "loss": loss
        }
    
    @classmethod
    def from_pretrained(cls, pretrained_model_path, **kwargs):
        """Load pretrained model (for transfer learning)."""
        model = cls(**kwargs)
        state_dict = torch.load(pretrained_model_path)
        model.load_state_dict(state_dict)
        return model
