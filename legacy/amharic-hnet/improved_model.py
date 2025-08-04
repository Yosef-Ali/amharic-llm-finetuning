import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Positional encoding for the transformer model.
    
    This adds positional information to the input embeddings.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register buffer (not a parameter, but should be saved and moved with the model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape [seq_len, batch_size, embedding_dim]
            
        Returns:
            Output tensor with positional encoding added
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with improved efficiency and stability."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Q, K, V, and output
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.d_k]))
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute multi-head attention.
        
        Args:
            query: Query tensor [batch_size, seq_len_q, d_model]
            key: Key tensor [batch_size, seq_len_k, d_model]
            value: Value tensor [batch_size, seq_len_v, d_model]
            mask: Optional mask tensor [batch_size, 1, 1, seq_len_k]
            
        Returns:
            Output tensor [batch_size, seq_len_q, d_model]
        """
        batch_size = query.shape[0]
        
        # Linear projections and reshape for multi-head attention
        Q = self.q_proj(query).view(batch_size, -1, self.n_heads, self.d_k).permute(0, 2, 1, 3)  # [B, H, Lq, Dk]
        K = self.k_proj(key).view(batch_size, -1, self.n_heads, self.d_k).permute(0, 2, 1, 3)    # [B, H, Lk, Dk]
        V = self.v_proj(value).view(batch_size, -1, self.n_heads, self.d_k).permute(0, 2, 1, 3)   # [B, H, Lv, Dk]
        
        # Scale dot-product attention
        scale = self.scale.to(query.device)
        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) / scale  # [B, H, Lq, Lk]
        
        # Apply mask if provided
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attention = self.dropout(F.softmax(attention, dim=-1))  # [B, H, Lq, Lk]
        
        # Apply attention to values
        output = torch.matmul(attention, V)  # [B, H, Lq, Dk]
        
        # Reshape and apply output projection
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)  # [B, Lq, D]
        output = self.out_proj(output)
        
        return output


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation and dropout."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward network.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        return self.fc2(self.dropout(self.gelu(self.fc1(x))))


class EncoderLayer(nn.Module):
    """Transformer encoder layer with multi-head attention and feed-forward network."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply encoder layer.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional mask tensor [batch_size, 1, 1, seq_len]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Self-attention with residual connection and layer normalization
        attn_output = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection and layer normalization
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        
        return x


class DecoderLayer(nn.Module):
    """Transformer decoder layer with masked multi-head attention, 
    cross-attention, and feed-forward network."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                self_mask: Optional[torch.Tensor] = None, 
                cross_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply decoder layer.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            encoder_output: Encoder output tensor [batch_size, src_len, d_model]
            self_mask: Optional self-attention mask tensor [batch_size, 1, seq_len, seq_len]
            cross_mask: Optional cross-attention mask tensor [batch_size, 1, seq_len, src_len]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Self-attention with residual connection and layer normalization
        self_attn_output = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), self_mask)
        x = x + self.dropout(self_attn_output)
        
        # Cross-attention with residual connection and layer normalization
        cross_attn_output = self.cross_attn(self.norm2(x), encoder_output, encoder_output, cross_mask)
        x = x + self.dropout(cross_attn_output)
        
        # Feed-forward with residual connection and layer normalization
        ff_output = self.feed_forward(self.norm3(x))
        x = x + self.dropout(ff_output)
        
        return x


class HNetTransformer(nn.Module):
    """Improved Amharic language model with hybrid transformer architecture.
    
    This model combines the best features of the original HNet model with modern
    transformer improvements for better Amharic text generation.
    """
    def __init__(self, vocab_size: int, d_model: int = 512, n_layers: int = 6, 
                 n_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1, 
                 pad_idx: int = 0, max_len: int = 5000, use_decoder: bool = True):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.use_decoder = use_decoder
        
        # Token embedding and positional encoding
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        
        # Decoder layers (if used)
        if use_decoder:
            self.decoder_layers = nn.ModuleList([
                DecoderLayer(d_model, n_heads, d_ff, dropout) 
                for _ in range(n_layers)
            ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Layer normalization
        self.encoder_norm = nn.LayerNorm(d_model)
        if use_decoder:
            self.decoder_norm = nn.LayerNorm(d_model)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _create_padding_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Create padding mask for source sequence.
        
        Args:
            src: Source sequence tensor [batch_size, src_len]
            
        Returns:
            Padding mask tensor [batch_size, 1, 1, src_len]
        """
        # Create mask for padding tokens (1 for tokens, 0 for padding)
        mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return mask
    
    def _create_causal_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """Create causal mask for target sequence.
        
        Args:
            tgt: Target sequence tensor [batch_size, tgt_len]
            
        Returns:
            Causal mask tensor [batch_size, 1, tgt_len, tgt_len]
        """
        # Create mask to prevent attending to future tokens
        batch_size, tgt_len = tgt.shape
        mask = torch.tril(torch.ones((tgt_len, tgt_len))).bool()
        mask = mask.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, tgt_len, tgt_len)
        return mask.to(tgt.device)
    
    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode source sequence.
        
        Args:
            src: Source sequence tensor [batch_size, src_len]
            src_mask: Optional source mask tensor [batch_size, 1, 1, src_len]
            
        Returns:
            Encoder output tensor [batch_size, src_len, d_model]
        """
        # Create padding mask if not provided
        if src_mask is None:
            src_mask = self._create_padding_mask(src)
        
        # Embed tokens and add positional encoding
        src_embedded = self.token_embedding(src) * math.sqrt(self.d_model)
        src_embedded = self.pos_encoding(src_embedded.transpose(0, 1)).transpose(0, 1)
        
        # Apply encoder layers
        encoder_output = src_embedded
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, src_mask)
        
        # Apply final layer normalization
        encoder_output = self.encoder_norm(encoder_output)
        
        return encoder_output
    
    def decode(self, tgt: torch.Tensor, encoder_output: torch.Tensor, 
               tgt_mask: Optional[torch.Tensor] = None, 
               src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode target sequence.
        
        Args:
            tgt: Target sequence tensor [batch_size, tgt_len]
            encoder_output: Encoder output tensor [batch_size, src_len, d_model]
            tgt_mask: Optional target mask tensor [batch_size, 1, tgt_len, tgt_len]
            src_mask: Optional source mask tensor [batch_size, 1, 1, src_len]
            
        Returns:
            Decoder output tensor [batch_size, tgt_len, d_model]
        """
        if not self.use_decoder:
            raise ValueError("Decoder is not enabled for this model")
        
        # Create masks if not provided
        if tgt_mask is None:
            tgt_mask = self._create_causal_mask(tgt)
        
        # Embed tokens and add positional encoding
        tgt_embedded = self.token_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.pos_encoding(tgt_embedded.transpose(0, 1)).transpose(0, 1)
        
        # Apply decoder layers
        decoder_output = tgt_embedded
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output, tgt_mask, src_mask)
        
        # Apply final layer normalization
        decoder_output = self.decoder_norm(decoder_output)
        
        return decoder_output
    
    def forward(self, src: torch.Tensor, tgt: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            src: Source sequence tensor [batch_size, src_len]
            tgt: Optional target sequence tensor [batch_size, tgt_len]
            
        Returns:
            Output logits tensor [batch_size, seq_len, vocab_size]
        """
        # Create padding mask for source sequence
        src_mask = self._create_padding_mask(src)
        
        # Encode source sequence
        encoder_output = self.encode(src, src_mask)
        
        # If decoder is used and target is provided, use it; otherwise use source as target
        if self.use_decoder and tgt is not None:
            # Create causal mask for target sequence
            tgt_mask = self._create_causal_mask(tgt)
            
            # Decode target sequence
            decoder_output = self.decode(tgt, encoder_output, tgt_mask, src_mask)
            
            # Project to vocabulary
            output = self.output_projection(decoder_output)
        else:
            # Project encoder output to vocabulary (encoder-only model)
            output = self.output_projection(encoder_output)
        
        return output
    
    def generate(self, src: torch.Tensor, max_len: int = 100, temperature: float = 1.0,
                 top_k: int = 0, top_p: float = 0.9, repetition_penalty: float = 1.0,
                 bos_token_id: int = 2, eos_token_id: int = 3) -> torch.Tensor:
        """Generate text using nucleus sampling with repetition penalty.
        
        Args:
            src: Source sequence tensor [batch_size, src_len]
            max_len: Maximum length of generated sequence
            temperature: Sampling temperature
            top_k: Number of highest probability tokens to keep (0 to disable)
            top_p: Cumulative probability for nucleus sampling
            repetition_penalty: Penalty for repeating tokens
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID
            
        Returns:
            Generated sequence tensor [batch_size, max_len]
        """
        batch_size = src.shape[0]
        device = src.device
        
        # Encode source sequence
        src_mask = self._create_padding_mask(src)
        encoder_output = self.encode(src, src_mask)
        
        # Initialize generated sequences with BOS token
        generated = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
        
        # Track which sequences have completed
        completed_sequences = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Generate tokens one by one
        for _ in range(max_len - 1):
            if self.use_decoder:
                # Create causal mask for target sequence
                tgt_mask = self._create_causal_mask(generated)
                
                # Decode current sequence
                decoder_output = self.decode(generated, encoder_output, tgt_mask, src_mask)
                
                # Get logits for next token (last position only)
                logits = self.output_projection(decoder_output[:, -1, :])
            else:
                # For encoder-only model, use the last token's representation
                last_token_idx = generated.shape[1] - 1
                last_token_embedding = self.token_embedding(generated[:, last_token_idx:last_token_idx+1]) * math.sqrt(self.d_model)
                last_token_pos_encoding = self.pos_encoding(last_token_embedding.transpose(0, 1)).transpose(0, 1)
                
                # Process through encoder layers
                last_token_output = last_token_pos_encoding
                for layer in self.encoder_layers:
                    last_token_output = layer(last_token_output, None)
                
                # Get logits for next token
                logits = self.output_projection(last_token_output.squeeze(1))
            
            # Apply temperature
            logits = logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in generated[i]:
                        if token_id in range(len(logits[i])):
                            logits[i, token_id] /= repetition_penalty
            
            # Apply top-k filtering
            if top_k > 0:
                top_k = min(top_k, logits.size(-1))
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                
                # Shift the indices to the right to keep the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Add next token to generated sequence
            generated = torch.cat((generated, next_token), dim=1)
            
            # Mark sequences that generated EOS token as completed
            completed_sequences = completed_sequences | (next_token.squeeze(-1) == eos_token_id)
            
            # Stop if all sequences have completed
            if completed_sequences.all():
                break
        
        return generated
    
    @classmethod
    def from_pretrained(cls, model_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> 'HNetTransformer':
        """Load a pretrained model from a checkpoint.
        
        Args:
            model_path: Path to the model checkpoint
            device: Device to load the model on
            
        Returns:
            Loaded model instance
        """
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Get model config from checkpoint
        config = checkpoint['config']
        
        # Create model instance
        model = cls(
            vocab_size=config['vocab_size'],
            d_model=config.get('d_model', 512),
            n_layers=config.get('n_layers', 6),
            n_heads=config.get('n_heads', 8),
            d_ff=config.get('d_ff', 2048),
            dropout=config.get('dropout', 0.1),
            pad_idx=config.get('pad_idx', 0),
            max_len=config.get('max_len', 5000),
            use_decoder=config.get('use_decoder', True)
        )
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move model to device
        model = model.to(device)
        
        return model


class ImprovedHNetTrainer:
    """Trainer for the improved HNetTransformer model."""
    def __init__(self, model: HNetTransformer, tokenizer, device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 5e-5, weight_decay: float = 0.01, 
                 warmup_steps: int = 1000, max_grad_norm: float = 1.0,
                 use_mixed_precision: bool = True):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        
        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = self._get_scheduler(warmup_steps)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=model.pad_idx)
        
        # Gradient clipping
        self.max_grad_norm = max_grad_norm
        
        # Mixed precision training
        self.use_mixed_precision = use_mixed_precision
        self.scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
    
    def _get_scheduler(self, warmup_steps: int):
        """Create a learning rate scheduler with warmup."""
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(1.0 - current_step / 100000))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_step(self, src: torch.Tensor, tgt: torch.Tensor) -> float:
        """Perform a single training step.
        
        Args:
            src: Source sequence tensor [batch_size, src_len]
            tgt: Target sequence tensor [batch_size, tgt_len]
            
        Returns:
            Loss value
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move tensors to device
        src = src.to(self.device)
        tgt = tgt.to(self.device)
        
        # For decoder models, input is all but last token, target is all but first token
        if self.model.use_decoder:
            decoder_input = tgt[:, :-1]
            decoder_target = tgt[:, 1:]
            
            # Forward pass with mixed precision if enabled
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    output = self.model(src, decoder_input)
                    # Reshape for loss calculation
                    output = output.contiguous().view(-1, self.model.vocab_size)
                    decoder_target = decoder_target.contiguous().view(-1)
                    loss = self.criterion(output, decoder_target)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(src, decoder_input)
                # Reshape for loss calculation
                output = output.contiguous().view(-1, self.model.vocab_size)
                decoder_target = decoder_target.contiguous().view(-1)
                loss = self.criterion(output, decoder_target)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
        else:
            # For encoder-only models, predict next token for each position
            input_seq = src[:, :-1]
            target_seq = src[:, 1:]
            
            # Forward pass with mixed precision if enabled
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    output = self.model(input_seq)
                    # Reshape for loss calculation
                    output = output.contiguous().view(-1, self.model.vocab_size)
                    target_seq = target_seq.contiguous().view(-1)
                    loss = self.criterion(output, target_seq)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(input_seq)
                # Reshape for loss calculation
                output = output.contiguous().view(-1, self.model.vocab_size)
                target_seq = target_seq.contiguous().view(-1)
                loss = self.criterion(output, target_seq)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
        # Update learning rate
        self.scheduler.step()
        
        return loss.item()
    
    def validate(self, val_dataloader) -> float:
        """Validate the model on a validation dataset.
        
        Args:
            val_dataloader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                src, tgt = batch
                
                # Move tensors to device
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                
                # For decoder models, input is all but last token, target is all but first token
                if self.model.use_decoder:
                    decoder_input = tgt[:, :-1]
                    decoder_target = tgt[:, 1:]
                    
                    # Forward pass
                    output = self.model(src, decoder_input)
                    
                    # Reshape for loss calculation
                    output = output.contiguous().view(-1, self.model.vocab_size)
                    decoder_target = decoder_target.contiguous().view(-1)
                    
                    # Calculate loss
                    loss = self.criterion(output, decoder_target)
                else:
                    # For encoder-only models, predict next token for each position
                    input_seq = src[:, :-1]
                    target_seq = src[:, 1:]
                    
                    # Forward pass
                    output = self.model(input_seq)
                    
                    # Reshape for loss calculation
                    output = output.contiguous().view(-1, self.model.vocab_size)
                    target_seq = target_seq.contiguous().view(-1)
                    
                    # Calculate loss
                    loss = self.criterion(output, target_seq)
                
                # Accumulate loss
                batch_size = src.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        # Calculate average loss
        avg_loss = total_loss / total_samples
        
        return avg_loss
    
    def save_checkpoint(self, path: str, epoch: int, val_loss: float):
        """Save model checkpoint.
        
        Args:
            path: Path to save the checkpoint
            epoch: Current epoch number
            val_loss: Validation loss
        """
        # Create model config
        config = {
            'vocab_size': self.model.vocab_size,
            'd_model': self.model.d_model,
            'n_layers': len(self.model.encoder_layers),
            'n_heads': self.model.encoder_layers[0].self_attn.n_heads,
            'd_ff': self.model.encoder_layers[0].feed_forward.fc1.out_features,
            'dropout': self.model.encoder_layers[0].dropout.p,
            'pad_idx': self.model.pad_idx,
            'use_decoder': self.model.use_decoder
        }
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': config
        }, path)
        
        logger.info(f"Model checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint.
        
        Args:
            path: Path to the checkpoint
            
        Returns:
            Epoch number and validation loss
        """
        # Load checkpoint
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Get epoch and validation loss
        epoch = checkpoint.get('epoch', 0)
        val_loss = checkpoint.get('val_loss', float('inf'))
        
        logger.info(f"Loaded checkpoint from {path} (epoch {epoch}, val_loss {val_loss:.4f})")
        
        return epoch, val_loss


def main():
    """Main function for testing the model."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test the improved HNetTransformer model')
    parser.add_argument('--vocab-size', type=int, default=32000, help='Vocabulary size')
    parser.add_argument('--d-model', type=int, default=512, help='Model dimension')
    parser.add_argument('--n-layers', type=int, default=6, help='Number of layers')
    parser.add_argument('--n-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--d-ff', type=int, default=2048, help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--use-decoder', action='store_true', help='Use decoder architecture')
    
    args = parser.parse_args()
    
    # Create model
    model = HNetTransformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        use_decoder=args.use_decoder
    )
    
    # Print model summary
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Architecture: {'Encoder-Decoder' if args.use_decoder else 'Encoder-only'}")
    print(f"Layers: {args.n_layers}, Heads: {args.n_heads}, Dimension: {args.d_model}")
    
    # Test with random input
    batch_size = 2
    seq_len = 10
    src = torch.randint(0, args.vocab_size, (batch_size, seq_len))
    
    if args.use_decoder:
        tgt = torch.randint(0, args.vocab_size, (batch_size, seq_len))
        output = model(src, tgt[:, :-1])
        print(f"Output shape: {output.shape}")
    else:
        output = model(src)
        print(f"Output shape: {output.shape}")
    
    # Test generation
    generated = model.generate(src, max_len=20)
    print(f"Generated shape: {generated.shape}")


if __name__ == "__main__":
    main()