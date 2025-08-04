import torch
import torch.nn as nn
import torch.nn.functional as F

class CPUAmharicHNet(nn.Module):
    """CPU-optimized H-Net for Amharic - VERY SMALL for local training"""
    
    def __init__(self):
        super().__init__()
        # TINY model that can train on CPU
        self.d_model = 64        # Very small
        self.vocab_size = 256    # Byte-level (0-255)
        self.max_seq_len = 64    # Short sequences
        
        # Simple 1-stage hierarchy only
        self.byte_embedding = nn.Embedding(256, self.d_model)
        
        # Encoder: 2 layers only (CPU constraint)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=4,  # Small attention heads
                dim_feedforward=128,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Dynamic chunking (simplified)
        self.router = nn.Linear(self.d_model, 1)
        
        # Main network: TINY transformer
        self.main_net = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=4,
                dim_feedforward=128,
                batch_first=True
            ),
            num_layers=3
        )
        
        # Decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=4,
                dim_feedforward=128,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Output head
        self.output_head = nn.Linear(self.d_model, 256)
        
    def forward(self, input_bytes):
        # Convert bytes to embeddings
        x = self.byte_embedding(input_bytes)
        
        # Encode
        encoded = self.encoder(x)
        
        # Simple dynamic chunking (every 4 bytes)
        batch_size, seq_len, d_model = encoded.shape
        chunk_size = 4
        chunks = []
        
        for i in range(0, seq_len, chunk_size):
            chunk = encoded[:, i:i+chunk_size].mean(dim=1, keepdim=True)
            chunks.append(chunk)
        
        chunked = torch.cat(chunks, dim=1)
        
        # Process in main network
        processed = self.main_net(chunked)
        
        # Decode back to original length
        # Simple upsampling (repeat each chunk)
        upsampled = processed.repeat_interleave(chunk_size, dim=1)
        upsampled = upsampled[:, :seq_len, :]  # Trim to original length
        
        # Output logits
        logits = self.output_head(upsampled)
        
        return {
            'logits': logits,
            'loss': None,  # Add loss computation
            'compression_ratio': seq_len / processed.shape[1]
        }
    
    def generate(self, input_bytes, max_length=32, temperature=1.0):
        """Generate text from input bytes"""
        self.eval()
        
        with torch.no_grad():
            # Start with input
            current_input = input_bytes.clone()
            
            for _ in range(max_length - input_bytes.shape[1]):
                # Forward pass
                outputs = self.forward(current_input)
                logits = outputs['logits']
                
                # Get next token probabilities
                next_token_logits = logits[:, -1, :] / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(probs, 1)
                
                # Append to sequence
                current_input = torch.cat([current_input, next_token], dim=1)
                
                # Stop if we hit padding token
                if next_token.item() == 0:
                    break
            
            return current_input

class HNetConfig:
    """Configuration for H-Net model"""
    def __init__(self):
        self.d_model = 64
        self.vocab_size = 256
        self.max_seq_len = 64
        self.num_encoder_layers = 2
        self.num_decoder_layers = 2
        self.num_main_layers = 3
        self.nhead = 4
        self.dim_feedforward = 128
        self.chunk_size = 4
        self.dropout = 0.1

def create_cpu_hnet(config=None):
    """Factory function to create CPU H-Net model"""
    if config is None:
        config = HNetConfig()
    
    model = CPUAmharicHNet()
    return model

def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test the model
    model = create_cpu_hnet()
    print(f"🧠 Model created with {count_parameters(model):,} parameters")
    
    # Test forward pass
    test_input = torch.randint(0, 256, (2, 32))  # Batch of 2, sequence length 32
    outputs = model(test_input)
    
    print(f"📊 Input shape: {test_input.shape}")
    print(f"📊 Output logits shape: {outputs['logits'].shape}")
    print(f"📊 Compression ratio: {outputs['compression_ratio']:.2f}x")
    
    # Test generation
    test_bytes = torch.tensor([[72, 101, 108, 108, 111] + [0] * 27]).long()  # "Hello" + padding
    generated = model.generate(test_bytes, max_length=16)
    print(f"🎭 Generated shape: {generated.shape}")