# Amharic H-Net Configuration
import os

class AmharicConfig:
    # Model architecture
    d_model = 512
    n_layers = 12
    n_heads = 8
    dropout = 0.1
    
    # Training parameters
    batch_size = 16
    gradient_accumulation_steps = 2
    use_fp16 = True
    learning_rate = 5e-5
    num_epochs = 10
    patience = 3
    
    # Amharic-specific settings
    amharic_spaces = True
    amharic_punctuation = True
    
    # Paths
    data_path = "processed_articles/amharic_corpus.txt"
    model_path = "models"
    
    @classmethod
    def validate(cls):
        """Validate configuration."""
        assert cls.d_model % cls.n_heads == 0, "d_model must be divisible by n_heads"
        print("✅ Configuration validated")
