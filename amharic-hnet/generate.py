import torch
from hybrid_tokenizer import HybridAmharicTokenizer
from hnet import HNetAmharic
from config import AmharicConfig

def generate_text(prompt, max_length=50, model_path=None):
    """Generates Amharic text from a prompt."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = HybridAmharicTokenizer()
    model = HNetAmharic(
        vocab_size=len(tokenizer.vocab),
        d_model=AmharicConfig.d_model,
        n_layers=AmharicConfig.n_layers,
        n_heads=AmharicConfig.n_heads,
        dropout=AmharicConfig.dropout
    )
    
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.to(device)
    model.eval()
    
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor(tokens, device=device).unsqueeze(0)
    
    with torch.no_grad():
        for _ in range(max_length - len(tokens)):
            outputs = model(input_ids)
            logits = outputs["logits"][:, -1, :]
            next_token = torch.argmax(logits, dim=-1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            if next_token.item() == tokenizer.vocab.get("።", -1):
                break
    
    return tokenizer.decode(input_ids[0].cpu().numpy())

if __name__ == "__main__":
    prompt = "ኢትዮጵያ"
    model_path = "models/amharic_hnet_best.pt"
    generated_text = generate_text(prompt, model_path=model_path)
    
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
