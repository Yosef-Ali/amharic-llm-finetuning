import os
import time
import torch
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from config import AmharicConfig
from dataset import get_dataloader
from hybrid_tokenizer import HybridAmharicTokenizer
from hnet import HNetAmharic

def evaluate(model, dataloader, device):
    """Evaluates the model on a validation set."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            with autocast(enabled=AmharicConfig.use_fp16):
                outputs = model(input_ids, labels=labels)
                total_loss += outputs["loss"].item()
    return total_loss / len(dataloader)

def train():
    """Main training loop."""
    AmharicConfig.validate()
    
    tokenizer = HybridAmharicTokenizer()
    model = HNetAmharic(
        vocab_size=len(tokenizer.vocab),
        d_model=AmharicConfig.d_model,
        n_layers=AmharicConfig.n_layers,
        n_heads=AmharicConfig.n_heads,
        dropout=AmharicConfig.dropout
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=AmharicConfig.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=AmharicConfig.num_epochs)
    scaler = GradScaler(enabled=AmharicConfig.use_fp16)

    train_dataloader = get_dataloader(tokenizer, AmharicConfig.batch_size, split='train')
    val_dataloader = get_dataloader(tokenizer, AmharicConfig.batch_size, split='val')
    
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(AmharicConfig.num_epochs):
        model.train()
        epoch_loss = 0
        progress = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{AmharicConfig.num_epochs}")

        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with autocast(enabled=AmharicConfig.use_fp16):
                outputs = model(input_ids, labels=labels)
                loss = outputs["loss"] / AmharicConfig.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (progress.n + 1) % AmharicConfig.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * AmharicConfig.gradient_accumulation_steps
            progress.set_postfix({"loss": f"{loss.item() * AmharicConfig.gradient_accumulation_steps:.4f}"})

        scheduler.step()
        
        val_loss = evaluate(model, val_dataloader, device)
        print(f"\nEpoch {epoch+1} | Val Loss: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model_path = os.path.join(AmharicConfig.model_path, "amharic_hnet_best.pt")
            torch.save(model.state_dict(), model_path)
            print(f"✅ Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= AmharicConfig.patience:
            print(f"EARLY STOPPING after {patience_counter} epochs with no improvement.")
            break

if __name__ == "__main__":
    print("="*50)
    print("AMHARIC H-NET TRAINING")
    print("="*50)
    
    os.makedirs(AmharicConfig.model_path, exist_ok=True)
    train()
