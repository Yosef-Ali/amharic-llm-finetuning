import re
import torch
from torch.utils.data import Dataset, DataLoader
from config import AmharicConfig

class AmharicDataset(Dataset):
    def __init__(self, tokenizer, examples, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]
        tokens = self.tokenizer.encode(text)
        
        input_ids = tokens[:-1]
        target_ids = tokens[1:]
        
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            target_ids = target_ids + [-100] * padding_length
        
        return {
            "input_ids": torch.tensor(input_ids[:self.max_length]),
            "labels": torch.tensor(target_ids[:self.max_length])
        }

def get_dataloader(tokenizer, batch_size, split='train', val_split=0.1):
    """Creates a DataLoader with a train/val split."""
    try:
        with open(AmharicConfig.data_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = [
            "ኢትዮጵያዊያን እንኳን ደስ አላችሁ።",
            "ቡና የኢትዮጵያ ባህላዊ ማህበራዊ ሥርዓት ነው።",
            "የሀገሪቱ የደን ሽፋን ከዚህ በፊት ሦስት በመቶ ነበር።",
            "አማርኛ የኢትዮጵያ ሰፊ የሚነገር ቋንቋ ነው።",
            "ህዝብ አንድ ነው። ሀገር አንድ ነው። ቋንቋ አንድ ነው።"
        ]
        print("⚠️ Using sample data.")

    examples = []
    for line in lines:
        clean_line = re.sub(r'[^\u1200-\u137F\s\u1361-\u1368]', '', line)
        clean_line = re.sub(r'\s+', ' ', clean_line).strip()
        if clean_line and len(clean_line) > 10:
            examples.append(clean_line)
            
    split_idx = int(len(examples) * (1 - val_split))
    if split == 'train':
        dataset = AmharicDataset(tokenizer, examples[:split_idx])
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        dataset = AmharicDataset(tokenizer, examples[split_idx:])
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
