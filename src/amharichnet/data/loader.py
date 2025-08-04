from __future__ import annotations
from dataclasses import dataclass
from typing import List, Iterator, Optional

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except Exception:  # pragma: no cover
    torch, Dataset, DataLoader = None, object, None  # type: ignore

from .jsonl import read_jsonl
from .tokenizer import BasicCharTokenizer
from .amharic_tokenizer import AmharicSubwordTokenizer

@dataclass
class Batch:
    input_ids: "torch.Tensor"  # type: ignore

class JsonlTextDataset(Dataset):  # type: ignore[misc]
    def __init__(self, path: str, tokenizer=None, max_len: int = 64, tokenizer_type: str = "basic"):
        if torch is None:
            raise RuntimeError("PyTorch not available")
        self.path = path
        self.rows: List[str] = []
        self.max_len = max_len
        self.tokenizer_type = tokenizer_type
        
        # Choose tokenizer based on type
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif tokenizer_type == "amharic":
            # Load trained Amharic tokenizer
            self.tokenizer = AmharicSubwordTokenizer()
            try:
                self.tokenizer.load_vocab("models/tokenizer/amharic_vocab.json")
            except FileNotFoundError:
                print("⚠️  Amharic vocab not found, falling back to basic tokenizer")
                self.tokenizer = BasicCharTokenizer()
        else:
            self.tokenizer = BasicCharTokenizer()
        for row in read_jsonl(path):
            txt = row.get("text") or row.get("prompt") or row.get("input")
            if isinstance(txt, str) and txt.strip():
                self.rows.append(txt.strip())
        if not self.rows:
            # fallback tiny synthetic rows
            self.rows = [f"synthetic {i}" for i in range(32)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        text = self.rows[idx]
        ids = self.tokenizer.encode(text, max_len=64)
        import torch as T
        return T.tensor(ids, dtype=T.long)


def make_dataloader(path: str, batch_size: int = 16, num_workers: int = 0, 
                   tokenizer_type: str = "basic", max_len: int = 128) -> DataLoader:
    if torch is None or DataLoader is None:
        raise RuntimeError("PyTorch not available")
    ds = JsonlTextDataset(path, tokenizer_type=tokenizer_type, max_len=max_len)
    def collate(batch):
        import torch as T
        return Batch(input_ids=T.stack(batch, dim=0))
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate)
