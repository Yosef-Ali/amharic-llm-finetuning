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

@dataclass
class Batch:
    input_ids: "torch.Tensor"  # type: ignore

class JsonlTextDataset(Dataset):  # type: ignore[misc]
    def __init__(self, path: str, tokenizer: Optional[BasicCharTokenizer] = None, max_len: int = 64):
        if torch is None:
            raise RuntimeError("PyTorch not available")
        self.path = path
        self.rows: List[str] = []
        self.tokenizer = tokenizer or BasicCharTokenizer()
        self.max_len = max_len
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


def make_dataloader(path: str, batch_size: int = 16, num_workers: int = 0) -> DataLoader:
    if torch is None or DataLoader is None:
        raise RuntimeError("PyTorch not available")
    ds = JsonlTextDataset(path)
    def collate(batch):
        import torch as T
        return Batch(input_ids=T.stack(batch, dim=0))
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate)
