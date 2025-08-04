from __future__ import annotations
from typing import List

class BasicCharTokenizer:
    """Very basic tokenizer that splits into unicode code points and maps to ids.
    Placeholder for a real tokenizer; keeps interface simple for migration.
    """
    def __init__(self):
        self.vocab = {"<pad>": 0, "<unk>": 1}
        self._next_id = 2

    def _add(self, ch: str) -> int:
        if ch not in self.vocab:
            self.vocab[ch] = self._next_id
            self._next_id += 1
        return self.vocab[ch]

    def encode(self, text: str, max_len: int = 64) -> List[int]:
        ids = [self._add(ch) for ch in list(text)]
        ids = ids[:max_len]
        # right-pad
        while len(ids) < max_len:
            ids.append(0)
        return ids
