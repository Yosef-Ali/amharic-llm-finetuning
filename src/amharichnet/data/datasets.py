from __future__ import annotations
from dataclasses import dataclass
from typing import List, Iterator

@dataclass
class Sample:
    text: str

class SyntheticDataset:
    """Tiny synthetic dataset useful for smoke tests and CPU training."""
    def __init__(self, size: int = 32) -> None:
        self.samples: List[Sample] = [Sample(text=f"sample {i}") for i in range(size)]

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)
