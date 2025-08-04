from __future__ import annotations
from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch, nn = None, None  # type: ignore

@dataclass
class HNetConfig:
    hidden_dim: int = 128
    num_layers: int = 2

class TinyHNet:
    """A tiny toy model standing in for H-Net while migrating code."""
    def __init__(self, cfg: HNetConfig) -> None:
        if torch is None or nn is None:
            self.available = False
            return
        self.available = True
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )

    def parameters(self):  # pragma: no cover
        if not self.available:
            return []
        return self.net.parameters()

    def step(self):  # pragma: no cover
        if not self.available:
            return 0.0
        import torch
        x = torch.randn(8, self.cfg.hidden_dim)
        out = self.net(x).mean()
        loss = (out ** 2)
        return loss
