import os
import random
from typing import Optional

import numpy as np
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def set_seed(seed: int = 1337) -> None:
    """Set seeds for python, numpy, and torch if available."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():  # type: ignore[attr-defined]
                torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]
            # CUDNN determinism
            if hasattr(torch, 'backends') and hasattr(torch.backends, 'cudnn'):
                torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
                torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
        except Exception:
            pass


def seed_from_env(default: int = 1337) -> int:
    return int(os.getenv("SEED", default))
