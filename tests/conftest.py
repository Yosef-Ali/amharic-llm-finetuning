import os
import random
import numpy as np

try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore


def pytest_configure(config):
    seed = int(os.getenv("TEST_SEED", "1337"))
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
      try:
        import torch as T
        T.manual_seed(seed)
        if T.cuda.is_available():
            T.cuda.manual_seed_all(seed)
        if hasattr(T, 'backends') and hasattr(T.backends, 'cudnn'):
            T.backends.cudnn.deterministic = True
            T.backends.cudnn.benchmark = False
      except Exception:
        pass


def pytest_addoption(parser):
    parser.addoption("--heavy", action="store_true", help="run heavy (GPU/data) tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--heavy") or os.getenv("RUN_HEAVY_TESTS") == "1":
        return
    import pytest
    skip_heavy = pytest.mark.skip(reason="need --heavy or RUN_HEAVY_TESTS=1")
    for item in items:
        if "heavy" in item.keywords:
            item.add_marker(skip_heavy)
