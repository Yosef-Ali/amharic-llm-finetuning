from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from ..utils.config import Config
from ..utils.repro import set_seed


@dataclass
class TrainArtifacts:
    output_dir: Path
    config_dump: Path


def prepare_output_dir(base: str) -> TrainArtifacts:
    out = Path(base)
    out.mkdir(parents=True, exist_ok=True)
    cfg_dump = out / "used_config.txt"
    return TrainArtifacts(output_dir=out, config_dump=cfg_dump)


from typing import Optional

def validate(cfg: Config, model) -> float:
    """Tiny validation that mirrors the train dummy loss on val data.
    Returns a float val_loss.
    """
    try:
        from ..data.loader import make_dataloader
        import torch as T
    except Exception:
        return 0.0
    try:
        val_loader = make_dataloader(cfg.data.val_path, batch_size=max(1, cfg.data.batch_size//2), num_workers=0)
    except Exception:
        return 0.0
    total = 0.0
    count = 0
    for i, batch in enumerate(val_loader):
        x = batch.input_ids.float()
        hidden = cfg.model.hidden_dim
        if x.shape[1] < hidden:
            pad = T.zeros((x.shape[0], hidden - x.shape[1]))
            x = T.cat([x, pad], dim=1)
        else:
            x = x[:, :hidden]
        out = model.net(x).mean() if hasattr(model, "net") else x.mean()
        loss = float((out ** 2).detach().item() if hasattr(out, "detach") else out**2)
        total += loss
        count += 1
        if i >= 5:
            break
    return float(total / max(1, count))

def run_training(cfg: Config) -> TrainArtifacts:

from typing import Any, Dict


def load_checkpoint(arts: TrainArtifacts, cfg: Config, model, optimizer):
    """Attempt to load checkpoint from cfg.model.checkpoint if provided.
    Returns (loaded, message).
    """
    ckpt_path = cfg.model.checkpoint
    if not ckpt_path:
        return False, "no checkpoint"
    try:
        import torch as T
        data = T.load(ckpt_path, map_location="cpu")
        if hasattr(model, "net") and data.get("model_state") is not None:
            model.net.load_state_dict(data["model_state"])
        if hasattr(optimizer, "load_state_dict") and data.get("optimizer_state") is not None:
            optimizer.load_state_dict(data["optimizer_state"])
        return True, f"loaded {ckpt_path}"
    except Exception as e:
        return False, f"failed to load {ckpt_path}: {e}"



def save_checkpoint(arts: TrainArtifacts, cfg: Config, model, optimizer, step: int, last_loss: float) -> None:
    ckpt_dir = arts.output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    meta: Dict[str, Any] = {
        "step": step,
        "final_loss": float(last_loss),
        "model": {
            "name": cfg.model.name,
            "hidden_dim": cfg.model.hidden_dim,
            "num_layers": cfg.model.num_layers,
        },
    }
    # Save torch states if available
    try:
        import torch as T
        if hasattr(model, 'net') and hasattr(model, 'available') and model.available:
            meta_path = ckpt_dir / "meta.json"
            meta_path.write_text(json.dumps(meta, indent=2))
            T.save({
                "model_state": getattr(model, 'net').state_dict(),
                "optimizer_state": getattr(optimizer, 'state_dict', lambda: {})(),
                "meta": meta,
            }, ckpt_dir / "ckpt.pt")
        else:
            # Fallback meta only
            (ckpt_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    except Exception:
        # Fallback meta only
        (ckpt_dir / "meta.json").write_text(json.dumps(meta, indent=2))


    """
    Minimal training scaffold. Replace with real training once migrated.
    - Sets seed
    - Prepares output dir
    - Dumps config text for reproducibility
    """
    set_seed(cfg.train.seed)
    arts = prepare_output_dir(cfg.train.output_dir)

    # Save a simple config dump for traceability
    try:
        from pydantic import BaseModel
        def _dump_model(m: BaseModel) -> str:
            return m.model_dump_json(indent=2)  # type: ignore[attr-defined]
        text = (
            "# Training Config\n"\
            + _dump_model(cfg.data) + "\n"\
            + _dump_model(cfg.model) + "\n"\
            + _dump_model(cfg.train) + "\n"
        )
    except Exception:
        text = str(cfg)
    arts.config_dump.write_text(text)

    
    # Minimal model + optimizer + a couple of dummy steps
    from ..models.hnet import HNetConfig, TinyHNet

    model_cfg = HNetConfig(hidden_dim=cfg.model.hidden_dim, num_layers=cfg.model.num_layers)
    model = TinyHNet(model_cfg)

    if torch is None or not getattr(model, 'available', False):
        # Torch not available; skip heavy steps
        
    # Save metrics and checkpoint
    try:
        (arts.output_dir / 'metrics.json').write_text(json.dumps({'steps': int(step_count), 'final_loss': float(last_loss)}, indent=2))
    except Exception:
        pass
    try:
        save_checkpoint(arts, cfg, model, optimizer, step=step_count, last_loss=last_loss)
    except Exception:
        pass
    return arts
    

    import torch.optim as optim
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    # Resume from checkpoint if provided
    try:
        loaded, msg = load_checkpoint(arts, cfg, model, optimizer)
        if loaded:
            print('[RESUME]', msg)
        else:
            
            if 'no checkpoint' not in msg:
                print('[RESUME]', msg)
    except Exception as e:
        print('[RESUME] error', e)


    last_loss = 0.0
    step_count = 0

    steps = max(2, min(10, cfg.train.epochs * 2))
    if train_loader is None:
        for _ in range(steps):
            optimizer.zero_grad()
            loss = model.step()
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().item() if hasattr(loss, 'detach') else loss)
            step_count += 1
    else:
        import torch as T
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            # Dummy loss using token ids as floats
            x = batch.input_ids.float()
            hidden = cfg.model.hidden_dim
            if x.shape[1] < hidden:
                pad = T.zeros((x.shape[0], hidden - x.shape[1]))
                x = T.cat([x, pad], dim=1)
            else:
                x = x[:, :hidden]
            out = model.net(x).mean() if hasattr(model, 'net') else x.mean()
            loss = (out ** 2)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().item() if hasattr(loss, 'detach') else loss)
            step_count += 1
            if i >= steps:
                break
    
    if torch is not None:
        # quick CPU dummy op
        import math
        x = 0.0
        for i in range(10000):
            x += math.sin(i) * 1e-6
    
    # Save metrics and checkpoint
    try:
        (arts.output_dir / 'metrics.json').write_text(json.dumps({'steps': int(step_count), 'final_loss': float(last_loss)}, indent=2))
    except Exception:
        pass
    try:
        save_checkpoint(arts, cfg, model, optimizer, step=step_count, last_loss=last_loss)
    except Exception:
        pass
    return arts
    
