import argparse
import sys

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore

from .utils.config import Config
from .utils.repro import set_seed


def load_config(path: str) -> Config:
    if yaml is None:
        raise RuntimeError("pyyaml is required. Install with `pip install pyyaml`.\n")
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return Config(**raw)


def main(argv=None):
    argv = argv or sys.argv[1:]
    p = argparse.ArgumentParser(prog="hnet", description="H-Net CLI")
    sub = p.add_subparsers(dest="cmd")

    p_train = sub.add_parser("train", help="Run training")
    p_train.add_argument("--config", required=True, help="Path to YAML config")

    p_infer = sub.add_parser("infer", help="Run inference or start API")
    p_infer.add_argument("--config", required=True)
    p_infer.add_argument("--prompt", required=False)

    p_eval = sub.add_parser("eval", help="Run evaluation")
    p_eval.add_argument("--config", required=True)

    args = p.parse_args(argv)

    if args.cmd == "train":
        cfg = load_config(args.config)
        set_seed(cfg.train.seed)
        from .train.training_loop import run_training
        run_training(cfg)
        print("[OK] Training scaffold executed. Artifacts in:", cfg.train.output_dir)
    elif args.cmd == "infer":
        cfg = load_config(args.config)
        set_seed(1337)
        if args.prompt:
            print("[DRY-RUN] Inference for prompt:", args.prompt)
        else:
            print("[DRY-RUN] Would start API server with config:", args.config)
    elif args.cmd == "eval":
        cfg = load_config(args.config)
        set_seed(cfg.train.seed)
        print("[DRY-RUN] Evaluation would run.")
    else:
        p.print_help()

if __name__ == "__main__":
    main()
