from pathlib import Path
from amharichnet.cli import main

def test_cli_help(capsys):
    main(["--help"])  # should print help without crash
    captured = capsys.readouterr()
    assert "H-Net CLI" in captured.out


def test_cli_train_dry_run(tmp_path, capsys):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text((Path("configs/base.yaml").read_text()))
    main(["train", "--config", str(cfg)])
    out = capsys.readouterr().out
    assert "[DRY-RUN] Training" in out
