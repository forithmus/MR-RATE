"""Stage transitions: an explicit RESUME (previous stage) must not override this stage's own progress."""
from pathlib import Path

from mr_dino.train_fsdp import resolve_checkpoint


def make_ckpt(root: Path, step: int, complete: bool = True) -> Path:
    path = root / "checkpoints" / f"step_{step:08d}"
    path.mkdir(parents=True)
    if complete:
        (path / "COMPLETE").touch()
    (root / "checkpoints" / "latest.txt").write_text(path.name + "\n")
    return path


def test_previous_stage_checkpoint_until_own_progress(tmp_path):
    previous = make_ckpt(tmp_path / "pretrain", 50_000)
    gram = tmp_path / "gram"
    assert resolve_checkpoint(gram, str(previous)) == previous          # first start of phase 2
    own = make_ckpt(gram, 500)
    assert resolve_checkpoint(gram, str(previous)) == own               # requeue continues phase 2
    assert resolve_checkpoint(gram, "latest") == own
    assert resolve_checkpoint(gram, "none") is None


def test_incomplete_own_checkpoint_is_ignored(tmp_path):
    previous = make_ckpt(tmp_path / "pretrain", 50_000)
    make_ckpt(tmp_path / "gram", 500, complete=False)
    assert resolve_checkpoint(tmp_path / "gram", str(previous)) == previous
    assert resolve_checkpoint(tmp_path / "fresh", "latest") is None
