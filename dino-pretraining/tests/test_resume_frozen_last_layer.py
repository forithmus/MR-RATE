"""Resume from a checkpoint saved during freeze_last_layer (GPU smoke 941555 failed on this):
the frozen prototype layer never gets a gradient, so AdamW has no state for it and the checkpoint lacks
`optimizer.state.<...>.last_layer.*`. Only those keys may be missing; anything else must still fail loudly."""
import pytest
import torch
import torch.distributed as dist
from torch import nn

from dinov3.checkpointer import load_checkpoint, save_checkpoint
from mr_dino.recipe import cancel_last_layer_gradients, missing_frozen_optimizer_keys


@pytest.fixture(autouse=True)
def single_process_group(tmp_path):
    """dinov3's DCP checkpointer needs a process group; a 1-rank gloo group is enough on CPU."""
    if not dist.is_initialized():
        dist.init_process_group("gloo", init_method=f"file://{tmp_path}/pg", rank=0, world_size=1)
    yield
    dist.destroy_process_group()


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.dino_head = nn.ModuleDict({"mlp": nn.Linear(4, 4), "last_layer": nn.Linear(4, 3, bias=False)})

    def forward(self, x):
        return self.dino_head["last_layer"](self.dino_head["mlp"](self.backbone(x)))


def _train_step(model, opt, freeze=True, drop=None):
    opt.zero_grad(set_to_none=True)
    model(torch.randn(2, 4)).sum().backward()
    if freeze:
        cancel_last_layer_gradients(model)
    if drop is not None:   # simulate another parameter without optimizer state
        dict(model.named_parameters())[drop].grad = None
    opt.step()


def test_resume_during_freeze_allows_only_last_layer_state(tmp_path):
    torch.manual_seed(0)
    model = Tiny()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    _train_step(model, opt)
    save_checkpoint(tmp_path / "ckpt", iteration=1, model=model, optimizer=opt)

    fresh = Tiny()
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=1e-2)
    missing = missing_frozen_optimizer_keys(tmp_path / "ckpt", fresh, fresh_opt)
    assert missing and all(k.startswith("optimizer.state.") and ".last_layer." in k for k in missing)
    with pytest.raises(BaseException, match="Missing key in checkpoint state_dict.*last_layer"):   # old strict resume = GPU smoke 941555 failure
        strict_model = Tiny()
        load_checkpoint(tmp_path / "ckpt", model=strict_model, optimizer=torch.optim.AdamW(strict_model.parameters()), strict_loading=True)
    it = load_checkpoint(tmp_path / "ckpt", model=fresh, optimizer=fresh_opt, strict_loading=not missing)
    assert it == 1
    for (n, a), (_, b) in zip(model.state_dict().items(), fresh.state_dict().items()):
        assert torch.equal(a, b), n


def test_other_missing_state_still_raises(tmp_path):
    torch.manual_seed(0)
    model = Tiny()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    _train_step(model, opt, drop="dino_head.mlp.weight")
    save_checkpoint(tmp_path / "ckpt", iteration=1, model=model, optimizer=opt)
    fresh = Tiny()
    with pytest.raises(RuntimeError, match="lacks"):
        missing_frozen_optimizer_keys(tmp_path / "ckpt", fresh, torch.optim.AdamW(fresh.parameters(), lr=1e-2))
