"""Check replay gradients against a dense bag, including trainable features."""
import copy

import torch
from torch import nn

from mr_dino.mil_head import SparseClassifyThenAggregate
from mr_dino.mil_stream import LastBlockAdapter, backward, evidence
from mr_dino.model import DinoVisionTransformer3D


def test_replayed_head_and_feature_gradients_match_dense_global_softmax():
    torch.manual_seed(3)
    head=SparseClassifyThenAggregate(16,3,hidden_dim=8,mlp_hidden_dim=7,use_norm=True,checkpoint_chunks=False)
    head.route_bias_raw=nn.Parameter(torch.tensor([1.,2.,3.]))
    transform=nn.Linear(16,16)
    reference=copy.deepcopy(head);reference_transform=copy.deepcopy(transform)
    parts=[torch.randn(7,16),torch.randn(3,16),torch.randn(11,16)]
    masks=[torch.rand(len(x),3)>.4 for x in parts]
    def factory():
        for x,m in zip(parts,masks):yield transform(x),m
    target=torch.tensor([1.,0.,1.])
    loss,logits=backward(head,factory,target)
    all_tokens=reference_transform(torch.cat(parts))
    scores,patch=reference._chunk_outputs(all_tokens,torch.arange(3))
    scores=scores-(~torch.cat(masks)).float()*torch.nn.functional.softplus(reference.route_bias_raw)
    expected=(scores.softmax(0)*patch).sum(0)*reference.output_scale+reference.output_bias
    expected_loss=torch.nn.functional.binary_cross_entropy_with_logits(expected,target)
    expected_loss.backward()
    torch.testing.assert_close(logits[0],expected,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(loss,expected_loss)
    for actual,ref in zip(head.parameters(),reference.parameters()):
        torch.testing.assert_close(actual.grad,ref.grad,atol=2e-6,rtol=2e-5)
    for actual,ref in zip(transform.parameters(),reference_transform.parameters()):
        torch.testing.assert_close(actual.grad,ref.grad,atol=2e-6,rtol=2e-5)
    expected_evidence=(scores.softmax(0)*patch*reference.output_scale).clamp_min(0)
    torch.testing.assert_close(torch.cat(list(evidence(head,factory))),expected_evidence)


def test_mri_last_block_matches_backbone_with_dynamic_prefix_and_grid():
    torch.manual_seed(0)
    model=DinoVisionTransformer3D(embed_dim=32,depth=2,num_heads=4,
                                  volume_size=(4,32,32),patch_size=(2,16,16),
                                  n_storage_tokens=2,drop_path_rate=.3)
    model.init_weights();model.eval()
    x=torch.randn(1,1,4,32,32)
    tokens,grid=model.prepare_tokens_with_masks(x)
    rope=model.rope_embed(D=grid[0],H=grid[1],W=grid[2])
    for block in model.blocks[:-1]:tokens=block(tokens,rope)
    model.requires_grad_(False)
    adapter=LastBlockAdapter(model)
    adapter.train()
    actual=adapter(tokens,grid)
    expected=model.forward_features(x)["x_norm_patchtokens"]
    torch.testing.assert_close(actual,expected)
    actual.square().sum().backward()
    assert adapter.block.attn.qkv.weight.grad is not None
    assert adapter.prefix==3
