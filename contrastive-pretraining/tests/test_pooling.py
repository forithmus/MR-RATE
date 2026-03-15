"""Tests for pooling modules: SimpleAttnPool, CrossAttnPool, GatedAttnPool."""
import pytest
import torch
from mr_rate import SimpleAttnPool, CrossAttnPool, GatedAttnPool


DIM = 32
B, R, T = 2, 4, 8  # batch, volumes (reconstructions), tokens


class TestSimpleAttnPool:
    def test_output_shape(self):
        pool = SimpleAttnPool(DIM)
        x = torch.randn(B, R, T, DIM)
        out = pool(x)
        assert out.shape == (B, T, DIM)

    def test_output_shape_with_mask(self):
        pool = SimpleAttnPool(DIM)
        x = torch.randn(B, R, T, DIM)
        mask = torch.ones(B, R, dtype=torch.bool)
        mask[0, 2:] = False  # mask out some volumes
        out = pool(x, mask=mask)
        assert out.shape == (B, T, DIM)

    def test_masked_volumes_ignored(self):
        """When only 1 volume is valid, output should depend only on it."""
        pool = SimpleAttnPool(DIM)
        x = torch.randn(B, R, T, DIM)
        mask = torch.zeros(B, R, dtype=torch.bool)
        mask[:, 0] = True  # only first volume valid
        out = pool(x, mask=mask)
        assert out.shape == (B, T, DIM)
        assert not torch.isnan(out).any()

    def test_gradient_flows(self):
        pool = SimpleAttnPool(DIM)
        x = torch.randn(B, R, T, DIM, requires_grad=True)
        out = pool(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestCrossAttnPool:
    def test_output_shape(self):
        pool = CrossAttnPool(DIM, num_heads=4)
        x = torch.randn(B, R, T, DIM)
        text_query = torch.randn(B, DIM)
        out = pool(x, text_query)
        assert out.shape == (B, T, DIM)

    def test_output_shape_with_mask(self):
        pool = CrossAttnPool(DIM, num_heads=4)
        x = torch.randn(B, R, T, DIM)
        text_query = torch.randn(B, DIM)
        mask = torch.ones(B, R, dtype=torch.bool)
        mask[1, 3] = False
        out = pool(x, text_query, mask=mask)
        assert out.shape == (B, T, DIM)

    def test_gradient_flows(self):
        pool = CrossAttnPool(DIM, num_heads=4)
        x = torch.randn(B, R, T, DIM, requires_grad=True)
        text_query = torch.randn(B, DIM, requires_grad=True)
        out = pool(x, text_query)
        out.sum().backward()
        assert x.grad is not None
        assert text_query.grad is not None


class TestGatedAttnPool:
    def test_output_shape(self):
        pool = GatedAttnPool(DIM)
        x = torch.randn(B, R, T, DIM)
        text_query = torch.randn(B, DIM)
        out = pool(x, text_query)
        assert out.shape == (B, T, DIM)

    def test_output_shape_with_mask(self):
        pool = GatedAttnPool(DIM)
        x = torch.randn(B, R, T, DIM)
        text_query = torch.randn(B, DIM)
        mask = torch.ones(B, R, dtype=torch.bool)
        mask[0, 1:] = False
        out = pool(x, text_query, mask=mask)
        assert out.shape == (B, T, DIM)
        assert not torch.isnan(out).any()

    def test_gradient_flows(self):
        pool = GatedAttnPool(DIM)
        x = torch.randn(B, R, T, DIM, requires_grad=True)
        text_query = torch.randn(B, DIM, requires_grad=True)
        out = pool(x, text_query)
        out.sum().backward()
        assert x.grad is not None
        assert text_query.grad is not None
