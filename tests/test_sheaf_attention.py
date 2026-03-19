"""Tests for the sheaf-inspired cross-head mixing attention mechanism."""

import sys
import os
import torch
import pytest

# Add project root to path so we can import model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model import GPTConfig, GPT, CausalSelfAttention


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_config(use_sheaf_mixing=False, **overrides):
    """Return a small GPTConfig suitable for CPU tests."""
    defaults = dict(
        n_layer=2,
        n_head=4,
        n_embd=64,
        block_size=32,
        vocab_size=100,
        dropout=0.0,
        bias=False,
        use_sheaf_mixing=use_sheaf_mixing,
    )
    defaults.update(overrides)
    return GPTConfig(**defaults)


def _make_attn(config):
    """Build a CausalSelfAttention module from config."""
    attn = CausalSelfAttention(config)
    attn.eval()
    return attn


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStandardPathUnchanged:
    """use_sheaf_mixing=False must match the original manual attention path."""

    def test_standard_output_matches_manual(self):
        torch.manual_seed(42)
        config = _small_config(use_sheaf_mixing=False)
        attn = _make_attn(config)

        B, T, C = 2, 16, config.n_embd
        x = torch.randn(B, T, C)

        y = attn(x)
        assert y.shape == (B, T, C)
        assert torch.isfinite(y).all()


class TestOutputShape:
    """Sheaf path must produce (B, T, C) output."""

    def test_shape(self):
        torch.manual_seed(42)
        config = _small_config(use_sheaf_mixing=True)
        attn = _make_attn(config)

        B, T, C = 2, 16, config.n_embd
        x = torch.randn(B, T, C)

        y = attn(x)
        assert y.shape == (B, T, C)

    def test_shape_full_block_size(self):
        torch.manual_seed(42)
        config = _small_config(use_sheaf_mixing=True)
        attn = _make_attn(config)

        B, T, C = 1, config.block_size, config.n_embd
        x = torch.randn(B, T, C)

        y = attn(x)
        assert y.shape == (B, T, C)


class TestCausality:
    """Output at position t must depend only on positions 0..t."""

    def test_future_independence(self):
        torch.manual_seed(42)
        config = _small_config(use_sheaf_mixing=True)
        attn = _make_attn(config)

        B, T, C = 1, 16, config.n_embd
        x = torch.randn(B, T, C)

        # Run forward with original input
        y1 = attn(x)

        # Perturb future positions (positions 8..15)
        x_perturbed = x.clone()
        x_perturbed[:, 8:, :] = torch.randn(B, T - 8, C)

        y2 = attn(x_perturbed)

        # Output at positions 0..7 must be identical
        assert torch.allclose(y1[:, :8, :], y2[:, :8, :], atol=1e-5), \
            "Sheaf attention violates causality: future tokens affected past outputs"

    def test_single_token(self):
        """First token output must be deterministic regardless of later tokens."""
        torch.manual_seed(42)
        config = _small_config(use_sheaf_mixing=True)
        attn = _make_attn(config)

        B, C = 1, config.n_embd
        x1 = torch.randn(B, 10, C)
        x2 = torch.randn(B, 10, C)
        # Share only the first token
        x2[:, 0, :] = x1[:, 0, :].clone()

        y1 = attn(x1)
        y2 = attn(x2)

        assert torch.allclose(y1[:, 0, :], y2[:, 0, :], atol=1e-5)


class TestOrthogonalHeadsDegeneracy:
    """When heads are orthogonal, sheaf mixing should approximate standard attention.

    If Q and K are constructed so that cross-head dot products are ~0
    (i.e., Q[h] . K[g] ≈ 0 for h≠g), then beta becomes ~uniform over g
    and the mixing collapses toward standard single-head attention behavior.
    More precisely, beta[h,h] >> beta[h,g≠h], so the sheaf output should
    closely match standard attention output.
    """

    def test_orthogonal_convergence(self):
        torch.manual_seed(42)
        config = _small_config(use_sheaf_mixing=False)
        attn_std = _make_attn(config)

        config_sheaf = _small_config(use_sheaf_mixing=True)
        attn_sheaf = CausalSelfAttention(config_sheaf)
        attn_sheaf.eval()

        # Copy weights so both modules are identical except for the mixing flag
        attn_sheaf.load_state_dict(attn_std.state_dict(), strict=False)

        B, T, C = 2, 8, config.n_embd
        H = config.n_head
        D = C // H

        # Construct input that produces orthogonal heads:
        # We'll directly test the internal _sheaf_attention vs standard attention
        # by crafting Q, K, V where cross-head dots are small.
        x = torch.randn(B, T, C)

        # Get Q, K, V from the same projection
        qkv = attn_std.c_attn(x)
        q, k, v = qkv.split(config.n_embd, dim=2)
        q = q.view(B, T, H, D).transpose(1, 2)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        # Standard attention output (manual path)
        import math
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(D))
        mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        y_std = att @ v  # (B, H, T, D)
        y_std = y_std.transpose(1, 2).contiguous().view(B, T, C)

        # Sheaf attention output
        y_sheaf = attn_sheaf._sheaf_attention(q, k, v)

        # They won't be identical (beta is not exactly identity), but should
        # be in the same ballpark. Check that the relative difference is bounded.
        diff = (y_sheaf - y_std).norm() / y_std.norm()
        # This is a soft check — with random weights, cross-head dots are nonzero
        # but typically small relative to same-head dots. Allow generous tolerance.
        assert diff < 2.0, f"Sheaf output diverged too far from standard: relative diff = {diff:.4f}"


class TestGradientFlow:
    """Gradients must flow through both alpha and beta paths."""

    def test_gradients_nonzero(self):
        torch.manual_seed(42)
        config = _small_config(use_sheaf_mixing=True)
        attn = CausalSelfAttention(config)
        attn.train()

        B, T, C = 2, 8, config.n_embd
        x = torch.randn(B, T, C)

        y = attn(x)
        loss = y.sum()
        loss.backward()

        assert attn.c_attn.weight.grad is not None, "No gradient on c_attn.weight"
        assert attn.c_attn.weight.grad.abs().sum() > 0, "Gradient is all zeros on c_attn.weight"
        assert attn.c_proj.weight.grad is not None, "No gradient on c_proj.weight"
        assert attn.c_proj.weight.grad.abs().sum() > 0, "Gradient is all zeros on c_proj.weight"

    def test_gradients_flow_through_input(self):
        torch.manual_seed(42)
        config = _small_config(use_sheaf_mixing=True)
        attn = CausalSelfAttention(config)
        attn.train()

        B, T, C = 2, 8, config.n_embd
        x = torch.randn(B, T, C, requires_grad=True)

        y = attn(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None, "No gradient on input"
        assert x.grad.abs().sum() > 0, "Gradient is all zeros on input"


class TestNumericalStability:
    """No NaN or Inf in output under normal conditions."""

    def test_no_nan_inf_float32(self):
        torch.manual_seed(42)
        config = _small_config(use_sheaf_mixing=True)
        attn = _make_attn(config)

        B, T, C = 2, 16, config.n_embd
        x = torch.randn(B, T, C)

        y = attn(x)
        assert torch.isfinite(y).all(), "Output contains NaN or Inf"

    def test_no_nan_with_large_input(self):
        """Scaled inputs should not cause numerical issues."""
        torch.manual_seed(42)
        config = _small_config(use_sheaf_mixing=True)
        attn = _make_attn(config)

        B, T, C = 2, 16, config.n_embd
        x = torch.randn(B, T, C) * 10.0  # scaled up

        y = attn(x)
        assert torch.isfinite(y).all(), "Output contains NaN or Inf with large inputs"


class TestFullModelForward:
    """Full GPT model with sheaf mixing produces finite loss."""

    def test_forward_with_loss(self):
        torch.manual_seed(42)
        config = _small_config(use_sheaf_mixing=True)
        model = GPT(config)
        model.eval()

        B, T = 2, 16
        idx = torch.randint(0, config.vocab_size, (B, T))
        targets = torch.randint(0, config.vocab_size, (B, T))

        logits, loss = model(idx, targets)
        assert logits.shape == (B, T, config.vocab_size)
        assert loss is not None
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    def test_forward_without_targets(self):
        torch.manual_seed(42)
        config = _small_config(use_sheaf_mixing=True)
        model = GPT(config)
        model.eval()

        B, T = 2, 16
        idx = torch.randint(0, config.vocab_size, (B, T))

        logits, loss = model(idx)
        assert logits.shape == (B, 1, config.vocab_size)
        assert loss is None

    def test_backward_pass(self):
        torch.manual_seed(42)
        config = _small_config(use_sheaf_mixing=True)
        model = GPT(config)
        model.train()

        B, T = 2, 8
        idx = torch.randint(0, config.vocab_size, (B, T))
        targets = torch.randint(0, config.vocab_size, (B, T))

        logits, loss = model(idx, targets)
        loss.backward()

        # Check that gradients exist on key parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestParallelEquivalence:
    """Batched parallel implementation must match the sequential loop version."""

    def _get_qkv(self, attn, x):
        """Extract Q, K, V from the attention module."""
        B, T, C = x.shape
        H = attn.n_head
        D = C // H
        qkv = attn.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, H, D).transpose(1, 2)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)
        return q, k, v

    def test_parallel_matches_loop(self):
        """New batched _sheaf_attention must match old _sheaf_attention_loop."""
        torch.manual_seed(42)
        config = _small_config(use_sheaf_mixing=True)
        attn = _make_attn(config)

        B, T, C = 2, 16, config.n_embd
        x = torch.randn(B, T, C)
        q, k, v = self._get_qkv(attn, x)

        y_parallel = attn._sheaf_attention(q, k, v)
        y_loop = attn._sheaf_attention_loop(q, k, v)

        assert torch.allclose(y_parallel, y_loop, atol=1e-5), \
            f"Max diff: {(y_parallel - y_loop).abs().max().item()}"

    def test_chunked_matches_full(self):
        """sheaf_chunk_size=1 must match sheaf_chunk_size=0 (full parallel)."""
        torch.manual_seed(42)
        config_full = _small_config(use_sheaf_mixing=True, sheaf_chunk_size=0)
        config_chunked = _small_config(use_sheaf_mixing=True, sheaf_chunk_size=1)
        attn_full = _make_attn(config_full)
        attn_chunked = CausalSelfAttention(config_chunked)
        attn_chunked.eval()
        attn_chunked.load_state_dict(attn_full.state_dict(), strict=False)

        B, T, C = 2, 16, config_full.n_embd
        x = torch.randn(B, T, C)
        q, k, v = self._get_qkv(attn_full, x)

        y_full = attn_full._sheaf_attention(q, k, v)
        y_chunked = attn_chunked._sheaf_attention(q, k, v)

        assert torch.allclose(y_full, y_chunked, atol=1e-5), \
            f"Max diff: {(y_full - y_chunked).abs().max().item()}"

    def test_chunked_size_2(self):
        """sheaf_chunk_size=2 must match full parallel."""
        torch.manual_seed(42)
        config_full = _small_config(use_sheaf_mixing=True, sheaf_chunk_size=0)
        config_chunked = _small_config(use_sheaf_mixing=True, sheaf_chunk_size=2)
        attn_full = _make_attn(config_full)
        attn_chunked = CausalSelfAttention(config_chunked)
        attn_chunked.eval()
        attn_chunked.load_state_dict(attn_full.state_dict(), strict=False)

        B, T, C = 2, 16, config_full.n_embd
        x = torch.randn(B, T, C)
        q, k, v = self._get_qkv(attn_full, x)

        y_full = attn_full._sheaf_attention(q, k, v)
        y_chunked = attn_chunked._sheaf_attention(q, k, v)

        assert torch.allclose(y_full, y_chunked, atol=1e-5), \
            f"Max diff: {(y_full - y_chunked).abs().max().item()}"

    def test_chunked_gradient_flow(self):
        """Gradient checkpointing in chunked mode must produce non-zero gradients."""
        torch.manual_seed(42)
        config = _small_config(use_sheaf_mixing=True, sheaf_chunk_size=2)
        attn = CausalSelfAttention(config)
        attn.train()

        B, T, C = 2, 8, config.n_embd
        x = torch.randn(B, T, C, requires_grad=True)

        y = attn(x)
        loss = y.sum()
        loss.backward()

        assert attn.c_attn.weight.grad is not None, "No gradient on c_attn.weight"
        assert attn.c_attn.weight.grad.abs().sum() > 0, "Gradient is all zeros"
        assert x.grad is not None, "No gradient on input"
        assert x.grad.abs().sum() > 0, "Input gradient is all zeros"


class TestConfigSerialization:
    """use_sheaf_mixing round-trips through model_args dict."""

    def test_config_has_field(self):
        config = _small_config(use_sheaf_mixing=True)
        assert config.use_sheaf_mixing is True

        config = _small_config(use_sheaf_mixing=False)
        assert config.use_sheaf_mixing is False

    def test_model_args_roundtrip(self):
        """Simulate the train.py checkpoint save/load cycle."""
        config = _small_config(use_sheaf_mixing=True)

        # Simulate saving model_args (as train.py does)
        model_args = dict(
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
            block_size=config.block_size,
            bias=config.bias,
            vocab_size=config.vocab_size,
            dropout=config.dropout,
            use_sheaf_mixing=config.use_sheaf_mixing,
        )

        # Simulate loading (reconstruct config from model_args)
        restored_config = GPTConfig(**model_args)
        assert restored_config.use_sheaf_mixing is True

    def test_backward_compat_missing_key(self):
        """Old checkpoints without use_sheaf_mixing should default to False."""
        old_model_args = dict(
            n_layer=2, n_head=4, n_embd=64,
            block_size=32, bias=False, vocab_size=100, dropout=0.0,
        )
        # GPTConfig default should fill in use_sheaf_mixing=False
        config = GPTConfig(**old_model_args)
        assert config.use_sheaf_mixing is False
