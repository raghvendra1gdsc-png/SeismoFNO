"""
test_baselines.py — Unit Tests for LSTM and MLP Baseline Architectures.
"""

import pytest
import torch
from src.models.lstm_baseline import LSTMBaseline
from src.models.mlp_baseline import MLPBaseline


def test_lstm_baseline_forward_shape():
    batch_size = 4
    in_channels = 10
    out_channels = 3
    seq_len = 256

    model = LSTMBaseline(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_dim=64,
        num_layers=2,
        bidirectional=False,
    )

    x = torch.randn(batch_size, in_channels, seq_len)
    out = model(x)

    assert out.shape == (batch_size, out_channels, seq_len), f"Expected {(batch_size, out_channels, seq_len)}, got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN"


def test_lstm_baseline_backward():
    batch_size = 2
    in_channels = 12
    out_channels = 3
    seq_len = 128

    model = LSTMBaseline(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_dim=32,
        num_layers=2,
    )

    x = torch.randn(batch_size, in_channels, seq_len, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "Gradients not computed for input"
    assert model.get_num_parameters() > 0


def test_mlp_baseline_forward_shape():
    batch_size = 4
    in_channels = 10
    out_channels = 3
    seq_len = 256

    model = MLPBaseline(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_dim=64,
        num_layers=3,
    )

    x = torch.randn(batch_size, in_channels, seq_len)
    out = model(x)

    assert out.shape == (batch_size, out_channels, seq_len), f"Expected {(batch_size, out_channels, seq_len)}, got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN"


def test_mlp_baseline_backward():
    batch_size = 2
    in_channels = 10
    out_channels = 3
    seq_len = 128

    model = MLPBaseline(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_dim=32,
        num_layers=2,
    )

    x = torch.randn(batch_size, in_channels, seq_len, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "Gradients not computed for input"
    assert model.get_num_parameters() > 0
