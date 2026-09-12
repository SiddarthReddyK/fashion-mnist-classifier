import torch
from src.model import SimpleCNN


def test_output_shape():
    model = SimpleCNN()
    fake_input = torch.randn(1, 1, 28, 28)
    output = model(fake_input)
    assert output.shape == (1, 10)


def test_batch_output_shape():
    model = SimpleCNN()
    fake_batch = torch.randn(32, 1, 28, 28)
    output = model(fake_batch)
    assert output.shape == (32, 10)


def test_model_runs_without_error():
    model = SimpleCNN()
    fake_input = torch.randn(1, 1, 28, 28)
    try:
        model(fake_input)
    except Exception as e:
        assert False, f"Model forward pass failed: {e}"