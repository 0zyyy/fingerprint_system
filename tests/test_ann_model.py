"""
Unit tests for FingerprintANN (PyTorch model).

Tests here never touch the filesystem or real dataset — purely structural.
"""
import pytest
import torch
import numpy as np
from models.ann_model import FingerprintANN, TrainingConfig

INPUT_DIM = 1764
OUTPUT_DIM = 10
BATCH = 8


@pytest.fixture
def model():
    return FingerprintANN(input_dim=INPUT_DIM, hidden_layers=[512, 256], output_dim=OUTPUT_DIM)


@pytest.fixture
def batch(model):
    return torch.randn(BATCH, INPUT_DIM)


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class TestArchitecture:
    def test_output_shape(self, model, batch):
        out = model(batch)
        assert out.shape == (BATCH, OUTPUT_DIM)

    def test_single_sample_output_shape(self, model):
        x = torch.randn(1, INPUT_DIM)
        out = model(x)
        assert out.shape == (1, OUTPUT_DIM)

    def test_logits_not_probabilities(self, model, batch):
        """Forward pass returns logits — should NOT sum to 1."""
        model.eval()
        with torch.no_grad():
            out = model(batch)
        sums = out.sum(dim=1)
        assert not torch.allclose(sums, torch.ones(BATCH), atol=0.01), \
            "Expected raw logits, got probabilities"

    def test_softmax_sums_to_one(self, model, batch):
        model.eval()
        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(BATCH), atol=1e-5)

    def test_output_has_correct_num_classes(self, model, batch):
        out = model(batch)
        assert out.shape[1] == OUTPUT_DIM


# ---------------------------------------------------------------------------
# Backpropagation
# ---------------------------------------------------------------------------

class TestBackprop:
    def test_gradients_flow_to_all_layers(self, model, batch):
        """All parameters must receive gradients after backward."""
        model.train()
        labels = torch.randint(0, OUTPUT_DIM, (BATCH,))
        loss = torch.nn.CrossEntropyLoss()(model(batch), labels)
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for: {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in: {name}"

    def test_loss_is_scalar(self, model, batch):
        labels = torch.randint(0, OUTPUT_DIM, (BATCH,))
        loss = torch.nn.CrossEntropyLoss()(model(batch), labels)
        assert loss.ndim == 0

    def test_loss_decreases_after_one_step(self, model):
        """Single gradient step should reduce loss on same batch."""
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        x = torch.randn(16, INPUT_DIM)
        y = torch.randint(0, OUTPUT_DIM, (16,))

        loss_before = criterion(model(x), y).item()
        optimizer.zero_grad()
        criterion(model(x), y).backward()
        optimizer.step()
        loss_after = criterion(model(x), y).item()

        assert loss_after < loss_before, \
            f"Loss did not decrease: {loss_before:.4f} → {loss_after:.4f}"


# ---------------------------------------------------------------------------
# Train / eval mode
# ---------------------------------------------------------------------------

class TestTrainEvalMode:
    def test_dropout_active_in_train_mode(self, model):
        """Outputs differ between two forward passes in train mode (dropout active)."""
        model.train()
        x = torch.randn(64, INPUT_DIM)
        out1 = model(x)
        out2 = model(x)
        assert not torch.allclose(out1, out2), "Dropout should cause different outputs in train mode"

    def test_dropout_inactive_in_eval_mode(self, model):
        """Outputs are identical in eval mode (dropout disabled)."""
        model.eval()
        x = torch.randn(64, INPUT_DIM)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2), "Outputs should be deterministic in eval mode"

    def test_batchnorm_uses_running_stats_in_eval(self, model):
        """BatchNorm should not raise errors in eval mode with batch size 1."""
        model.eval()
        x = torch.randn(1, INPUT_DIM)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, OUTPUT_DIM)


# ---------------------------------------------------------------------------
# predict() interface
# ---------------------------------------------------------------------------

class TestPredictInterface:
    def test_predict_returns_tuple(self, model):
        x = torch.randn(INPUT_DIM)
        result = model.predict(x)
        assert isinstance(result, tuple) and len(result) == 2

    def test_predict_confidence_in_range(self, model):
        x = torch.randn(INPUT_DIM)
        confidence, _ = model.predict(x)
        assert 0.0 <= confidence <= 1.0

    def test_predict_class_id_in_range(self, model):
        x = torch.randn(INPUT_DIM)
        _, class_id = model.predict(x)
        assert 0 <= class_id < OUTPUT_DIM

    def test_predict_accepts_1d_input(self, model):
        x = torch.randn(INPUT_DIM)       # 1D — no batch dim
        confidence, class_id = model.predict(x)
        assert isinstance(confidence, float)
        assert isinstance(class_id, int)

    def test_predict_accepts_2d_input(self, model):
        x = torch.randn(1, INPUT_DIM)    # 2D — with batch dim
        confidence, class_id = model.predict(x)
        assert isinstance(confidence, float)

    def test_predict_numpy_compatible(self, model):
        """predict() result types are compatible with downstream code expecting Python scalars."""
        x = torch.randn(INPUT_DIM)
        confidence, class_id = model.predict(x)
        assert isinstance(confidence, float)
        assert isinstance(class_id, int)


# ---------------------------------------------------------------------------
# Config variations
# ---------------------------------------------------------------------------

class TestConfigVariations:
    @pytest.mark.parametrize("hidden", [[256], [512, 256], [512, 256, 128]])
    def test_different_hidden_layer_configs(self, hidden):
        model = FingerprintANN(input_dim=INPUT_DIM, hidden_layers=hidden, output_dim=OUTPUT_DIM)
        out = model(torch.randn(4, INPUT_DIM))
        assert out.shape == (4, OUTPUT_DIM)

    @pytest.mark.parametrize("dropout", [0.0, 0.3, 0.5])
    def test_different_dropout_values(self, dropout):
        model = FingerprintANN(dropout=dropout)
        model.eval()
        out = model(torch.randn(4, INPUT_DIM))
        assert out.shape == (4, OUTPUT_DIM)
