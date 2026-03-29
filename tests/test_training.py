"""
Training convergence and ONNX parity tests.

Convergence: loss must decrease on the real dataset over N epochs.
ONNX parity: exported model output must match PyTorch output within tolerance.

These tests load real BMP files — they are skipped if the dataset is absent.
"""
import os
import tempfile
import pytest
import numpy as np
import torch
from pathlib import Path

DATASET_ROOT = Path("SHIDIQ JARI TENGAH KNN/SHIDIQ JARI TENGAH KNN")
DATASET_AVAILABLE = DATASET_ROOT.exists() and any(DATASET_ROOT.iterdir())

skip_no_dataset = pytest.mark.skipif(
    not DATASET_AVAILABLE,
    reason="Dataset not found — run on machine with fingerprint BMP files"
)

skip_no_onnxruntime = pytest.mark.skipif(
    not pytest.importorskip("onnxruntime", reason="onnxruntime not installed"),
    reason="onnxruntime not installed"
)


# ---------------------------------------------------------------------------
# Convergence smoke test
# ---------------------------------------------------------------------------

class TestTrainingConvergence:
    @skip_no_dataset
    def test_loss_decreases_over_epochs(self):
        """
        Train for 5 epochs on the real dataset.
        Final loss must be lower than initial loss.
        Uses a small LR so we don't accidentally skip over the minimum.
        """
        from train import load_dataset
        from models.ann_model import FingerprintANN, TrainingConfig

        X, y = load_dataset(DATASET_ROOT)
        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y)

        model = FingerprintANN()
        model.train()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Record initial loss
        with torch.no_grad():
            initial_loss = criterion(model(X_t), y_t).item()

        for _ in range(5):
            optimizer.zero_grad()
            loss = criterion(model(X_t), y_t)
            loss.backward()
            optimizer.step()

        final_loss = criterion(model(X_t), y_t).item()
        assert final_loss < initial_loss, (
            f"Loss did not decrease after 5 epochs: {initial_loss:.4f} → {final_loss:.4f}"
        )

    @skip_no_dataset
    def test_accuracy_improves_over_epochs(self):
        """Train for 10 epochs — accuracy on training data must improve."""
        from train import load_dataset
        from models.ann_model import FingerprintANN

        X, y = load_dataset(DATASET_ROOT)
        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y)

        model = FingerprintANN()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        def accuracy():
            model.eval()
            with torch.no_grad():
                preds = model(X_t).argmax(1)
            return (preds == y_t).float().mean().item()

        model.train()
        initial_acc = accuracy()

        for _ in range(10):
            model.train()
            optimizer.zero_grad()
            criterion(model(X_t), y_t).backward()
            optimizer.step()

        final_acc = accuracy()
        assert final_acc > initial_acc, (
            f"Accuracy did not improve: {initial_acc:.4f} → {final_acc:.4f}"
        )

    @skip_no_dataset
    def test_all_subjects_in_dataset(self):
        """Dataset must contain all 10 expected subjects."""
        from train import load_dataset
        X, y = load_dataset(DATASET_ROOT)
        assert len(np.unique(y)) == 10, f"Expected 10 subjects, found {len(np.unique(y))}"

    @skip_no_dataset
    def test_feature_vector_dimension(self):
        """HOG vector must be 1764-dimensional."""
        from train import load_dataset
        X, _ = load_dataset(DATASET_ROOT)
        assert X.shape[1] == 1764, f"Expected 1764 features, got {X.shape[1]}"

    @skip_no_dataset
    def test_feature_vectors_are_finite(self):
        """No NaN or Inf in extracted HOG features."""
        from train import load_dataset
        X, _ = load_dataset(DATASET_ROOT)
        assert np.isfinite(X).all(), "HOG features contain NaN or Inf"


# ---------------------------------------------------------------------------
# ONNX parity
# ---------------------------------------------------------------------------

class TestONNXParity:
    @pytest.fixture
    def trained_model(self):
        """A model trained for a few epochs on synthetic data."""
        from models.ann_model import FingerprintANN
        model = FingerprintANN()
        # Quick synthetic training so weights are non-trivial
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(3):
            x = torch.randn(32, 1764)
            y = torch.randint(0, 10, (32,))
            optimizer.zero_grad()
            torch.nn.CrossEntropyLoss()(model(x), y).backward()
            optimizer.step()
        model.eval()
        return model

    def test_onnx_export_creates_file(self, trained_model, tmp_path):
        from train import export_onnx
        output = str(tmp_path / "test.onnx")
        export_onnx(trained_model, output)
        assert os.path.exists(output)
        assert os.path.getsize(output) > 0

    def test_onnx_output_matches_pytorch(self, trained_model, tmp_path):
        """ONNX and PyTorch must produce identical logits (within float32 tolerance)."""
        import onnxruntime as ort
        from train import export_onnx

        onnx_path = str(tmp_path / "parity.onnx")
        export_onnx(trained_model, onnx_path)

        x = torch.randn(4, 1764)

        # PyTorch output
        trained_model.eval()
        with torch.no_grad():
            pt_out = trained_model(x).numpy()

        # ONNX output
        sess = ort.InferenceSession(onnx_path)
        input_name = sess.get_inputs()[0].name
        ort_out = sess.run(None, {input_name: x.numpy()})[0]

        np.testing.assert_allclose(pt_out, ort_out, rtol=1e-4, atol=1e-5)

    def test_onnx_inference_wrapper(self, trained_model, tmp_path):
        """FingerprintONNX.predict() returns valid (confidence, class_id) from a HOG vector."""
        from train import export_onnx
        from models.onnx_inference import FingerprintONNX

        onnx_path = str(tmp_path / "wrapper.onnx")
        export_onnx(trained_model, onnx_path)

        wrapper = FingerprintONNX(model_path=onnx_path)
        hog_vector = np.random.rand(1764).astype(np.float32)

        confidence, class_id = wrapper.predict(hog_vector)

        assert 0.0 <= confidence <= 1.0
        assert 0 <= class_id < 10

    def test_onnx_single_vs_batch_consistency(self, trained_model, tmp_path):
        """Single sample and batched sample should give the same result."""
        import onnxruntime as ort
        from train import export_onnx

        onnx_path = str(tmp_path / "batch.onnx")
        export_onnx(trained_model, onnx_path)

        sess = ort.InferenceSession(onnx_path)
        input_name = sess.get_inputs()[0].name

        x_single = np.random.rand(1, 1764).astype(np.float32)
        x_batch = np.repeat(x_single, 4, axis=0)

        out_single = sess.run(None, {input_name: x_single})[0]
        out_batch = sess.run(None, {input_name: x_batch})[0]

        np.testing.assert_allclose(out_single[0], out_batch[0], rtol=1e-4)
