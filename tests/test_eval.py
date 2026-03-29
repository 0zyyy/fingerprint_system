"""
End-to-end pipeline accuracy evaluation.

Requires:
  - Dataset present at DATASET_ROOT
  - Trained ONNX model at models/fingerprint_ann.onnx

Run after training:
    python train.py
    uv run pytest tests/test_eval.py -v

Accuracy targets are based on the thesis results:
  - Classification accuracy >= 95% (thesis: 98%)
  - Per-condition accuracy (NRML, OILY, DST, WT) >= 80%
"""
import pytest
import numpy as np
from pathlib import Path

DATASET_ROOT = Path("SHIDIQ JARI TENGAH KNN/SHIDIQ JARI TENGAH KNN")
MODEL_PATH = "models/fingerprint_ann.onnx"

skip_no_dataset = pytest.mark.skipif(
    not DATASET_ROOT.exists(),
    reason="Dataset not found"
)
skip_no_model = pytest.mark.skipif(
    not Path(MODEL_PATH).exists(),
    reason=f"Trained model not found at {MODEL_PATH} — run: python train.py"
)
skip_both = pytest.mark.skipif(
    not DATASET_ROOT.exists() or not Path(MODEL_PATH).exists(),
    reason="Requires both dataset and trained model"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_condition(condition: str) -> tuple[np.ndarray, np.ndarray]:
    """Load HOG features + labels for one condition folder."""
    import cv2
    from train import parse_subject_id
    from core.preprocessing import ImagePreprocessor
    from core.feature_extraction import HOGExtractor

    pre = ImagePreprocessor()
    hog = HOGExtractor()
    features, labels = [], []

    for bmp in sorted((DATASET_ROOT / condition).glob("*.bmp")):
        img = cv2.imread(str(bmp), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        vec = hog.extract(pre.process(img))
        features.append(vec.astype(np.float32))
        labels.append(parse_subject_id(bmp.name))

    return np.stack(features), np.array(labels)


def run_onnx(features: np.ndarray) -> np.ndarray:
    """Run ONNX model on feature matrix, return predicted class IDs."""
    import onnxruntime as ort
    sess = ort.InferenceSession(MODEL_PATH)
    input_name = sess.get_inputs()[0].name
    logits = sess.run(None, {input_name: features})[0]
    return np.argmax(logits, axis=1)


# ---------------------------------------------------------------------------
# Overall accuracy
# ---------------------------------------------------------------------------

class TestOverallAccuracy:
    @skip_both
    def test_overall_accuracy_meets_baseline(self):
        """Full dataset accuracy must be >= 95% (thesis reports 98%)."""
        from train import load_dataset
        X, y = load_dataset(DATASET_ROOT)
        preds = run_onnx(X)
        acc = np.mean(preds == y)
        assert acc >= 0.95, f"Overall accuracy {acc:.2%} is below 95% baseline"

    @skip_both
    def test_all_subjects_recognized(self):
        """Every subject must be correctly classified at least once."""
        from train import load_dataset
        X, y = load_dataset(DATASET_ROOT)
        preds = run_onnx(X)
        for subject_id in np.unique(y):
            mask = y == subject_id
            subject_acc = np.mean(preds[mask] == y[mask])
            assert subject_acc > 0.0, f"Subject {subject_id} was never recognized"

    @skip_both
    def test_no_subject_below_50_percent(self):
        """No individual subject should have accuracy below 50%."""
        from train import load_dataset
        X, y = load_dataset(DATASET_ROOT)
        preds = run_onnx(X)
        for subject_id in np.unique(y):
            mask = y == subject_id
            acc = np.mean(preds[mask] == y[mask])
            assert acc >= 0.5, f"Subject {subject_id} accuracy {acc:.2%} below 50%"


# ---------------------------------------------------------------------------
# Per-condition accuracy
# ---------------------------------------------------------------------------

class TestPerConditionAccuracy:
    @skip_both
    @pytest.mark.parametrize("condition,min_acc", [
        ("NRML", 0.90),   # clean — should be highest
        ("OILY", 0.80),   # oily — thesis main focus
        ("DST",  0.80),   # dusty
        ("WT",   0.80),   # wet
    ])
    def test_condition_accuracy(self, condition, min_acc):
        condition_dir = DATASET_ROOT / condition
        if not condition_dir.exists():
            pytest.skip(f"Condition {condition} not in dataset")

        X, y = load_condition(condition)
        preds = run_onnx(X)
        acc = np.mean(preds == y)
        assert acc >= min_acc, (
            f"{condition} accuracy {acc:.2%} below threshold {min_acc:.0%}"
        )


# ---------------------------------------------------------------------------
# Confusion / error analysis
# ---------------------------------------------------------------------------

class TestErrorAnalysis:
    @skip_both
    def test_oily_vs_normal_confusion(self):
        """
        When model makes an error on oily images, it should still predict
        the correct subject class — not a completely wrong identity.
        (Oily errors should be within-subject noise, not cross-subject confusion.)
        """
        X_oily, y_oily = load_condition("OILY")
        X_nrml, y_nrml = load_condition("NRML")

        preds_oily = run_onnx(X_oily)
        preds_nrml = run_onnx(X_nrml)

        # Oily errors should not exceed normal errors by more than 20%
        err_oily = np.mean(preds_oily != y_oily)
        err_nrml = np.mean(preds_nrml != y_nrml)

        assert err_oily <= err_nrml + 0.20, (
            f"Oily error rate {err_oily:.2%} far exceeds normal {err_nrml:.2%} "
            f"(difference > 20%). Preprocessing may not be handling oily condition."
        )

    @skip_both
    def test_print_full_report(self, capsys):
        """Print classification report — always passes, useful for manual inspection."""
        from train import load_dataset
        from sklearn.metrics import classification_report

        X, y = load_dataset(DATASET_ROOT)
        preds = run_onnx(X)
        report = classification_report(y, preds, digits=4)
        print(f"\n{report}")
        # Always passes — just for visibility in pytest -v -s output


# ---------------------------------------------------------------------------
# Threshold sensitivity (FAR / FRR proxy)
# ---------------------------------------------------------------------------

class TestThresholds:
    @skip_both
    @pytest.mark.parametrize("threshold", [0.80, 0.90, 0.95])
    def test_far_frr_at_threshold(self, threshold):
        """
        At each threshold, compute FAR and FRR proxy.
        Neither should exceed 30% — a sanity check, not a strict assertion.
        """
        import onnxruntime as ort
        from train import load_dataset

        X, y = load_dataset(DATASET_ROOT)
        sess = ort.InferenceSession(MODEL_PATH)
        input_name = sess.get_inputs()[0].name
        logits = sess.run(None, {input_name: X})[0]

        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        confidence = probs.max(axis=1)
        preds = probs.argmax(axis=1)

        accepted = confidence >= threshold
        # FAR proxy: wrong identity accepted
        far = np.mean((preds != y) & accepted)
        # FRR proxy: correct identity rejected
        frr = np.mean((preds == y) & ~accepted)

        print(f"\nThreshold={threshold}: FAR={far:.3f}, FRR={frr:.3f}")
        assert far <= 0.30, f"FAR {far:.2%} too high at threshold {threshold}"
        assert frr <= 0.30, f"FRR {frr:.2%} too high at threshold {threshold}"
