"""
Unit tests for FingerprintVerifier — aggregation logic with mocked CNN.

The ONNX model is patched so tests never need a real .onnx file.
Each test controls cnn_score + cnn_user independently of HOG.
"""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from tests.conftest import make_fingerprint_image, make_hog_vector


def make_verifier(tmp_db, cnn_score=0.95, cnn_user=1):
    """Build a FingerprintVerifier with a mocked CNN returning fixed values."""
    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [MagicMock(name="input")]
    probs = np.zeros(10, dtype=np.float32)
    probs[cnn_user] = cnn_score
    mock_session.run.return_value = [np.expand_dims(probs, 0)]

    with patch("onnxruntime.InferenceSession", return_value=mock_session), \
         patch("os.path.exists", return_value=True):
        from core.verifier import FingerprintVerifier
        return FingerprintVerifier(db=tmp_db)


def _patch_cnn(verifier, cnn_score, cnn_user):
    """Swap out the CNN mock on an existing verifier instance."""
    probs = np.zeros(10, dtype=np.float32)
    if cnn_user >= 0:
        probs[cnn_user] = cnn_score
    verifier.cnn.session.run.return_value = [np.expand_dims(probs, 0)]


class TestEnroll:
    def test_enroll_stores_template(self, tmp_db, sample_image):
        tmp_db.add_user(1, "Alice")
        v = make_verifier(tmp_db)
        assert v.enroll(1, sample_image)
        assert len(tmp_db.get_templates()) == 1

    def test_enroll_multiple_samples(self, tmp_db):
        tmp_db.add_user(1, "Alice")
        v = make_verifier(tmp_db)
        for seed in range(3):
            v.enroll(1, make_fingerprint_image(seed=seed))
        assert len(tmp_db.get_templates()) == 3

    def test_enroll_reloads_matcher(self, tmp_db, sample_image):
        tmp_db.add_user(1, "Alice")
        v = make_verifier(tmp_db)
        v.enroll(1, sample_image)
        assert len(v.matcher._templates) == 1


class TestVerifyGranted:
    def test_both_pass_same_user_grants(self, tmp_db, sample_image):
        tmp_db.add_user(1, "Alice")
        v = make_verifier(tmp_db, cnn_score=0.95, cnn_user=1)
        v.enroll(1, sample_image)

        result = v.verify(sample_image)
        assert result["granted"] is True
        assert result["user_id"] == 1
        assert result["conflict"] is False

    def test_result_contains_all_keys(self, tmp_db, sample_image):
        tmp_db.add_user(1, "Alice")
        v = make_verifier(tmp_db, cnn_score=0.95, cnn_user=1)
        v.enroll(1, sample_image)

        result = v.verify(sample_image)
        for key in ("granted", "user_id", "cnn_score", "hog_score", "conflict"):
            assert key in result


class TestVerifyDenied:
    def test_cnn_fails_denies(self, tmp_db, sample_image):
        tmp_db.add_user(1, "Alice")
        v = make_verifier(tmp_db, cnn_score=0.50, cnn_user=1)
        v.enroll(1, sample_image)

        result = v.verify(sample_image)
        assert result["granted"] is False

    def test_hog_fails_denies(self, tmp_db, sample_image):
        """Enroll user 2, query as user 1 HOG → HOG won't match."""
        tmp_db.add_user(1, "Alice")
        tmp_db.add_user(2, "Bob")
        v = make_verifier(tmp_db, cnn_score=0.95, cnn_user=1)

        # Enroll a different image so HOG match for sample_image fails
        v.enroll(2, make_fingerprint_image(seed=99))

        result = v.verify(sample_image)
        assert result["granted"] is False

    def test_no_templates_denies(self, tmp_db, sample_image):
        v = make_verifier(tmp_db, cnn_score=0.95, cnn_user=1)
        result = v.verify(sample_image)
        assert result["granted"] is False


class TestConflict:
    def test_cnn_and_hog_disagree_flags_conflict(self, tmp_db, sample_image):
        """
        HOG matches user 1 (same image enrolled).
        CNN is patched to return user 2 with high confidence.
        → conflict should be True, granted should be False.
        """
        tmp_db.add_user(1, "Alice")
        tmp_db.add_user(2, "Bob")

        v = make_verifier(tmp_db, cnn_score=0.95, cnn_user=2)
        v.enroll(1, sample_image)  # HOG will match user 1

        result = v.verify(sample_image)
        assert result["conflict"] is True
        assert result["granted"] is False


class TestLogging:
    def test_verify_writes_to_access_log(self, tmp_db, sample_image):
        tmp_db.add_user(1, "Alice")
        v = make_verifier(tmp_db, cnn_score=0.95, cnn_user=1)
        v.enroll(1, sample_image)
        v.verify(sample_image)

        logs = tmp_db.get_logs()
        assert len(logs) == 1

    def test_log_records_both_scores(self, tmp_db, sample_image):
        tmp_db.add_user(1, "Alice")
        v = make_verifier(tmp_db, cnn_score=0.95, cnn_user=1)
        v.enroll(1, sample_image)
        v.verify(sample_image)

        log = tmp_db.get_logs()[0]
        assert log["cnn_score"] == pytest.approx(0.95, abs=0.01)
        assert log["hog_score"] > 0.0

    def test_conflict_logged(self, tmp_db, sample_image):
        tmp_db.add_user(1, "Alice")
        tmp_db.add_user(2, "Bob")
        v = make_verifier(tmp_db, cnn_score=0.95, cnn_user=2)
        v.enroll(1, sample_image)
        v.verify(sample_image)

        log = tmp_db.get_logs()[0]
        assert log["conflict"] == 1
