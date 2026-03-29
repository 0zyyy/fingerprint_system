"""
Integration tests — full enroll → verify → audit pipeline.

These tests exercise the real preprocessing + HOG path end-to-end,
only mocking the CNN and GPIO hardware.
"""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from tests.conftest import make_fingerprint_image, make_hog_vector


def build_system(tmp_db, cnn_responses: list[tuple[float, int]]):
    """
    Build a FingerprintVerifier whose CNN returns cnn_responses in order.
    cnn_responses: [(score, user_id), ...]
    """
    call_iter = iter(cnn_responses)

    def side_effect(inputs, feed):
        score, user_id = next(call_iter, (0.0, -1))
        probs = np.zeros(10, dtype=np.float32)
        if 0 <= user_id < 10:
            probs[user_id] = score
        return [np.expand_dims(probs, 0)]

    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [MagicMock(name="input")]
    mock_session.run.side_effect = side_effect

    with patch("onnxruntime.InferenceSession", return_value=mock_session), \
         patch("os.path.exists", return_value=True):
        from core.verifier import FingerprintVerifier
        return FingerprintVerifier(db=tmp_db)


class TestEnrollThenVerify:
    def test_enrolled_user_is_granted(self, tmp_db):
        img = make_fingerprint_image(seed=1)
        tmp_db.add_user(1, "Alice")

        verifier = build_system(tmp_db, cnn_responses=[(0.95, 1), (0.95, 1)])
        verifier.enroll(1, img)

        result = verifier.verify(img)
        assert result["granted"] is True
        assert result["user_id"] == 1

    def test_unenrolled_user_is_denied(self, tmp_db):
        alice_img = make_fingerprint_image(seed=1)
        intruder_img = make_fingerprint_image(seed=99)

        tmp_db.add_user(1, "Alice")
        verifier = build_system(tmp_db, cnn_responses=[(0.5, -1)])
        verifier.enroll(1, alice_img)

        result = verifier.verify(intruder_img)
        assert result["granted"] is False

    def test_multiple_users_correct_routing(self, tmp_db):
        alice_img = make_fingerprint_image(seed=10)
        bob_img = make_fingerprint_image(seed=20)

        tmp_db.add_user(1, "Alice")
        tmp_db.add_user(2, "Bob")

        verifier = build_system(tmp_db, cnn_responses=[
            (0.95, 1),  # enroll alice (not used for verify)
            (0.95, 2),  # enroll bob
            (0.95, 1),  # verify alice
            (0.95, 2),  # verify bob
        ])
        verifier.enroll(1, alice_img)
        verifier.enroll(2, bob_img)

        r_alice = verifier.verify(alice_img)
        r_bob = verifier.verify(bob_img)

        assert r_alice["user_id"] == 1
        assert r_bob["user_id"] == 2


class TestMultiSampleEnrollment:
    def test_three_samples_per_user(self, tmp_db):
        """Enrolling multiple samples increases template count."""
        tmp_db.add_user(1, "Alice")
        verifier = build_system(tmp_db, cnn_responses=[(0.95, 1)] * 10)

        for seed in range(3):
            verifier.enroll(1, make_fingerprint_image(seed=seed))

        assert len(tmp_db.get_templates()) == 3
        assert len(verifier.matcher._templates) == 3


class TestAuditTrail:
    def test_every_verify_creates_log_entry(self, tmp_db):
        img = make_fingerprint_image(seed=1)
        tmp_db.add_user(1, "Alice")

        verifier = build_system(tmp_db, cnn_responses=[(0.95, 1)] * 5)
        verifier.enroll(1, img)

        for _ in range(3):
            verifier.verify(img)

        assert len(tmp_db.get_logs()) == 3

    def test_denied_attempts_logged(self, tmp_db):
        intruder = make_fingerprint_image(seed=99)
        verifier = build_system(tmp_db, cnn_responses=[(0.1, -1)] * 3)

        for _ in range(3):
            verifier.verify(intruder)

        logs = tmp_db.get_logs()
        assert all(log["granted"] == 0 for log in logs)

    def test_conflict_events_logged(self, tmp_db):
        img = make_fingerprint_image(seed=1)
        tmp_db.add_user(1, "Alice")
        tmp_db.add_user(2, "Bob")

        # HOG will match user 1; CNN says user 2 → conflict
        verifier = build_system(tmp_db, cnn_responses=[(0.95, 2)])
        verifier.enroll(1, img)
        verifier.verify(img)

        logs = tmp_db.get_logs()
        assert logs[0]["conflict"] == 1
        assert logs[0]["granted"] == 0

    def test_log_contains_user_name(self, tmp_db):
        img = make_fingerprint_image(seed=1)
        tmp_db.add_user(1, "Alice")

        verifier = build_system(tmp_db, cnn_responses=[(0.95, 1)])
        verifier.enroll(1, img)
        verifier.verify(img)

        log = tmp_db.get_logs()[0]
        assert log["user_name"] == "Alice"


class TestScoreThresholds:
    @pytest.mark.parametrize("cnn_score,expected_granted", [
        (0.85, True),
        (0.80, True),   # exactly at threshold
        (0.79, False),  # just below
        (0.50, False),
    ])
    def test_cnn_threshold_boundary(self, tmp_db, cnn_score, expected_granted):
        img = make_fingerprint_image(seed=1)
        tmp_db.add_user(1, "Alice")

        verifier = build_system(tmp_db, cnn_responses=[(cnn_score, 1)])
        verifier.enroll(1, img)

        result = verifier.verify(img)
        assert result["granted"] is expected_granted
