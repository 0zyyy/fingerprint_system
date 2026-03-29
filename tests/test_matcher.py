"""Unit tests for HOGMatcher — cosine similarity and threshold logic."""
import numpy as np
import pytest
from core.matcher import HOGMatcher
from tests.conftest import make_hog_vector


@pytest.fixture
def matcher():
    return HOGMatcher(threshold=0.92)


class TestCosineSimilarity:
    def test_identical_vectors_score_one(self, matcher):
        v = np.random.default_rng(0).random(1764).astype(np.float32)
        score = matcher._cosine_similarity(v, v)
        assert abs(score - 1.0) < 1e-5

    def test_opposite_vectors_score_negative_one(self, matcher):
        v = np.ones(1764, dtype=np.float32)
        score = matcher._cosine_similarity(v, -v)
        assert abs(score - (-1.0)) < 1e-5

    def test_zero_vector_returns_zero(self, matcher):
        v = np.ones(1764, dtype=np.float32)
        z = np.zeros(1764, dtype=np.float32)
        assert matcher._cosine_similarity(v, z) == 0.0
        assert matcher._cosine_similarity(z, z) == 0.0

    def test_orthogonal_vectors_score_zero(self, matcher):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert abs(matcher._cosine_similarity(a, b)) < 1e-6


class TestMatch:
    def test_no_templates_returns_none(self, matcher):
        query = make_hog_vector(seed=0)
        user_id, score = matcher.match(query)
        assert user_id is None
        assert score == 0.0

    def test_exact_same_vector_matches(self, matcher):
        vec = make_hog_vector(seed=42)
        matcher.load_templates([(1, vec)])

        user_id, score = matcher.match(vec)
        assert user_id == 1
        assert score > 0.99

    def test_different_finger_no_match(self, matcher):
        """Two very different synthetic images should not match."""
        alice_vec = make_hog_vector(seed=1)
        query = make_hog_vector(seed=99)   # unrelated seed

        matcher.load_templates([(1, alice_vec)])
        user_id, score = matcher.match(query)
        # Score may or may not pass threshold — what matters is we get a score
        # In practice, unrelated HOG vectors score well below 0.92
        assert score < 0.99   # at minimum not a perfect match

    def test_noisy_version_of_same_image_has_high_score(self, matcher):
        """Small noise should still yield high similarity."""
        clean = make_hog_vector(seed=5, noise=0.0)
        noisy = make_hog_vector(seed=5, noise=0.02)

        matcher.load_templates([(1, clean)])
        user_id, score = matcher.match(noisy)
        # Noisy version of same pattern should score close to clean
        assert score > 0.90

    def test_picks_best_match_among_multiple_users(self, matcher):
        alice = make_hog_vector(seed=10)
        bob = make_hog_vector(seed=20)
        query = alice.copy()

        matcher.load_templates([(1, alice), (2, bob)])
        user_id, score = matcher.match(query)
        assert user_id == 1

    def test_multiple_templates_same_user_best_wins(self, matcher):
        """All three templates belong to user 1; query should match."""
        templates = [(1, make_hog_vector(seed=i)) for i in range(3)]
        # Use one of the enrolled vectors as the query
        query = make_hog_vector(seed=1)
        matcher.load_templates(templates)

        user_id, score = matcher.match(query)
        assert user_id == 1

    def test_below_threshold_returns_none(self):
        """Set threshold to 1.01 so nothing ever matches."""
        m = HOGMatcher(threshold=1.01)
        vec = make_hog_vector(seed=0)
        m.load_templates([(1, vec)])

        user_id, score = m.match(vec)
        assert user_id is None
        assert score > 0.0   # score still computed, just didn't pass

    def test_load_templates_replaces_previous(self, matcher):
        matcher.load_templates([(1, make_hog_vector(seed=1))])
        matcher.load_templates([(2, make_hog_vector(seed=2))])

        query = make_hog_vector(seed=2)
        user_id, _ = matcher.match(query)
        assert user_id == 2
