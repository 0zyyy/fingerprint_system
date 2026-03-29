"""Unit tests for DatabaseManager — users, templates, access logs."""
import numpy as np
import pytest


class TestUsers:
    def test_add_and_get_user(self, tmp_db):
        assert tmp_db.add_user(1, "Alice")
        user = tmp_db.get_user(1)
        assert user is not None
        assert user["name"] == "Alice"
        assert user["id"] == 1

    def test_get_nonexistent_user_returns_none(self, tmp_db):
        assert tmp_db.get_user(999) is None

    def test_add_user_replace(self, tmp_db):
        tmp_db.add_user(1, "Alice")
        tmp_db.add_user(1, "Alice Renamed")
        assert tmp_db.get_user(1)["name"] == "Alice Renamed"

    def test_list_users(self, tmp_db):
        tmp_db.add_user(1, "Alice")
        tmp_db.add_user(2, "Bob")
        users = tmp_db.list_users()
        assert len(users) == 2
        assert {u["name"] for u in users} == {"Alice", "Bob"}

    def test_list_users_empty(self, tmp_db):
        assert tmp_db.list_users() == []


class TestTemplates:
    def test_add_and_retrieve_template(self, tmp_db, sample_vector):
        tmp_db.add_user(1, "Alice")
        assert tmp_db.add_template(1, sample_vector)

        templates = tmp_db.get_templates()
        assert len(templates) == 1

        user_id, stored = templates[0]
        assert user_id == 1
        np.testing.assert_allclose(stored, sample_vector.astype(np.float32))

    def test_multiple_templates_same_user(self, tmp_db):
        tmp_db.add_user(1, "Alice")
        for seed in range(3):
            from tests.conftest import make_hog_vector
            tmp_db.add_template(1, make_hog_vector(seed=seed))

        templates = tmp_db.get_templates()
        assert len(templates) == 3
        assert all(uid == 1 for uid, _ in templates)

    def test_templates_multiple_users(self, tmp_db):
        from tests.conftest import make_hog_vector
        tmp_db.add_user(1, "Alice")
        tmp_db.add_user(2, "Bob")
        tmp_db.add_template(1, make_hog_vector(seed=1))
        tmp_db.add_template(2, make_hog_vector(seed=2))

        templates = tmp_db.get_templates()
        assert len(templates) == 2
        user_ids = {uid for uid, _ in templates}
        assert user_ids == {1, 2}

    def test_delete_templates(self, tmp_db, sample_vector):
        tmp_db.add_user(1, "Alice")
        tmp_db.add_template(1, sample_vector)
        assert len(tmp_db.get_templates()) == 1

        tmp_db.delete_templates(1)
        assert tmp_db.get_templates() == []

    def test_vector_roundtrip_precision(self, tmp_db):
        """float32 roundtrip should be lossless."""
        tmp_db.add_user(1, "Alice")
        vec = np.random.default_rng(0).random(1764).astype(np.float32)
        tmp_db.add_template(1, vec)

        _, stored = tmp_db.get_templates()[0]
        np.testing.assert_array_equal(stored, vec)


class TestAccessLogs:
    def test_log_granted(self, tmp_db):
        tmp_db.add_user(1, "Alice")
        tmp_db.log_access(user_id=1, cnn_score=0.95, hog_score=0.97, granted=True)

        logs = tmp_db.get_logs()
        assert len(logs) == 1
        assert logs[0]["granted"] == 1
        assert logs[0]["conflict"] == 0
        assert logs[0]["user_name"] == "Alice"

    def test_log_denied(self, tmp_db):
        tmp_db.log_access(user_id=None, cnn_score=0.4, hog_score=0.3, granted=False)
        logs = tmp_db.get_logs()
        assert logs[0]["granted"] == 0
        assert logs[0]["user_name"] is None

    def test_log_conflict(self, tmp_db):
        tmp_db.log_access(user_id=1, cnn_score=0.9, hog_score=0.93, granted=False, conflict=True)
        logs = tmp_db.get_logs()
        assert logs[0]["conflict"] == 1

    def test_get_logs_limit(self, tmp_db):
        for i in range(10):
            tmp_db.log_access(user_id=None, cnn_score=0.1, hog_score=0.1, granted=False)
        assert len(tmp_db.get_logs(limit=5)) == 5

    def test_logs_ordered_newest_first(self, tmp_db):
        tmp_db.add_user(1, "Alice")
        tmp_db.add_user(2, "Bob")
        tmp_db.log_access(user_id=1, cnn_score=0.9, hog_score=0.9, granted=True)
        tmp_db.log_access(user_id=2, cnn_score=0.9, hog_score=0.9, granted=True)

        logs = tmp_db.get_logs()
        assert logs[0]["user_name"] == "Bob"
        assert logs[1]["user_name"] == "Alice"
