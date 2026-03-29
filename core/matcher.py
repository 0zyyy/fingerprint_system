import numpy as np
import logging


class HOGMatcher:
    """Matches a HOG feature vector against stored templates using cosine similarity."""

    def __init__(self, threshold: float = 0.92):
        self.logger = logging.getLogger("HOGMatcher")
        self.threshold = threshold
        self._templates: list[tuple[int, np.ndarray]] = []

    def load_templates(self, templates: list[tuple[int, np.ndarray]]):
        """Load (user_id, vector) pairs from the database."""
        self._templates = templates
        self.logger.info(f"Loaded {len(templates)} template(s)")

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def match(self, query: np.ndarray) -> tuple[int | None, float]:
        """
        Compare query vector against all loaded templates.
        Returns (best_user_id, best_score). user_id is None if below threshold.
        """
        if not self._templates:
            self.logger.warning("No templates loaded — match will always fail")
            return None, 0.0

        best_user, best_score = None, 0.0

        for user_id, template in self._templates:
            score = self._cosine_similarity(query, template)
            if score > best_score:
                best_score = score
                best_user = user_id

        if best_score >= self.threshold:
            self.logger.debug(f"HOG match: user={best_user}, score={best_score:.4f}")
            return best_user, best_score

        self.logger.debug(f"HOG no match: best_score={best_score:.4f} < {self.threshold}")
        return None, best_score
