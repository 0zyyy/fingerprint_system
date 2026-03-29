import logging
import numpy as np

from core.preprocessing import ImagePreprocessor
from core.feature_extraction import HOGExtractor
from core.matcher import HOGMatcher
from models.onnx_inference import FingerprintONNX
from database.db_manager import DatabaseManager
from config import MODEL_CONFIG


class FingerprintVerifier:
    """
    Aggregates CNN (ONNX) and HOG cosine-similarity scores.

    Decision rule:
      GRANT  — both scores pass their thresholds AND both agree on the same user
      DENY   — either score fails its threshold
      CONFLICT — scores pass but disagree on user (logged separately)
    """

    CNN_THRESHOLD = MODEL_CONFIG.get('THRESHOLD', 0.80)
    HOG_THRESHOLD = MODEL_CONFIG.get('HOG_THRESHOLD', 0.92)

    def __init__(self, db: DatabaseManager):
        self.logger = logging.getLogger("FingerprintVerifier")
        self.db = db

        self.preprocessor = ImagePreprocessor()
        self.hog = HOGExtractor()
        self.matcher = HOGMatcher(threshold=self.HOG_THRESHOLD)
        self.cnn = FingerprintONNX(model_path=MODEL_CONFIG.get('MODEL_PATH', 'models/fingerprint_ann.onnx'))

        self._reload_templates()

    def _reload_templates(self):
        templates = self.db.get_templates()
        self.matcher.load_templates(templates)

    def extract_vector(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image and extract HOG feature vector."""
        processed = self.preprocessor.process(image)
        return self.hog.extract(processed)

    def enroll(self, user_id: int, image: np.ndarray) -> bool:
        """
        Enroll a fingerprint. Extracts HOG vector and stores it.
        Call multiple times with different samples for robustness.
        """
        vector = self.extract_vector(image)
        ok = self.db.add_template(user_id, vector)
        if ok:
            self._reload_templates()
        return ok

    def verify(self, image: np.ndarray) -> dict:
        """
        Run both ANN (ONNX) and HOG cosine-similarity verification.

        HOG vector is extracted once and shared between both paths —
        CNN now also receives the HOG vector (not the raw image).

        Returns a dict with:
          granted  (bool)
          user_id  (int | None)
          cnn_score (float)
          hog_score (float)
          conflict  (bool)
        """
        # Extract HOG vector once — used by both paths
        vector = self.extract_vector(image)

        # --- ANN path (via ONNX, input = HOG vector) ---
        cnn_score, cnn_user = self.cnn.predict(vector)

        # --- Cosine similarity path ---
        hog_user, hog_score = self.matcher.match(vector)

        # --- Aggregate ---
        cnn_pass = cnn_score >= self.CNN_THRESHOLD and cnn_user != -1
        hog_pass = hog_user is not None

        conflict = cnn_pass and hog_pass and (cnn_user != hog_user)
        granted = cnn_pass and hog_pass and not conflict

        # Resolve the reported user_id
        if granted:
            user_id = hog_user  # both agree, either works
        elif hog_pass:
            user_id = hog_user
        elif cnn_pass:
            user_id = cnn_user
        else:
            user_id = None

        result = {
            "granted": granted,
            "user_id": user_id,
            "cnn_score": cnn_score,
            "hog_score": hog_score,
            "conflict": conflict,
        }

        self.db.log_access(
            user_id=user_id,
            cnn_score=cnn_score,
            hog_score=hog_score,
            granted=granted,
            conflict=conflict,
        )

        if conflict:
            self.logger.warning(
                f"Score conflict — CNN says user {cnn_user} ({cnn_score:.3f}), "
                f"HOG says user {hog_user} ({hog_score:.3f})"
            )

        return result
