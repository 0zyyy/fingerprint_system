import onnxruntime as ort
import numpy as np
import logging
import os


class FingerprintONNX:
    """
    ONNX inference wrapper for the HOG+ANN model.

    Input:  (1, 1764) float32 HOG feature vector
    Output: (class_id, confidence)

    The model exported from train.py outputs raw logits;
    softmax is applied here so confidence is a proper probability.
    """

    def __init__(self, model_path: str = "models/fingerprint_ann.onnx"):
        self.logger = logging.getLogger("FingerprintONNX")
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.load_model()

    def load_model(self):
        if not os.path.exists(self.model_path):
            self.logger.warning(f"Model not found at {self.model_path}. Inference will fail.")
            return
        try:
            self.session = ort.InferenceSession(self.model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.logger.info(f"ONNX model loaded from {self.model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load ONNX model: {e}")

    def predict_from_vector(self, hog_vector: np.ndarray) -> tuple[float, int]:
        """
        Run inference on a pre-extracted HOG feature vector.

        Args:
            hog_vector: float32 array of shape (1764,) or (1, 1764)

        Returns:
            (confidence, class_id)
        """
        if self.session is None:
            self.logger.error("Model not loaded")
            return 0.0, -1

        try:
            vec = hog_vector.astype(np.float32)
            if vec.ndim == 1:
                vec = vec.reshape(1, -1)

            logits = self.session.run(None, {self.input_name: vec})[0][0]

            # Softmax
            exp_logits = np.exp(logits - logits.max())
            probs = exp_logits / exp_logits.sum()

            class_id = int(np.argmax(probs))
            confidence = float(probs[class_id])
            return confidence, class_id

        except Exception as e:
            self.logger.error(f"Inference error: {e}")
            return 0.0, -1

    def predict(self, hog_vector: np.ndarray) -> tuple[float, int]:
        """Alias kept for interface compatibility."""
        return self.predict_from_vector(hog_vector)
