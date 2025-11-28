import numpy as np
import tensorflow as tf
import cv2
import logging
import os

class FingerprintClassifier:
    def __init__(self, model_path="models/fingerprint_cnn.h5"):
        self.logger = logging.getLogger("FingerprintClassifier")
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            if not os.path.exists(self.model_path):
                self.logger.warning(f"Model not found at {self.model_path}. Inference will fail.")
                return

            self.model = tf.keras.models.load_model(self.model_path)
            self.logger.info(f"Keras model loaded from {self.model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load Keras model: {e}")

    def preprocess(self, image):
        # Resize to 128x128
        img = cv2.resize(image, (128, 128))
        
        # Convert to RGB if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Normalize to [-1, 1] (MobileNetV2 standard)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        
        # Add batch dimension (1, H, W, 3)
        img = np.expand_dims(img, axis=0)
        
        return img

    def predict(self, image):
        if self.model is None:
            self.logger.error("Model not loaded")
            return 0.0, -1

        try:
            processed_img = self.preprocess(image)
            
            # Run inference
            predictions = self.model.predict(processed_img, verbose=0)
            probabilities = predictions[0]
            
            class_id = np.argmax(probabilities)
            confidence = float(probabilities[class_id])
            
            return confidence, class_id
            
        except Exception as e:
            self.logger.error(f"Inference error: {e}")
            return 0.0, -1
