import onnxruntime as ort
import numpy as np
import cv2
import logging
import os

class FingerprintONNX:
    def __init__(self, model_path="models/fingerprint_cnn.onnx"):
        self.logger = logging.getLogger("FingerprintONNX")
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.load_model()

    def load_model(self):
        try:
            if not os.path.exists(self.model_path):
                self.logger.warning(f"Model not found at {self.model_path}. Inference will fail.")
                return

            self.session = ort.InferenceSession(self.model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.logger.info(f"ONNX model loaded from {self.model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load ONNX model: {e}")

    def preprocess(self, image):
        # Resize to 128x128
        img = cv2.resize(image, (128, 128))
        
        # Convert to RGB (duplicate channels)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Normalize to [-1, 1] (MobileNetV2 standard)
        img = (img.astype('float32') / 127.5) - 1.0
        
        # Add batch dimension (1, H, W, 3)
        img = np.expand_dims(img, axis=0)
        
        return img

    def predict(self, image):
        if self.session is None:
            self.logger.error("Model not loaded")
            return 0.0, -1

        try:
            processed_img = self.preprocess(image)
            
            # Run inference
            outputs = self.session.run(None, {self.input_name: processed_img})
            probabilities = outputs[0][0]
            
            class_id = np.argmax(probabilities)
            confidence = float(probabilities[class_id])
            
            return confidence, class_id
            
        except Exception as e:
            self.logger.error(f"Inference error: {e}")
            return 0.0, -1
