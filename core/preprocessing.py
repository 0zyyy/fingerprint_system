"""Image preprocessing module"""

import cv2
import numpy as np
from config import HOG_CONFIG

class ImagePreprocessor:
    def __init__(self):
        """Initialize preprocessor"""
        pass
    
    def process(self, image):
        """Apply preprocessing pipeline"""
        # Gaussian blur
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Normalize
        normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)
        
        # CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(normalized.astype(np.uint8))
        
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        # Median filter
        denoised = cv2.medianBlur(binary, 3)
        
        return denoised

if __name__ == "__main__":
    pre = ImagePreprocessor()