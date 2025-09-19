"""Fingerprint sensor interface module"""

import numpy as np
from pyfingerprint.pyfingerprint import PyFingerprint
from config import SENSOR_CONFIG
import logging

class FingerprintSensor:
    def __init__(self):
        """Initialize fingerprint sensor"""
        self.logger = logging.getLogger('FingerprintSensor')
        self.sensor = None
        self.connect()
    
    def connect(self):
        """Connect to fingerprint sensor"""
        try:
            self.sensor = PyFingerprint(
                SENSOR_CONFIG['PORT'],
                SENSOR_CONFIG['BAUDRATE'],
                0xFFFFFFFF,
                0x00000000
            )
            
            if not self.sensor.verifyPassword():
                raise ValueError('Fingerprint sensor password verification failed')
                
            self.logger.info(f"Sensor connected. Capacity: {self.sensor.getStorageCapacity()}")
            
        except Exception as e:
            self.logger.error(f"Failed to connect sensor: {e}")
            raise
    
    def capture_image(self):
        """Capture fingerprint image"""
        try:
            self.logger.debug("Waiting for finger...")
            
            # Wait for finger
            while not self.sensor.readImage():
                pass
            
            # Download image
            image_data = self.sensor.downloadImage()
            
            # Convert to numpy array
            image = np.array(image_data).reshape(
                SENSOR_CONFIG['IMAGE_HEIGHT'],
                SENSOR_CONFIG['IMAGE_WIDTH']
            )
            
            self.logger.debug("Image captured successfully")
            return image
            
        except Exception as e:
            self.logger.error(f"Failed to capture image: {e}")
            return None