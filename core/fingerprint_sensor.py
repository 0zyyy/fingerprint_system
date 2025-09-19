import numpy as np
from pyfingerprint.pyfingerprint import PyFingerprint
from config import SENSOR_CONFIG
import logging

class FingerprintSensor:
    def __init__(self):
        self.logger = logging.getLogger('FingerprintSensor')
        self.sensor = None
        self.connect()
    
    def connect(self):
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
        try:
            self.logger.debug("Waiting for finger...")
            
            while not self.sensor.readImage():
                pass
            
            image_data = self.sensor.downloadImage()
            
            image = np.array(image_data).reshape(
                SENSOR_CONFIG['IMAGE_HEIGHT'],
                SENSOR_CONFIG['IMAGE_WIDTH']
            )
            
            self.logger.debug("Image captured successfully")
            return image
            
        except Exception as e:
            self.logger.error(f"Failed to capture image: {e}")
            return None