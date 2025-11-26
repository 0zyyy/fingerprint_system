# TODO: Fix import, malaz
import numpy as np
import serial
import adafruit_fingerprint
from config import SENSOR_CONFIG
import logging

class FingerprintSensor:
    def __init__(self):
        self.logger = logging.getLogger('FingerprintSensor')
        self.sensor = None
        self.connect()

    def connect(self):
        try:
            uart = serial.Serial(SENSOR_CONFIG["PORT"],baudrate=SENSOR_CONFIG['BAUDRATE'],timeout=-1)
            self.sensor = adafruit_fingerprint.Adafruit_Fingerprint(uart)

            if not self.sensor.verify_password():
                raise ValueError('Fingerprint sensor password verification failed')

            self.logger.info(f"Sensor connected. Capacity: {self.sensor.getStorageCapacity()}")

        except Exception as e:
            self.logger.error(f"Failed to connect sensor: {e}")
            raise

    def capture_image(self):
        try:
            self.logger.debug("Waiting for finger...")

            while self.sensor.get_image() != adafruit_fingerprint.OK:
                pass

            image_data = self.sensor.get_fpdata("image", 2)

            image_array = np.zeros(73728, np.uint8)

            for i, val in enumerate(image_data):
                image_array[(i * 2)] = val & 240
                image_array[(i * 2) + 1] = (val & 15) * 16

            image = np.array(image_array).reshape(
                SENSOR_CONFIG['IMAGE_HEIGHT'],
                SENSOR_CONFIG['IMAGE_WIDTH']
            )

            self.logger.debug("Image captured successfully")
            return image

        except Exception as e:
            self.logger.error(f"Failed to capture image: {e}")
            return None
