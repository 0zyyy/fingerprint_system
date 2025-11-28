#!/usr/bin/env python3
import signal
import sys
import time

from config import SYSTEM_CONFIG, MODEL_CONFIG
from core.fingerprint_sensor import FingerprintSensor
from core.gpio_controller import GPIOController
from models.tf_inference import FingerprintClassifier
from utils.logger import setup_logger


class FingerprintSystem:
    def __init__(self):
        self.logger = setup_logger("FingerprintSystem")
        self.logger.info("Initializing Fingerprint System...")

        self.sensor = FingerprintSensor()
        self.gpio = GPIOController()
        self.model = FingerprintClassifier()

        self.running = False

        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

    def start(self):
        self.running = True
        self.logger.info("System started")
        self.main_loop()

    def main_loop(self):
        self.logger.info("Ready for verification...")
        while self.running:
            # 1. Capture Image
            image = self.sensor.capture_image()
            
            if image is not None:
                # 2. Predict using TensorFlow
                confidence, user_id = self.model.predict(image)
                
                self.logger.info(f"Predicted User: {user_id}, Confidence: {confidence:.2f}")
                
                # 3. Verify
                if confidence > MODEL_CONFIG.get('THRESHOLD', 0.8):
                    self.grant_access()
                else:
                    self.deny_access()
            
            # Small delay
            time.sleep(0.1)

    def grant_access(self):
        self.logger.info("Access GRANTED")
        
        # Visual indicator
        self.gpio.led_on()
        
        # Open door (Servo to 90 degrees)
        self.logger.info("Opening door...")
        self.gpio.setAngle(90)
        
        # Keep open for 5 seconds
        time.sleep(5)
        
        # Close door (Servo to 0 degrees)
        self.logger.info("Closing door...")
        self.gpio.setAngle(0)
        
        # Turn off indicator
        self.gpio.led_off()

    def deny_access(self):
        self.logger.warning("Access DENIED")
        
        # Blink LED 3 times
        for _ in range(3):
            self.gpio.led_on()
            time.sleep(0.2)
            self.gpio.led_off()
            time.sleep(0.2)

    def shutdown(self, signum, frame):
        self.logger.info("Shutting down system...")
        self.running = False
        self.gpio.cleanup()
        sys.exit(0)


if __name__ == "__main__":
    system = FingerprintSystem()
    system.start()
