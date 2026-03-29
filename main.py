#!/usr/bin/env python3
import signal
import sys
import time
import adafruit_fingerprint

from config import SYSTEM_CONFIG
from core.fingerprint_sensor import FingerprintSensor
from core.gpio_controller import GPIOController
from core.verifier import FingerprintVerifier
from database.db_manager import DatabaseManager
from utils.logger import setup_logger


class FingerprintSystem:
    def __init__(self):
        self.logger = setup_logger("FingerprintSystem")
        self.logger.info("Initializing Fingerprint System...")

        self.db = DatabaseManager()
        self.sensor = FingerprintSensor()
        self.gpio = GPIOController()
        self.verifier = FingerprintVerifier(db=self.db)

        self.running = False

        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

    def start(self):
        self.running = True
        self.logger.info("System started. Waiting for finger...")
        self.main_loop()

    def main_loop(self):
        while self.running:
            # Blocks here until a finger is placed (handled inside capture_image)
            image = self.sensor.capture_image()

            if image is None:
                continue

            result = self.verifier.verify(image)

            if result["granted"]:
                self.grant_access(result["user_id"])
            else:
                self.deny_access(conflict=result["conflict"])

            # Wait for finger to lift before next scan
            self._wait_for_finger_lift()

    def _wait_for_finger_lift(self):
        """Block until the sensor reports no finger present."""
        while self.running:
            if self.sensor.sensor.get_image() == adafruit_fingerprint.NOFINGER:
                break
            time.sleep(0.05)

    def grant_access(self, user_id):
        user = self.db.get_user(user_id)
        name = user["name"] if user else f"User {user_id}"
        self.logger.info(f"Access GRANTED — {name}")

        self.gpio.led_on()
        self.gpio.setAngle(90)
        time.sleep(5)
        self.gpio.setAngle(0)
        self.gpio.led_off()

    def deny_access(self, conflict: bool = False):
        if conflict:
            self.logger.warning("Access DENIED — score conflict (possible spoof attempt)")
        else:
            self.logger.warning("Access DENIED")

        for _ in range(3):
            self.gpio.led_on()
            time.sleep(0.2)
            self.gpio.led_off()
            time.sleep(0.2)

    def shutdown(self, signum, frame):
        self.logger.info("Shutting down...")
        self.running = False
        self.gpio.setAngle(0)   # ensure door is closed
        self.gpio.led_off()
        self.gpio.cleanup()
        sys.exit(0)


if __name__ == "__main__":
    system = FingerprintSystem()
    system.start()
