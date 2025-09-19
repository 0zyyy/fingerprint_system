#!/usr/bin/env python3
import time
import threading
import signal
import sys
from core.fingerprint_sensor import FingerprintSensor
from core.gpio_controller import GPIOController
from core.preprocessing import ImagePreprocessor
from core.feature_extraction import HOGExtractor
from models.ann_model import FingerprintClassifier
from database.db_manager import DatabaseManager
from utils.logger import setup_logger
from config import SYSTEM_CONFIG, MODEL_CONFIG

class FingerprintSystem:
    def __init__(self):
        """Initialize the fingerprint recognition system"""
        self.logger = setup_logger('FingerprintSystem')
        self.logger.info("Initializing Fingerprint System...")
        
        # Initialize components
        self.sensor = FingerprintSensor()
        self.gpio = GPIOController()
        self.preprocessor = ImagePreprocessor()
        self.feature_extractor = HOGExtractor()
        self.classifier = FingerprintClassifier()
        self.db = DatabaseManager()
        
        # System state
        self.running = False
        self.machine_running = False
        self.verification_timer = None
        
        # # Load trained model
        # self.classifier.load_model('models/trained_model.pkl')
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        
    def start(self):
        """Start the fingerprint system"""
        self.running = True
        self.logger.info("System started")
            
        # Main loop
        self.main_loop()
        
    def main_loop(self):
        """Main operational loop"""
        while self.running:
            # Check button press
            if self.gpio.is_button_pressed():
                self.start_machine()
            
            # Check if verification needed
            if self.machine_running and self.needs_verification():
                self.request_verification()
            
            time.sleep(0.1)
    
    def start_machine(self):
        """Start machine operation"""
        self.logger.info("Starting machine")
        self.machine_running = True
        self.gpio.led_on()
        self.gpio.relay_on(0)  # Start main relay
        self.gpio.display_message("Machine Running", "Ready")
        self.schedule_verification()
        
    def schedule_verification(self):
        """Schedule next verification"""
        self.verification_timer = threading.Timer(
            SYSTEM_CONFIG['VERIFICATION_INTERVAL'],
            self.request_verification
        )
        self.verification_timer.start()
        
    def request_verification(self):
        """Request fingerprint verification"""
        self.logger.info("Requesting verification")
        self.gpio.buzzer_pattern([0.5, 0.5, 0.5])  # Beep pattern
        self.gpio.display_message("Verification", "Place finger")
        
        # Wait for fingerprint
        success = self.capture_and_verify()
        
        if success:
            self.gpio.buzzer_off()
            self.gpio.display_message("Verified", "Access granted")
            self.schedule_verification()
        else:
            self.stop_machine()
            self.gpio.display_message("Failed", "Access denied")
    
    def capture_and_verify(self, timeout=10):
        """Capture and verify fingerprint"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Try to capture fingerprint
            image = self.sensor.capture_image()
            
            if image is not None:
                # Process image
                processed = self.preprocessor.process(image)
                features = self.feature_extractor.extract(processed)
                
                # Classify
                confidence, user_id = self.classifier.predict(features)
                
                # Log attempt
                self.db.log_access(
                    user_id=user_id,
                    status='success' if confidence > MODEL_CONFIG['THRESHOLD'] else 'failed',
                    confidence=confidence
                )
                
                if confidence > MODEL_CONFIG['THRESHOLD']:
                    self.logger.info(f"User {user_id} verified with confidence {confidence:.2f}")
                    return True
                else:
                    self.logger.warning(f"Verification failed. Confidence: {confidence:.2f}")
                    return False
            
            time.sleep(0.5)
        
        self.logger.warning("Verification timeout")
        return False
    
    def stop_machine(self):
        """Stop machine operation"""
        self.logger.info("Stopping machine")
        self.machine_running = False
        self.gpio.led_off()
        self.gpio.all_relays_off()
        self.gpio.display_message("Machine Stopped", "")
        
        if self.verification_timer:
            self.verification_timer.cancel()
    
    def needs_verification(self):
        """Check if verification is needed"""
        # This would be called by timer in real implementation
        return False
        
    def shutdown(self, signum, frame):
        """Graceful shutdown"""
        self.logger.info("Shutting down system...")
        self.running = False
        self.stop_machine()
        self.gpio.cleanup()
        self.db.close()
        sys.exit(0)

if __name__ == "__main__":
    system = FingerprintSystem()
    system.start()