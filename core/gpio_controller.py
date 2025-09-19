#import RPi.GPIO as GPIO
import time
from smbus2 import SMBus
from config import GPIO_CONFIG

class GPIOController:
    def __init__(self):
        """Initialize GPIO pins"""
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        GPIO.setup(GPIO_CONFIG['LED_PIN'], GPIO.OUT)
        GPIO.setup(GPIO_CONFIG['BUZZER_PIN'], GPIO.OUT)
        GPIO.setup(GPIO_CONFIG['BUTTON_PIN'], GPIO.IN, pull_up_down=GPIO.PUD_UP)
        
        for pin in GPIO_CONFIG['RELAY_PINS']:
           GPIO.setup(pin, GPIO.OUT)
           GPIO.output(pin, GPIO.LOW)
        
        self.bus = SMBus(1)
        self.lcd_addr = GPIO_CONFIG['LCD_I2C_ADDR']
        pass
        
    def led_on(self):
        GPIO.output(GPIO_CONFIG['LED_PIN'], GPIO.HIGH)
    
    def led_off(self):
        GPIO.output(GPIO_CONFIG['LED_PIN'], GPIO.LOW)
    
    def buzzer_on(self):
        GPIO.output(GPIO_CONFIG['BUZZER_PIN'], GPIO.HIGH)
    
    def buzzer_off(self):
        GPIO.output(GPIO_CONFIG['BUZZER_PIN'], GPIO.LOW)
    
    def buzzer_pattern(self, pattern):
        """Play buzzer pattern"""
        for duration in pattern:
            self.buzzer_on()
            time.sleep(duration)
            self.buzzer_off()
            time.sleep(0.1)
    
    def is_button_pressed(self):
        return GPIO.input(GPIO_CONFIG['BUTTON_PIN']) == GPIO.LOW
    
    def relay_on(self, index):
        if 0 <= index < len(GPIO_CONFIG['RELAY_PINS']):
            GPIO.output(GPIO_CONFIG['RELAY_PINS'][index], GPIO.HIGH)
    
    def relay_off(self, index):
        if 0 <= index < len(GPIO_CONFIG['RELAY_PINS']):
            GPIO.output(GPIO_CONFIG['RELAY_PINS'][index], GPIO.LOW)
    
    def all_relays_off(self):
        for pin in GPIO_CONFIG['RELAY_PINS']:
            GPIO.output(pin, GPIO.LOW)
    
    def display_message(self, line1, line2=""):
        """Display message on LCD (simplified)"""
        # This would need proper I2C LCD library implementation
        print(f"LCD: {line1} | {line2}")
    
    def cleanup(self):
        """Clean up GPIO"""
        GPIO.cleanup()
