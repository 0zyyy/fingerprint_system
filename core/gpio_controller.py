import RPi.GPIO as GPIO
from config import GPIO_CONFIG
from time import sleep


class GPIOController:
    def __init__(self):
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)

        GPIO.setup(GPIO_CONFIG['SERVO_PIN'], GPIO.OUT)
        GPIO.setup(GPIO_CONFIG['LED_PIN'], GPIO.OUT)

        self.pwm = GPIO.PWM(GPIO_CONFIG['SERVO_PIN'],50)
        self.pwm.start(0)

    def led_on(self):
        GPIO.output(GPIO_CONFIG['LED_PIN'], GPIO.HIGH)

    def led_off(self):
        GPIO.output(GPIO_CONFIG['LED_PIN'], GPIO.LOW)

    def setAngle(self, angle):
        duty = angle / 18 + 2
        GPIO.output(GPIO_CONFIG["SERVO_PIN"], GPIO.HIGH)
        self.pwm.ChangeDutyCycle(duty)
        sleep(1)
        GPIO.output(GPIO_CONFIG['SERVO_PIN'],GPIO.LOW)
        self.pwm.ChangeDutyCycle(duty)

    def stop(self):
        self.pwm.stop()

    def cleanup(self):
        GPIO.cleanup()
