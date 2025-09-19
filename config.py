GPIO_CONFIG = {
    'LED_PIN': 17,
    'BUZZER_PIN': 18,
    'BUTTON_PIN': 27,
    'RELAY_PINS': [22, 23, 24, 25],
    'LCD_I2C_ADDR': 0x27
}

SENSOR_CONFIG = {
    'PORT': '/dev/ttyUSB0',
    'BAUDRATE': 57600,
    'IMAGE_WIDTH': 256,
    'IMAGE_HEIGHT': 288
}

HOG_CONFIG = {
    'CELL_SIZE': (16, 16),
    'BLOCK_SIZE': (2, 2),
    'BINS': 9,
    'RESIZE_DIM': (128, 128)
}

MODEL_CONFIG = {
    'INPUT_DIM': 1764,
    'HIDDEN_LAYERS': [512, 256],
    'OUTPUT_DIM': 10,
    'THRESHOLD': 0.80
}

SYSTEM_CONFIG = {
    'VERIFICATION_INTERVAL': 30,
    'DATABASE_PATH': 'data/fingerprint.db',
    'LOG_PATH': 'logs/system.log',
    'WEB_PORT': 5000
}