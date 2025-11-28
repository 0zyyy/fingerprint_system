GPIO_CONFIG = {
    'SERVO_PIN': 11,
    'LED_PIN': 13,
}

import platform

SENSOR_CONFIG = {
    'PORT': 'COM3' if platform.system() == 'Windows' else '/dev/ttyUSB0',
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
