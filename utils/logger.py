import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path

LOG_DIR = Path('logs')
LOG_DIR.mkdir(exist_ok=True)

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DETAILED_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'

def setup_logger(name='FingerprintSystem', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.handlers:
        return logger
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(LOG_FORMAT)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / 'system.log',
        maxBytes=10*1024*1024,
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(DETAILED_FORMAT)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    error_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / 'errors.log',
        maxBytes=10*1024*1024,
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    logger.addHandler(error_handler)
    
    return logger

def get_logger(module_name):
    return logging.getLogger(f'FingerprintSystem.{module_name}')

def setup_access_logger():
    logger = logging.getLogger('AccessLog')
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        return logger
    
    handler = logging.handlers.TimedRotatingFileHandler(
        LOG_DIR / 'access.log',
        when='midnight',
        interval=1,
        backupCount=30
    )
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

def log_access_attempt(user_id, success, confidence, message=""):
    access_logger = setup_access_logger()
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    status = "SUCCESS" if success else "FAILED"
    
    log_message = f"[{status}] User: {user_id}, Confidence: {confidence:.2f}, Time: {timestamp}"
    if message:
        log_message += f", Note: {message}"
    
    if success:
        access_logger.info(log_message)
    else:
        access_logger.warning(log_message)

def log_training_metrics(epoch, train_loss, train_acc, val_loss=None, val_acc=None):
    logger = get_logger('Training')
    
    message = f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}"
    if val_loss is not None:
        message += f", Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
    
    logger.info(message)

def log_performance(operation, duration, details=None):
    logger = get_logger('Performance')
    
    message = f"{operation} completed in {duration:.3f}s"
    if details:
        message += f" - {details}"
    
    logger.debug(message)

main_logger = setup_logger()