"""
Logging utilities for the Fingerprint Recognition System
"""

import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path

# Create logs directory if it doesn't exist
LOG_DIR = Path('logs')
LOG_DIR.mkdir(exist_ok=True)

# Define log format
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DETAILED_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'

def setup_logger(name='FingerprintSystem', level=logging.INFO):
    """
    Set up logger for the fingerprint system
    
    Args:
        name: Logger name
        level: Logging level
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(LOG_FORMAT)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler - general logs
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / 'system.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(DETAILED_FORMAT)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Error file handler
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
    """Get logger for a specific module"""
    return logging.getLogger(f'FingerprintSystem.{module_name}')

# Access logger for security events
def setup_access_logger():
    """Set up access logger for authentication events"""
    logger = logging.getLogger('AccessLog')
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        return logger
    
    # Daily rotating file for access logs
    handler = logging.handlers.TimedRotatingFileHandler(
        LOG_DIR / 'access.log',
        when='midnight',
        interval=1,
        backupCount=30  # Keep 30 days
    )
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

def log_access_attempt(user_id, success, confidence, message=""):
    """
    Log fingerprint access attempt
    
    Args:
        user_id: User identifier
        success: Boolean indicating success
        confidence: Confidence score
        message: Additional message
    """
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
    """Log training metrics"""
    logger = get_logger('Training')
    
    message = f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}"
    if val_loss is not None:
        message += f", Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
    
    logger.info(message)

def log_performance(operation, duration, details=None):
    """Log performance metrics"""
    logger = get_logger('Performance')
    
    message = f"{operation} completed in {duration:.3f}s"
    if details:
        message += f" - {details}"
    
    logger.debug(message)

# Initialize main logger when module is imported
main_logger = setup_logger()