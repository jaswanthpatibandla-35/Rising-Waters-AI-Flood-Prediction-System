import os
import logging

# Set up logs root folder
LOG_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR, 'app.log')

# Setup formatter
log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')

# Stream Handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

# File Handler
file_handler = logging.FileHandler(LOG_FILE_PATH, encoding='utf-8')
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

# Configure Root Logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
if root_logger.hasHandlers():
    root_logger.handlers.clear()
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

def get_logger(name):
    """Retrieve named logger."""
    return logging.getLogger(name)

# Database Logging Handler Hook
class DBLoggingHandler(logging.Handler):
    """
    Saves LogRecords in system_logs table via SQLAlchemy.
    Added to root logger in app.py after DB initialization.
    """
    def __init__(self, db_session, log_model):
        super().__init__()
        self.db_session = db_session
        self.log_model = log_model

    def emit(self, record):
        try:
            db_log = self.log_model(
                log_level=record.levelname,
                logger_name=record.name,
                message=record.getMessage()
            )
            self.db_session.add(db_log)
            self.db_session.commit()
        except Exception:
            self.handleError(record)
