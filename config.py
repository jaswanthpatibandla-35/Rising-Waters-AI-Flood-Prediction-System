import os

class Config:
    # Flask Settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'RisingWatersEnterpriseSecretKey_98765_!@#$%')
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    PORT = int(os.environ.get('PORT', 5000))
    HOST = os.environ.get('HOST', '0.0.0.0')

    # Security settings
    SESSION_COOKIE_SECURE = False  # Set to True in production with HTTPS
    SESSION_COOKIE_HTTPONLY = True
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload limit
    WTF_CSRF_ENABLED = True

    # Database settings
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL', 
        f"sqlite:///{os.path.join(BASE_DIR, 'database', 'flood.db')}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    DATABASE_PATH = os.path.join(BASE_DIR, 'database', 'flood.db')
    BACKUP_DIR = os.path.join(BASE_DIR, 'database', 'backups')

    # Uploads, Reports, and Logs root-level directories
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    REPORT_FOLDER = os.path.join(BASE_DIR, 'reports')
    LOG_FOLDER = os.path.join(BASE_DIR, 'logs')
    LOG_FILE_PATH = os.path.join(LOG_FOLDER, 'app.log')

    # Model settings
    MODEL_DIR = os.path.join(BASE_DIR, 'model')
    MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')
    SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
    LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')
    FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, 'feature_names.pkl')
    MODEL_METRICS_PATH = os.path.join(MODEL_DIR, 'metrics.json')
    MODEL_VERSION = "2.1.0"

    # Dataset settings
    DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
    RAW_DATA_PATH = os.path.join(DATASET_DIR, 'raw', 'flood_raw.csv')
    PROCESSED_DATA_PATH = os.path.join(DATASET_DIR, 'processed', 'flood_processed.csv')

    # Visualizations path
    STATIC_IMAGES_DIR = os.path.join(BASE_DIR, 'static', 'images')

    # API Keys
    OPENWEATHER_API_KEY = os.environ.get('OPENWEATHER_API_KEY', '')

    # Environmental Features
    FEATURES = [
        'Annual_Rainfall',
        'Monthly_Rainfall',
        'Temperature',
        'Humidity',
        'Pressure',
        'Cloud_Cover',
        'Wind_Speed',
        'River_Water_Level',
        'Ground_Water_Level',
        'Visibility',
        'Latitude',
        'Longitude',
        'Season',
        'Month'
    ]

    # Categorical Mappings
    SEASON_MAPPING = {
        'Summer': 0,
        'Monsoon': 1,
        'Winter': 2,
        'Spring': 3
    }
    
    MONTH_MAPPING = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
