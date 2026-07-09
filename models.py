import bcrypt
from datetime import datetime
from flask_login import UserMixin
from database import db

# -------------------------------------------------------------
# 1. USER MODEL
# -------------------------------------------------------------
class User(db.Model, UserMixin):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    is_verified = db.Column(db.Boolean, default=False)
    reset_token = db.Column(db.String(100), nullable=True)
    theme_preference = db.Column(db.String(10), default='light')
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    predictions = db.relationship('Prediction', backref='user', lazy=True, cascade="all, delete-orphan")
    reports = db.relationship('Report', backref='user', lazy=True, cascade="all, delete-orphan")

    def set_password(self, password):
        """Hash password before saving."""
        salt = bcrypt.gensalt()
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

    def check_password(self, password):
        """Check entered password against hashed password."""
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))

# -------------------------------------------------------------
# 2. PREDICTION RECORD MODEL
# -------------------------------------------------------------
class Prediction(db.Model):
    __tablename__ = 'predictions'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=True)
    date = db.Column(db.String(30), default=lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    user_name = db.Column(db.String(100), nullable=False)
    location = db.Column(db.String(100), nullable=False)
    latitude = db.Column(db.Float, default=0.0)
    longitude = db.Column(db.Float, default=0.0)
    
    # Inputs
    annual_rainfall = db.Column(db.Float, nullable=False)
    monthly_rainfall = db.Column(db.Float, nullable=False)
    temperature = db.Column(db.Float, nullable=False)
    humidity = db.Column(db.Float, nullable=False)
    pressure = db.Column(db.Float, nullable=False)
    cloud_cover = db.Column(db.Float, nullable=False)
    wind_speed = db.Column(db.Float, nullable=False)
    river_water_level = db.Column(db.Float, nullable=False)
    ground_water_level = db.Column(db.Float, nullable=False)
    visibility = db.Column(db.Float, nullable=False)
    season = db.Column(db.String(20), nullable=False)
    month = db.Column(db.String(20), nullable=False)
    district = db.Column(db.String(50), default='Unknown')
    state = db.Column(db.String(50), default='Unknown')
    
    # Outputs
    prediction = db.Column(db.Integer, nullable=False)  # 0 or 1
    probability = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(10), nullable=False)  # Low, Medium, High
    confidence = db.Column(db.Float, nullable=False)

# -------------------------------------------------------------
# 3. SYSTEM LOGS MODEL
# -------------------------------------------------------------
class SystemLog(db.Model):
    __tablename__ = 'system_logs'

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    log_level = db.Column(db.String(20), nullable=False)
    logger_name = db.Column(db.String(80), nullable=False)
    message = db.Column(db.Text, nullable=False)

# -------------------------------------------------------------
# 4. REPORTS GENERATED MODEL
# -------------------------------------------------------------
class Report(db.Model):
    __tablename__ = 'reports'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    filename = db.Column(db.String(150), nullable=False)
    report_type = db.Column(db.String(20), nullable=False)  # 'PDF', 'CSV', 'EXCEL'
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

# -------------------------------------------------------------
# 5. SYSTEM SETTINGS MODEL
# -------------------------------------------------------------
class SystemSetting(db.Model):
    __tablename__ = 'system_settings'

    id = db.Column(db.Integer, primary_key=True)
    setting_key = db.Column(db.String(50), unique=True, nullable=False)
    setting_value = db.Column(db.String(255), nullable=False)
