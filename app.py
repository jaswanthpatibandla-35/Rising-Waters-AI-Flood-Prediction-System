import os
import json
import logging
from datetime import datetime
from functools import wraps

from flask import (Flask, render_template, request, redirect, 
                   url_for, flash, send_from_directory, jsonify)
from flask_login import LoginManager, login_user, logout_user, login_required, current_user

from config import Config
from database import db
from models import User, Prediction, SystemLog, Report, SystemSetting
from logger import get_logger, DBLoggingHandler
from prediction import FloodPredictor
from batch_predict import BatchInferenceEngine
from utils import (get_weather_data, generate_pdf_report, 
                   export_predictions_data, backup_database, restore_database)
from api import api_blueprint
from forms import (LoginForm, RegistrationForm, PredictForm, 
                   ProfileForm, ForgotPasswordForm, ResetPasswordForm)

# -------------------------------------------------------------
# 1. SETUP LOGGER
# -------------------------------------------------------------
logger = get_logger("app")
logger.info("Initializing Rising Waters Web Platform...")

# -------------------------------------------------------------
# 2. FLASK SERVER SETUP
# -------------------------------------------------------------
app = Flask(__name__)
app.config.from_object(Config)

# Register API Blueprint
app.register_blueprint(api_blueprint, url_prefix='/api')

# Initialize DB with App
db.init_app(app)

# Configure Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'warning'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Ensure required directories exist
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(Config.REPORT_FOLDER, exist_ok=True)
os.makedirs(Config.BACKUP_DIR, exist_ok=True)
os.makedirs(Config.LOG_FOLDER, exist_ok=True)

# -------------------------------------------------------------
# 3. DB SEEDING & DYNAMIC LOGGER ATTACHMENT
# -------------------------------------------------------------
with app.app_context():
    db.create_all()
    
    # Attach database logging handler
    db_handler = DBLoggingHandler(db.session, SystemLog)
    db_handler.setLevel(logging.WARNING)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    db_handler.setFormatter(formatter)
    logging.getLogger().addHandler(db_handler)
    logger.info("Database logging connected.")

    # Seed Admin user if database is empty
    admin_exists = User.query.filter_by(is_admin=True).first()
    if not admin_exists:
        admin = User(
            username='admin',
            email='admin@risingwaters.ml',
            is_admin=True,
            is_verified=True
        )
        admin.set_password('admin123')
        db.session.add(admin)
        
        # Seed basic system settings
        default_settings = [
            SystemSetting(setting_key='api_autofill', setting_value='enabled'),
            SystemSetting(setting_key='system_alerts', setting_value='active'),
            SystemSetting(setting_key='default_risk_threshold', setting_value='0.5')
        ]
        for s in default_settings:
            db.session.add(s)
            
        db.session.commit()
        logger.info("Admin account seeded successfully (admin/admin123).")

# Instantiate Inference Engines
predictor = FloodPredictor()
batch_engine = BatchInferenceEngine()

# -------------------------------------------------------------
# 4. CUSTOM SECURITY DECORATORS
# -------------------------------------------------------------
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            flash("Authentication required. Please login first.", "danger")
            return redirect(url_for('login'))
        if not current_user.is_admin:
            flash("Access denied. Administrator privileges required.", "danger")
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

# Context processor to inject global variables
@app.context_processor
def inject_global_vars():
    best_model = "Not Trained"
    if os.path.exists(Config.MODEL_METRICS_PATH):
        try:
            with open(Config.MODEL_METRICS_PATH, 'r') as f:
                best_model = json.load(f).get('best_model_name', 'Not Trained')
        except Exception:
            pass
    
    theme = 'light'
    if current_user.is_authenticated:
        theme = current_user.theme_preference

    return {
        'best_model_name': best_model,
        'global_theme': theme,
        'current_year': datetime.now().year
    }

# -------------------------------------------------------------
# 5. AUTHENTICATION VIEWS (Flask-Login & Flask-WTF)
# -------------------------------------------------------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    form = RegistrationForm()
    if form.validate_on_submit():
        username = form.username.data.strip()
        email = form.email.data.strip()
        password = form.password.data

        user_exists = User.query.filter((User.username == username) | (User.email == email)).first()
        if user_exists:
            flash("Username or Email already registered.", "danger")
            return redirect(url_for('register'))

        try:
            new_user = User(username=username, email=email)
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()
            
            flash("Registration successful! You can now log in.", "success")
            return redirect(url_for('login'))
        except Exception as e:
            logger.error(f"User registration error: {e}")
            flash("An error occurred during account creation.", "danger")
            return redirect(url_for('register'))

    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data.strip()
        password = form.password.data
        remember = form.remember_me.data

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user, remember=remember)
            flash(f"Welcome back, {user.username}!", "success")
            # Handle redirection next parameter if present
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        else:
            flash("Invalid username or password.", "danger")
            return redirect(url_for('login'))

    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("You have successfully logged out.", "info")
    return redirect(url_for('home'))

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    form = ForgotPasswordForm()
    if form.validate_on_submit():
        email = form.email.data.strip()
        user = User.query.filter_by(email=email).first()
        if user:
            user.reset_token = os.urandom(16).hex()
            db.session.commit()
            flash(f"A password reset link has been dispatched to {email}. (Mock: Token set to {user.reset_token})", "info")
            return redirect(url_for('login'))
        else:
            flash("No account associated with that email address.", "danger")
    return render_template('forgot_password.html', form=form)

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    user = User.query.filter_by(reset_token=token).first_or_404()
    form = ResetPasswordForm()
    if form.validate_on_submit():
        password = form.password.data
        user.set_password(password)
        user.reset_token = None
        db.session.commit()
        flash("Your password has been successfully reset.", "success")
        return redirect(url_for('login'))
    return render_template('reset_password.html', form=form, token=token)

# -------------------------------------------------------------
# 6. WEB APPLICATION VIEWS
# -------------------------------------------------------------
@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/dashboard')
@login_required
def dashboard():
    records = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.id.desc()).all()
    total_preds = len(records)
    
    high_risk = sum(1 for r in records if r.risk_level == 'High')
    med_risk = sum(1 for r in records if r.risk_level == 'Medium')
    low_risk = sum(1 for r in records if r.risk_level == 'Low')
    
    avg_conf = 0.0
    if total_preds > 0:
        avg_conf = round(sum(r.confidence for r in records) / total_preds, 1)

    return render_template(
        'dashboard.html',
        total_predictions=total_preds,
        high_risk_count=high_risk,
        med_risk_count=med_risk,
        low_risk_count=low_risk,
        avg_confidence=avg_conf,
        recent_predictions=records[:5]
    )

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    form = PredictForm()
    
    # Check if weather lookup button was clicked (autofill triggered via POST)
    city_lookup = request.form.get('city_lookup', '').strip()
    if request.method == 'POST' and city_lookup:
        weather = get_weather_data(city_lookup)
        if weather.get('success'):
            # Pre-populate WTForm fields with API values
            form.location.data = weather['city']
            form.Latitude.data = weather['latitude']
            form.Longitude.data = weather['longitude']
            form.Temperature.data = weather['temperature']
            form.Humidity.data = weather['humidity']
            form.Pressure.data = weather['pressure']
            form.Cloud_Cover.data = weather['cloud_cover']
            form.Wind_Speed.data = weather['wind_speed']
            form.Visibility.data = weather['visibility']
            flash(f"Fetched weather telemetry for {weather['city']}.", "success")
            return render_template('predict.html', form=form, weather=weather)
        else:
            flash("Could not fetch meteorological logs for that city.", "warning")
            return render_template('predict.html', form=form)

    # Main predict action
    if form.validate_on_submit():
        try:
            # Build input dictionary
            input_dict = {
                'Annual_Rainfall': float(form.Annual_Rainfall.data),
                'Monthly_Rainfall': float(form.Monthly_Rainfall.data),
                'Temperature': float(form.Temperature.data),
                'Humidity': float(form.Humidity.data),
                'Pressure': float(form.Pressure.data),
                'Cloud_Cover': float(form.Cloud_Cover.data),
                'Wind_Speed': float(form.Wind_Speed.data),
                'River_Water_Level': float(form.River_Water_Level.data),
                'Ground_Water_Level': float(form.Ground_Water_Level.data),
                'Visibility': float(form.Visibility.data),
                'Latitude': float(form.Latitude.data),
                'Longitude': float(form.Longitude.data),
                'Season': form.Season.data,
                'Month': form.Month.data
            }
            
            district = form.District.data.strip() if form.District.data else 'Unknown'
            state = form.State.data.strip() if form.State.data else 'Unknown'

            # Predict
            res = predictor.predict(input_dict)
            
            # Save Record
            record = Prediction(
                user_id=current_user.id,
                user_name=current_user.username,
                location=form.location.data.strip(),
                latitude=input_dict['Latitude'],
                longitude=input_dict['Longitude'],
                annual_rainfall=input_dict['Annual_Rainfall'],
                monthly_rainfall=input_dict['Monthly_Rainfall'],
                temperature=input_dict['Temperature'],
                humidity=input_dict['Humidity'],
                pressure=input_dict['Pressure'],
                cloud_cover=input_dict['Cloud_Cover'],
                wind_speed=input_dict['Wind_Speed'],
                river_water_level=input_dict['River_Water_Level'],
                ground_water_level=input_dict['Ground_Water_Level'],
                visibility=input_dict['Visibility'],
                season=input_dict['Season'],
                month=input_dict['Month'],
                district=district,
                state=state,
                prediction=res['prediction'],
                probability=res['probability'],
                risk_level=res['risk_level'],
                confidence=res['confidence']
            )
            db.session.add(record)
            db.session.commit()

            flash("Risk analysis model executed successfully.", "success")
            return redirect(url_for('result', record_id=record.id))

        except FileNotFoundError:
            flash("ML model files are missing. Please build models in train_model.py.", "danger")
        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            flash("Error during model evaluation. Verify input formatting.", "danger")

    return render_template('predict.html', form=form)

@app.route('/result/<int:record_id>')
@login_required
def result(record_id):
    record = Prediction.query.filter_by(id=record_id, user_id=current_user.id).first_or_404()
    
    explanation = predictor._explain_prediction(
        None, 
        [record.annual_rainfall, record.monthly_rainfall, record.temperature, record.humidity, 
         record.pressure, record.cloud_cover, record.wind_speed, record.river_water_level, 
         record.ground_water_level, record.visibility, record.latitude, record.longitude, 0.0, 0.0]
    )
    
    return render_template('result.html', record=record, explanation=explanation)

@app.route('/result/<int:record_id>/pdf')
@login_required
def download_pdf(record_id):
    record = Prediction.query.filter_by(id=record_id, user_id=current_user.id).first_or_404()
    
    best_model = "Stacking Classifier"
    if os.path.exists(Config.MODEL_METRICS_PATH):
        try:
            with open(Config.MODEL_METRICS_PATH, 'r') as f:
                best_model = json.load(f).get('best_model_name', 'Stacking Classifier')
        except Exception:
            pass
            
    try:
        filename = generate_pdf_report(record, best_model)
        db_report = Report(user_id=current_user.id, filename=filename, report_type='PDF')
        db.session.add(db_report)
        db.session.commit()
        return send_from_directory(Config.REPORT_FOLDER, filename, as_attachment=True)
    except Exception as e:
        logger.error(f"Error generating PDF download: {e}")
        flash("Could not compile report PDF.", "danger")
        return redirect(url_for('result', record_id=record_id))

@app.route('/history')
@login_required
def history():
    records = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.id.desc()).all()
    
    # Check export actions
    export_format = request.args.get('export', '').upper()
    if export_format in ['CSV', 'EXCEL']:
        try:
            filename = export_predictions_data(records, export_format)
            db_report = Report(user_id=current_user.id, filename=filename, report_type=export_format)
            db.session.add(db_report)
            db.session.commit()
            return send_from_directory(Config.REPORT_FOLDER, filename, as_attachment=True)
        except Exception as e:
            logger.error(f"Export history error: {e}")
            flash("Failed to export dataset records.", "danger")

    return render_template('history.html', records=records)

@app.route('/delete_record/<int:record_id>', methods=['POST'])
@login_required
def delete_record(record_id):
    record = Prediction.query.filter_by(id=record_id, user_id=current_user.id).first_or_404()
    try:
        db.session.delete(record)
        db.session.commit()
        flash("Record removed from history logs.", "success")
    except Exception as e:
        logger.error(f"Delete record error: {e}")
        flash("An error occurred while deleting the record.", "danger")
    return redirect(url_for('history'))

@app.route('/batch_predict', methods=['GET', 'POST'])
@login_required
def batch_predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash("No file part found.", "danger")
            return redirect(request.url)
            
        file = request.files['file']
        if file.filename == '':
            flash("No file selected.", "danger")
            return redirect(request.url)

        if not (file.filename.endswith('.csv') or file.filename.endswith('.xlsx') or file.filename.endswith('.xls')):
            flash("Unsupported file format. Please upload CSV or Excel spreadsheet.", "danger")
            return redirect(request.url)

        # Save temporarily in uploads
        temp_path = os.path.join(Config.UPLOAD_FOLDER, file.filename)
        file.save(temp_path)

        try:
            report_name = batch_engine.process_file(
                temp_path, 
                user_id=current_user.id, 
                user_name=current_user.username, 
                location_prefix="Batch Upload"
            )
            # Log report
            db_report = Report(user_id=current_user.id, filename=report_name, report_type='BATCH_LOG')
            db.session.add(db_report)
            db.session.commit()
            
            # Remove temp upload
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            flash("Batch predictions processed successfully!", "success")
            return render_template('batch_predict.html', report_name=report_name)

        except Exception as e:
            logger.error(f"Batch inference exception: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            flash(f"Batch execution failed: {str(e)}", "danger")
            return redirect(request.url)

    return render_template('batch_predict.html')

@app.route('/analytics')
@login_required
def analytics():
    records = Prediction.query.filter_by(user_id=current_user.id).all()
    
    # Calculate stats
    low_c = sum(1 for r in records if r.risk_level == 'Low')
    med_c = sum(1 for r in records if r.risk_level == 'Medium')
    high_c = sum(1 for r in records if r.risk_level == 'High')
    
    timeline_records = records[-15:]
    dates = [r.date.split(' ')[0] for r in timeline_records]
    probs = [r.probability for r in timeline_records]
    confs = [r.confidence for r in timeline_records]

    return render_template(
        'analytics.html',
        low_count=low_c,
        med_count=med_c,
        high_count=high_c,
        dates=dates,
        probabilities=probs,
        confidences=confs
    )

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        logger.warning(f"Contact form message from {name} ({email}): {message}")
        flash("Your message has been logged. Thank you!", "success")
        return redirect(url_for('contact'))
    return render_template('contact.html')

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    form = ProfileForm()
    user = User.query.get_or_404(current_user.id)
    
    # Pre-populate theme preference
    if request.method == 'GET':
        form.theme_preference.data = user.theme_preference

    if form.validate_on_submit():
        user.theme_preference = form.theme_preference.data
        db.session.commit()
        
        old_pass = form.old_password.data
        new_pass = form.new_password.data
        
        if old_pass and new_pass:
            if user.check_password(old_pass):
                user.set_password(new_pass)
                db.session.commit()
                flash("Password updated successfully.", "success")
            else:
                flash("Incorrect current password.", "danger")
        else:
            flash("Profile preferences updated successfully.", "success")
        return redirect(url_for('profile'))
        
    return render_template('profile.html', user=user, form=form)

# -------------------------------------------------------------
# 7. ADMINISTRATOR PANEL ROUTES
# -------------------------------------------------------------
@app.route('/admin')
@admin_required
def admin_panel():
    users = User.query.all()
    predictions = Prediction.query.order_by(Prediction.id.desc()).all()
    logs = SystemLog.query.order_by(SystemLog.id.desc()).limit(20).all()
    
    backups = []
    if os.path.exists(Config.BACKUP_DIR):
        backups = [f for f in os.listdir(Config.BACKUP_DIR) if f.endswith('.db')]

    return render_template(
        'admin.html',
        users=users,
        predictions=predictions,
        logs=logs,
        backups=backups
    )

@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
@admin_required
def delete_user(user_id):
    if user_id == current_user.id:
        flash("You cannot delete your own admin account.", "danger")
        return redirect(url_for('admin_panel'))
        
    user = User.query.get_or_404(user_id)
    try:
        db.session.delete(user)
        db.session.commit()
        flash(f"User '{user.username}' successfully deleted.", "success")
    except Exception as e:
        logger.error(f"Error deleting user ID {user_id}: {e}")
        flash("Failed to delete user.", "danger")
    return redirect(url_for('admin_panel'))

@app.route('/admin/delete_prediction/<int:pred_id>', methods=['POST'])
@admin_required
def delete_prediction(pred_id):
    record = Prediction.query.get_or_404(pred_id)
    try:
        db.session.delete(record)
        db.session.commit()
        flash("Prediction log record successfully deleted.", "success")
    except Exception as e:
        logger.error(f"Admin delete prediction error: {e}")
        flash("Failed to delete prediction record.", "danger")
    return redirect(url_for('admin_panel'))

@app.route('/admin/backup', methods=['POST'])
@admin_required
def trigger_backup():
    try:
        filename = backup_database()
        if filename:
            flash(f"Database backup created successfully: {filename}", "success")
        else:
            flash("Backup creation failed.", "danger")
    except Exception as e:
        logger.error(f"Backup creation error: {e}")
        flash("Backup failed.", "danger")
    return redirect(url_for('admin_panel'))

@app.route('/admin/restore/<string:filename>', methods=['POST'])
@admin_required
def trigger_restore(filename):
    try:
        success = restore_database(filename)
        if success:
            flash(f"Database restored successfully from: {filename}", "success")
        else:
            flash("Database restoration failed. File missing.", "danger")
    except Exception as e:
        logger.error(f"Database restore error: {e}")
        flash("Database restoration failed.", "danger")
    return redirect(url_for('admin_panel'))

# -------------------------------------------------------------
# 8. ERROR HANDLERS
# -------------------------------------------------------------
@app.errorhandler(404)
def page_not_found(e):
    logger.warning(f"Route 404: {request.url}")
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    logger.error(f"Route 500: {e}", exc_info=True)
    return render_template('500.html'), 500

# -------------------------------------------------------------
# 9. RUN APPLICATION
# -------------------------------------------------------------
if __name__ == '__main__':
    print("====================================================")
    print("RISING WATERS - ENTERPRISE EDITION FLASK SERVER")
    print(f"Running locally at http://{Config.HOST}:{Config.PORT}")
    print("====================================================")
    
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )
