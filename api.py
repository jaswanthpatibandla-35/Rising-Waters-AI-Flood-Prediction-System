import os
import logging
from flask import Blueprint, jsonify, request, url_for
from config import Config
from database import db
from models import Prediction
from prediction import FloodPredictor
from batch_predict import BatchInferenceEngine

logger = logging.getLogger("api")
api_blueprint = Blueprint('api', __name__)
predictor = FloodPredictor()
batch_engine = BatchInferenceEngine()

# Helper to load metrics
def load_metrics_details():
    if os.path.exists(Config.MODEL_METRICS_PATH):
        try:
            with open(Config.MODEL_METRICS_PATH, 'r') as f:
                import json
                return json.load(f)
        except Exception:
            pass
    return {'best_model_name': 'Unknown', 'metrics': {}}

# -------------------------------------------------------------
# 1. POST /api/predict
# -------------------------------------------------------------
@api_blueprint.route('/predict', methods=['POST'])
def api_predict():
    """
    JSON Inference Endpoint.
    Expects JSON body matching Config.FEATURES.
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No input parameters provided.'}), 400

    required_fields = [
        'Annual_Rainfall', 'Monthly_Rainfall', 'Temperature', 'Humidity', 
        'Pressure', 'Cloud_Cover', 'Wind_Speed', 'River_Water_Level', 
        'Ground_Water_Level', 'Visibility', 'Season', 'Month', 
        'user_name', 'location'
    ]

    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({'error': f"Missing fields: {', '.join(missing)}"}), 400

    try:
        # Perform Inference
        res = predictor.predict(data)
        
        # Save to database
        record = Prediction(
            user_name=data['user_name'],
            location=data['location'],
            latitude=float(data.get('latitude', 0.0)),
            longitude=float(data.get('longitude', 0.0)),
            annual_rainfall=float(data['Annual_Rainfall']),
            monthly_rainfall=float(data['Monthly_Rainfall']),
            temperature=float(data['Temperature']),
            humidity=float(data['Humidity']),
            pressure=float(data['Pressure']),
            cloud_cover=float(data['Cloud_Cover']),
            wind_speed=float(data['Wind_Speed']),
            river_water_level=float(data['River_Water_Level']),
            ground_water_level=float(data['Ground_Water_Level']),
            visibility=float(data['Visibility']),
            season=data['Season'],
            month=data['Month'],
            prediction=res['prediction'],
            probability=res['probability'],
            risk_level=res['risk_level'],
            confidence=res['confidence']
        )
        
        db.session.add(record)
        db.session.commit()

        # Build Response
        return jsonify({
            'success': True,
            'prediction_id': record.id,
            'prediction': res['prediction'],
            'probability': res['probability'],
            'risk_level': res['risk_level'],
            'confidence': res['confidence'],
            'explanation': res['contributions']
        })

    except Exception as e:
        logger.error(f"API predict exception: {e}")
        return jsonify({'error': str(e)}), 500

# -------------------------------------------------------------
# 2. POST /api/batch_predict
# -------------------------------------------------------------
@api_blueprint.route('/batch_predict', methods=['POST'])
def api_batch_predict():
    """Processes uploaded CSV files and returns json report links."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file element in request.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400

    if not (file.filename.endswith('.csv') or file.filename.endswith('.xlsx') or file.filename.endswith('.xls')):
        return jsonify({'error': 'Unsupported file format. Upload CSV or Excel.'}), 400

    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    temp_path = os.path.join(Config.UPLOAD_FOLDER, file.filename)
    file.save(temp_path)

    try:
        report_filename = batch_engine.process_file(
            temp_path, 
            user_id=None, 
            user_name='API Batch Client', 
            location_prefix='API Batch'
        )
        # Cleanup upload
        if os.path.exists(temp_path):
            os.remove(temp_path)

        report_url = url_for('static', filename=f"reports/{report_filename}", _external=True)

        return jsonify({
            'success': True,
            'message': 'Batch predictions processed successfully.',
            'report_file': report_filename,
            'download_url': report_url
        })
    except Exception as e:
        logger.error(f"API batch predict exception: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(e)}), 500

# -------------------------------------------------------------
# 3. GET /api/history
# -------------------------------------------------------------
@api_blueprint.route('/history', methods=['GET'])
def api_history():
    """Returns prediction log list from database."""
    try:
        records = Prediction.query.order_by(Prediction.id.desc()).all()
        history_list = []
        for r in records:
            history_list.append({
                'id': r.id,
                'date': r.date,
                'user_name': r.user_name,
                'location': r.location,
                'probability': r.probability,
                'risk_level': r.risk_level,
                'confidence': r.confidence
            })
        return jsonify(history_list)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -------------------------------------------------------------
# 4. GET /api/analytics
# -------------------------------------------------------------
@api_blueprint.route('/analytics', methods=['GET'])
def api_analytics():
    """Returns aggregated stats for system reporting dashboards."""
    try:
        records = Prediction.query.all()
        total = len(records)
        
        low_c = sum(1 for r in records if r.risk_level == 'Low')
        med_c = sum(1 for r in records if r.risk_level == 'Medium')
        high_c = sum(1 for r in records if r.risk_level == 'High')
        
        avg_prob = sum(r.probability for r in records) / total if total > 0 else 0.0
        avg_conf = sum(r.confidence for r in records) / total if total > 0 else 0.0

        return jsonify({
            'total_predictions': total,
            'risk_level_counts': {
                'Low': low_c,
                'Medium': med_c,
                'High': high_c
            },
            'averages': {
                'probability_pct': round(avg_prob, 2),
                'confidence_pct': round(avg_conf, 2)
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -------------------------------------------------------------
# 5. GET /api/model_info
# -------------------------------------------------------------
@api_blueprint.route('/model_info', methods=['GET'])
def api_model_info():
    """Returns model metrics parameters."""
    info = load_metrics_details()
    return jsonify({
        'best_model_name': info['best_model_name'],
        'model_version': Config.MODEL_VERSION,
        'features': Config.FEATURES,
        'metrics': info['metrics']
    })
