import os
import joblib
import numpy as np
import pandas as pd
import logging
from config import Config

logger = logging.getLogger("prediction")

class FloodPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.pipeline_meta = None
        self.label_encoder = None
        self.load_artifacts()

    def load_artifacts(self):
        """Loads optimal models, scalers, and label encoder maps from disk."""
        if not all(os.path.exists(p) for p in [Config.MODEL_PATH, Config.SCALER_PATH, 
                                               Config.FEATURE_NAMES_PATH, Config.LABEL_ENCODER_PATH]):
            logger.warning("Pipeline binaries missing. Please execute train_model.py first.")
            return False

        try:
            self.model = joblib.load(Config.MODEL_PATH)
            self.scaler = joblib.load(Config.SCALER_PATH)
            self.pipeline_meta = joblib.load(Config.FEATURE_NAMES_PATH)
            self.label_encoder = joblib.load(Config.LABEL_ENCODER_PATH)
            logger.info("Successfully loaded all machine learning artifacts.")
            return True
        except Exception as e:
            logger.error(f"Error loading artifacts: {e}", exc_info=True)
            return False

    def predict(self, raw_input):
        """
        Calculates flood probability, risk level, confidence score, and feature contributions.
        raw_input format:
        {
            'Annual_Rainfall': float,
            'Monthly_Rainfall': float,
            'Temperature': float,
            'Humidity': float,
            'Pressure': float,
            'Cloud_Cover': float,
            'Wind_Speed': float,
            'River_Water_Level': float,
            'Ground_Water_Level': float,
            'Visibility': float,
            'Latitude': float,
            'Longitude': float,
            'Season': str,
            'Month': str
        }
        """
        if self.model is None or self.scaler is None:
            success = self.load_artifacts()
            if not success:
                raise FileNotFoundError("Classifier artifacts not loaded. Run train_model.py.")

        try:
            # 1. Preprocess & Map Categoricals
            season_code = Config.SEASON_MAPPING.get(raw_input['Season'], 0)
            month_code = Config.MONTH_MAPPING.get(raw_input['Month'], 1)

            # Build DataFrame following exact feature sequence
            df_in = pd.DataFrame([{
                'Annual_Rainfall': float(raw_input['Annual_Rainfall']),
                'Monthly_Rainfall': float(raw_input['Monthly_Rainfall']),
                'Temperature': float(raw_input['Temperature']),
                'Humidity': float(raw_input['Humidity']),
                'Pressure': float(raw_input['Pressure']),
                'Cloud_Cover': float(raw_input['Cloud_Cover']),
                'Wind_Speed': float(raw_input['Wind_Speed']),
                'River_Water_Level': float(raw_input['River_Water_Level']),
                'Ground_Water_Level': float(raw_input['Ground_Water_Level']),
                'Visibility': float(raw_input['Visibility']),
                'Latitude': float(raw_input['Latitude']),
                'Longitude': float(raw_input['Longitude']),
                'Season': season_code,
                'Month': month_code
            }])

            # Apply Outlier Clipping based on metadata
            outlier_bounds = self.pipeline_meta.get('outlier_bounds', {})
            for col, bounds in outlier_bounds.items():
                lower, upper = bounds
                df_in[col] = np.clip(df_in[col], lower, upper)

            # Scale Inputs
            X_scaled = self.scaler.transform(df_in[Config.FEATURES])

            # 2. Predict Probability & Class
            if hasattr(self.model, "predict_proba"):
                prob = self.model.predict_proba(X_scaled)[0][1]
            else:
                decision = self.model.decision_function(X_scaled)[0]
                prob = 1 / (1 + np.exp(-decision))

            pred_class = int(self.model.predict(X_scaled)[0])
            prob_pct = round(float(prob) * 100, 2)

            # Confidence = distance from 50% decision boundary (scaled to 0-100%)
            # e.g., if probability is 95%, confidence is 90%. If prob is 50%, confidence is 0%
            confidence = round(abs(prob - 0.5) * 2 * 100, 1)
            # Give a base confidence floor of 50% for standard classifications
            confidence = max(50.0, confidence)

            # Risk Classification
            if prob_pct < 30.0:
                risk_level = "Low"
            elif prob_pct < 70.0:
                risk_level = "Medium"
            else:
                risk_level = "High"

            # 3. Calculate Feature Contributions (SHAP/LIME approximation)
            contributions = self._explain_prediction(df_in.iloc[0], X_scaled[0])

            return {
                'prediction': pred_class,
                'probability': prob_pct,
                'risk_level': risk_level,
                'confidence': confidence,
                'contributions': contributions
            }

        except Exception as e:
            logger.error(f"Error during prediction: {e}", exc_info=True)
            raise e

    def _explain_prediction(self, raw_series, scaled_row):
        """
        Estimates the contribution of each feature to the final prediction.
        This provides a fast, stable, and compilation-free alternative to SHAP/LIME.
        """
        contributions = {}
        try:
            # We approximate feature impact using: (Scaled Value * Feature Weight proxy)
            # Fetch model weights or importances
            if hasattr(self.model, 'feature_importances_'):
                weights = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                weights = self.model.coef_[0]
            else:
                # Fallback flat weights
                weights = np.ones(len(Config.FEATURES)) / len(Config.FEATURES)

            # Normalize weights to sum to 1.0
            total_w = np.sum(np.abs(weights))
            if total_w > 0:
                weights = weights / total_w

            # Contribution = Scaled value * normalized weight
            # Large positive scaled value * positive weight = strong positive contributor
            for idx, col in enumerate(Config.FEATURES):
                val_scaled = scaled_row[idx]
                contrib_score = float(val_scaled * weights[idx])
                contributions[col] = round(contrib_score * 100, 2)

            # Sort by absolute contribution score
            sorted_contrib = dict(sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True))
            return sorted_contrib
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            # Fallback explanation
            return {col: 0.0 for col in Config.FEATURES}
