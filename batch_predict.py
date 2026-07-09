import os
import pandas as pd
import logging
from datetime import datetime
from config import Config
from prediction import FloodPredictor
from database import db
from models import Prediction

logger = logging.getLogger("batch_predict")

class BatchInferenceEngine:
    def __init__(self):
        self.predictor = FloodPredictor()

    def process_file(self, filepath, user_id=None, user_name='System Batch', location_prefix='Batch Import'):
        """
        Reads CSV/Excel from path, runs model inference on each row,
        logs entries in database, and returns the path to the output reports file.
        """
        logger.info(f"Processing batch file: {filepath}")
        
        # Load file
        ext = os.path.splitext(filepath)[1].lower()
        try:
            if ext == '.xlsx' or ext == '.xls':
                df = pd.read_excel(filepath)
            else:
                df = pd.read_csv(filepath)
        except Exception as e:
            logger.error(f"Error reading batch file: {e}")
            raise ValueError("Invalid file format. Could not parse data sheet.")

        # Check columns
        required_cols = [
            'Annual_Rainfall', 'Monthly_Rainfall', 'Temperature', 'Humidity', 
            'Pressure', 'Cloud_Cover', 'Wind_Speed', 'River_Water_Level', 
            'Ground_Water_Level', 'Visibility', 'Season', 'Month'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns in CSV: {missing_cols}")
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

        # Autocomplete Latitude / Longitude if missing
        if 'Latitude' not in df.columns:
            df['Latitude'] = 0.0
        if 'Longitude' not in df.columns:
            df['Longitude'] = 0.0

        # Output predictions lists
        predictions = []
        probabilities = []
        risk_levels = []
        confidences = []

        # Ensure predictor is loaded
        if self.predictor.model is None:
            self.predictor.load_artifacts()

        # Iterate rows
        for idx, row in df.iterrows():
            try:
                # Build raw input dict
                input_data = {
                    'Annual_Rainfall': float(row['Annual_Rainfall']),
                    'Monthly_Rainfall': float(row['Monthly_Rainfall']),
                    'Temperature': float(row['Temperature']),
                    'Humidity': float(row['Humidity']),
                    'Pressure': float(row['Pressure']),
                    'Cloud_Cover': float(row['Cloud_Cover']),
                    'Wind_Speed': float(row['Wind_Speed']),
                    'River_Water_Level': float(row['River_Water_Level']),
                    'Ground_Water_Level': float(row['Ground_Water_Level']),
                    'Visibility': float(row['Visibility']),
                    'Latitude': float(row['Latitude']),
                    'Longitude': float(row['Longitude']),
                    'Season': str(row['Season']).strip(),
                    'Month': str(row['Month']).strip()
                }

                # Predict
                res = self.predictor.predict(input_data)
                
                predictions.append(res['prediction'])
                probabilities.append(res['probability'])
                risk_levels.append(res['risk_level'])
                confidences.append(res['confidence'])

                # Log to DB if user_id is provided
                if user_id is not None:
                    db_record = Prediction(
                        user_id=user_id,
                        user_name=user_name,
                        location=f"{location_prefix} [Row {idx+1}]",
                        latitude=input_data['Latitude'],
                        longitude=input_data['Longitude'],
                        annual_rainfall=input_data['Annual_Rainfall'],
                        monthly_rainfall=input_data['Monthly_Rainfall'],
                        temperature=input_data['Temperature'],
                        humidity=input_data['Humidity'],
                        pressure=input_data['Pressure'],
                        cloud_cover=input_data['Cloud_Cover'],
                        wind_speed=input_data['Wind_Speed'],
                        river_water_level=input_data['River_Water_Level'],
                        ground_water_level=input_data['Ground_Water_Level'],
                        visibility=input_data['Visibility'],
                        season=input_data['Season'],
                        month=input_data['Month'],
                        prediction=res['prediction'],
                        probability=res['probability'],
                        risk_level=res['risk_level'],
                        confidence=res['confidence']
                    )
                    db.session.add(db_record)

            except Exception as e:
                logger.warning(f"Error predicting row {idx+1}: {e}")
                predictions.append(-1)
                probabilities.append(0.0)
                risk_levels.append("Error")
                confidences.append(0.0)

        # Save DB transaction in batch
        if user_id is not None:
            db.session.commit()

        # Add predictions to DataFrame
        df['Flood_Risk_Probability'] = probabilities
        df['Predicted_Risk_Level'] = risk_levels
        df['Model_Confidence'] = confidences
        df['Flood_Target_Prediction'] = predictions

        # Save outputs report file
        os.makedirs(Config.REPORT_FOLDER, exist_ok=True)
        report_filename = f"Batch_Report_{datetime.now().strftime('%Y%m%d%H%M%S')}{ext}"
        report_filepath = os.path.join(Config.REPORT_FOLDER, report_filename)

        if ext == '.xlsx' or ext == '.xls':
            df.to_excel(report_filepath, index=False)
        else:
            df.to_csv(report_filepath, index=False)

        logger.info(f"Batch processing completed. Output saved: {report_filepath}")
        return report_filename
