import os
import joblib
import numpy as np
import pandas as pd
from config import Config

class FeaturePipeline:
    def __init__(self):
        self.scaler = None
        self.outlier_bounds = {}
        self.medians = {}

    def fit(self, df):
        """Fit preprocessing transformers, finding medians, outlier bounds, and scaler."""
        # 1. Store Medians for Imputation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            med_val = df[col].median()
            self.medians[col] = float(med_val) if not pd.isna(med_val) else 0.0

        # 2. Find IQR Outlier Boundaries
        # We find boundaries but don't clip yet, we do it in transform
        for col in Config.FEATURES:
            if col in numeric_cols and col not in ['Latitude', 'Longitude', 'Season', 'Month']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                self.outlier_bounds[col] = (float(Q1 - 1.5 * IQR), float(Q3 + 1.5 * IQR))

        # 3. Transform data to fit the StandardScaler
        df_transformed = self.transform(df.copy(), is_training=True)
        
        # 4. Fit StandardScaler
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X = df_transformed[Config.FEATURES]
        self.scaler.fit(X)

    def transform(self, df, is_training=False):
        """Preprocesses input dataframe according to fitted parameters."""
        df_out = df.copy()

        # 1. Impute Missing Values
        for col in df_out.columns:
            if col in self.medians:
                df_out[col] = df_out[col].fillna(self.medians[col])
            elif df_out[col].isnull().any():
                # Categorical fallback
                mode_val = df_out[col].mode()[0] if not df_out[col].mode().empty else 'Unknown'
                df_out[col] = df_out[col].fillna(mode_val)

        # 2. Encode Categoricals (Season, Month)
        if 'Season' in df_out.columns:
            df_out['Season'] = df_out['Season'].map(lambda x: Config.SEASON_MAPPING.get(x, 0))
        if 'Month' in df_out.columns:
            df_out['Month'] = df_out['Month'].map(lambda x: Config.MONTH_MAPPING.get(x, 1))

        # 3. Outlier Clipping
        for col, bounds in self.outlier_bounds.items():
            if col in df_out.columns:
                lower, upper = bounds
                df_out[col] = np.clip(df_out[col], lower, upper)

        # 4. Standard Scaling (if scaler exists)
        if self.scaler is not None and not is_training:
            X = df_out[Config.FEATURES]
            X_scaled = self.scaler.transform(X)
            # Recreate scaled DataFrame
            df_scaled = pd.DataFrame(X_scaled, columns=Config.FEATURES)
            # Retain non-feature metadata if needed, but return scaled features
            return df_scaled

        return df_out

    def save(self):
        """Saves fitted artifacts to the model folder."""
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
        # Save pipeline parameters
        joblib.dump(self.scaler, Config.SCALER_PATH)
        # Save other properties (outliers, medians) into feature_names
        pipeline_meta = {
            'outlier_bounds': self.outlier_bounds,
            'medians': self.medians,
            'features': Config.FEATURES
        }
        joblib.dump(pipeline_meta, Config.FEATURE_NAMES_PATH)
        print(f"Feature engineering artifacts saved successfully.")

    def load(self):
        """Loads fitted artifacts from disk."""
        if not os.path.exists(Config.SCALER_PATH) or not os.path.exists(Config.FEATURE_NAMES_PATH):
            return False
        
        self.scaler = joblib.load(Config.SCALER_PATH)
        meta = joblib.load(Config.FEATURE_NAMES_PATH)
        self.outlier_bounds = meta.get('outlier_bounds', {})
        self.medians = meta.get('medians', {})
        return True
